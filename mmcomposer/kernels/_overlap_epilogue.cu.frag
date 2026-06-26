                // ── Overlap epilogue drain (TMEM → SMEM → GMEM), shared ─────
                // Spliced into the overlap epilogue-warp loop of every warp-spec
                // tier's drain marker, right after `trow` (the tier-specific TMEM
                // lane base) and `LDW` are in scope.  The skeleton supplies three
                // contract macros for the per-tier bits:
                //   EPI_OUT_ROW                  this CTA's GMEM row base
                //   EPI_OUT_COL_BASE             this CTA's GMEM column base
                //   signal_sync(buf)    release the drained TMEM buffer
                // EPILOGUE_TMA_PIPELINED picks the Paul-v6-style path:
                // chunk BN into STORE_N=64 columns, stage each chunk into one
                // of TMA_STORE_STAGES compact swizzled SMEM buffers, and
                // launch TMA stores.
                // EPILOGUE_SPLIT (constexpr) picks the two-pass half-BN writeback,
                // which stages one BN/2 column panel at a time (EPI_STAGE_COLS=BN/2)
                // so the epilogue SMEM shrinks enough for an extra K-loop stage.
                //
                // EPILOGUE_L1_NO_ALLOC (knob): the write-once C store bypasses L1
                // allocation (`st...L1::no_allocate`) so it doesn't evict A/B from
                // L1.  Measured win when the epilogue is exposed (low K), null at
                // high K — so it's a sweep knob, not always-on.
#if EPILOGUE_TMA_PIPELINED
                {
                    constexpr int LOADS_PER_CHUNK = STORE_N / 8;
                    constexpr int LOADS_PER_WARP = LOADS_PER_CHUNK / COL_GROUPS;
                    constexpr int NUM_CHUNKS = BN / STORE_N;
                    static_assert(STORE_N == 64, "pipelined TMA store assumes STORE_N=64");
                    static_assert(NUM_CHUNKS * STORE_N == BN, "BN must divide into STORE_N chunks");
                    static_assert(LOADS_PER_WARP * COL_GROUPS == LOADS_PER_CHUNK,
                                  "STORE_N/8 chunks must divide across column warp groups");
                    int store_stage = 0;

                    #pragma unroll
                    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                        float t[LOADS_PER_WARP][8];
                        #pragma unroll
                        for (int n = 0; n < LOADS_PER_WARP; n++) {
                            const int local_n = col_warp * LOADS_PER_WARP + n;
                            tcgen05_ld_32x32b_x8(trow + (uint32_t)(chunk * STORE_N + local_n * 8), t[n]);
                        }
                        tcgen05_wait_ld();

                        // The TMEM->reg load above doesn't touch the store buffer, so the
                        // free-store-slot wait below is deferred to just before the buffer write
                        // (and stays before the bar.sync so every warp observes the ew==0 wait).
#if SINGLE_TMEM_ACCUM
                        if (chunk == NUM_CHUNKS - 1)
                            tcgen05_fence_before_thread_sync();

                        if (ew == 0)
                            tma_wait_group<TMA_STORE_STAGES - 1>();

                        asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));

                        if (chunk == NUM_CHUNKS - 1) {
                            if (ew == 0 && elect_sync())
                                signal_sync(buf);
                        }
#else
                        if (chunk == NUM_CHUNKS - 1) {
                            tcgen05_fence_before_thread_sync();
                            if (ew == 0 && elect_sync())
                                signal_sync(buf);
                        }

                        if (ew == 0)
                            tma_wait_group<TMA_STORE_STAGES - 1>();

                        asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));
#endif

                        #pragma unroll
                        for (int n = 0; n < LOADS_PER_WARP; n++) {
                            __nv_bfloat162 pk[4];
                            #pragma unroll
                            for (int i = 0; i < 4; i++)
                                pk[i] = __floats2bfloat162_rn(t[n][2 * i], t[n][2 * i + 1]);
                            const int local_n = col_warp * LOADS_PER_WARP + n;
                            const int swizzled_n = local_n ^ (my_row & 7);
                            __nv_bfloat16* write_ptr =
                                C_store + store_stage * BM * STORE_N + my_row * STORE_N + swizzled_n * 8;
                            *reinterpret_cast<int4*>(write_ptr) = *reinterpret_cast<int4*>(pk);
                        }

                        __syncwarp();
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));

                        if (ew == 0 && elect_sync()) {
                            const uint32_t src = STORE_SMEM_BASE + store_stage * STORE_BUF_BYTES;
                            tma_2d_store(C_tmap_ptr, src,
                                         EPI_OUT_COL_BASE + chunk * STORE_N, EPI_OUT_ROW);
                            tma_commit_group();
                        }

#if TMA_STORE_STAGES == 1
                        store_stage = 0;
#elif TMA_STORE_STAGES == 2
                        store_stage ^= 1;
#else
                        store_stage = (store_stage + 1) % TMA_STORE_STAGES;
#endif
                    }
                }
#else
#if EPILOGUE_L1_NO_ALLOC
#define EPI_ST_I4(DST, VAL) do { int4 _v = (VAL); \
        asm volatile("st.relaxed.cta.global.L1::no_allocate.v4.b32 [%0], {%1,%2,%3,%4};" \
            :: "l"(DST), "r"(_v.x), "r"(_v.y), "r"(_v.z), "r"(_v.w) : "memory"); } while (0)
#else
#define EPI_ST_I4(DST, VAL) (*reinterpret_cast<int4*>(DST) = (VAL))
#endif
#if EPILOGUE_SPLIT
                {
                    static_assert((BN / 2) % 8 == 0, "split epilogue needs int4-aligned columns");
                    static_assert((BN / 2) % COL_GROUPS == 0,
                                  "split epilogue panel must divide across column warp groups");
                    constexpr int SPLIT_COLS_PER_WARP = (BN / 2) / COL_GROUPS;
                    static_assert(SPLIT_COLS_PER_WARP % LDW == 0,
                                  "split epilogue per-warp column span must divide by LDW");
                    #pragma unroll
                    for (int split = 0; split < 2; split++) {
                        const int split_base = split * EPI_STAGE_COLS;
                        const int panel_col_base = split_base + col_warp * SPLIT_COLS_PER_WARP;
                        #pragma unroll
                        for (int n = panel_col_base; n < panel_col_base + SPLIT_COLS_PER_WARP; n += LDW) {
                            float t[LDW];
#if TCGEN05_LD_WIDTH == 8
                            tcgen05_ld_32x32b_x8 (trow + (uint32_t)n, t);
#else
                            tcgen05_ld_32x32b_x16(trow + (uint32_t)n, t);
#endif
                            tcgen05_wait_ld();
                            __nv_bfloat162 pk[LDW / 2];
                            #pragma unroll
                            for (int i = 0; i < LDW / 2; i++)
                                pk[i] = __floats2bfloat162_rn(t[2 * i], t[2 * i + 1]);
                            #pragma unroll
                            for (int c = 0; c < LDW; c += 8)
                                *reinterpret_cast<int4*>(&C_sh[my_row][n - split_base + c]) =
                                    *reinterpret_cast<int4*>(&pk[c / 2]);
                        }
                        asm volatile("bar.sync 1, %0;" :: "n"(NUM_WARPS * 32));

                        // Once split 1 is staged, this tile's TMEM buffer is no
                        // longer needed.  Release before the split-1 GMEM store so
                        // the MMA warp can start the next tile earlier.
                        if (split == 1 && ew == 0 && elect_sync())
                            signal_sync(buf);

                        constexpr int CHUNKS = (BN / 2) / 8;
                        constexpr int STORES = BM * (BN / 2) / (EPI_THREADS * 8);
                        static_assert(STORES * EPI_THREADS * 8 == BM * (BN / 2),
                                      "split epilogue tile must divide across epilogue threads");
                        #pragma unroll
                        for (int s = 0; s < STORES; s++) {
                            const int flat = etid + s * EPI_THREADS;
                            const int row = flat / CHUNKS;
                            const int col = (flat % CHUNKS) * 8;
                            EPI_ST_I4(&C_ptr[(EPI_OUT_ROW + row) * N + EPI_OUT_COL_BASE + split_base + col],
                                      *reinterpret_cast<const int4*>(&C_sh[row][col]));
                        }
                        asm volatile("bar.sync 1, %0;" :: "n"(NUM_WARPS * 32));
                    }
                }
#else
                {
                    #pragma unroll
                    for (int n = col_base; n < col_base + COLS_PER_WARP; n += LDW) {
                        float t[LDW];
#if TCGEN05_LD_WIDTH == 8
                        tcgen05_ld_32x32b_x8 (trow + (uint32_t)n, t);
#else
                        tcgen05_ld_32x32b_x16(trow + (uint32_t)n, t);
#endif
                        tcgen05_wait_ld();
                        __nv_bfloat162 pk[LDW / 2];
                        #pragma unroll
                        for (int i = 0; i < LDW / 2; i++)
                            pk[i] = __floats2bfloat162_rn(t[2 * i], t[2 * i + 1]);
                        #pragma unroll
                        for (int c = 0; c < LDW; c += 8)
                            *reinterpret_cast<int4*>(&C_sh[my_row][n + c]) =
                                *reinterpret_cast<int4*>(&pk[c / 2]);
                    }
                    asm volatile("bar.sync 1, %0;" :: "n"(NUM_WARPS * 32));
                    if (ew == 0 && elect_sync())
                        signal_sync(buf);
                    constexpr int CHUNKS = BN / 8;
                    constexpr int STORES = BM * BN / (EPI_THREADS * 8);
                    #pragma unroll
                    for (int s = 0; s < STORES; s++) {
                        int flat = etid + s * EPI_THREADS;
                        int row = flat / CHUNKS;
                        int col = (flat % CHUNKS) * 8;
                        EPI_ST_I4(&C_ptr[(EPI_OUT_ROW + row) * N + EPI_OUT_COL_BASE + col],
                                  *reinterpret_cast<const int4*>(&C_sh[row][col]));
                    }
                    asm volatile("bar.sync 1, %0;" :: "n"(NUM_WARPS * 32));
                }
#endif
#undef EPI_ST_I4
#endif
