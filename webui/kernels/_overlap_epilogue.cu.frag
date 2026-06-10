                // ── Overlap epilogue drain (TMEM → SMEM → GMEM), shared ─────
                // Spliced into the overlap epilogue-warp loop of every warp-spec
                // tier's drain marker, right after `trow` (the tier-specific TMEM
                // lane base) and `LDW` are in scope.  The skeleton supplies three
                // contract macros for the per-tier bits:
                //   EPI_OUT_ROW                  this CTA's GMEM row base
                //   EPI_OUT_COL_BASE             this CTA's GMEM column base
                //   EPI_TMEM_EMPTY_ARRIVE(buf)   release the drained TMEM buffer
                // EPILOGUE_SPLIT (constexpr) picks the two-pass half-BN writeback,
                // which stages one BN/2 column panel at a time (EPI_STAGE_COLS=BN/2)
                // so the epilogue SMEM shrinks enough for an extra K-loop stage.
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
                            EPI_TMEM_EMPTY_ARRIVE(buf);

                        constexpr int CHUNKS = (BN / 2) / 8;
                        constexpr int STORES = BM * (BN / 2) / (EPI_THREADS * 8);
                        static_assert(STORES * EPI_THREADS * 8 == BM * (BN / 2),
                                      "split epilogue tile must divide across epilogue threads");
                        #pragma unroll
                        for (int s = 0; s < STORES; s++) {
                            const int flat = etid + s * EPI_THREADS;
                            const int row = flat / CHUNKS;
                            const int col = (flat % CHUNKS) * 8;
                            *reinterpret_cast<int4*>(&C_ptr[(EPI_OUT_ROW + row) * N + EPI_OUT_COL_BASE + split_base + col]) =
                                *reinterpret_cast<const int4*>(&C_sh[row][col]);
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
                        EPI_TMEM_EMPTY_ARRIVE(buf);
                    constexpr int CHUNKS = BN / 8;
                    constexpr int STORES = BM * BN / (EPI_THREADS * 8);
                    #pragma unroll
                    for (int s = 0; s < STORES; s++) {
                        int flat = etid + s * EPI_THREADS;
                        int row = flat / CHUNKS;
                        int col = (flat % CHUNKS) * 8;
                        *reinterpret_cast<int4*>(&C_ptr[(EPI_OUT_ROW + row) * N + EPI_OUT_COL_BASE + col]) =
                            *reinterpret_cast<const int4*>(&C_sh[row][col]);
                    }
                    asm volatile("bar.sync 1, %0;" :: "n"(NUM_WARPS * 32));
                }
#endif