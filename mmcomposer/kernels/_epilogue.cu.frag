    // ── Shared epilogue (TMEM → SMEM → GMEM) ────────────────────────
    // mvp_core stitches this fragment into every tier's kernel.cu at the
    // epilogue marker, so the epilogue lives in exactly one place.  Each
    // tier supplies a small contract just before the marker:
    //   cta_rank, off_m_cluster, off_n   — tile-origin primitives
    //       (single-CTA tiers set cta_rank=0, off_m_cluster=off_m)
    //   C_tmap_ptr                       — const CUtensorMap* for the store
    //   EPI_DEALLOC(taddr, n)            — that tier's tcgen05 dealloc
    //       (single-CTA: tcgen05_dealloc; cluster: tcgen05_dealloc_g2)
    //
    // EPILOGUE_L1_NO_ALLOC (knob): write-once C store bypasses L1 allocation so
    // it doesn't evict A/B from L1.  Win when the epilogue is exposed (low K),
    // null at high K — a sweep knob.
#if EPILOGUE_L1_NO_ALLOC
#define EPI_ST_I4(DST, VAL) do { int4 _v = (VAL); \
        asm volatile("st.relaxed.cta.global.L1::no_allocate.v4.b32 [%0], {%1,%2,%3,%4};" \
            :: "l"(DST), "r"(_v.x), "r"(_v.y), "r"(_v.z), "r"(_v.w) : "memory"); } while (0)
#else
#define EPI_ST_I4(DST, VAL) (*reinterpret_cast<int4*>(DST) = (VAL))
#endif
    constexpr int EPI_LD = BN + 8;   // +8 bank-pad for the columnar Phase-1 stores
    auto C_sh = reinterpret_cast<__nv_bfloat16(*)[EPI_LD]>(smem);

    tcgen05_fence_after_thread_sync();

    // ── Phase 1: TMEM → SMEM, generalized variable-warp 2D grid ──────
    // Partition NUM_WARPS as ROW_STRIPS (BM/32) row groups × COL_GROUPS
    // column slices so every warp works even at NW=8/16.  The cluster
    // adds a cta_rank*BM logical-row offset into this CTA's TMEM.
    constexpr int ROW_STRIPS    = BM / 32;
    constexpr int COL_GROUPS    = NUM_WARPS / ROW_STRIPS;
    constexpr int COLS_PER_WARP = BN / COL_GROUPS;

    const int row_warp = warp_id % ROW_STRIPS;
    const int col_warp = warp_id / ROW_STRIPS;
    const int my_row   = row_warp * 32 + lane;
    const uint32_t taddr_row =
        taddr + ((uint32_t)(cta_rank * BM + row_warp * 32) << 16);
    const int col_base = col_warp * COLS_PER_WARP;

    // TCGEN05_LD_WIDTH (8/16) = 32-bit elems/lane per tcgen05.ld.
    // Wider = fewer loads + fewer wait_ld syncs (more registers, free while
    // we're SMEM-occupancy-bound).
    constexpr int LDW = TCGEN05_LD_WIDTH;
    #pragma unroll
    for (int n = col_base; n < col_base + COLS_PER_WARP; n += LDW) {
        float tmp[LDW];
#if TCGEN05_LD_WIDTH == 8
        tcgen05_ld_32x32b_x8 (taddr_row + (uint32_t)n, tmp);
#else
        tcgen05_ld_32x32b_x16(taddr_row + (uint32_t)n, tmp);
#endif
        tcgen05_wait_ld();

        __nv_bfloat162 packed[LDW / 2];
        #pragma unroll
        for (int i = 0; i < LDW / 2; i++) {
            packed[i] = __floats2bfloat162_rn(tmp[2 * i], tmp[2 * i + 1]);
        }
        // SMEM stores — int4 = 16 B = 8 BF16 = 4 BF16x2, one per 8 columns.
        #pragma unroll
        for (int c = 0; c < LDW; c += 8) {
            *reinterpret_cast<int4*>(&C_sh[my_row][n + c]) =
                *reinterpret_cast<int4*>(&packed[c / 2]);
        }
    }

    __syncthreads();   // all Phase-1 SMEM writes visible before Phase 2

    const int out_m_base = off_m_cluster + cta_rank * BM;

    {
        // ── Phase 2: flat thread-major coalesced int4 stores ────────
        if (warp_id == 0 && elect_sync()) {
            EPI_DEALLOC(taddr, BN);
        }
        constexpr int CHUNK_BF16        = 8;
        constexpr int CHUNKS_PER_ROW    = BN / CHUNK_BF16;
        constexpr int STORES_PER_THREAD = (BM * BN) / (THREADS * CHUNK_BF16);
        static_assert(STORES_PER_THREAD * THREADS * CHUNK_BF16 == BM * BN,
                      "BM*BN must be a multiple of THREADS*8 for the flat tile-walk");
        #pragma unroll
        for (int s = 0; s < STORES_PER_THREAD; s++) {
            const int flat = tid + s * THREADS;
            const int row  = flat / CHUNKS_PER_ROW;
            const int col  = (flat % CHUNKS_PER_ROW) * CHUNK_BF16;
            const int gr   = out_m_base + row;
            const int gc   = off_n + col;
            EPI_ST_I4(&C_ptr[gr * N + gc], *reinterpret_cast<const int4*>(&C_sh[row][col]));
        }
    }
#undef EPI_ST_I4
