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
    // TMA_STORE (0/1) picks Phase 2: a flat int4 store loop, or one async
    // TMA store per CTA.  The TMA store needs a tightly-packed SMEM source
    // (no +8 bank-pad), so EPI_LD switches accordingly.
    constexpr int EPI_LD = TMA_STORE ? BN : (BN + 8);
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

    #pragma unroll
    for (int n = col_base; n < col_base + COLS_PER_WARP; n += 8) {
        float tmp[8];
        tcgen05_ld_32x32b_x8(taddr_row + (uint32_t)n, tmp);
        tcgen05_wait_ld();

        __nv_bfloat162 packed[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            packed[i] = __floats2bfloat162_rn(tmp[2 * i], tmp[2 * i + 1]);
        }
        // SMEM store — int4 = 16 B = 8 BF16, one chunk of this row.
        *reinterpret_cast<int4*>(&C_sh[my_row][n]) =
            *reinterpret_cast<int4*>(packed);
    }

    __syncthreads();   // all Phase-1 SMEM writes visible before Phase 2

    const int out_m_base = off_m_cluster + cta_rank * BM;

    if constexpr (TMA_STORE) {
        // ── Phase 2a: one async TMA store per CTA ───────────────────
        // Phase 1 wrote SMEM via the GENERIC proxy (st.shared); the TMA
        // store engine reads via the ASYNC proxy — fence between them.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        if (warp_id == 0 && elect_sync()) {
            tma_2d_store(C_tmap_ptr,
                         (uint32_t)__cvta_generic_to_shared(&C_sh[0][0]),
                         /*x=*/ off_n, /*y=*/ out_m_base);
            tma_commit_group();
            tma_wait_group<0>();   // drain before we touch TMEM
        }
        // dealloc must come AFTER the store drains — reversing it
        // deadlocks the bulk-copy engine (bisected in ch13).
        if (warp_id == 0 && elect_sync()) {
            EPI_DEALLOC(taddr, BN);
        }
    } else {
        // ── Phase 2b: flat thread-major coalesced int4 stores ───────
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
            if constexpr (EPILOGUE_L1_NO_ALLOCATE) {
                // C is write-once: hint L1 not to allocate for the store so it
                // doesn't evict A/B.  Measured ~perf-neutral on B200 (the reuse
                // that matters is L2, which this L1 hint doesn't touch), kept
                // as a free, honest knob.
                const int4 _v = *reinterpret_cast<const int4*>(&C_sh[row][col]);
                asm volatile(
                    "st.relaxed.cta.global.L1::no_allocate.v4.b32 [%0], {%1,%2,%3,%4};"
                    :: "l"(&C_ptr[gr * N + gc]),
                       "r"(_v.x), "r"(_v.y), "r"(_v.z), "r"(_v.w) : "memory");
            } else {
                *reinterpret_cast<int4*>(&C_ptr[gr * N + gc]) =
                    *reinterpret_cast<const int4*>(&C_sh[row][col]);
            }
        }
    }
