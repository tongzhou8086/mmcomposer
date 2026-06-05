// Runnable companion for Chapter 07 — coalesced SMEM-staged epilogue.
//
// Identical to chapter 06 (K-major B, multi-stage, multi-CTA, warp-
// specialized) through the K-loop.  The only thing that changes is
// the epilogue: instead of writing TMEM directly to GMEM in the
// natural per-thread layout (which scatters 32 stores per warp across
// 32 cache lines), we split into two phases:
//
//   phase 1:  TMEM → SMEM   in the natural "lane L owns row L" layout
//             (with the BN+8 padding to avoid bank conflicts).
//   phase 2:  SMEM → GMEM   re-indexed flat thread-major so
//             consecutive lanes hit consecutive 16-byte chunks
//             → one coalesced 512-byte GMEM transaction per warp.
//
// The SMEM ring used by the multi-stage A/B tiles is reused as the
// staging buffer — no extra SMEM allocation needed.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

// ── User-tunable constants (the webui substitutes these six) ────────
constexpr int BM           = 128;
constexpr int BN           = 256;
constexpr int BK           = 64;
constexpr int NS           = 2;       // multi-stage SMEM ring depth
constexpr int GROUP_SIZE_M = 8;       // CTA-swizzle chunk (1 = no swizzle)
constexpr int NUM_WARPS    = 4;       // total warps per CTA

// ── Derived constants (do not edit) ─────────────────────────────────
constexpr int MMA_K   = 16;
constexpr int K_MMAS  = BK / MMA_K;          // 4

constexpr int BF16_BYTES        = 2;        // byte size of the operand element
constexpr int SWIZZLE_ROW_BYTES = 128;      // one 128B-swizzle atom row

// ── SWIZZLE_128B constraint on K-major B ────────────────────────────
//
// TMA with SWIZZLE_128B requires the box's INNERMOST dimension to be
// exactly SWIZZLE_ROW_BYTES (= 128 B) — one swizzle atom.  For native
// (K, N) row-major B, N is the contiguous dim, so the inner box covers
// N and must be capped at SWIZZLE_ROW_BYTES / BF16_BYTES = 64 BF16.
// Loading the full BN requires BN / 64 sub-tile TMA calls, each
// writing into a different SMEM offset.  (Ch05 dodged this by
// host-transposing B so K — also 64 BF16 — was the inner dim.)
//
// The literal 64 below is therefore not a free parameter; it's
// `SWIZZLE_ROW_BYTES / BF16_BYTES`.

constexpr int A_SLOT_BYTES = BM * BK * BF16_BYTES;     // 16 KB
constexpr int B_SLOT_BYTES = BN * BK * BF16_BYTES;     // 32 KB
constexpr int SLOT_BYTES   = A_SLOT_BYTES + B_SLOT_BYTES;          // 48 KB / slot

constexpr int WARP_SIZE = 32;
constexpr int THREADS   = NUM_WARPS * WARP_SIZE;


// ── helpers (identical to ch05) ─────────────────────────────────────
__device__ __forceinline__ bool elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t .reg .pred px;\n\t"
        "elect.sync _|px, %1;\n\t"
        "@px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred) : "r"(0xFFFFFFFF));
    return pred;
}

__device__ __forceinline__ void tma_2d_load(
    uint32_t smem_dst, const void* tmap, int x, int y, uint32_t mbar
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar) : "memory");
}

// A's descriptor — unchanged from earlier chapters (MN-major).
__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    uint64_t b = ((SBO)              >> 4) & 0x3FFFULL;
    return a | (b << 32) | (1ULL << 46) | (2ULL << 61);
}

// NEW: K-major B descriptor.  The only additional field versus make_desc
// is the LBO (leading byte offset) at bits [16, 29] — telling the tensor
// cores how far apart consecutive N-sub-tiles sit in SMEM.
__device__ __forceinline__ uint64_t make_desc_K_major(
    uint32_t smem_addr, int lbo_bytes
) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a   = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    uint64_t lbo = ((uint64_t)lbo_bytes >> 4) & 0x3FFFULL;
    uint64_t b   = ((SBO)               >> 4) & 0x3FFFULL;
    return a | (lbo << 16) | (b << 32) | (1ULL << 46) | (2ULL << 61);
}

// idesc with bit 16 = 1: B is K-major.
__device__ __forceinline__ uint32_t make_idesc_bf16_kmajor_b(int m, int n) {
    uint32_t d = 0;
    d |= (1u << 4);                                    // c_format = F32
    d |= (1u << 7);                                    // a_format = BF16
    d |= (1u << 10);                                   // b_format = BF16
    d |= (1u << 16);                                   // B is K-major  ← new
    d |= (((uint32_t)(n >> 3) & 0x3F) << 17);
    d |= (((uint32_t)(m >> 4) & 0x1F) << 24);
    return d;
}

__device__ __forceinline__ void tcgen05_alloc(uint32_t smem_dst, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_dst), "r"(n_cols) : "memory");
}
__device__ __forceinline__ void tcgen05_dealloc(uint32_t taddr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(taddr), "r"(n_cols) : "memory");
}
__device__ __forceinline__ void tcgen05_mma(
    uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
    uint32_t idesc, bool enable_d
) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "setp.ne.b32 P, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, P;\n\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
           "r"((uint32_t)enable_d) : "memory");
}
__device__ __forceinline__ void tcgen05_commit(uint32_t smem_bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(smem_bar) : "memory");
}
__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}
__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tcgen05_ld_32x32b_x8(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
          "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7])
        : "r"(taddr));
}

__device__ __forceinline__ void mbarrier_init(uint32_t mb, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mb), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive_no_tx(uint32_t mb) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(mb) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint32_t mb, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(mb), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbarrier_wait_phase(uint32_t mb, uint32_t phase) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE_%=;\n\t bra WAIT_%=;\n\t DONE_%=:\n\t }"
        :: "r"(mb), "r"(phase) : "memory");
}


// ── Kernel ──────────────────────────────────────────────────────────
extern "C" __global__ void matmul_coalesced_epilogue(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    __nv_bfloat16* __restrict__ C_ptr,
    int M, int N, int K
) {
    // ── CTA-swizzle: derive (bid_m, bid_n) from blockIdx.x  ─────────
    // Chunked walk over GROUP_SIZE_M block-rows for L2 reuse on B.
    // GROUP_SIZE_M=1 collapses to the natural N-fast walk
    // (bid_m = blockIdx.x / grid_n, bid_n = blockIdx.x % grid_n).
    const int grid_m = M / BM;
    const int grid_n = N / BN;
    const int num_block_in_group = GROUP_SIZE_M * grid_n;
    const int group_id           = blockIdx.x / num_block_in_group;
    const int first_block_m      = group_id * GROUP_SIZE_M;
    const int gsm    = min(grid_m - first_block_m, GROUP_SIZE_M);
    const int bid_m  = first_block_m + (blockIdx.x % gsm);
    const int bid_n  = (blockIdx.x % num_block_in_group) / gsm;
    const int off_m  = bid_m * BM;
    const int off_n  = bid_n * BN;

    extern __shared__ __align__(1024) char smem[];
    const uint32_t SMEM_BASE = (uint32_t)__cvta_generic_to_shared(smem);
    auto A_base = [SMEM_BASE](int s) -> uint32_t {
        return SMEM_BASE + s * SLOT_BYTES;
    };
    // B_base points at sub-tile 0 of slot s.  Sub-tile at N-offset n
    // (n in {0, 64, 128, 192}) sits at B_base(s) + n * BK * BF16_BYTES.
    auto B_base = [SMEM_BASE](int s) -> uint32_t {
        return SMEM_BASE + s * SLOT_BYTES + A_SLOT_BYTES;
    };

    __shared__ uint64_t tile_ready[NS];
    __shared__ uint64_t mma_done[NS];
    __shared__ uint64_t all_mmas_done;
    __shared__ uint32_t tmem_addr_holder[1];

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane    = tid % WARP_SIZE;

    if (warp_id == 0) {
        if (elect_sync()) {
            #pragma unroll
            for (int s = 0; s < NS; s++) {
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&tile_ready[s]), 1);
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mma_done[s]),   1);
            }
            mbarrier_init((uint32_t)__cvta_generic_to_shared(&all_mmas_done), 1);
            mbarrier_arrive_no_tx(
                (uint32_t)__cvta_generic_to_shared(&mma_done[NS - 1]));
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        tcgen05_alloc((uint32_t)__cvta_generic_to_shared(tmem_addr_holder), BN);
    }
    __syncthreads();
    const uint32_t taddr = tmem_addr_holder[0];
    const uint32_t idesc = make_idesc_bf16_kmajor_b(BM, BN);   // bit 16 = 1

    const int num_k_iters = K / BK;

    // ── TMA warp ────────────────────────────────────────────────────
    //
    // A: one bulk per stage (unchanged from ch05).
    // B: BN/64 bulks per stage, one per 64-N-col sub-tile.  Each
    //    bulk loads (64 N-cols × BK K-rows) from native (K, N) GMEM.
    if (warp_id == 0 && elect_sync()) {
        uint32_t mma_done_phase[NS] = {};

        // Prologue
        #pragma unroll
        for (int s = 0; s < NS - 1; s++) {
            const uint32_t mb = (uint32_t)__cvta_generic_to_shared(&tile_ready[s]);
            tma_2d_load(A_base(s), &A_tmap, /*x=*/ s * BK, /*y=*/ off_m, mb);
            // BN/64 sub-tile loads — see SWIZZLE_128B constraint above.
            #pragma unroll
            for (int n = 0; n < BN; n += 64) {
                tma_2d_load(B_base(s) + n * BK * BF16_BYTES,    // SMEM offset of this sub-tile
                            &B_tmap,
                            /*x=*/ off_n + n,                    // N-coord (innermost)
                            /*y=*/ s * BK,                       // K-coord
                            mb);
            }
            mbarrier_arrive_expect_tx(mb, SLOT_BYTES);
        }

        // Steady-state
        for (int k = 0; k < num_k_iters - (NS - 1); k++) {
            const int slot = (k + NS - 1) % NS;
            const uint32_t done_mb  = (uint32_t)__cvta_generic_to_shared(&mma_done[slot]);
            const uint32_t ready_mb = (uint32_t)__cvta_generic_to_shared(&tile_ready[slot]);

            mbarrier_wait_phase(done_mb, mma_done_phase[slot]);
            tma_2d_load(A_base(slot), &A_tmap,
                        /*x=*/ (k + NS - 1) * BK, /*y=*/ off_m, ready_mb);
            #pragma unroll
            for (int n = 0; n < BN; n += 64) {
                tma_2d_load(B_base(slot) + n * BK * BF16_BYTES,
                            &B_tmap,
                            /*x=*/ off_n + n,
                            /*y=*/ (k + NS - 1) * BK,
                            ready_mb);
            }
            mbarrier_arrive_expect_tx(ready_mb, SLOT_BYTES);
            mma_done_phase[slot] ^= 1;
        }
    }

    // ── MMA warp ────────────────────────────────────────────────────
    //
    // B descriptor now points at sub-tile 0 at the current K-strip; the
    // MMA hardware uses LBO = BK * 128 to walk to sub-tiles 1..3 on
    // its own.  One MMA still consumes 16 K-rows × all-BN of B.
    else if (warp_id == 1 && elect_sync()) {
        uint32_t tile_ready_phase[NS] = {};

        for (int k = 0; k < num_k_iters; k++) {
            const int slot = k % NS;
            const uint32_t ready_mb = (uint32_t)__cvta_generic_to_shared(&tile_ready[slot]);
            const uint32_t done_mb  = (uint32_t)__cvta_generic_to_shared(&mma_done[slot]);

            mbarrier_wait_phase(ready_mb, tile_ready_phase[slot]);
            tcgen05_fence_after_thread_sync();

            #pragma unroll
            for (int kk = 0; kk < K_MMAS; kk++) {
                // A K-step = MMA_K elements = MMA_K * BF16_BYTES bytes,
                // walking within one swizzle atom row.
                const uint64_t a_desc = make_desc(
                    A_base(slot) + kk * MMA_K * BF16_BYTES);
                // K-major B: addr points at sub-tile 0, K-row kk*MMA_K
                // (each K-row is one swizzle atom).  LBO = BK rows ×
                // one swizzle atom — tells MMA how to walk to sub-tiles
                // 1..3.
                const uint64_t b_desc = make_desc_K_major(
                    B_base(slot) + kk * MMA_K * SWIZZLE_ROW_BYTES,
                    BK * SWIZZLE_ROW_BYTES);
                const bool first_ever = (k == 0) && (kk == 0);
                tcgen05_mma(taddr, a_desc, b_desc, idesc, /*enable_d=*/ !first_ever);
            }
            tcgen05_commit(done_mb);
            tile_ready_phase[slot] ^= 1;
        }
        tcgen05_commit((uint32_t)__cvta_generic_to_shared(&all_mmas_done));
    }

    mbarrier_wait_phase((uint32_t)__cvta_generic_to_shared(&all_mmas_done), 0);

    // ── Two-phase epilogue: TMEM → SMEM → coalesced GMEM ────────────
    //
    // The K-loop is done, so the multi-stage SMEM ring is free to
    // reuse as a staging buffer.  We alias the same dynamic SMEM as
    // a [BM][BN_PAD] BF16 array.  The +8 padding shifts each row's
    // bank-base so phase 1's columnar stores hit 8 distinct banks
    // instead of all colliding on one (32-way → 4-way conflict).
    constexpr int BN_PAD = BN + 8;
    auto C_sh = reinterpret_cast<__nv_bfloat16(*)[BN_PAD]>(smem);

    tcgen05_fence_after_thread_sync();   // mma_done → tcgen05.ld

    // ── Phase 1: TMEM → SMEM, generalized variable-warp 2D grid ──────
    // Partition the NUM_WARPS warps as ROW_STRIPS row groups (BM/32) ×
    // COL_GROUPS column slices, so every warp does real work even at
    // NW=8/16 (NW=4 → 4 row × 1 col → each warp all BN cols).
    constexpr int ROW_STRIPS    = BM / 32;
    constexpr int COL_GROUPS    = NUM_WARPS / ROW_STRIPS;
    constexpr int COLS_PER_WARP = BN / COL_GROUPS;

    const int row_warp = warp_id % ROW_STRIPS;
    const int col_warp = warp_id / ROW_STRIPS;
    const int my_row   = row_warp * 32 + lane;
    const uint32_t taddr_row = taddr + ((uint32_t)(row_warp * 32) << 16);
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

    __syncthreads();
    if (warp_id == 0 && elect_sync()) {
        tcgen05_dealloc(taddr, BN);
    }

    // ── Phase 2: SMEM → GMEM with flat thread-major indexing
    //
    // Consecutive lanes write consecutive 16-byte int4 chunks to GMEM
    // → one coalesced 512-byte transaction per warp per store.
    constexpr int CHUNK_BF16        = 8;
    constexpr int CHUNKS_PER_ROW    = BN / CHUNK_BF16;                       // 32
    constexpr int STORES_PER_THREAD = (BM * BN) / (THREADS * CHUNK_BF16);   // 32
    static_assert(STORES_PER_THREAD * THREADS * CHUNK_BF16 == BM * BN,
                  "BM*BN must be a multiple of THREADS*8 for the flat tile-walk");

    #pragma unroll
    for (int s = 0; s < STORES_PER_THREAD; s++) {
        const int flat = tid + s * THREADS;
        const int row  = flat / CHUNKS_PER_ROW;
        const int col  = (flat % CHUNKS_PER_ROW) * CHUNK_BF16;
        const int gr   = off_m + row;
        const int gc   = off_n + col;
        *reinterpret_cast<int4*>(&C_ptr[gr * N + gc]) =
            *reinterpret_cast<const int4*>(&C_sh[row][col]);
    }
}
