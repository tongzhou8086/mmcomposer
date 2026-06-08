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
constexpr int TMA_STORE    = 0;       // epilogue Phase 2: 0 = int4 stores, 1 = async TMA store

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
// ── TMA store helpers (epilogue Phase 2 when TMA_STORE=1) ───────────
__device__ __forceinline__ void tma_2d_store(
    const void* tmap, uint32_t smem_src, int x, int y
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
        "[%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_src) : "memory");
}
__device__ __forceinline__ void tma_commit_group() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void tma_wait_group() {
    asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory");
}

// MMA-issue building block (shared fragment).  MMA_ISSUE picks the
// single-CTA g1 instruction; the cluster tier supplies the g2 variant.
#define MMA_ISSUE(t, a, b, i, e) tcgen05_mma((t), (a), (b), (i), (e))
// @@MMA_CHAIN@@
#undef MMA_ISSUE

extern "C" __global__ void matmul_coalesced_epilogue(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ CUtensorMap C_tmap,
    __nv_bfloat16* __restrict__ C_ptr,
    int M, int N, int K
) {
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

    // ── Persistent grid (Step A: persistent scheduling, no overlap) ──
    // TMEM is allocated ONCE here and reused across every output tile
    // this CTA visits — cycling alloc/dealloc per tile deadlocks the
    // allocator.  Launched with grid = num_tiles the loop runs exactly
    // once per CTA (bit-identical to the non-persistent schedule);
    // launched with grid = #SMs each CTA walks a strided run of tiles.
    if (warp_id == 0) {
        tcgen05_alloc((uint32_t)__cvta_generic_to_shared(tmem_addr_holder), BN);
    }
    __syncthreads();
    const uint32_t taddr = tmem_addr_holder[0];
    const uint32_t idesc = make_idesc_bf16_kmajor_b(BM, BN);   // bit 16 = 1
    const int num_k_iters = K / BK;

    // Loop-invariant swizzle geometry (see chunked-walk note below).
    const int grid_m             = M / BM;
    const int grid_n             = N / BN;
    const int num_block_in_group = GROUP_SIZE_M * grid_n;
    const int num_tiles          = grid_m * grid_n;

    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
        // ── CTA-swizzle: derive (bid_m, bid_n) from the tile index ──
        // Chunked walk over GROUP_SIZE_M block-rows for L2 reuse on B.
        // GROUP_SIZE_M=1 collapses to the natural N-fast walk
        // (bid_m = tile / grid_n, bid_n = tile % grid_n).
        const int group_id      = tile / num_block_in_group;
        const int first_block_m = group_id * GROUP_SIZE_M;
        const int gsm   = min(grid_m - first_block_m, GROUP_SIZE_M);
        const int bid_m = first_block_m + (tile % gsm);
        const int bid_n = (tile % num_block_in_group) / gsm;
        const int off_m = bid_m * BM;
        const int off_n = bid_n * BN;

        // Per-tile mbarrier (re)init.  Safe to reset every tile because
        // the previous tile's epilogue + __syncthreads drained them all.
        if (warp_id == 0 && elect_sync()) {
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
        __syncthreads();

        // ── TMA warp ────────────────────────────────────────────────
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

        // ── MMA warp ────────────────────────────────────────────────
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

                issue_mma_chain(taddr, A_base(slot), B_base(slot), idesc, /*first_k_tile=*/ (k == 0));
                tcgen05_commit(done_mb);
                tile_ready_phase[slot] ^= 1;
            }
            tcgen05_commit((uint32_t)__cvta_generic_to_shared(&all_mmas_done));
        }

        mbarrier_wait_phase((uint32_t)__cvta_generic_to_shared(&all_mmas_done), 0);

        // ── Two-phase epilogue: TMEM → SMEM → coalesced GMEM ────────
        //
        // The K-loop is done, so the multi-stage SMEM ring is free to
        // reuse as a staging buffer.  We alias the same dynamic SMEM as
        // a [BM][BN_PAD] BF16 array.  The +8 padding shifts each row's
        // bank-base so phase 1's columnar stores hit 8 distinct banks
        // instead of all colliding on one (32-way → 4-way conflict).
        // ── Epilogue contract (single-CTA) + shared fragment splice ─
        const int cta_rank      = 0;       // single CTA
        const int off_m_cluster = off_m;   // single-CTA tile origin
        const CUtensorMap* C_tmap_ptr = &C_tmap;
        // Persistent: TMEM outlives the tile, so the epilogue must NOT
        // free it — we dealloc once after the loop.
#define EPI_DEALLOC(t, n) ((void)0)
        // @@EPILOGUE@@
#undef EPI_DEALLOC

        // Drain this tile fully (TMEM reads + SMEM staging) before the
        // next iteration reuses the same SMEM ring and TMEM accumulator.
        __syncthreads();
    }

    // Free the accumulator once, after every tile this CTA owns is done.
    if (warp_id == 0 && elect_sync()) {
        tcgen05_dealloc(taddr, BN);
    }
}
