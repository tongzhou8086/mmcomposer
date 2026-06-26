#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

// ── User-tunable constants ──────────────────────────────────────────
// These six lines are the contract between the webui and the kernel.
// The webui regex-substitutes them when the user changes a value in
// the UI; everything else in the file is *derived* from these.
constexpr int BM           = 128;
constexpr int BN           = 256;
constexpr int BK           = 64;
constexpr int NS           = 2;       // SMEM ring depth (2 = double buffer; >2 = multi-stage)
constexpr int GROUP_SIZE_M = 8;       // CTA-swizzle chunk (1 = no swizzle)
constexpr int NUM_WARPS    = 4;       // total warps per CTA
constexpr int TCGEN05_LD_WIDTH = 8;  // TMEM->reg epilogue load width: 8 or 16 (32-bit elems per lane)

// ── Derived constants (do not edit) ─────────────────────────────────
constexpr int MMA_K             = 16;
constexpr int K_MMAS            = BK / MMA_K;
constexpr int BF16_BYTES        = 2;
constexpr int SWIZZLE_ROW_BYTES = 128;

constexpr int A_SLOT_BYTES = BM * BK * BF16_BYTES;
constexpr int B_SLOT_BYTES = BN * BK * BF16_BYTES;
constexpr int SLOT_BYTES   = A_SLOT_BYTES + B_SLOT_BYTES;

constexpr int WARP_SIZE = 32;
constexpr int THREADS   = NUM_WARPS * WARP_SIZE;


// ── helpers (same as ch07) ──────────────────────────────────────────
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

__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    uint64_t b = ((SBO)              >> 4) & 0x3FFFULL;
    // NOTE: this descriptor is correct for BM=128, the only single-CTA
    // value that works (see the BM=128 invariant at the top of file).
    // BM was never the SBO's fault — the SBO/bit-46 are layout-correct
    // for any BM.  The real blocker is the MMA M-atom + TMEM lanes:
    // tcgen05.mma.kind::f16 single-CTA computes M=128, and BM=64 makes
    // it read past the 64-row A tile (CUDA_ERROR_ILLEGAL_ADDRESS).
    return a | (b << 32) | (1ULL << 46) | (2ULL << 61);
}

__device__ __forceinline__ uint64_t make_desc_K_major(
    uint32_t smem_addr, int lbo_bytes
) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a   = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    uint64_t lbo = ((uint64_t)lbo_bytes >> 4) & 0x3FFFULL;
    uint64_t b   = ((SBO)               >> 4) & 0x3FFFULL;
    return a | (lbo << 16) | (b << 32) | (1ULL << 46) | (2ULL << 61);
}

__device__ __forceinline__ uint32_t make_idesc_bf16_kmajor_b(int m, int n) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (1u << 16);                                  // B is K-major
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
// @@TCGEN05_LD@@

__device__ __forceinline__ void mbarrier_init(uint32_t mb, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mb), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive_no_tx(uint32_t mb) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(mb) : "memory");
}
__device__ __forceinline__ void signal_on_bytes_loaded(uint32_t mb, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(mb), "r"(bytes) : "memory");
}
__device__ __forceinline__ void wait_phase(uint32_t mb, uint32_t phase) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE_%=;\n\t bra WAIT_%=;\n\t DONE_%=:\n\t }"
        :: "r"(mb), "r"(phase) : "memory");
}


// ── Kernel ──────────────────────────────────────────────────────────
//
// GROUP_SIZE_M is now a constexpr at the top of the file (not a
// template) — this is what lets the webui's regex substitution treat
// every knob uniformly.  GROUP_SIZE_M=1 collapses to the natural
// N-fast walk; >1 swaps to a chunked M-fast walk for L2 reuse on B.
// MMA-issue building block (shared fragment).  MMA_ISSUE picks the
// single-CTA g1 instruction; the cluster tier supplies the g2 variant.
#define MMA_ISSUE(t, a, b, i, e) tcgen05_mma((t), (a), (b), (i), (e))
// @@MMA_CHAIN@@
#undef MMA_ISSUE

extern "C" __global__ void matmul_dbuf(
    const __grid_constant__ CUtensorMap A_tmap_,
    const __grid_constant__ CUtensorMap B_tmap_,
    const __grid_constant__ CUtensorMap C_tmap_,
    __nv_bfloat16* __restrict__ C_ptr,
    int M, int N, int K
) {
    const CUtensorMap* A_tmap = &A_tmap_;
    const CUtensorMap* B_tmap = &B_tmap_;
    const CUtensorMap* C_tmap_ptr = &C_tmap_;
    // ── CTA-swizzle: derive (bid_m, bid_n) from blockIdx.x  ─────────
    //
    // Same chunked walk as ch09 but at the single-CTA granularity
    // (no cluster division).  GSM=1 collapses to the natural
    // bid_m = blockIdx.x / grid_n, bid_n = blockIdx.x % grid_n.
    const int grid_m = M / BM;
    const int grid_n = N / BN;
    const int num_block_in_group = GROUP_SIZE_M * grid_n;
    const int group_id           = blockIdx.x / num_block_in_group;
    const int first_block_m      = group_id * GROUP_SIZE_M;
    const int gsm   = min(grid_m - first_block_m, GROUP_SIZE_M);
    const int bid_m = first_block_m + (blockIdx.x % gsm);
    const int bid_n = (blockIdx.x % num_block_in_group) / gsm;
    const int off_m = bid_m * BM;
    const int off_n = bid_n * BN;

    extern __shared__ __align__(1024) char smem[];
    const uint32_t SMEM_BASE = (uint32_t)__cvta_generic_to_shared(smem);
    auto A_base = [SMEM_BASE](int s) -> uint32_t {
        return SMEM_BASE + s * SLOT_BYTES;
    };
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
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        tcgen05_alloc((uint32_t)__cvta_generic_to_shared(tmem_addr_holder), BN);
    }
    __syncthreads();
    const uint32_t taddr = tmem_addr_holder[0];
    const uint32_t idesc = make_idesc_bf16_kmajor_b(BM, BN);

    const int num_k_iters = K / BK;

    // ── Single-warp sync K-loop with NS=2 double buffer ─────────────
    //
    // Pattern per iteration (warp 0):
    //   1. Wait on tile_ready[slot]   (slot's TMA must have landed)
    //   2. Issue K_MMAS tcgen05.mma's into TMEM
    //   3. tcgen05_commit(mma_done[slot])      → mma_done fires when MMAs drain
    //   4. Issue next-iter's TMA into slot     (overlaps with running MMAs)
    //   5. signal_on_bytes_loaded(tile_ready[next_slot], SLOT_BYTES)
    //   6. Wait on mma_done[next_slot] before that slot's NEXT TMA       (skipped here)
    //
    // The "double-buffer" trick: with NS=2 we maintain one slot in
    // flight while MMAs run on the other.  Concurrency comes from
    // (a) tcgen05.mma being async (warp doesn't wait), (b) the TMA
    // bulk being async, (c) the two slots holding two K-tiles at
    // once.  *Without* warp specialization the single warp must still
    // serialize its instruction stream — that's the cost we pay vs ch07.
    if (warp_id == 0 && elect_sync()) {
        // Phase tracking for mbarriers (toggled per slot completion).
        uint32_t tile_ready_phase[NS] = {};
        uint32_t mma_done_phase[NS]   = {};

        // Prologue: prefetch NS K-tiles.
        #pragma unroll
        for (int s = 0; s < NS; s++) {
            if (s < num_k_iters) {
                const uint32_t mb = (uint32_t)__cvta_generic_to_shared(&tile_ready[s]);
                tma_2d_load(A_base(s), A_tmap, /*x=*/ s * BK, /*y=*/ off_m, mb);
                #pragma unroll
                for (int n = 0; n < BN; n += 64) {
                    tma_2d_load(B_base(s) + n * BK * BF16_BYTES,
                                B_tmap, /*x=*/ off_n + n, /*y=*/ s * BK, mb);
                }
                signal_on_bytes_loaded(mb, SLOT_BYTES);
            }
        }

        // Steady-state.  Each iter: wait the current slot, MMA, commit,
        // then prefetch the (k + NS)-th tile into that slot.
        for (int k = 0; k < num_k_iters; k++) {
            const int slot = k % NS;
            const uint32_t ready_mb = (uint32_t)__cvta_generic_to_shared(&tile_ready[slot]);
            const uint32_t done_mb  = (uint32_t)__cvta_generic_to_shared(&mma_done[slot]);

            wait_phase(ready_mb, tile_ready_phase[slot]);
            tile_ready_phase[slot] ^= 1;
            tcgen05_fence_after_thread_sync();

            issue_mma_chain(taddr, A_base(slot), B_base(slot), idesc, /*first_k_tile=*/ (k == 0));
            tcgen05_commit(done_mb);

            // Prefetch (k + NS)-th tile into the freed slot.
            const int next_k = k + NS;
            if (next_k < num_k_iters) {
                wait_phase(done_mb, mma_done_phase[slot]);
                mma_done_phase[slot] ^= 1;
                tma_2d_load(A_base(slot), A_tmap,
                            /*x=*/ next_k * BK, /*y=*/ off_m, ready_mb);
                #pragma unroll
                for (int n = 0; n < BN; n += 64) {
                    tma_2d_load(B_base(slot) + n * BK * BF16_BYTES,
                                B_tmap,
                                /*x=*/ off_n + n,
                                /*y=*/ next_k * BK, ready_mb);
                }
                signal_on_bytes_loaded(ready_mb, SLOT_BYTES);
            }
        }

        tcgen05_commit((uint32_t)__cvta_generic_to_shared(&all_mmas_done));
    }

    wait_phase(
        (uint32_t)__cvta_generic_to_shared(&all_mmas_done), 0);

    // ── Coalesced 2-phase epilogue (row × col warp grid, from ch10) ──
    //
    // TMEM read binding: `tcgen05.ld` ties a warp to one 32-row TMEM
    // group, indexed by (warp_id % ROW_STRIPS) — a warp's lanes are
    // wired to those 32 rows and the ld address's row field selects
    // *which* of the ROW_STRIPS groups, but only the group matching
    // (warp_id % ROW_STRIPS) returns valid data.  (Verified: a layout
    // where a warp reads a stripe ≠ its own group gives garbage; the
    // hardware hands back the warp's own rows regardless.)
    //
    // So we partition the NUM_WARPS warps as a 2D grid:
    //   row_warp = warp_id % ROW_STRIPS   → which 32-row strip (0..3)
    //   col_warp = warp_id / ROW_STRIPS   → which column slice
    // Columns ARE divided across warps along col_warp — every warp
    // does real Phase-1 work even at NW=8/16.  This mirrors ch10.
    //
    //   BM=128, NW=4  → 4 row × 1 col  → each warp all BN cols
    //   BM=128, NW=8  → 4 row × 2 col  → each warp BN/2 cols
    //   BM=128, NW=16 → 4 row × 4 col  → each warp BN/4 cols
    // ── Epilogue contract (single-CTA) + shared fragment splice ─────
    const int cta_rank      = 0;       // single CTA
    const int off_m_cluster = off_m;   // single-CTA tile origin
#define EPI_DEALLOC(t, n) tcgen05_dealloc((t), (n))
    // C_tmap_ptr already in scope (declared at top of kernel).
    // @@EPILOGUE@@
#undef EPI_DEALLOC
}


// No multi-launcher dispatch table — there is exactly one kernel
// symbol (`matmul_dbuf`) compiled at the GROUP_SIZE_M baked in above.
// The webui generates a fresh source file per user click; the user
// recompiles locally to get a binary with their chosen knobs.
