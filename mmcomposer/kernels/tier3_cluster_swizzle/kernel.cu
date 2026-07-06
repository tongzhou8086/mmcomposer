#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

// ── User-tunable constants (the webui substitutes these) ────────────
constexpr int BM           = 128;
constexpr int BN           = 256;
constexpr int BK           = 64;
constexpr int NS           = 5;       // multi-stage SMEM ring depth
constexpr int GROUP_SIZE_M = 8;       // CTA-swizzle chunk (1 = no swizzle)
constexpr int NUM_WARPS    = 4;       // total warps per CTA
constexpr int TCGEN05_LD_WIDTH = 8;  // TMEM->reg epilogue load width: 8 or 16 (32-bit elems per lane)
constexpr int EPILOGUE_OVERLAP = 0;  // 1 = persistent 2-CTA cluster + epilogue/K-loop overlap
constexpr int EPILOGUE_SPLIT   = 0;  // 1 = split overlapped int4 writeback into two half-BN passes
constexpr int EPILOGUE_TMA_PIPELINED = 0;  // 1 = chunked staged TMA-store overlap epilogue
constexpr int SINGLE_TMEM_ACCUM = 0;  // 1 = overlap path synchronizes epilogue drain before reusing one TMEM accumulator
constexpr int SEGMENTED_PANELS = 0;  // 1 = BN512 segmented panel schedule (SEG = NS k-tiles per segment)
constexpr int TWO_CTA          = 1;  // 1 = 2-CTA cluster MMA (cta_group::2); 0 = single-CTA

// ── Derived constants (do not edit) ─────────────────────────────────
constexpr int MMA_K     = 16;
constexpr int BF16_BYTES = 2;
constexpr int K_MMAS    = BK / MMA_K;        // 4

constexpr int CTA_GROUP        = TWO_CTA ? 2 : 1;    // 2-CTA cluster vs single-CTA
constexpr int BN_LOCAL         = BN / CTA_GROUP;     // per-CTA N width of B (=BN single-CTA)
constexpr int SWIZZLE_ROW_BYTES = 128;               // one 128B-swizzle atom row
constexpr int STORE_N          = 64;                 // TMA-store chunk width
constexpr int TMA_STORE_STAGES = 2;                  // TMA-store SMEM buffers

// Per-stage SMEM per CTA: A = BM*BK*2 = 16 KB; B = BN_LOCAL*BK*2 = 16 KB.
// Total 32 KB / stage / CTA — half of ch07's 48 KB / stage / CTA.
constexpr int A_SLOT_BYTES = BM       * BK * BF16_BYTES;       // 16 KB
constexpr int B_SLOT_BYTES = BN_LOCAL * BK * BF16_BYTES;       // 16 KB
constexpr int SLOT_BYTES   = A_SLOT_BYTES + B_SLOT_BYTES;      // 32 KB / slot

#if SEGMENTED_PANELS
// ── Segmented panel schedule (BN=512 only) ──────────────────────────
// Process the K-loop in segments of SEG = NS k-tiles.  Within a segment run
// ALL of panel 0 (SEG MMAs into TMEM [0,256)), then ALL of panel 1 (SEG MMAs
// into [256,512)) reusing the segment's resident A tiles (A loaded ONCE — the
// arithmetic intensity of the baseline is preserved).  Because the two panels
// are serialized in time, B0 and B1 tiles never coexist: one FIFO B ring is
// recycled between the two streams instead of holding both (2x NS in the
// packed layout).  panel 0's accumulator half is fully summed at the global
// last k, so the epilogue drains [0,256) while the MMA warp still computes
// the last segment's panel 1 — hiding ~half the single-TMEM reuse delay.
//
// SMEM = [ A ring SEG_NA | B ring SEG_NB | C_store ], all 16 KB tiles:
//   SEG_NA = SEG + 1        segment residency + 1 slot so the next segment's
//                           first A can prefetch without waiting on panel 1
//   SEG_NB = budget-fill    B is a pure FIFO stream (1 tile per MMA step);
//                           its depth is TMA-latency hiding, so give it the
//                           rest of the 14-tile budget (14 * 16 KB + 1 KB =
//                           230400 B fits the 227 KB / 232448 B opt-in cap).
constexpr int SEG             = NS;
constexpr int SEG_NA          = SEG + 1;
constexpr int SEG_NB          = 14 - TMA_STORE_STAGES - SEG_NA;
constexpr int SEG_B_SLOT_BYTES = (BN / 2 / CTA_GROUP) * BK * BF16_BYTES;   // one B panel tile (16 KB)
constexpr int SEG_RING_BYTES  = SEG_NA * A_SLOT_BYTES + SEG_NB * SEG_B_SLOT_BYTES;
static_assert(BN == 512, "SEGMENTED_PANELS is the BN=512 two-panel schedule");
static_assert(SINGLE_TMEM_ACCUM == 1, "SEGMENTED_PANELS drains one shared accumulator");
static_assert(SEG_NB >= 2, "segmented B ring needs >= 2 slots");
#endif

// ── Important: dynamic SMEM is used in TWO non-overlapping phases ──
//
// 1.  During the K-loop, the kernel uses `NS * SLOT_BYTES` bytes —
//     NS slots × (A + B) per slot — as the multi-stage ring buffer.
// 2.  During the epilogue, the same dynamic SMEM is REINTERPRETED as
//     a `[BM][BN+8]` BF16 staging buffer for the coalesced writeback
//     (see ch07).  Its size is `EPILOGUE_STAGING_BYTES` below.
//
// The two phases never overlap in time (`all_mmas_done` separates
// them), so SMEM can be reused.  But the launcher MUST size the
// dynamic SMEM allocation to the MAX of the two phases:
//
//     shared_bytes = max(NS * SLOT_BYTES, EPILOGUE_STAGING_BYTES)
//                  + padding for __align__(1024)
//
// In ch07 (single-CTA) the K-loop term always dominated, so we never
// had to think about this.  In ch08, the per-CTA B-slot SMEM cost
// drops from 32 KB to 16 KB (cluster splits B), which means at low
// NS the K-loop SMEM can fall *below* the staging buffer's needs.
// Specifically, at NS=2, `NS * SLOT_BYTES = 64 KB < 67584 B` and the
// staging dominates.  Allocate too little dynamic SMEM and the
// epilogue scribbles past it → CUDA_ERROR_ILLEGAL_ADDRESS.
//
// See `shared_for()` in `main.py` for the launcher-side computation,
// and the README's "Sizing the dynamic SMEM" subsection for the
// full discussion.
constexpr int WARP_SIZE = 32;
constexpr int THREADS   = NUM_WARPS * WARP_SIZE;  // epilogue worker threads
#if EPILOGUE_OVERLAP
constexpr int LAUNCH_THREADS = (NUM_WARPS + 4) * WARP_SIZE;
#else
constexpr int LAUNCH_THREADS = NUM_WARPS * WARP_SIZE;
#endif


// ── helpers ─────────────────────────────────────────────────────────
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

// TMA load — `.cta_group::2` is the key new modifier.  The tx-count
// is bookkept against a cluster-wide mbar (the peer-CTA arrival is
// what makes both CTAs' arrivals count toward CTA 0's SMEM-compute-full
// mbar).  Without it, peer-CTA loads silently fail to advance the
// mbar and the kernel deadlocks.
#if TWO_CTA
__device__ __forceinline__ void tma_2d_load_g2(
    uint32_t smem_dst, const void* tmap, int x, int y, uint32_t mbar
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar) : "memory");
}
#else
__device__ __forceinline__ void tma_2d_load_g2(
    uint32_t smem_dst, const void* tmap, int x, int y, uint32_t mbar
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar) : "memory");
}
#endif


// A's descriptor — MN-major, unchanged from earlier chapters.
__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    uint64_t b = ((SBO)              >> 4) & 0x3FFFULL;
    return a | (b << 32) | (1ULL << 46) | (2ULL << 61);   // SWIZZLE_128B
}

// K-major B descriptor (same as ch06/07).
__device__ __forceinline__ uint64_t make_desc_K_major(
    uint32_t smem_addr, int lbo_bytes
) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a   = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    uint64_t lbo = ((uint64_t)lbo_bytes >> 4) & 0x3FFFULL;
    uint64_t b   = ((SBO)               >> 4) & 0x3FFFULL;
    return a | (lbo << 16) | (b << 32) | (1ULL << 46) | (2ULL << 61);
}

// idesc with M = CTA_GROUP * BM (cluster spans both CTAs in M),
// bit 16 = 1 (B is K-major).
__device__ __forceinline__ uint32_t make_idesc_bf16_cluster(int m, int n) {
    uint32_t d = 0;
    d |= (1u << 4);                                    // c_format = F32
    d |= (1u << 7);                                    // a_format = BF16
    d |= (1u << 10);                                   // b_format = BF16
    d |= (1u << 16);                                   // B is K-major
    d |= (((uint32_t)(n >> 3) & 0x3F) << 17);          // n_dim
    d |= (((uint32_t)(m >> 4) & 0x1F) << 24);          // m_dim
    return d;
}


// ── tcgen05 MMA wrappers (cta_group::2 cluster / cta_group::1 single) ─
// Same names + signatures under both TWO_CTA arms so the call sites (and the
// MMA_ISSUE macro) are identical — TWO_CTA=1 renders byte-for-byte as the
// cluster tier; TWO_CTA=0 swaps in the single-CTA cta_group::1 instructions.
#if TWO_CTA
__device__ __forceinline__ void tcgen05_alloc_g2(uint32_t smem_dst, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_dst), "r"(n_cols) : "memory");
}
__device__ __forceinline__ void tcgen05_dealloc_g2(uint32_t taddr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(taddr), "r"(n_cols) : "memory");
}
__device__ __forceinline__ void tcgen05_mma_g2(
    uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
    uint32_t idesc, bool enable_d
) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "setp.ne.b32 P, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, P;\n\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
           "r"((uint32_t)enable_d) : "memory");
}
// Multicast commit: arrives on the supplied mbar in every CTA whose bit is set in
// the mask.  cta_mask = (1 << CTA_GROUP) - 1 = 0b11 → both CTAs.
__device__ __forceinline__ void signal_on_mma_completion(uint32_t smem_bar, int16_t cta_mask) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
        "[%0], %1;"
        :: "r"(smem_bar), "h"(cta_mask) : "memory");
}
#else
__device__ __forceinline__ void tcgen05_alloc_g2(uint32_t smem_dst, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_dst), "r"(n_cols) : "memory");
}
__device__ __forceinline__ void tcgen05_dealloc_g2(uint32_t taddr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(taddr), "r"(n_cols) : "memory");
}
__device__ __forceinline__ void tcgen05_mma_g2(
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
// Single-CTA: one arrive, no multicast (cta_mask ignored).
__device__ __forceinline__ void signal_on_mma_completion(uint32_t smem_bar, int16_t cta_mask) {
    (void)cta_mask;
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(smem_bar) : "memory");
}
#endif

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}
__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
// @@TCGEN05_LD@@


// ── mbarrier helpers ────────────────────────────────────────────────
__device__ __forceinline__ void mbarrier_init(uint32_t mb, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mb), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive_no_tx(uint32_t mb) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(mb) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive_no_tx_cluster(uint32_t mb) {
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                 :: "r"(mb) : "memory");
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


// ── Kernel (NS, GROUP_SIZE_M are file-level constexpr knobs) ────────
// ── TMA store helpers (pipelined TMA-store epilogue) ────────────────
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
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory");
}

// MMA-issue building block (shared fragment).  The cluster tier uses the
// g2 (cta_group::2) MMA instruction.
#define MMA_ISSUE(t, a, b, i, e) tcgen05_mma_g2((t), (a), (b), (i), (e))
// @@MMA_CHAIN@@
#undef MMA_ISSUE

__device__ __forceinline__ void matmul_cluster_impl(
    const CUtensorMap* A_tmap,
    const CUtensorMap* B_tmap,
    const CUtensorMap* C_tmap_ptr,
    __nv_bfloat16* __restrict__ C_ptr,
    int M, int N, int K
#if MMC_N_EXTRA >= 1
    , const __nv_bfloat16* __restrict__ mmc_c0   // phase-2 extra epilogue input [M,N]
#endif
) {
    // ── Per-cluster + per-CTA tile coords ───────────────────────────
    //
    // Grid is ceil(M / (CTA_GROUP*BM)) * ceil(N / BN) flat CTA ids.  Each
    // *pair* of CTAs forms one cluster; cta_rank picks which CTA in
    // the pair owns which half.  Ragged edge tiles are clipped by TMA.
    //
    // bid (the cluster id derived from blockIdx.x / CTA_GROUP) is what
    // we'd normally call the grid coordinate; the cluster handles a
    // 2*BM × BN output tile.
#if TWO_CTA
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
#else
    const int cta_rank = 0;   // single-CTA: this CTA is rank 0
#endif

    // Tile coords (the GSM chunked-walk swizzle) are computed PER-TILE
    // inside each path's persistent loop below — both the overlap and the
    // non-overlap branch derive (cluster_m, cluster_n) from their own
    // cluster id, so there are no tile-specific coords at this scope.

    // ── SMEM (per CTA — B is now half-width) ────────────────────────
    extern __shared__ __align__(1024) char smem[];
    const uint32_t SMEM_BASE = (uint32_t)__cvta_generic_to_shared(smem);
#if SEGMENTED_PANELS
    // Split rings: [ A ring SEG_NA | B ring SEG_NB | C_store ].  Each ring has
    // its own data_ready / buffer_free mbarriers — A is freed by panel 1 (the
    // LAST reader, one segment after panel 0), B by whichever panel consumes it.
    auto A_base = [SMEM_BASE](int s) -> uint32_t {
        return SMEM_BASE + s * A_SLOT_BYTES;
    };
    auto B_base = [SMEM_BASE](int s) -> uint32_t {
        return SMEM_BASE + SEG_NA * A_SLOT_BYTES + s * SEG_B_SLOT_BYTES;
    };

    __shared__ uint64_t mbar_a_data_ready[SEG_NA];
    __shared__ uint64_t mbar_a_buffer_free[SEG_NA];
    __shared__ uint64_t mbar_b_data_ready[SEG_NB];
    __shared__ uint64_t mbar_b_buffer_free[SEG_NB];
#else
    auto A_base = [SMEM_BASE](int s) -> uint32_t {
        return SMEM_BASE + s * SLOT_BYTES;
    };
    auto B_base = [SMEM_BASE](int s) -> uint32_t {
        return SMEM_BASE + s * SLOT_BYTES + A_SLOT_BYTES;
    };

    __shared__ uint64_t mbar_compute_data_ready[NS];
    __shared__ uint64_t mbar_compute_buffer_free[NS];
#endif
    __shared__ uint64_t all_mmas_done;
    __shared__ uint32_t tmem_addr_holder[1];

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane    = tid % WARP_SIZE;

#if EPILOGUE_OVERLAP
    {
        // Persistent cluster pipeline: both CTAs stream A/B, CTA 0 issues
        // cta_group::2 MMA into a two-buffer TMEM accumulator, and every CTA
        // drains its own BM x BN output half while the next cluster tile runs.
        __shared__ uint64_t mbar_tmem_data_ready[2];
        __shared__ uint64_t mbar_tmem_buffer_free[2];
#if EPILOGUE_TMA_PIPELINED
        // Pipelined TMA-store mode keeps the K-loop ring intact and reserves
        // compact 128B-swizzled SMEM buffers for chunked TMA stores.
        constexpr int STORE_BUF_BYTES = BM * STORE_N * BF16_BYTES;
#if SEGMENTED_PANELS
        const uint32_t STORE_SMEM_BASE = SMEM_BASE + SEG_RING_BYTES;
        auto C_store = reinterpret_cast<__nv_bfloat16*>(smem + SEG_RING_BYTES);
#else
        const uint32_t STORE_SMEM_BASE = SMEM_BASE + NS * SLOT_BYTES;
        auto C_store = reinterpret_cast<__nv_bfloat16*>(smem + NS * SLOT_BYTES);
#endif
#else
        // Split mode stages one half-BN column panel at a time, reducing
        // epilogue SMEM enough to make room for one extra K-loop stage.
#if EPILOGUE_SPLIT
        constexpr int EPI_STAGE_COLS = BN / 2;
#else
        constexpr int EPI_STAGE_COLS = BN;
#endif
        auto C_sh = reinterpret_cast<__nv_bfloat16(*)[EPI_STAGE_COLS + 8]>(smem + NS * SLOT_BYTES);
#endif

        if (warp_id == 0) {
#if SINGLE_TMEM_ACCUM
            tcgen05_alloc_g2((uint32_t)__cvta_generic_to_shared(tmem_addr_holder), BN);
#else
            tcgen05_alloc_g2((uint32_t)__cvta_generic_to_shared(tmem_addr_holder), 2 * BN);
#endif
        }
        if (warp_id == 0 && elect_sync()) {
#if SEGMENTED_PANELS
            #pragma unroll
            for (int s = 0; s < SEG_NA; s++) {
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_a_data_ready[s]), CTA_GROUP);
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_a_buffer_free[s]), 1);
                mbarrier_arrive_no_tx((uint32_t)__cvta_generic_to_shared(&mbar_a_buffer_free[s]));
            }
            #pragma unroll
            for (int s = 0; s < SEG_NB; s++) {
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_b_data_ready[s]), CTA_GROUP);
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_b_buffer_free[s]), 1);
                mbarrier_arrive_no_tx((uint32_t)__cvta_generic_to_shared(&mbar_b_buffer_free[s]));
            }
#else
            #pragma unroll
            for (int s = 0; s < NS; s++) {
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_compute_data_ready[s]), CTA_GROUP);
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_compute_buffer_free[s]), 1);
                mbarrier_arrive_no_tx((uint32_t)__cvta_generic_to_shared(&mbar_compute_buffer_free[s]));
            }
#endif
            #pragma unroll
            for (int b = 0; b < 2; b++) {
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_tmem_data_ready[b]), 1);
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_tmem_buffer_free[b]), CTA_GROUP);
                uint32_t tmem_buffer_free_cta0 =
                    ((uint32_t)__cvta_generic_to_shared(&mbar_tmem_buffer_free[b])) & 0xFEFFFFFFu;
                mbarrier_arrive_no_tx_cluster(tmem_buffer_free_cta0);
            }
            asm volatile("fence.mbarrier_init.release.cluster;");
        }

#if TWO_CTA
        asm volatile("barrier.cluster.arrive.release.aligned;");
        asm volatile("barrier.cluster.wait.acquire.aligned;");
#else
        __syncthreads();
#endif

        const uint32_t taddr = tmem_addr_holder[0];
#if BN == 512
        constexpr int BN_PANEL = 256;
        constexpr int BN_PANEL_LOCAL = BN_PANEL / CTA_GROUP;
        const uint32_t idesc = make_idesc_bf16_cluster(CTA_GROUP * BM, BN_PANEL);
#else
        const uint32_t idesc = make_idesc_bf16_cluster(CTA_GROUP * BM, BN);
#endif
        const int num_k = K / BK;
#if SEGMENTED_PANELS
        const int num_seg = (num_k + SEG - 1) / SEG;   // last segment may be partial
#endif
        constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;

        // ceil-div tile counts: a ragged M/N launches partial edge tiles whose
        // TMA box is clipped out of bounds (zero-fill on load, masked on store).
        const int grid_m_clusters = (M + CTA_GROUP * BM - 1) / (CTA_GROUP * BM);
        const int grid_n          = (N + BN - 1) / BN;
        const int num_cluster_in_group = GROUP_SIZE_M * grid_n;
        const int num_clusters = grid_m_clusters * grid_n;
        const int cluster_pid = (int)blockIdx.x / CTA_GROUP;
        const int cluster_stride = (int)gridDim.x / CTA_GROUP;
        const int num_my = (cluster_pid >= num_clusters) ? 0
                         : (num_clusters - cluster_pid + cluster_stride - 1) / cluster_stride;

        auto map_off = [&](int ti, int& base_m, int& base_n, int& local_m, int& local_n) {
            int tile = cluster_pid + ti * cluster_stride;
            int group = tile / num_cluster_in_group;
            int first = group * GROUP_SIZE_M;
            int gsm_i = min(grid_m_clusters - first, GROUP_SIZE_M);
            int cm = first + (tile % gsm_i);
            int cn = (tile % num_cluster_in_group) / gsm_i;
            base_m = cm * (CTA_GROUP * BM);
            base_n = cn * BN;
            local_m = base_m + cta_rank * BM;
            local_n = base_n + cta_rank * BN_LOCAL;
        };

        if (warp_id == 0 && elect_sync()) {
#if SEGMENTED_PANELS
            // Segmented loader.  Per segment: stream A[k] + B0[k] (the panel-0
            // phase), then B1[k] (the panel-1 phase) into B-ring slots recycled
            // from B0.  Both rings are FIFOs consumed in exactly this order by
            // the MMA warp, so slot indices are just running counters mod depth.
            uint32_t a_buffer_free_phase[SEG_NA] = {};
            uint32_t b_buffer_free_phase[SEG_NB] = {};
            long ga = 0, gb = 0;
            for (int ti = 0; ti < num_my; ti++) {
                int base_m, base_n, local_m, local_n;
                map_off(ti, base_m, base_n, local_m, local_n);
                for (int seg = 0; seg < num_seg; seg++) {
                    const int base_k = seg * SEG;
                    const int seg_len = min(SEG, num_k - base_k);
                    for (int j = 0; j < seg_len; j++) {      // panel-0 stream: A[k] + B0[k]
                        const int k = base_k + j;
                        const int as = (int)(ga % SEG_NA);
                        uint32_t a_buffer_free_addr =
                            (uint32_t)__cvta_generic_to_shared(&mbar_a_buffer_free[as]);
                        uint32_t a_data_ready_cta0 =
                            ((uint32_t)__cvta_generic_to_shared(&mbar_a_data_ready[as])) & 0xFEFFFFFFu;
                        wait_phase(a_buffer_free_addr, a_buffer_free_phase[as]);
                        tma_2d_load_g2(A_base(as), A_tmap, k * BK, local_m, a_data_ready_cta0);
                        signal_on_bytes_loaded(a_data_ready_cta0, A_SLOT_BYTES);
                        a_buffer_free_phase[as] ^= 1;
                        ga++;

                        const int bs = (int)(gb % SEG_NB);
                        uint32_t b_buffer_free_addr =
                            (uint32_t)__cvta_generic_to_shared(&mbar_b_buffer_free[bs]);
                        uint32_t b_data_ready_cta0 =
                            ((uint32_t)__cvta_generic_to_shared(&mbar_b_data_ready[bs])) & 0xFEFFFFFFu;
                        wait_phase(b_buffer_free_addr, b_buffer_free_phase[bs]);
                        #pragma unroll
                        for (int n = 0; n < BN_PANEL_LOCAL; n += 64)
                            tma_2d_load_g2(B_base(bs) + n * BK * BF16_BYTES, B_tmap,
                                           base_n + 0 * BN_PANEL + cta_rank * BN_PANEL_LOCAL + n,
                                           k * BK, b_data_ready_cta0);
                        signal_on_bytes_loaded(b_data_ready_cta0, SEG_B_SLOT_BYTES);
                        b_buffer_free_phase[bs] ^= 1;
                        gb++;
                    }
                    for (int j = 0; j < seg_len; j++) {      // panel-1 stream: B1[k]
                        const int k = base_k + j;
                        const int bs = (int)(gb % SEG_NB);
                        uint32_t b_buffer_free_addr =
                            (uint32_t)__cvta_generic_to_shared(&mbar_b_buffer_free[bs]);
                        uint32_t b_data_ready_cta0 =
                            ((uint32_t)__cvta_generic_to_shared(&mbar_b_data_ready[bs])) & 0xFEFFFFFFu;
                        wait_phase(b_buffer_free_addr, b_buffer_free_phase[bs]);
                        #pragma unroll
                        for (int n = 0; n < BN_PANEL_LOCAL; n += 64)
                            tma_2d_load_g2(B_base(bs) + n * BK * BF16_BYTES, B_tmap,
                                           base_n + 1 * BN_PANEL + cta_rank * BN_PANEL_LOCAL + n,
                                           k * BK, b_data_ready_cta0);
                        signal_on_bytes_loaded(b_data_ready_cta0, SEG_B_SLOT_BYTES);
                        b_buffer_free_phase[bs] ^= 1;
                        gb++;
                    }
                }
            }
#else
            uint32_t compute_buffer_free_phase[NS] = {};
            long gk = 0;
            for (int ti = 0; ti < num_my; ti++) {
                int base_m, base_n, local_m, local_n;
                map_off(ti, base_m, base_n, local_m, local_n);
                for (int k = 0; k < num_k; k++) {
                    int slot = gk % NS;
                    uint32_t compute_buffer_free_addr =
                        (uint32_t)__cvta_generic_to_shared(&mbar_compute_buffer_free[slot]);
                    uint32_t compute_data_ready_cta0 =
                        ((uint32_t)__cvta_generic_to_shared(&mbar_compute_data_ready[slot])) & 0xFEFFFFFFu;
                    wait_phase(compute_buffer_free_addr, compute_buffer_free_phase[slot]);
                    tma_2d_load_g2(A_base(slot), A_tmap, k * BK, local_m, compute_data_ready_cta0);
#if BN == 512
                    #pragma unroll
                    for (int panel = 0; panel < 2; panel++) {
                        #pragma unroll
                        for (int n = 0; n < BN_PANEL_LOCAL; n += 64) {
                            tma_2d_load_g2(
                                B_base(slot) + (panel * BN_PANEL_LOCAL + n) * BK * BF16_BYTES,
                                B_tmap,
                                base_n + panel * BN_PANEL + cta_rank * BN_PANEL_LOCAL + n,
                                k * BK,
                                compute_data_ready_cta0);
                        }
                    }
#else
                    #pragma unroll
                    for (int n = 0; n < BN_LOCAL; n += 64) {
                        tma_2d_load_g2(B_base(slot) + n * BK * BF16_BYTES,
                                       B_tmap, local_n + n, k * BK, compute_data_ready_cta0);
                    }
#endif
                    signal_on_bytes_loaded(compute_data_ready_cta0, SLOT_BYTES);
                    compute_buffer_free_phase[slot] ^= 1;
                    gk++;
                }
            }
#endif
        } else if (cta_rank == 0 && warp_id == 1 && elect_sync()) {
#if SEGMENTED_PANELS
            // Segmented MMA.  Per segment: panel 0 (seg_len MMAs into [0,256),
            // consuming A[k]+B0[k], freeing each B0 slot), then panel 1 (seg_len
            // MMAs into [256,512) reusing the resident A[k], freeing each B1
            // slot AND the A slot — panel 1 is A's last reader).
            // tmem_data_ready[0] fires at the GLOBAL last k (panel 0 fully
            // accumulated) so the epilogue drains [0,256) concurrently with the
            // last segment's panel-1 MMAs; tmem_data_ready[1] fires after them.
            uint32_t a_data_ready_phase[SEG_NA] = {};
            uint32_t b_data_ready_phase[SEG_NB] = {};
            uint32_t tmem_buffer_free_phase[2] = {};
            long ga = 0, gb = 0;
            for (int ti = 0; ti < num_my; ti++) {
                // [0,256) is free as soon as the previous tile's pass-0 drain
                // finished — typically during its last panel-1 segment, so this
                // wait is usually satisfied already.
                wait_phase((uint32_t)__cvta_generic_to_shared(&mbar_tmem_buffer_free[0]),
                           tmem_buffer_free_phase[0]);
                tmem_buffer_free_phase[0] ^= 1;
                for (int seg = 0; seg < num_seg; seg++) {
                    const int base_k = seg * SEG;
                    const int seg_len = min(SEG, num_k - base_k);
                    for (int j = 0; j < seg_len; j++) {      // panel 0 -> [0,256)
                        const int k = base_k + j;
                        const int as = (int)(ga % SEG_NA);
                        const int bs = (int)(gb % SEG_NB);
                        wait_phase((uint32_t)__cvta_generic_to_shared(&mbar_a_data_ready[as]),
                                   a_data_ready_phase[as]);
                        wait_phase((uint32_t)__cvta_generic_to_shared(&mbar_b_data_ready[bs]),
                                   b_data_ready_phase[bs]);
                        tcgen05_fence_after_thread_sync();
                        issue_mma_chain(taddr, A_base(as), B_base(bs), idesc,
                                        /*first_k_tile=*/ k == 0);
                        a_data_ready_phase[as] ^= 1;
                        b_data_ready_phase[bs] ^= 1;
                        signal_on_mma_completion(
                            (uint32_t)__cvta_generic_to_shared(&mbar_b_buffer_free[bs]), cta_mask);
                        ga++;
                        gb++;
                        if (k == num_k - 1)                   // panel 0 complete -> drain may start
                            signal_on_mma_completion(
                                (uint32_t)__cvta_generic_to_shared(&mbar_tmem_data_ready[0]), cta_mask);
                    }
                    if (seg == 0) {
                        // [256,512) may still be draining the PREVIOUS tile's
                        // panel 1 (its drain overlaps this tile's first panel-0
                        // segment above).  Gate this tile's first panel-1 MMA on
                        // that drain's completion.
                        wait_phase((uint32_t)__cvta_generic_to_shared(&mbar_tmem_buffer_free[1]),
                                   tmem_buffer_free_phase[1]);
                        tmem_buffer_free_phase[1] ^= 1;
                    }
                    for (int j = 0; j < seg_len; j++) {      // panel 1 -> [256,512)
                        const int k = base_k + j;
                        const int as = (int)((ti * (long)num_k + k) % SEG_NA);
                        const int bs = (int)(gb % SEG_NB);
                        wait_phase((uint32_t)__cvta_generic_to_shared(&mbar_b_data_ready[bs]),
                                   b_data_ready_phase[bs]);
                        tcgen05_fence_after_thread_sync();
                        issue_mma_chain(taddr + BN_PANEL, A_base(as), B_base(bs), idesc,
                                        /*first_k_tile=*/ k == 0);
                        b_data_ready_phase[bs] ^= 1;
                        signal_on_mma_completion(
                            (uint32_t)__cvta_generic_to_shared(&mbar_b_buffer_free[bs]), cta_mask);
                        signal_on_mma_completion(
                            (uint32_t)__cvta_generic_to_shared(&mbar_a_buffer_free[as]), cta_mask);
                        gb++;
                    }
                }
                signal_on_mma_completion(
                    (uint32_t)__cvta_generic_to_shared(&mbar_tmem_data_ready[1]), cta_mask);
            }
#else
            uint32_t compute_data_ready_phase[NS] = {};
            uint32_t tmem_buffer_free_phase[2] = {};
            long gk = 0;
            for (int ti = 0; ti < num_my; ti++) {
#if SINGLE_TMEM_ACCUM
                const int buf = 0;
                uint32_t d_tmem = taddr;
#else
                int buf = ti & 1;
                uint32_t d_tmem = taddr + buf * BN;
#endif
                wait_phase((uint32_t)__cvta_generic_to_shared(&mbar_tmem_buffer_free[buf]),
                                    tmem_buffer_free_phase[buf]);
                tmem_buffer_free_phase[buf] ^= 1;
                for (int k = 0; k < num_k; k++) {
                    int slot = gk % NS;
                    uint32_t compute_data_ready_addr =
                        (uint32_t)__cvta_generic_to_shared(&mbar_compute_data_ready[slot]);
                    uint32_t compute_buffer_free_addr =
                        (uint32_t)__cvta_generic_to_shared(&mbar_compute_buffer_free[slot]);
                    wait_phase(compute_data_ready_addr, compute_data_ready_phase[slot]);
                    tcgen05_fence_after_thread_sync();
#if BN == 512
                    issue_mma_chain(d_tmem,
                                    A_base(slot),
                                    B_base(slot),
                                    idesc,
                                    /*first_k_tile=*/ k == 0);
                    issue_mma_chain(d_tmem + BN_PANEL,
                                    A_base(slot),
                                    B_base(slot) + BN_PANEL_LOCAL * BK * BF16_BYTES,
                                    idesc,
                                    /*first_k_tile=*/ k == 0);
#else
                    issue_mma_chain(d_tmem, A_base(slot), B_base(slot), idesc, /*first_k_tile=*/ k == 0);
#endif
                    signal_on_mma_completion(compute_buffer_free_addr, cta_mask);
                    compute_data_ready_phase[slot] ^= 1;
                    gk++;
                }
                signal_on_mma_completion((uint32_t)__cvta_generic_to_shared(&mbar_tmem_data_ready[buf]), cta_mask);
            }
#endif
        } else if (warp_id >= 4 && warp_id < NUM_WARPS + 4) {
            // Contract for the shared overlap-drain fragment: cluster tier writes
            // this CTA's BM x BN output half (local_m / base_n) and releases the
            // TMEM buffer with a CTA-0-masked cluster arrive.
#define EPI_OUT_ROW                 local_m
#define EPI_OUT_COL_BASE            base_n
#define signal_sync(buf)   do { uint32_t _f = ((uint32_t)__cvta_generic_to_shared(&mbar_tmem_buffer_free[buf])) & 0xFEFFFFFFu; mbarrier_arrive_no_tx_cluster(_f); } while (0)
            constexpr int ROW_STRIPS    = BM / 32;
            constexpr int COL_GROUPS    = NUM_WARPS / ROW_STRIPS;
            constexpr int COLS_PER_WARP = BN / COL_GROUPS;
            constexpr int EPI_THREADS   = NUM_WARPS * 32;
            const int ew = warp_id - 4;
            const int row_warp = ew % ROW_STRIPS;
            const int col_warp = ew / ROW_STRIPS;
            const int my_row = row_warp * 32 + lane;
            const int col_base = col_warp * COLS_PER_WARP;
            const int etid = ew * 32 + lane;
            uint32_t full[2] = {};
#if SEGMENTED_PANELS
            // Two-pass drain: panel 0's [0,256) is fully accumulated (and its
            // tmem_data_ready[0] fired) while the MMA warp still computes the
            // last segment's panel 1 — pass 0 overlaps that compute.  Each half
            // is released as soon as ITS drain completes (the fragment fires
            // signal_sync(pass)), so pass 1's drain overlaps the NEXT tile's
            // first panel-0 segment.  store_stage is continuous across the two
            // passes (8 STORE_N chunks per tile).
            for (int ti = 0; ti < num_my; ti++) {
                int base_m, base_n, local_m, local_n;
                map_off(ti, base_m, base_n, local_m, local_n);
                int store_stage = 0;
                for (int pass = 0; pass < 2; pass++) {
                    wait_phase((uint32_t)__cvta_generic_to_shared(&mbar_tmem_data_ready[pass]), full[pass]);
                    full[pass] ^= 1;
                    tcgen05_fence_after_thread_sync();
                    const uint32_t trow =
                        (taddr + pass * BN_PANEL) + ((uint32_t)(cta_rank * BM + row_warp * 32) << 16);
                    constexpr int LDW = TCGEN05_LD_WIDTH;

                    // @@OVERLAP_EPILOGUE@@
                }
            }
#else
            for (int ti = 0; ti < num_my; ti++) {
                int base_m, base_n, local_m, local_n;
#if SINGLE_TMEM_ACCUM
                const int buf = 0;
#else
                int buf = ti & 1;
#endif
                map_off(ti, base_m, base_n, local_m, local_n);
                wait_phase((uint32_t)__cvta_generic_to_shared(&mbar_tmem_data_ready[buf]), full[buf]);
                full[buf] ^= 1;
                tcgen05_fence_after_thread_sync();
                const uint32_t trow =
                    (taddr + buf * BN) + ((uint32_t)(cta_rank * BM + row_warp * 32) << 16);
                constexpr int LDW = TCGEN05_LD_WIDTH;

                // @@OVERLAP_EPILOGUE@@
            }
#endif
#if EPILOGUE_TMA_PIPELINED
            if (ew == 0)
                tma_wait_group<0>();
            asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));
#endif
#undef EPI_OUT_ROW
#undef EPI_OUT_COL_BASE
#undef signal_sync
        }

        __syncthreads();
        if (warp_id == 0 && elect_sync()) {
#if SINGLE_TMEM_ACCUM
            tcgen05_dealloc_g2(taddr, BN);
#else
            tcgen05_dealloc_g2(taddr, 2 * BN);
#endif
        }
        return;
    }
#else
    {

    // ── Persistent grid (Step A: persistent scheduling, no overlap) ──
    //
    // Mirrors the overlap path's persistent loop but drains each tile's
    // epilogue inline (no cross-tile overlap).  TWO_CTA is just a knob
    // here: the cluster barriers / multicast commits degenerate to a
    // single CTA at CTA_GROUP=1, so persistent + non-overlap works for
    // BOTH the single-CTA and 2-CTA arms — there is no hardware reason for
    // the old "the cluster path needs overlap to be persistent" rule.
    //
    // TMEM is allocated ONCE and reused across every tile this CTA visits
    // — cycling alloc/dealloc per tile deadlocks the allocator.  Launched
    // with grid = num_clusters * CTA_GROUP the loop runs exactly once per
    // cluster (bit-identical to the non-persistent schedule); launched with
    // grid = #SMs each cluster walks a strided run of tiles.
    if (warp_id == 0)
        tcgen05_alloc_g2((uint32_t)__cvta_generic_to_shared(tmem_addr_holder), BN);
#if TWO_CTA
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
#else
    __syncthreads();
#endif

    const uint32_t taddr = tmem_addr_holder[0];
    const uint32_t idesc = make_idesc_bf16_cluster(CTA_GROUP * BM, BN);
    const int num_k_iters = K / BK;
    constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;     // 0b11 cluster / 0b1 single

    // Loop-invariant chunked-walk geometry (GSM swizzle; see overlap path).
    // ceil-div (see overlap path): edge tiles rely on TMA out-of-bounds clipping.
    const int grid_n               = (N + BN - 1) / BN;
    const int grid_m_clusters      = (M + CTA_GROUP * BM - 1) / (CTA_GROUP * BM);
    const int num_cluster_in_group = GROUP_SIZE_M * grid_n;
    const int num_clusters         = grid_m_clusters * grid_n;
    const int cluster_stride       = (int)gridDim.x / CTA_GROUP;

    for (int cluster_id = (int)blockIdx.x / CTA_GROUP;
         cluster_id < num_clusters; cluster_id += cluster_stride) {

        // ── Per-tile cluster-swizzle coords (Triton chunked walk; GSM=1
        //    collapses to the natural N-fast walk) ────────────────────
        const int group_id        = cluster_id / num_cluster_in_group;
        const int first_cluster_m  = group_id * GROUP_SIZE_M;
        const int gsm              = min(grid_m_clusters - first_cluster_m, GROUP_SIZE_M);
        const int cluster_m        = first_cluster_m + (cluster_id % gsm);
        const int cluster_n        = (cluster_id % num_cluster_in_group) / gsm;
        const int off_m_cluster    = cluster_m * (CTA_GROUP * BM);
        const int off_n            = cluster_n * BN;            // shared by both CTAs
        const int off_m_local      = off_m_cluster + cta_rank * BM;
        const int off_n_local      = off_n + cta_rank * BN_LOCAL;   // each CTA owns BN/2 cols

        // ── Per-tile mbarrier (re)init.  Safe to reset every tile: the
        //    previous tile's epilogue + barrier drained them all.
        //    mbar_compute_data_ready[s] count = CTA_GROUP (both CTAs' TMA arrivals);
        //    mbar_compute_buffer_free[s] count = 1 (one multicast commit fires both CTAs).
        if (warp_id == 0 && elect_sync()) {
            #pragma unroll
            for (int s = 0; s < NS; s++) {
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_compute_data_ready[s]), CTA_GROUP);
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_compute_buffer_free[s]), 1);
            }
            mbarrier_init((uint32_t)__cvta_generic_to_shared(&all_mmas_done), 1);
            mbarrier_arrive_no_tx((uint32_t)__cvta_generic_to_shared(&mbar_compute_buffer_free[NS - 1]));
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
#if TWO_CTA
        asm volatile("barrier.cluster.arrive.release.aligned;");
        asm volatile("barrier.cluster.wait.acquire.aligned;");
#else
        __syncthreads();
#endif

    // ── TMA warp ────────────────────────────────────────────────────
    //
    // Each CTA loads:
    //   * Its half of A: BM rows starting at off_m_local, full BK K-cols.
    //   * Its half of B: BN_LOCAL N-cols starting at off_n_local, full BK K-rows.
    //
    // expect_tx routes to CTA 0's SMEM-compute-full mbar via the &0xFEFFFFFFu
    // mask — both CTAs arrive at the same (CTA 0's) mbar.
    if (warp_id == 0 && elect_sync()) {
        uint32_t compute_buffer_free_phase[NS] = {};

        // Prologue: front-load NS-1 tiles unconditionally
        #pragma unroll
        for (int s = 0; s < NS - 1; s++) {
            const uint32_t compute_data_ready_local =
                (uint32_t)__cvta_generic_to_shared(&mbar_compute_data_ready[s]);
            const uint32_t compute_data_ready_cta0 =
                compute_data_ready_local & 0xFEFFFFFFu;
            tma_2d_load_g2(A_base(s), A_tmap,
                           /*x=*/ s * BK, /*y=*/ off_m_local, compute_data_ready_cta0);
            #pragma unroll
            for (int n = 0; n < BN_LOCAL; n += 64) {
                tma_2d_load_g2(B_base(s) + n * BK * BF16_BYTES,
                               B_tmap,
                               /*x=*/ off_n_local + n,
                               /*y=*/ s * BK,
                               compute_data_ready_cta0);
            }
            signal_on_bytes_loaded(compute_data_ready_cta0, SLOT_BYTES);
        }

        // Steady-state
        for (int k = 0; k < num_k_iters - (NS - 1); k++) {
            const int slot = (k + NS - 1) % NS;
            const uint32_t compute_buffer_free_addr =
                (uint32_t)__cvta_generic_to_shared(&mbar_compute_buffer_free[slot]);
            const uint32_t compute_data_ready_local =
                (uint32_t)__cvta_generic_to_shared(&mbar_compute_data_ready[slot]);
            const uint32_t compute_data_ready_cta0 =
                compute_data_ready_local & 0xFEFFFFFFu;

            wait_phase(compute_buffer_free_addr, compute_buffer_free_phase[slot]);
            tma_2d_load_g2(A_base(slot), A_tmap,
                           /*x=*/ (k + NS - 1) * BK, /*y=*/ off_m_local,
                           compute_data_ready_cta0);
            #pragma unroll
            for (int n = 0; n < BN_LOCAL; n += 64) {
                tma_2d_load_g2(B_base(slot) + n * BK * BF16_BYTES,
                               B_tmap,
                               /*x=*/ off_n_local + n,
                               /*y=*/ (k + NS - 1) * BK,
                               compute_data_ready_cta0);
            }
            signal_on_bytes_loaded(compute_data_ready_cta0, SLOT_BYTES);
            compute_buffer_free_phase[slot] ^= 1;
        }
    }

    // ── MMA warp — only CTA 0 issues; cta_group::2 result lands in both CTAs' TMEM
    else if (cta_rank == 0 && warp_id == 1 && elect_sync()) {
        uint32_t compute_data_ready_phase[NS] = {};

        for (int k = 0; k < num_k_iters; k++) {
            const int slot = k % NS;
            const uint32_t compute_data_ready_addr =
                (uint32_t)__cvta_generic_to_shared(&mbar_compute_data_ready[slot]);
            const uint32_t compute_buffer_free_addr =
                (uint32_t)__cvta_generic_to_shared(&mbar_compute_buffer_free[slot]);

            wait_phase(compute_data_ready_addr, compute_data_ready_phase[slot]);
            tcgen05_fence_after_thread_sync();

            issue_mma_chain(taddr, A_base(slot), B_base(slot), idesc, /*first_k_tile=*/ (k == 0));
            signal_on_mma_completion(compute_buffer_free_addr, cta_mask);
            compute_data_ready_phase[slot] ^= 1;
        }
        signal_on_mma_completion(
            (uint32_t)__cvta_generic_to_shared(&all_mmas_done), cta_mask);
    }

    // ── Wait for the cluster's main loop to drain (the multicast commit
    //    fires all_mmas_done on both CTAs)
    wait_phase(
        (uint32_t)__cvta_generic_to_shared(&all_mmas_done), 0);

    // ── Epilogue — TMEM → SMEM → coalesced GMEM
    //
    // TMEM is cluster-wide: this CTA's TMEM holds physical rows
    // [0, BM), addressed cluster-logically as [cta_rank*BM, (cta_rank+1)*BM).
    //
    // C_sh aliases the dynamic SMEM (the same allocation that held
    // the multi-stage A/B ring during the K-loop).  Launcher must
    // have sized the dynamic SMEM ≥ EPILOGUE_STAGING_BYTES — see the
    // dual-use comment at the top of the file.
    // ── Epilogue contract + shared fragment splice ──────────────────
    // Persistent: TMEM outlives the tile, so the epilogue must NOT free
    // it — we dealloc once after the loop.  cta_rank, off_m_cluster, off_n
    // are in scope (recomputed per tile above).
#define EPI_DEALLOC(t, n) ((void)0)
        // @@EPILOGUE@@
#undef EPI_DEALLOC

        // Drain this tile fully (TMEM reads + SMEM staging) before the
        // next iteration reuses the same SMEM ring and TMEM accumulator.
#if TWO_CTA
        asm volatile("barrier.cluster.arrive.release.aligned;");
        asm volatile("barrier.cluster.wait.acquire.aligned;");
#else
        __syncthreads();
#endif
    }  // for cluster_id (persistent tile loop)

    // Free the accumulator once, after every tile this CTA owns is done.
    if (warp_id == 0 && elect_sync())
        tcgen05_dealloc_g2(taddr, BN);
    }
#endif
}


// ── Single entry symbol — NS and GROUP_SIZE_M are baked in from the
// constexpr knobs at the top of the file (the webui substitutes them).
#if TWO_CTA
extern "C" __global__ __cluster_dims__(CTA_GROUP, 1, 1) __launch_bounds__(LAUNCH_THREADS, 1)
#else
extern "C" __global__ __launch_bounds__(LAUNCH_THREADS, 1)
#endif
void matmul_cluster(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ CUtensorMap C_tmap,
    __nv_bfloat16* C_ptr, int M, int N, int K
#if MMC_N_EXTRA >= 1
    , const __nv_bfloat16* mmc_c0
#endif
)
{
    matmul_cluster_impl(&A_tmap, &B_tmap, &C_tmap, C_ptr, M, N, K
#if MMC_N_EXTRA >= 1
        , mmc_c0
#endif
    );
}
