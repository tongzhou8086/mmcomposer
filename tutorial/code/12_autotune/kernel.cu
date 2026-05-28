// Runnable companion for Chapter 12 — autotuning capstone.
//
// Same templated kernel as ch10/ch11 (CTA swizzle + 2-CTA cluster +
// multi-stage + warp-spec, parameterized on NS, GROUP_SIZE_M,
// NUM_WARPS, LD_X).  The new piece is on the host side: main.py
// builds an Autotuner that compiles a sweep of variants and picks
// the best per-shape.
//
// For this chapter we sweep two of the four knobs:
//   * NS in {3, 4, 5, 6, 7} (5 values)
//   * GROUP_SIZE_M in {1, 4, 8, 16} (4 values)
// at fixed NUM_WARPS = 4 and LD_X = 8 — ch10 showed those two don't
// move the needle much at 8192³, and keeping the sweep 2D keeps
// compile time honest (20 variants ≈ 3–5 min nvcc).  Extending to
// the full 4D cross is a one-line edit to MAKE_LAUNCHER below.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

constexpr int BM        = 128;
constexpr int BN        = 256;
constexpr int BK        = 64;
constexpr int MMA_K     = 16;
constexpr int BF16_BYTES = 2;
constexpr int K_MMAS    = BK / MMA_K;        // 4

constexpr int CTA_GROUP        = 2;
constexpr int BN_LOCAL         = BN / CTA_GROUP;     // 128 — per-CTA N width of B
constexpr int SWIZZLE_ROW_BYTES = 128;               // one 128B-swizzle atom row

// Per-stage SMEM per CTA: A = BM*BK*2 = 16 KB; B = BN_LOCAL*BK*2 = 16 KB.
// Total 32 KB / stage / CTA — half of ch07's 48 KB / stage / CTA.
constexpr int A_SLOT_BYTES = BM       * BK * BF16_BYTES;       // 16 KB
constexpr int B_SLOT_BYTES = BN_LOCAL * BK * BF16_BYTES;       // 16 KB
constexpr int SLOT_BYTES   = A_SLOT_BYTES + B_SLOT_BYTES;      // 32 KB / slot

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
constexpr int BN_PAD                 = BN + 8;
constexpr int EPILOGUE_STAGING_BYTES = BM * BN_PAD * BF16_BYTES;   // 67584 B ≈ 66 KB

constexpr int WARP_SIZE = 32;
// THREADS is no longer fixed — it's NUM_WARPS * WARP_SIZE per template
// instantiation.  See the launcher macros at the bottom.


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
// what makes both CTAs' arrivals count toward CTA 0's tile_ready
// mbar).  Without it, peer-CTA loads silently fail to advance the
// mbar and the kernel deadlocks.
__device__ __forceinline__ void tma_2d_load_g2(
    uint32_t smem_dst, const void* tmap, int x, int y, uint32_t mbar
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar) : "memory");
}


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


// ── tcgen05 cta_group::2 wrappers ───────────────────────────────────
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
// Multicast commit: fires mma_done on every CTA whose bit is set in
// the mask.  cta_mask = (1 << CTA_GROUP) - 1 = 0b11 → both CTAs.
__device__ __forceinline__ void tcgen05_commit_mcast_g2(uint32_t smem_bar, int16_t cta_mask) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
        "[%0], %1;"
        :: "r"(smem_bar), "h"(cta_mask) : "memory");
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
__device__ __forceinline__ void tcgen05_ld_32x32b_x16(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        : "=f"(out[0]),  "=f"(out[1]),  "=f"(out[2]),  "=f"(out[3]),
          "=f"(out[4]),  "=f"(out[5]),  "=f"(out[6]),  "=f"(out[7]),
          "=f"(out[8]),  "=f"(out[9]),  "=f"(out[10]), "=f"(out[11]),
          "=f"(out[12]), "=f"(out[13]), "=f"(out[14]), "=f"(out[15])
        : "r"(taddr));
}
__device__ __forceinline__ void tcgen05_ld_32x32b_x32(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
         "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=f"(out[0]),  "=f"(out[1]),  "=f"(out[2]),  "=f"(out[3]),
          "=f"(out[4]),  "=f"(out[5]),  "=f"(out[6]),  "=f"(out[7]),
          "=f"(out[8]),  "=f"(out[9]),  "=f"(out[10]), "=f"(out[11]),
          "=f"(out[12]), "=f"(out[13]), "=f"(out[14]), "=f"(out[15]),
          "=f"(out[16]), "=f"(out[17]), "=f"(out[18]), "=f"(out[19]),
          "=f"(out[20]), "=f"(out[21]), "=f"(out[22]), "=f"(out[23]),
          "=f"(out[24]), "=f"(out[25]), "=f"(out[26]), "=f"(out[27]),
          "=f"(out[28]), "=f"(out[29]), "=f"(out[30]), "=f"(out[31])
        : "r"(taddr));
}
__device__ __forceinline__ void tcgen05_ld_32x32b_x64(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
         "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
         "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
         "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
        : "=f"(out[0]),  "=f"(out[1]),  "=f"(out[2]),  "=f"(out[3]),
          "=f"(out[4]),  "=f"(out[5]),  "=f"(out[6]),  "=f"(out[7]),
          "=f"(out[8]),  "=f"(out[9]),  "=f"(out[10]), "=f"(out[11]),
          "=f"(out[12]), "=f"(out[13]), "=f"(out[14]), "=f"(out[15]),
          "=f"(out[16]), "=f"(out[17]), "=f"(out[18]), "=f"(out[19]),
          "=f"(out[20]), "=f"(out[21]), "=f"(out[22]), "=f"(out[23]),
          "=f"(out[24]), "=f"(out[25]), "=f"(out[26]), "=f"(out[27]),
          "=f"(out[28]), "=f"(out[29]), "=f"(out[30]), "=f"(out[31]),
          "=f"(out[32]), "=f"(out[33]), "=f"(out[34]), "=f"(out[35]),
          "=f"(out[36]), "=f"(out[37]), "=f"(out[38]), "=f"(out[39]),
          "=f"(out[40]), "=f"(out[41]), "=f"(out[42]), "=f"(out[43]),
          "=f"(out[44]), "=f"(out[45]), "=f"(out[46]), "=f"(out[47]),
          "=f"(out[48]), "=f"(out[49]), "=f"(out[50]), "=f"(out[51]),
          "=f"(out[52]), "=f"(out[53]), "=f"(out[54]), "=f"(out[55]),
          "=f"(out[56]), "=f"(out[57]), "=f"(out[58]), "=f"(out[59]),
          "=f"(out[60]), "=f"(out[61]), "=f"(out[62]), "=f"(out[63])
        : "r"(taddr));
}

// Templated dispatcher — pick the right asm wrapper for compile-time LD_X.
template <int LD_X>
__device__ __forceinline__ void tcgen05_ld_packed(uint32_t taddr, float* out) {
    if constexpr (LD_X ==  8) tcgen05_ld_32x32b_x8 (taddr, out);
    else if constexpr (LD_X == 16) tcgen05_ld_32x32b_x16(taddr, out);
    else if constexpr (LD_X == 32) tcgen05_ld_32x32b_x32(taddr, out);
    else if constexpr (LD_X == 64) tcgen05_ld_32x32b_x64(taddr, out);
    else static_assert(LD_X == 8 || LD_X == 16 || LD_X == 32 || LD_X == 64,
                       "LD_X must be one of 8, 16, 32, 64");
}


// ── mbarrier helpers ────────────────────────────────────────────────
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


// ── Kernel (templated on NS) ────────────────────────────────────────
template <int NS, int GROUP_SIZE_M, int NUM_WARPS, int LD_X>
__device__ __forceinline__ void matmul_cluster_impl(
    const CUtensorMap* A_tmap,
    const CUtensorMap* B_tmap,
    __nv_bfloat16* __restrict__ C_ptr,
    int M, int N, int K
) {
    // ── Per-cluster + per-CTA tile coords ───────────────────────────
    //
    // Grid is (M / (CTA_GROUP*BM)) * (N / BN) flat CTA ids.  Each
    // *pair* of CTAs forms one cluster; cta_rank picks which CTA in
    // the pair owns which half.
    //
    // bid (the cluster id derived from blockIdx.x / CTA_GROUP) is what
    // we'd normally call the grid coordinate; the cluster handles a
    // 2*BM × BN output tile.
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    const int cluster_id      = blockIdx.x / CTA_GROUP;
    const int grid_n          = N / BN;
    const int grid_m_clusters = M / (CTA_GROUP * BM);

    // ── Triton-style chunked walk in (cluster_m, cluster_n) ─────────
    //
    // Re-pack the grid so consecutive CTAs share a B-stripe (the
    // expensive thing to stream) instead of an A-stripe.  Within each
    // group of GROUP_SIZE_M × grid_n cluster IDs, walk M fast and N
    // slow; advance to the next group when the previous one is done.
    //
    // GSM = 1 reproduces ch08's N-fast walk exactly (verify: with
    // GSM=1, gsm=1, cluster_m = cluster_id / grid_n,
    //                cluster_n = cluster_id % grid_n).
    const int num_cluster_in_group = GROUP_SIZE_M * grid_n;
    const int group_id             = cluster_id / num_cluster_in_group;
    const int first_cluster_m      = group_id * GROUP_SIZE_M;
    // gsm shrinks for the (possibly ragged) last group so we stay
    // inside the M-grid: min(remaining cluster-rows, GROUP_SIZE_M).
    const int gsm        = min(grid_m_clusters - first_cluster_m, GROUP_SIZE_M);
    const int cluster_m  = first_cluster_m + (cluster_id % gsm);
    const int cluster_n  = (cluster_id % num_cluster_in_group) / gsm;

    const int off_m_cluster = cluster_m * (CTA_GROUP * BM);    // cluster M-base
    const int off_n         = cluster_n * BN;                  // shared by both CTAs
    const int off_m_local   = off_m_cluster + cta_rank * BM;
    const int off_n_local   = off_n + cta_rank * BN_LOCAL;     // each CTA owns BN/2 cols

    // ── SMEM (per CTA — B is now half-width) ────────────────────────
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

    // ── One-time setup ──────────────────────────────────────────────
    //
    // tile_ready[s] count = CTA_GROUP = 2: both CTAs' TMA arrivals
    // are required.  mma_done[s] count = 1: one multicast commit
    // per stage from CTA 0 fires both CTAs' mma_done via the
    // multicast::cluster modifier.
    if (warp_id == 0) {
        if (elect_sync()) {
            #pragma unroll
            for (int s = 0; s < NS; s++) {
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&tile_ready[s]),
                              CTA_GROUP);
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mma_done[s]), 1);
            }
            mbarrier_init((uint32_t)__cvta_generic_to_shared(&all_mmas_done), 1);
            mbarrier_arrive_no_tx(
                (uint32_t)__cvta_generic_to_shared(&mma_done[NS - 1]));
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        tcgen05_alloc_g2((uint32_t)__cvta_generic_to_shared(tmem_addr_holder), BN);
    }

    // ── Cluster barrier replaces __syncthreads at init ──────────────
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    const uint32_t taddr = tmem_addr_holder[0];
    const uint32_t idesc = make_idesc_bf16_cluster(CTA_GROUP * BM, BN);

    const int num_k_iters = K / BK;
    constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;     // 0b11

    // ── TMA warp ────────────────────────────────────────────────────
    //
    // Each CTA loads:
    //   * Its half of A: BM rows starting at off_m_local, full BK K-cols.
    //   * Its half of B: BN_LOCAL N-cols starting at off_n_local, full BK K-rows.
    //
    // expect_tx routes to CTA 0's tile_ready via the &0xFEFFFFFFu
    // mask — both CTAs arrive at the same (CTA 0's) mbar.
    if (warp_id == 0 && elect_sync()) {
        uint32_t mma_done_phase[NS] = {};

        // Prologue: front-load NS-1 tiles unconditionally
        #pragma unroll
        for (int s = 0; s < NS - 1; s++) {
            const uint32_t mb_local = (uint32_t)__cvta_generic_to_shared(&tile_ready[s]);
            const uint32_t mb_cta0  = mb_local & 0xFEFFFFFFu;
            tma_2d_load_g2(A_base(s), A_tmap,
                           /*x=*/ s * BK, /*y=*/ off_m_local, mb_cta0);
            #pragma unroll
            for (int n = 0; n < BN_LOCAL; n += 64) {
                tma_2d_load_g2(B_base(s) + n * BK * BF16_BYTES,
                               B_tmap,
                               /*x=*/ off_n_local + n,
                               /*y=*/ s * BK,
                               mb_cta0);
            }
            mbarrier_arrive_expect_tx(mb_cta0, SLOT_BYTES);
        }

        // Steady-state
        for (int k = 0; k < num_k_iters - (NS - 1); k++) {
            const int slot = (k + NS - 1) % NS;
            const uint32_t done_mb  =
                (uint32_t)__cvta_generic_to_shared(&mma_done[slot]);
            const uint32_t ready_mb_local =
                (uint32_t)__cvta_generic_to_shared(&tile_ready[slot]);
            const uint32_t ready_mb_cta0 = ready_mb_local & 0xFEFFFFFFu;

            mbarrier_wait_phase(done_mb, mma_done_phase[slot]);
            tma_2d_load_g2(A_base(slot), A_tmap,
                           /*x=*/ (k + NS - 1) * BK, /*y=*/ off_m_local,
                           ready_mb_cta0);
            #pragma unroll
            for (int n = 0; n < BN_LOCAL; n += 64) {
                tma_2d_load_g2(B_base(slot) + n * BK * BF16_BYTES,
                               B_tmap,
                               /*x=*/ off_n_local + n,
                               /*y=*/ (k + NS - 1) * BK,
                               ready_mb_cta0);
            }
            mbarrier_arrive_expect_tx(ready_mb_cta0, SLOT_BYTES);
            mma_done_phase[slot] ^= 1;
        }
    }

    // ── MMA warp — only CTA 0 issues; cta_group::2 result lands in both CTAs' TMEM
    else if (cta_rank == 0 && warp_id == 1 && elect_sync()) {
        uint32_t tile_ready_phase[NS] = {};

        for (int k = 0; k < num_k_iters; k++) {
            const int slot = k % NS;
            const uint32_t ready_mb =
                (uint32_t)__cvta_generic_to_shared(&tile_ready[slot]);
            const uint32_t done_mb  =
                (uint32_t)__cvta_generic_to_shared(&mma_done[slot]);

            mbarrier_wait_phase(ready_mb, tile_ready_phase[slot]);
            tcgen05_fence_after_thread_sync();

            #pragma unroll
            for (int kk = 0; kk < K_MMAS; kk++) {
                const uint64_t a_desc = make_desc(
                    A_base(slot) + kk * MMA_K * BF16_BYTES);
                const uint64_t b_desc = make_desc_K_major(
                    B_base(slot) + kk * MMA_K * SWIZZLE_ROW_BYTES,
                    BK * SWIZZLE_ROW_BYTES);
                const bool first_ever = (k == 0) && (kk == 0);
                tcgen05_mma_g2(taddr, a_desc, b_desc, idesc,
                               /*enable_d=*/ !first_ever);
            }
            tcgen05_commit_mcast_g2(done_mb, cta_mask);
            tile_ready_phase[slot] ^= 1;
        }
        tcgen05_commit_mcast_g2(
            (uint32_t)__cvta_generic_to_shared(&all_mmas_done), cta_mask);
    }

    // ── Wait for the cluster's main loop to drain (each CTA waits on
    //    its own mma_done — the multicast commit fires both)
    mbarrier_wait_phase(
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
    auto C_sh = reinterpret_cast<__nv_bfloat16(*)[BN_PAD]>(smem);

    tcgen05_fence_after_thread_sync();

    // ── Phase 1: TMEM → SMEM, parameterized by NUM_WARPS and LD_X ───
    //
    // 4 warps: each warp owns BM/4 = 32 rows × all BN cols.
    // 8 warps: row_warp ∈ {0..3} picks the 32-row strip;
    //          col_warp ∈ {0..1} picks the BN/2 col-half.
    //          Each warp covers 32 rows × BN/2 cols.
    // LD_X: how many TMEM cols each tcgen05.ld call fetches (8/16/32/64).
    constexpr int THREADS = NUM_WARPS * WARP_SIZE;
    int my_row, col_base, col_end;
    uint32_t taddr_row_base;
    if constexpr (NUM_WARPS == 4) {
        my_row         = warp_id * 32 + lane;            // 0..127
        taddr_row_base = taddr + ((uint32_t)(cta_rank * BM + warp_id * 32) << 16);
        col_base = 0;
        col_end  = BN;
    } else {  // NUM_WARPS == 8
        const int row_warp = warp_id & 3;                // 0..3
        const int col_warp = warp_id >> 2;               // 0..1
        my_row         = row_warp * 32 + lane;
        taddr_row_base = taddr + ((uint32_t)(cta_rank * BM + row_warp * 32) << 16);
        col_base = col_warp * (BN / 2);
        col_end  = col_base + (BN / 2);
    }

    #pragma unroll
    for (int n = col_base; n < col_end; n += LD_X) {
        float tmp[LD_X];
        tcgen05_ld_packed<LD_X>(taddr_row_base + (uint32_t)n, tmp);
        tcgen05_wait_ld();

        constexpr int N_PACKS = LD_X / 2;
        __nv_bfloat162 packed[N_PACKS];
        #pragma unroll
        for (int i = 0; i < N_PACKS; i++) {
            packed[i] = __floats2bfloat162_rn(tmp[2 * i], tmp[2 * i + 1]);
        }
        // Each LD_X-wide block produces LD_X/2 bfloat162 = LD_X bytes ×
        // 2 = 2*LD_X bytes.  We write it as LD_X/8 successive int4s.
        constexpr int N_INT4S = LD_X / 8;
        #pragma unroll
        for (int j = 0; j < N_INT4S; j++) {
            *reinterpret_cast<int4*>(&C_sh[my_row][n + j * 8]) =
                reinterpret_cast<int4*>(packed)[j];
        }
    }

    __syncthreads();
    if (warp_id == 0 && elect_sync()) {
        tcgen05_dealloc_g2(taddr, BN);
    }

    // ── Phase 2: SMEM → GMEM, coalesced
    //
    // TB_SIZE = NUM_WARPS * 32.  At NUM_WARPS=8, per-thread store
    // count halves vs NUM_WARPS=4 (16 stores instead of 32).
    constexpr int CHUNK_BF16        = 8;
    constexpr int CHUNKS_PER_ROW    = BN / CHUNK_BF16;
    constexpr int STORES_PER_THREAD = (BM * BN) / (THREADS * CHUNK_BF16);
    const int my_row_global_base = off_m_cluster + cta_rank * BM;

    #pragma unroll
    for (int s = 0; s < STORES_PER_THREAD; s++) {
        const int flat = tid + s * THREADS;
        const int row  = flat / CHUNKS_PER_ROW;
        const int col  = (flat % CHUNKS_PER_ROW) * CHUNK_BF16;
        const int gr   = my_row_global_base + row;
        const int gc   = off_n + col;
        *reinterpret_cast<int4*>(&C_ptr[gr * N + gc]) =
            *reinterpret_cast<const int4*>(&C_sh[row][col]);
    }
}


// ── Launchers — one per (NS, GSM) so main.py can sweep ──────────────
//
// For this chapter we fix NS = 5 (the ch08 sweet spot at 4096³) and
// only sweep GROUP_SIZE_M ∈ {1, 4, 8, 16}.  GSM = 1 is the
// no-swizzle baseline (matches ch08's walk verbatim).
// Launchers — full 4D sweep: NS × GSM × NUM_WARPS × LD_X.
// 5 × 4 × 2 × 4 = 160 variants.  main.py's Autotuner times all of them
// per problem shape and caches the winner.
#define MAKE_LAUNCHER(NS_, GSM_, NW_, LDX_)                                               \
extern "C" __global__ __cluster_dims__(CTA_GROUP, 1, 1)                                   \
__launch_bounds__(NW_ * WARP_SIZE, 1)                                                     \
void matmul_tune_ns##NS_##_gsm##GSM_##_nw##NW_##_ldx##LDX_(                               \
    const __grid_constant__ CUtensorMap A_tmap,                                           \
    const __grid_constant__ CUtensorMap B_tmap,                                            \
    __nv_bfloat16* C_ptr, int M, int N, int K)                                            \
{                                                                                          \
    matmul_cluster_impl<NS_, GSM_, NW_, LDX_>(&A_tmap, &B_tmap, C_ptr, M, N, K);          \
}

#define MAKE_LDX_ROW(NS_, GSM_, NW_)  \
    MAKE_LAUNCHER(NS_, GSM_, NW_, 8)   \
    MAKE_LAUNCHER(NS_, GSM_, NW_, 16)  \
    MAKE_LAUNCHER(NS_, GSM_, NW_, 32)  \
    MAKE_LAUNCHER(NS_, GSM_, NW_, 64)

#define MAKE_NW_ROW(NS_, GSM_)    \
    MAKE_LDX_ROW(NS_, GSM_, 4)    \
    MAKE_LDX_ROW(NS_, GSM_, 8)

#define MAKE_GSM_ROW(NS_)         \
    MAKE_NW_ROW(NS_, 1)            \
    MAKE_NW_ROW(NS_, 4)            \
    MAKE_NW_ROW(NS_, 8)            \
    MAKE_NW_ROW(NS_, 16)

MAKE_GSM_ROW(3)
MAKE_GSM_ROW(4)
MAKE_GSM_ROW(5)
MAKE_GSM_ROW(6)
MAKE_GSM_ROW(7)

#undef MAKE_GSM_ROW
#undef MAKE_NW_ROW
#undef MAKE_LDX_ROW
#undef MAKE_LAUNCHER
