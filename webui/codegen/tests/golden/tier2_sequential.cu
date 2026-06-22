#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

// ── User-tunable constants (the webui substitutes these) ────────────
constexpr int BM           = 128;
constexpr int BN           = 256;
constexpr int BK           = 64;
constexpr int NS           = 3;       // multi-stage SMEM ring depth
constexpr int GROUP_SIZE_M = 8;       // CTA-swizzle chunk (1 = no swizzle)
constexpr int NUM_WARPS    = 8;       // total warps per CTA
constexpr int TCGEN05_LD_WIDTH = 8;  // TMEM->reg epilogue load width: 8 or 16 (32-bit elems per lane)
constexpr int EPILOGUE_OVERLAP = 0;  // 1 = persistent 2-CTA cluster + epilogue/K-loop overlap
constexpr int EPILOGUE_SPLIT   = 0;  // 1 = split overlapped int4 writeback into two half-BN passes
constexpr int EPILOGUE_TMA_PIPELINED = 0;  // 1 = chunked staged TMA-store overlap epilogue
constexpr int SINGLE_TMEM_ACCUM = 0;  // 1 = overlap path synchronizes epilogue drain before reusing one TMEM accumulator
constexpr int TWO_CTA          = 0;  // 1 = 2-CTA cluster MMA (cta_group::2); 0 = single-CTA

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
constexpr int LAUNCH_THREADS = NUM_WARPS * WARP_SIZE;


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
__device__ __forceinline__ void tma_2d_load_g2(
    uint32_t smem_dst, const void* tmap, int x, int y, uint32_t mbar
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
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


// ── tcgen05 MMA wrappers (cta_group::2 cluster / cta_group::1 single) ─
// Same names + signatures under both TWO_CTA arms so the call sites (and the
// MMA_ISSUE macro) are identical — TWO_CTA=1 renders byte-for-byte as the
// cluster tier; TWO_CTA=0 swaps in the single-CTA cta_group::1 instructions.
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

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}
__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
// ── tcgen05.ld width helpers (building block) ───────────────────────
// mvp_core splices these at the TCGEN05_LD marker in every tier, so the
// TMEM->register load width (TCGEN05_LD_WIDTH = 8/16 32-bit elems per lane)
// is one knob with the asm in a single place.  Wider = fewer ld + fewer
// wait_ld syncs (more registers, but we're SMEM-occupancy-bound so it's free).
// The epilogue picks the variant via `#if` (resolved at generation time).

__device__ __forceinline__ void tcgen05_ld_32x32b_x8(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        :
          "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
          "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7])
        : "r"(taddr));
}

__device__ __forceinline__ void tcgen05_ld_32x32b_x16(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        :
          "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
          "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7]),
          "=f"(out[8]), "=f"(out[9]), "=f"(out[10]), "=f"(out[11]),
          "=f"(out[12]), "=f"(out[13]), "=f"(out[14]), "=f"(out[15])
        : "r"(taddr));
}


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
// ── MMA-issue chain (building block) ────────────────────────────────
// Issues the K_MMAS tcgen05 MMAs for one K-tile (slot) into the
// accumulator at `taddr`.  mvp_core stitches this into every tier at the
// MMA-chain marker, so the descriptor math + K-step loop live in exactly
// one place.  The only per-tier variation is the MMA instruction
// itself, supplied just before the marker as:
//   MMA_ISSUE(taddr, a_desc, b_desc, idesc, enable_d)
// → tcgen05_mma (single-CTA) or tcgen05_mma_g2 (2-CTA cluster).
__device__ __forceinline__ void issue_mma_chain(
    uint32_t taddr, uint32_t a_base_slot, uint32_t b_base_slot,
    uint32_t idesc, bool first_k_tile)
{
    #pragma unroll
    for (int kk = 0; kk < K_MMAS; kk++) {
        const uint64_t a_desc = make_desc(a_base_slot + kk * MMA_K * BF16_BYTES);
        const uint64_t b_desc = make_desc_K_major(
            b_base_slot + kk * MMA_K * SWIZZLE_ROW_BYTES, BK * SWIZZLE_ROW_BYTES);
        const bool first_ever = first_k_tile && (kk == 0);
        MMA_ISSUE(taddr, a_desc, b_desc, idesc, !first_ever);
    }
}
#undef MMA_ISSUE

__device__ __forceinline__ void matmul_cluster_impl(
    const CUtensorMap* A_tmap,
    const CUtensorMap* B_tmap,
    const CUtensorMap* C_tmap_ptr,
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
    const int cta_rank = 0;   // single-CTA: this CTA is rank 0

    // Tile coords (the GSM chunked-walk swizzle) are computed PER-TILE
    // inside each path's persistent loop below — both the overlap and the
    // non-overlap branch derive (cluster_m, cluster_n) from their own
    // cluster id, so there are no tile-specific coords at this scope.

    // ── SMEM (per CTA — B is now half-width) ────────────────────────
    extern __shared__ __align__(1024) char smem[];
    const uint32_t SMEM_BASE = (uint32_t)__cvta_generic_to_shared(smem);
    auto A_base = [SMEM_BASE](int s) -> uint32_t {
        return SMEM_BASE + s * SLOT_BYTES;
    };
    auto B_base = [SMEM_BASE](int s) -> uint32_t {
        return SMEM_BASE + s * SLOT_BYTES + A_SLOT_BYTES;
    };

    __shared__ uint64_t mbar_smem_compute_full[NS];
    __shared__ uint64_t mbar_smem_compute_free[NS];
    __shared__ uint64_t all_mmas_done;
    __shared__ uint32_t tmem_addr_holder[1];

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane    = tid % WARP_SIZE;

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
    __syncthreads();

    const uint32_t taddr = tmem_addr_holder[0];
    const uint32_t idesc = make_idesc_bf16_cluster(CTA_GROUP * BM, BN);
    const int num_k_iters = K / BK;
    constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;     // 0b11 cluster / 0b1 single

    // Loop-invariant chunked-walk geometry (GSM swizzle; see overlap path).
    const int grid_n               = N / BN;
    const int grid_m_clusters      = M / (CTA_GROUP * BM);
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
        //    mbar_smem_compute_full[s] count = CTA_GROUP (both CTAs' TMA arrivals);
        //    mbar_smem_compute_free[s] count = 1 (one multicast commit fires both CTAs).
        if (warp_id == 0 && elect_sync()) {
            #pragma unroll
            for (int s = 0; s < NS; s++) {
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_smem_compute_full[s]), CTA_GROUP);
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_smem_compute_free[s]), 1);
            }
            mbarrier_init((uint32_t)__cvta_generic_to_shared(&all_mmas_done), 1);
            mbarrier_arrive_no_tx((uint32_t)__cvta_generic_to_shared(&mbar_smem_compute_free[NS - 1]));
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        __syncthreads();

    // ── TMA warp ────────────────────────────────────────────────────
    //
    // Each CTA loads:
    //   * Its half of A: BM rows starting at off_m_local, full BK K-cols.
    //   * Its half of B: BN_LOCAL N-cols starting at off_n_local, full BK K-rows.
    //
    // expect_tx routes to CTA 0's SMEM-compute-full mbar via the &0xFEFFFFFFu
    // mask — both CTAs arrive at the same (CTA 0's) mbar.
    if (warp_id == 0 && elect_sync()) {
        uint32_t smem_compute_free_phase[NS] = {};

        // Prologue: front-load NS-1 tiles unconditionally
        #pragma unroll
        for (int s = 0; s < NS - 1; s++) {
            const uint32_t smem_compute_full_local =
                (uint32_t)__cvta_generic_to_shared(&mbar_smem_compute_full[s]);
            const uint32_t smem_compute_full_cta0 =
                smem_compute_full_local & 0xFEFFFFFFu;
            tma_2d_load_g2(A_base(s), A_tmap,
                           /*x=*/ s * BK, /*y=*/ off_m_local, smem_compute_full_cta0);
            #pragma unroll
            for (int n = 0; n < BN_LOCAL; n += 64) {
                tma_2d_load_g2(B_base(s) + n * BK * BF16_BYTES,
                               B_tmap,
                               /*x=*/ off_n_local + n,
                               /*y=*/ s * BK,
                               smem_compute_full_cta0);
            }
            signal_on_bytes_loaded(smem_compute_full_cta0, SLOT_BYTES);
        }

        // Steady-state
        for (int k = 0; k < num_k_iters - (NS - 1); k++) {
            const int slot = (k + NS - 1) % NS;
            const uint32_t smem_compute_free_addr =
                (uint32_t)__cvta_generic_to_shared(&mbar_smem_compute_free[slot]);
            const uint32_t smem_compute_full_local =
                (uint32_t)__cvta_generic_to_shared(&mbar_smem_compute_full[slot]);
            const uint32_t smem_compute_full_cta0 =
                smem_compute_full_local & 0xFEFFFFFFu;

            wait_phase(smem_compute_free_addr, smem_compute_free_phase[slot]);
            tma_2d_load_g2(A_base(slot), A_tmap,
                           /*x=*/ (k + NS - 1) * BK, /*y=*/ off_m_local,
                           smem_compute_full_cta0);
            #pragma unroll
            for (int n = 0; n < BN_LOCAL; n += 64) {
                tma_2d_load_g2(B_base(slot) + n * BK * BF16_BYTES,
                               B_tmap,
                               /*x=*/ off_n_local + n,
                               /*y=*/ (k + NS - 1) * BK,
                               smem_compute_full_cta0);
            }
            signal_on_bytes_loaded(smem_compute_full_cta0, SLOT_BYTES);
            smem_compute_free_phase[slot] ^= 1;
        }
    }

    // ── MMA warp — only CTA 0 issues; cta_group::2 result lands in both CTAs' TMEM
    else if (cta_rank == 0 && warp_id == 1 && elect_sync()) {
        uint32_t smem_compute_full_phase[NS] = {};

        for (int k = 0; k < num_k_iters; k++) {
            const int slot = k % NS;
            const uint32_t smem_compute_full_addr =
                (uint32_t)__cvta_generic_to_shared(&mbar_smem_compute_full[slot]);
            const uint32_t smem_compute_free_addr =
                (uint32_t)__cvta_generic_to_shared(&mbar_smem_compute_free[slot]);

            wait_phase(smem_compute_full_addr, smem_compute_full_phase[slot]);
            tcgen05_fence_after_thread_sync();

            issue_mma_chain(taddr, A_base(slot), B_base(slot), idesc, /*first_k_tile=*/ (k == 0));
            signal_on_mma_completion(smem_compute_free_addr, cta_mask);
            smem_compute_full_phase[slot] ^= 1;
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
#define EPI_ST_I4(DST, VAL) (*reinterpret_cast<int4*>(DST) = (VAL))
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
        tcgen05_ld_32x32b_x8 (taddr_row + (uint32_t)n, tmp);
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
#undef EPI_DEALLOC

        // Drain this tile fully (TMEM reads + SMEM staging) before the
        // next iteration reuses the same SMEM ring and TMEM accumulator.
        __syncthreads();
    }  // for cluster_id (persistent tile loop)

    // Free the accumulator once, after every tile this CTA owns is done.
    if (warp_id == 0 && elect_sync())
        tcgen05_dealloc_g2(taddr, BN);
    }
}


// ── Single entry symbol — NS and GROUP_SIZE_M are baked in from the
// constexpr knobs at the top of the file (the webui substitutes them).
extern "C" __global__ __launch_bounds__(LAUNCH_THREADS, 1)
void matmul_cluster(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ CUtensorMap C_tmap,
    __nv_bfloat16* C_ptr, int M, int N, int K)
{
    matmul_cluster_impl(&A_tmap, &B_tmap, &C_tmap, C_ptr, M, N, K);
}
