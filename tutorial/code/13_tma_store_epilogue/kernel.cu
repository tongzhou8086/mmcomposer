// Runnable companion for Chapter 13 — TMA store epilogue.
//
// One small change on top of ch12: Phase 2 (the SMEM → GMEM coalesced
// writeback) goes from 128-thread int4-store loop to a single async
// TMA store issued by one thread.  The mechanism mirrors the TMA load
// we've used since chapter 00, just in the opposite direction:
//
//     cp.async.bulk.tensor.2d.global.shared::cta.bulk_group  // store
//   vs.
//     cp.async.bulk.tensor.2d.shared::cluster.global         // load
//
// What the chapter introduces:
//
//   1. The store-side TMA PTX (cp.async.bulk.tensor.2d.global.shared).
//   2. The cross-proxy fence (`fence.proxy.async.shared::cta`).  Phase 1
//      writes SMEM via the GENERIC proxy (regular st.shared int4
//      stores).  The TMA store engine reads SMEM via the ASYNC proxy.
//      Without an explicit fence between them, the async-proxy read may
//      observe stale or partially-written bytes — visible as
//      nondeterministic data errors.
//   3. commit_group / wait_group.read for completion tracking — no
//      mbarrier needed because the store doesn't synchronize with any
//      other warp (we just drain at end of kernel before exit).
//
// What it costs: TMA's SMEM source must be tightly-packed (no row
// padding).  Ch12 used `BN_PAD = BN + 8 = 264` to defang 32-way bank
// conflicts on the Phase 1 int4 stores (see ch07).  TMA can't tolerate
// the extra 8-col gap (it'd write garbage into 8 cols of GMEM per row),
// so this chapter drops the padding.  Phase 1's bank conflicts climb
// back up to 32-way.  Whether the TMA-store win covers the Phase-1
// regression is a real perf question — answered in the README.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

constexpr int BM         = 128;
constexpr int BN         = 256;
constexpr int BK         = 64;
constexpr int MMA_K      = 16;
constexpr int BF16_BYTES = 2;
constexpr int K_MMAS     = BK / MMA_K;

constexpr int CTA_GROUP        = 2;
constexpr int BN_LOCAL         = BN / CTA_GROUP;
constexpr int SWIZZLE_ROW_BYTES = 128;

constexpr int A_SLOT_BYTES = BM       * BK * BF16_BYTES;
constexpr int B_SLOT_BYTES = BN_LOCAL * BK * BF16_BYTES;
constexpr int SLOT_BYTES   = A_SLOT_BYTES + B_SLOT_BYTES;

// Note: NO `BN + 8` padding here.  TMA wants the SMEM source
// tightly-packed.  See the comment block at the top.
constexpr int C_SH_BYTES   = BM * BN * BF16_BYTES;           // 65536 B = 64 KB

constexpr int WARP_SIZE = 32;

// ── helpers (same as ch12 except: + 3 new wrappers for TMA store) ────

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

__device__ __forceinline__ void tma_2d_load_g2(
    uint32_t smem_dst, const void* tmap, int x, int y, uint32_t mbar
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar) : "memory");
}

// NEW: async TMA store, SMEM → GMEM.  Mirror of tma_2d_load_g2 but the
// direction is reversed and there's no mbarrier — completion is tracked
// per warp via commit_group + wait_group.  (No cta_group::2 modifier
// because each CTA stores its own [BM × BN] section independently;
// there's nothing to multicast.)
__device__ __forceinline__ void tma_2d_store(
    const void* tmap, uint32_t smem_src, int x, int y
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
        "[%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_src) : "memory");
}

// NEW: enqueue a commit_group barrier after the TMA store(s).
// commit_group / wait_group is the bulk-store version of cp.async's
// completion barriers — no mbarrier needed, just a per-warp counter.
__device__ __forceinline__ void tma_commit_group() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

// NEW: wait for ALL outstanding bulk-store commit groups to drain.
// At `N` we'd allow `N` in-flight; at 0 we drain everything.
template <int N>
__device__ __forceinline__ void tma_wait_group() {
    asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory");
}

__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    uint64_t b = ((SBO)              >> 4) & 0x3FFFULL;
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

__device__ __forceinline__ uint32_t make_idesc_bf16_cluster(int m, int n) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (1u << 16);
    d |= (((uint32_t)(n >> 3) & 0x3F) << 17);
    d |= (((uint32_t)(m >> 4) & 0x1F) << 24);
    return d;
}

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
// Fixed config: NS=4, GSM=8, NUM_WARPS=8, LD_X=8 (the same combo ch12's
// autotuner tends to settle on at headline shapes).  No template sweep
// — ch13 is about a single structural change, not a knob.
template <int NS, int GROUP_SIZE_M, int NUM_WARPS, int LD_X>
__device__ __forceinline__ void matmul_tmast_impl(
    const CUtensorMap* A_tmap,
    const CUtensorMap* B_tmap,
    const CUtensorMap* C_tmap,                    // NEW: store-side descriptor
    int M, int N, int K
) {
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    const int cluster_id      = blockIdx.x / CTA_GROUP;
    const int grid_n          = N / BN;
    const int grid_m_clusters = M / (CTA_GROUP * BM);

    // CTA-swizzle / chunked grid walk (same math as ch09/ch12).
    const int num_cluster_in_group = GROUP_SIZE_M * grid_n;
    const int group_id             = cluster_id / num_cluster_in_group;
    const int first_cluster_m      = group_id * GROUP_SIZE_M;
    const int gsm        = min(grid_m_clusters - first_cluster_m, GROUP_SIZE_M);
    const int cluster_m  = first_cluster_m + (cluster_id % gsm);
    const int cluster_n  = (cluster_id % num_cluster_in_group) / gsm;

    const int off_m_cluster = cluster_m * (CTA_GROUP * BM);
    const int off_n         = cluster_n * BN;
    const int off_m_local   = off_m_cluster + cta_rank * BM;
    const int off_n_local   = off_n + cta_rank * BN_LOCAL;

    // SMEM allocation — same as ch12 (NS slots aliased with C_sh).
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

    // Setup: warp 0 inits mbarriers, warp 1 allocates TMEM.  Identical
    // to ch12 — no new mbarriers needed for TMA store.
    if (warp_id == 0 && elect_sync()) {
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
    } else if (warp_id == 1) {
        tcgen05_alloc_g2((uint32_t)__cvta_generic_to_shared(tmem_addr_holder), BN);
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    const uint32_t taddr = tmem_addr_holder[0];
    const uint32_t idesc = make_idesc_bf16_cluster(CTA_GROUP * BM, BN);
    const int num_k_iters = K / BK;
    constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;

    // ── TMA warp (verbatim from ch12) ───────────────────────────────
    if (warp_id == 0 && elect_sync()) {
        uint32_t mma_done_phase[NS] = {};
        #pragma unroll
        for (int s = 0; s < NS - 1; s++) {
            const uint32_t mb_local = (uint32_t)__cvta_generic_to_shared(&tile_ready[s]);
            const uint32_t mb_cta0  = mb_local & 0xFEFFFFFFu;
            tma_2d_load_g2(A_base(s), A_tmap, s * BK, off_m_local, mb_cta0);
            #pragma unroll
            for (int n = 0; n < BN_LOCAL; n += 64) {
                tma_2d_load_g2(B_base(s) + n * BK * BF16_BYTES,
                               B_tmap, off_n_local + n, s * BK, mb_cta0);
            }
            mbarrier_arrive_expect_tx(mb_cta0, SLOT_BYTES);
        }
        for (int k = 0; k < num_k_iters - (NS - 1); k++) {
            const int slot = (k + NS - 1) % NS;
            const uint32_t done_mb =
                (uint32_t)__cvta_generic_to_shared(&mma_done[slot]);
            const uint32_t ready_mb_local =
                (uint32_t)__cvta_generic_to_shared(&tile_ready[slot]);
            const uint32_t ready_mb_cta0 = ready_mb_local & 0xFEFFFFFFu;

            mbarrier_wait_phase(done_mb, mma_done_phase[slot]);
            tma_2d_load_g2(A_base(slot), A_tmap,
                           (k + NS - 1) * BK, off_m_local, ready_mb_cta0);
            #pragma unroll
            for (int n = 0; n < BN_LOCAL; n += 64) {
                tma_2d_load_g2(B_base(slot) + n * BK * BF16_BYTES,
                               B_tmap, off_n_local + n,
                               (k + NS - 1) * BK, ready_mb_cta0);
            }
            mbarrier_arrive_expect_tx(ready_mb_cta0, SLOT_BYTES);
            mma_done_phase[slot] ^= 1;
        }
    }

    // ── MMA warp (verbatim from ch12) ───────────────────────────────
    else if (cta_rank == 0 && warp_id == 1 && elect_sync()) {
        uint32_t tile_ready_phase[NS] = {};
        for (int k = 0; k < num_k_iters; k++) {
            const int slot = k % NS;
            const uint32_t ready_mb =
                (uint32_t)__cvta_generic_to_shared(&tile_ready[slot]);
            const uint32_t done_mb =
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
                tcgen05_mma_g2(taddr, a_desc, b_desc, idesc, !first_ever);
            }
            tcgen05_commit_mcast_g2(done_mb, cta_mask);
            tile_ready_phase[slot] ^= 1;
        }
        tcgen05_commit_mcast_g2(
            (uint32_t)__cvta_generic_to_shared(&all_mmas_done), cta_mask);
    }

    mbarrier_wait_phase(
        (uint32_t)__cvta_generic_to_shared(&all_mmas_done), 0);

    // ── Epilogue ────────────────────────────────────────────────────
    //
    // C_sh now tightly-packed: BM × BN, no +8 padding (TMA can't
    // tolerate a hole between rows of the SMEM source).  Phase 1's
    // int4 SMEM stores pay 32-way bank conflicts as a result; the
    // README discusses the trade-off.
    auto C_sh = reinterpret_cast<__nv_bfloat16(*)[BN]>(smem);

    tcgen05_fence_after_thread_sync();

    // Phase 1: TMEM → C_sh (same logic as ch12, just BN_PAD → BN).
    constexpr int THREADS = NUM_WARPS * WARP_SIZE;
    int my_row, col_base, col_end;
    uint32_t taddr_row_base;
    if constexpr (NUM_WARPS == 4) {
        my_row         = warp_id * 32 + lane;
        taddr_row_base = taddr + ((uint32_t)(cta_rank * BM + warp_id * 32) << 16);
        col_base = 0;
        col_end  = BN;
    } else if constexpr (NUM_WARPS == 8) {
        const int row_warp = warp_id & 3;
        const int col_warp = warp_id >> 2;
        my_row         = row_warp * 32 + lane;
        taddr_row_base = taddr + ((uint32_t)(cta_rank * BM + row_warp * 32) << 16);
        col_base = col_warp * (BN / 2);
        col_end  = col_base + (BN / 2);
    } else {  // NUM_WARPS == 16
        const int row_warp = warp_id & 3;
        const int col_warp = (warp_id >> 2) & 3;
        my_row         = row_warp * 32 + lane;
        taddr_row_base = taddr + ((uint32_t)(cta_rank * BM + row_warp * 32) << 16);
        col_base = col_warp * (BN / 4);
        col_end  = col_base + (BN / 4);
    }

    #pragma unroll
    for (int n = col_base; n < col_end; n += LD_X) {
        float tmp[LD_X];
        tcgen05_ld_32x32b_x8(taddr_row_base + (uint32_t)n, tmp);
        tcgen05_wait_ld();

        constexpr int N_PACKS = LD_X / 2;
        __nv_bfloat162 packed[N_PACKS];
        #pragma unroll
        for (int i = 0; i < N_PACKS; i++) {
            packed[i] = __floats2bfloat162_rn(tmp[2 * i], tmp[2 * i + 1]);
        }
        constexpr int N_INT4S = LD_X / 8;
        #pragma unroll
        for (int j = 0; j < N_INT4S; j++) {
            *reinterpret_cast<int4*>(&C_sh[my_row][n + j * 8]) =
                reinterpret_cast<int4*>(packed)[j];
        }
    }

    __syncthreads();

    // ── Cross-proxy fence ───────────────────────────────────────────
    // Phase 1 wrote C_sh via the GENERIC proxy (regular st.shared).
    // The TMA store below reads C_sh via the ASYNC proxy.  Without
    // this fence, the async-proxy read may observe stale or
    // partially-written bytes.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    // ── Phase 2: async TMA store, one PTX per CTA ──────────────────
    // Each CTA stores its own BM × BN section at (off_n, my_row_base).
    // commit_group + wait_group track completion — no mbarrier needed.
    const int my_row_global_base = off_m_cluster + cta_rank * BM;
    if (warp_id == 0 && elect_sync()) {
        tma_2d_store(C_tmap,
                     (uint32_t)__cvta_generic_to_shared(&C_sh[0][0]),
                     /*x=*/ off_n,
                     /*y=*/ my_row_global_base);
        tma_commit_group();
        tma_wait_group<0>();
    }

    // ── TMEM dealloc AFTER the TMA store completes ─────────────────
    //
    // Important ordering: tcgen05.dealloc.cta_group::2.sync.aligned
    // changes the cluster's tcgen05 state in a way that, if issued
    // BEFORE a subsequent TMA store + wait_group, makes the bulk-copy
    // engine deadlock (the wait_group never completes).  Discovered
    // by bisecting against ch12; both ordering combinations matter.
    //
    // The fix is simple: drain the TMA store first, then dealloc.
    // This is the only spot in the chapter where step order matters
    // for correctness rather than just for perf.
    if (warp_id == 0 && elect_sync()) {
        tcgen05_dealloc_g2(taddr, BN);
    }
}


// ── Launchers ───────────────────────────────────────────────────────
// Single config: NS=4, GSM=8, NW=8, LDX=8.  No autotune in this chapter.
#define MAKE_LAUNCHER(NS_, GSM_, NW_, LDX_)                                        \
extern "C" __global__ __cluster_dims__(CTA_GROUP, 1, 1)                            \
__launch_bounds__(NW_ * WARP_SIZE, 1)                                              \
void matmul_tmast_ns##NS_##_gsm##GSM_##_nw##NW_##_ldx##LDX_(                       \
    const __grid_constant__ CUtensorMap A_tmap,                                    \
    const __grid_constant__ CUtensorMap B_tmap,                                    \
    const __grid_constant__ CUtensorMap C_tmap,                                    \
    int M, int N, int K)                                                           \
{                                                                                   \
    matmul_tmast_impl<NS_, GSM_, NW_, LDX_>(&A_tmap, &B_tmap, &C_tmap, M, N, K);   \
}

MAKE_LAUNCHER(4, 8, 8, 8)

#undef MAKE_LAUNCHER
