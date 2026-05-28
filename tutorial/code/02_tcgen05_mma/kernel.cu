// Runnable companion for Chapter 02 — first tcgen05.mma.
//
// Problem:  C[M, N] = A[M, K] @ B[K, N]   with M=128, N=256, K=64.
//
// One CTA, 128 threads (4 warps).  No pipelining, no outer K-loop —
// the whole problem fits in a single SMEM tile and is consumed by one
// "stage" of MMAs.  This is the minimal kernel that exercises every
// piece chapter 02 introduced:
//
//   1.  TMEM alloc (256 cols, one warp issues)
//   2.  Two TMA bulk loads (A and B, single bulk per operand,
//       SWIZZLE_128B)
//   3.  tcgen05.fence::after_thread_sync between TMA proxy and MMA
//   4.  Four back-to-back tcgen05.mma calls (K = 16 each → K = 64
//       covered), with the accumulate predicate starting false then
//       true.
//   5.  tcgen05.commit + mbarrier wait
//   6.  TMEM → registers → GMEM via tcgen05.ld + wait::ld (direct
//       writeback, uncoalesced — SMEM-staged coalescing comes in a
//       later chapter).
//
// SMEM layout (TMA-produced, 128B-swizzled):
//   A: [M=128 rows of M][K=64 cols, innermost]    16 KB
//   B: [N=256 rows of N][K=64 cols, innermost]    32 KB     ← B is
//       transposed on the host so its SMEM ends up N-major-with-K-inner,
//       matching the descriptor convention (idesc bit 16 = 0).

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

constexpr int M       = 128;
constexpr int N       = 256;
constexpr int K       = 64;
constexpr int MMA_K   = 16;
constexpr int K_MMAS  = K / MMA_K;          // 4

constexpr int A_SMEM_BYTES = M * K * 2;     // 16384
constexpr int B_SMEM_BYTES = N * K * 2;     // 32768
constexpr int TILE_BYTES   = A_SMEM_BYTES + B_SMEM_BYTES;

constexpr int THREADS   = 128;
constexpr int WARP_SIZE = 32;


// ── tiny helper: one lane returns true per warp ─────────────────────
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


// ── TMA 2D load ─────────────────────────────────────────────────────
__device__ __forceinline__ void tma_2d_load(
    uint32_t smem_dst, const void* tmap, int x, int y, uint32_t mbar
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar) : "memory");
}


// ── Matrix descriptor (SMEM operand)  ───────────────────────────────
//   addr (>>4) | SBO=8*128 (>>4) | layout-mode bit 46 | SWIZZLE_128B (2<<61)
__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    uint64_t b = ((SBO)              >> 4) & 0x3FFFULL;
    return a | (b << 32) | (1ULL << 46) | (2ULL << 61);
}


// ── Instruction descriptor (shape + dtype) ──────────────────────────
__device__ __forceinline__ uint32_t make_idesc_bf16(int m, int n) {
    uint32_t d = 0;
    d |= (1u << 4);                                    // c_format = F32
    d |= (1u << 7);                                    // a_format = BF16
    d |= (1u << 10);                                   // b_format = BF16
    d |= (((uint32_t)(n >> 3) & 0x3F) << 17);          // n_dim = n/8
    d |= (((uint32_t)(m >> 4) & 0x1F) << 24);          // m_dim = m/16
    return d;
}


// ── tcgen05 PTX wrappers ────────────────────────────────────────────
__device__ __forceinline__ void tcgen05_alloc(uint32_t smem_dst, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_dst), "r"(n_cols) : "memory");
}
__device__ __forceinline__ void tcgen05_relinquish() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
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


// ── mbarrier helpers ────────────────────────────────────────────────
__device__ __forceinline__ void mbarrier_init(uint32_t mb, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mb), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint32_t mb, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(mb), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbarrier_wait_phase0(uint32_t mb) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], 0;\n\t"
        "@P bra DONE_%=;\n\t bra WAIT_%=;\n\t DONE_%=:\n\t }"
        :: "r"(mb) : "memory");
}


// ── Kernel ──────────────────────────────────────────────────────────
extern "C" __global__ void tcgen05_demo(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    __nv_bfloat16* __restrict__ C_ptr
) {
    // Rule-of-thumb alignment: 1024 absorbs any static __shared__ that
    // sits in front, so the swizzle pattern stays canonical.
    extern __shared__ __align__(1024) char smem[];
    const uint32_t SMEM_BASE = (uint32_t)__cvta_generic_to_shared(smem);
    const uint32_t A_BASE = SMEM_BASE;                       // [M][K] = 128×64 BF16
    const uint32_t B_BASE = SMEM_BASE + A_SMEM_BYTES;        // [N][K] = 256×64 BF16

    __shared__ uint64_t tile_ready;
    __shared__ uint64_t mma_done;
    __shared__ uint32_t tmem_addr_holder[1];

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane    = tid % WARP_SIZE;

    const uint32_t tile_ready_mb = (uint32_t)__cvta_generic_to_shared(&tile_ready);
    const uint32_t mma_done_mb   = (uint32_t)__cvta_generic_to_shared(&mma_done);

    // ── 1) One-time setup: mbar inits + TMEM alloc ──────────────────
    if (warp_id == 0 && elect_sync()) {
        mbarrier_init(tile_ready_mb, 1);
        mbarrier_init(mma_done_mb,   1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (warp_id == 1) {
        tcgen05_alloc((uint32_t)__cvta_generic_to_shared(tmem_addr_holder), N);
    }
    __syncthreads();
    const uint32_t taddr = tmem_addr_holder[0];
    const uint32_t idesc = make_idesc_bf16(M, N);

    // ── 2) TMA: one bulk per operand, then arrive.expect_tx for both ─
    if (warp_id == 0 && elect_sync()) {
        tma_2d_load(A_BASE, &A_tmap, /*x=*/0, /*y=*/0, tile_ready_mb);
        tma_2d_load(B_BASE, &B_tmap, /*x=*/0, /*y=*/0, tile_ready_mb);
        mbarrier_arrive_expect_tx(tile_ready_mb, TILE_BYTES);
    }

    // ── 3) All threads wait for SMEM, then publish to the MMA proxy ─
    //
    // These two lines synchronize *different things* and both are needed:
    //
    //   mbarrier_wait_phase0  — TMA (async proxy) → THREAD (generic proxy).
    //       When the wait returns, TMA's writes to SMEM are visible to
    //       this thread.  An ordinary ld.shared here would see the data.
    //
    //   tcgen05.fence::after_thread_sync — THREAD → TENSOR CORE (tcgen05
    //       proxy).  The MMA we're about to issue doesn't read SMEM from
    //       the thread's proxy; it tells the tensor-core unit to go fetch
    //       SMEM through *its own* pipeline.  Without this fence the
    //       tensor core could race ahead of the mbarrier's effect and
    //       grab stale/partial data.  The fence publishes the thread's
    //       prior synchronization to the tcgen05 proxy.
    //
    // mbarrier = cross-actor ordering (did the event happen?).
    // fence    = cross-proxy ordering (is the result visible to the other
    //            hardware unit?).  Different axes, both required.
    mbarrier_wait_phase0(tile_ready_mb);
    tcgen05_fence_after_thread_sync();

    // ── 4) Issue K_MMAS = 4 back-to-back MMAs (K = 16 each → K = 64) ─
    // Only one thread issues; the tensor core does the work async.
    if (warp_id == 1 && elect_sync()) {
        #pragma unroll
        for (int kk = 0; kk < K_MMAS; kk++) {
            const uint64_t a_desc = make_desc(A_BASE + kk * 32);   // K-step = 16 BF16 = 32 B
            const uint64_t b_desc = make_desc(B_BASE + kk * 32);
            // First MMA overwrites the accumulator (P = false); rest accumulate.
            tcgen05_mma(taddr, a_desc, b_desc, idesc, /*enable_d=*/ kk > 0);
        }
        tcgen05_commit(mma_done_mb);
    }

    // ── 5) Wait for all MMAs to finish ──────────────────────────────
    mbarrier_wait_phase0(mma_done_mb);

    // ── 6) TMEM → registers → GMEM (direct, uncoalesced) ────────────
    //
    //   4 warps × 32 lanes = 128 rows = M.  Each warp covers a 32-row
    //   strip of TMEM and loops over N in steps of 8 cols.  Each lane
    //   reads 8 float32 accumulators per call, packs to BF16, and
    //   writes a 16-byte int4 to its row's column window.
    //
    // Mirror of step 3's fence, opposite direction: mma_done mbar told
    // the thread "all MMAs finished," but tcgen05.ld reads TMEM through
    // the tcgen05 proxy — fence publishes the thread's mbar-wait order
    // to the tcgen05 unit so the ld sees the MMA's writes.
    tcgen05_fence_after_thread_sync();

    const int my_row = warp_id * 32 + lane;
    const uint32_t taddr_row_base = taddr + ((uint32_t)(warp_id * 32) << 16);

    #pragma unroll
    for (int n = 0; n < N; n += 8) {
        float tmp[8];
        tcgen05_ld_32x32b_x8(taddr_row_base + (uint32_t)n, tmp);
        tcgen05_wait_ld();

        __nv_bfloat162 packed[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            packed[i] = __floats2bfloat162_rn(tmp[2 * i], tmp[2 * i + 1]);
        }
        *reinterpret_cast<int4*>(&C_ptr[my_row * N + n]) =
            *reinterpret_cast<int4*>(packed);
    }

    __syncthreads();
    if (warp_id == 0 && elect_sync()) {
        tcgen05_dealloc(taddr, N);
    }
}
