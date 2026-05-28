// Runnable companion for Chapter 03 — outer K-loop.
//
// Problem:  C[M, N] = A[M, K] @ B[K, N]   with M = 128, N = 256, K = 512.
//
// Same single-CTA kernel as chapter 02, but with an outer K-loop that
// streams `K / BK` SMEM tiles through one slot, accumulating into the
// same TMEM accumulator across iterations.  Single-stage SMEM (no
// pipelining yet) — TMA and MMA strictly alternate.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

constexpr int BM      = 128;
constexpr int BN      = 256;
constexpr int BK      = 64;
constexpr int MMA_K   = 16;
constexpr int K_MMAS  = BK / MMA_K;          // 4 inner MMAs per K-tile

// Per-stage SMEM footprint (one A tile + one B tile).
constexpr int A_SMEM_BYTES = BM * BK * 2;    // 16 KB
constexpr int B_SMEM_BYTES = BN * BK * 2;    // 32 KB
constexpr int TILE_BYTES   = A_SMEM_BYTES + B_SMEM_BYTES;   // 48 KB

constexpr int THREADS   = 128;
constexpr int WARP_SIZE = 32;


// ── one lane returns true per warp ──────────────────────────────────
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


// ── Matrix descriptor + idesc (same as ch02) ────────────────────────
__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    uint64_t b = ((SBO)              >> 4) & 0x3FFFULL;
    return a | (b << 32) | (1ULL << 46) | (2ULL << 61);   // SWIZZLE_128B
}

__device__ __forceinline__ uint32_t make_idesc_bf16(int m, int n) {
    uint32_t d = 0;
    d |= (1u << 4);                                    // c_format = F32
    d |= (1u << 7);                                    // a_format = BF16
    d |= (1u << 10);                                   // b_format = BF16
    d |= (((uint32_t)(n >> 3) & 0x3F) << 17);          // n_dim = n/8
    d |= (((uint32_t)(m >> 4) & 0x1F) << 24);          // m_dim = m/16
    return d;
}


// ── tcgen05 PTX wrappers (same as ch02) ─────────────────────────────
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


// ── mbarrier helpers ────────────────────────────────────────────────
__device__ __forceinline__ void mbarrier_init(uint32_t mb, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mb), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint32_t mb, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(mb), "r"(bytes) : "memory");
}
// Wait helper now takes a phase: pass k_iter & 1 to alternate across iters.
__device__ __forceinline__ void mbarrier_wait_phase(uint32_t mb, uint32_t phase) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE_%=;\n\t bra WAIT_%=;\n\t DONE_%=:\n\t }"
        :: "r"(mb), "r"(phase) : "memory");
}


// ── Kernel ──────────────────────────────────────────────────────────
extern "C" __global__ void matmul_k_loop(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    __nv_bfloat16* __restrict__ C_ptr,
    int K                                       // runtime K
) {
    extern __shared__ __align__(1024) char smem[];
    const uint32_t SMEM_BASE = (uint32_t)__cvta_generic_to_shared(smem);
    const uint32_t A_BASE = SMEM_BASE;
    const uint32_t B_BASE = SMEM_BASE + A_SMEM_BYTES;

    __shared__ uint64_t tile_ready;
    __shared__ uint64_t mma_done;
    __shared__ uint32_t tmem_addr_holder[1];

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane    = tid % WARP_SIZE;

    const uint32_t tile_ready_mb = (uint32_t)__cvta_generic_to_shared(&tile_ready);
    const uint32_t mma_done_mb   = (uint32_t)__cvta_generic_to_shared(&mma_done);

    // ── 1) One-time setup (all done by warp 0) ──────────────────────
    if (warp_id == 0) {
        if (elect_sync()) {
            mbarrier_init(tile_ready_mb, 1);
            mbarrier_init(mma_done_mb,   1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        tcgen05_alloc((uint32_t)__cvta_generic_to_shared(tmem_addr_holder), BN);
    }
    __syncthreads();
    const uint32_t taddr = tmem_addr_holder[0];
    const uint32_t idesc = make_idesc_bf16(BM, BN);

    // ── 2) Outer K-loop ─────────────────────────────────────────────
    //
    // Each iter is one chapter-02-style mini-cycle (TMA → wait → fence
    // → 4 MMAs → wait), with three things changing across iters:
    //
    //   (a) `phase = k_iter & 1` alternates the mbar-wait operand
    //       (mbarrier parity flips every completion).
    //   (b) `enable_d = !(first_ever)` makes only the very first MMA
    //       of the very first iter overwrite TMEM; everything else
    //       accumulates.  The accumulator persists across iters.
    //   (c) TMA coord x = k_iter * BK steps the K-window forward each
    //       iter; y = 0 (we own a single M-block in this single-CTA
    //       kernel).
    const int num_k_iters = K / BK;

    for (int k_iter = 0; k_iter < num_k_iters; k_iter++) {
        const uint32_t phase = k_iter & 1;

        // ── TMA the next K-tile (single bulk per operand)
        if (warp_id == 0 && elect_sync()) {
            tma_2d_load(A_BASE, &A_tmap, k_iter * BK, 0, tile_ready_mb);
            tma_2d_load(B_BASE, &B_tmap, k_iter * BK, 0, tile_ready_mb);
            mbarrier_arrive_expect_tx(tile_ready_mb, TILE_BYTES);
        }

        // ── Wait for SMEM, publish to MMA proxy
        mbarrier_wait_phase(tile_ready_mb, phase);
        tcgen05_fence_after_thread_sync();

        // ── 4 MMAs covering BK = 64 (same issuing warp as TMA — they
        //    serialize anyway via the mbar waits above and below)
        if (warp_id == 0 && elect_sync()) {
            #pragma unroll
            for (int kk = 0; kk < K_MMAS; kk++) {
                const uint64_t a_desc = make_desc(A_BASE + kk * 32);
                const uint64_t b_desc = make_desc(B_BASE + kk * 32);
                const bool first_ever = (k_iter == 0) && (kk == 0);
                tcgen05_mma(taddr, a_desc, b_desc, idesc, /*enable_d=*/ !first_ever);
            }
            tcgen05_commit(mma_done_mb);
        }

        // ── Wait for MMAs to drain SMEM before next iter overwrites
        mbarrier_wait_phase(mma_done_mb, phase);
    }

    // ── 3) Epilogue (identical to chapter 02) ───────────────────────
    //
    // **Each thread owns one entire output row.**
    //
    //     each warp           →  32 rows × all N cols
    //     4 warps × 32 lanes  →  BM=128 rows × BN=256 = whole output tile
    tcgen05_fence_after_thread_sync();

    const int my_row = warp_id * 32 + lane;
    const uint32_t taddr_row_base = taddr + ((uint32_t)(warp_id * 32) << 16);

    #pragma unroll
    for (int n = 0; n < BN; n += 8) {
        float tmp[8];
        tcgen05_ld_32x32b_x8(taddr_row_base + (uint32_t)n, tmp);
        tcgen05_wait_ld();

        __nv_bfloat162 packed[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            packed[i] = __floats2bfloat162_rn(tmp[2 * i], tmp[2 * i + 1]);
        }
        *reinterpret_cast<int4*>(&C_ptr[my_row * BN + n]) =
            *reinterpret_cast<int4*>(packed);
    }

    __syncthreads();
    if (warp_id == 0 && elect_sync()) {
        tcgen05_dealloc(taddr, BN);
    }
}
