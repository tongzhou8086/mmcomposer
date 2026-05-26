// Runnable companion for Chapter 00 — A first TMA program.
//
// One CTA, 128 threads, a single 1D TMA bulk load of CHUNK_BYTES bytes
// from global memory into shared memory.  Each thread then reads one
// byte from SMEM and writes it to a unique GMEM slot (defeats DCE).
//
// Functionally: g_out[:CHUNK_BYTES] = g_in[:CHUNK_BYTES].
//
// Compiled and launched by main.py via cuda-python (NVRTC + driver API).

#include <cuda/std/cstdint>

constexpr unsigned CHUNK_BYTES = 128;


// ── TMA 1D bulk load ────────────────────────────────────────────────────────
//
// Issues a single cp.async.bulk.tensor.1d into SMEM.
// `complete_tx::bytes` modifier auto-decrements the named mbarrier's
// tx-count by the number of bytes that arrive.

__device__ __forceinline__ void tma_1d_load(
    uint32_t smem_dst, const void* tmap_ptr, int coord_x, uint32_t mbar_addr
) {
    asm volatile(
        "cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2}], [%3];"
        :: "r"(smem_dst), "l"(tmap_ptr), "r"(coord_x), "r"(mbar_addr)
        : "memory");
}


// ── mbarrier try-wait spin loop ─────────────────────────────────────────────
//
// Blocks the calling warp until the mbarrier's current parity flips
// to the opposite of `phase`.  For a single-shot load `phase` is 0.

__device__ __forceinline__ void mbarrier_wait(uint32_t mbar_addr, uint32_t phase) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE_%=;\n\t"
        "bra WAIT_%=;\n\t"
        "DONE_%=:\n\t"
        "}"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}


// ── Kernel ──────────────────────────────────────────────────────────────────

extern "C" __global__ void tma_demo(
    const __grid_constant__ CUtensorMap tmap,
    uint8_t* __restrict__ g_out
) {
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ uint64_t mbar;

    const uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(&mbar);
    const uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem);

    // 1) Initialize the mbarrier.
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar_addr));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    // 2) One thread issues the TMA load and declares the expected bytes.
    if (threadIdx.x == 0) {
        tma_1d_load(smem_addr, &tmap, /*coord_x=*/ 0, mbar_addr);
        asm volatile(
            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
            :: "r"(mbar_addr), "r"(CHUNK_BYTES) : "memory");
    }

    // 3) All threads wait for the load to complete.
    mbarrier_wait(mbar_addr, /*phase=*/ 0);

    // 4) SMEM now contains CHUNK_BYTES of data.  Each thread reads one
    //    byte and writes it to a unique GMEM slot to defeat DCE.
    if (threadIdx.x < CHUNK_BYTES) {
        g_out[threadIdx.x] = smem[threadIdx.x];
    }
}
