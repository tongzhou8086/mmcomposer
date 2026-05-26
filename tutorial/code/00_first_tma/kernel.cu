// Runnable companion for Chapter 00 — A first TMA program.
//
// One CTA, 128 threads, a single 1D TMA bulk load of CHUNK_BYTES bytes
// from global memory into shared memory.  Each thread then reads one
// byte from SMEM and writes it to a unique GMEM slot (defeats DCE).
//
// Functionally: g_out[:CHUNK_BYTES] = g_in[:CHUNK_BYTES].
//
// Style note: every PTX instruction is shown inline at its use site
// (no helper-function wrappers).  This keeps the entire kernel body
// readable top-to-bottom — what you see is what runs.
//
// Compiled and launched by main.py via cuda-python (NVRTC + driver API).

#include <cuda/std/cstdint>

constexpr unsigned CHUNK_BYTES = 128;


extern "C" __global__ void tma_demo(
    const __grid_constant__ CUtensorMap tmap,
    uint8_t* __restrict__ g_out
) {
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ uint64_t mbar;

    const uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(&mbar);
    const uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem);

    // ── 1) Initialize the mbarrier.
    //
    // The fence is required so that the async TMA proxy sees the
    // init before its complete_tx::bytes modifier touches the mbar.
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar_addr));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    // ── 2) One thread issues the TMA bulk load and declares the
    //       expected byte count.
    //
    // cp.async.bulk.tensor.1d takes a SMEM destination, a CUtensorMap
    // pointer + a coordinate vector ({coord_x} for 1D), and the
    // mbarrier whose tx-count it will decrement when bytes land.
    //
    // mbarrier.arrive.expect_tx adds CHUNK_BYTES to the mbarrier's
    // tx_count and decrements arrival_count by 1.
    if (threadIdx.x == 0) {
        const int coord_x = 0;
        asm volatile(
            "cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2}], [%3];"
            :: "r"(smem_addr), "l"(&tmap), "r"(coord_x), "r"(mbar_addr)
            : "memory");
        asm volatile(
            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
            :: "r"(mbar_addr), "r"(CHUNK_BYTES) : "memory");
    }

    // ── 3) All threads spin on try_wait.parity until the load lands.
    //
    // For a single-shot load `phase` is 0 (the parity bit will flip
    // from 0 → 1 when both counters reach zero, at which point
    // try_wait succeeds).
    const uint32_t phase = 0;
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE_%=;\n\t"
        "bra WAIT_%=;\n\t"
        "DONE_%=:\n\t"
        "}"
        :: "r"(mbar_addr), "r"(phase) : "memory");

    // ── 4) SMEM now contains CHUNK_BYTES of data.  Each thread reads
    //       one byte and writes it to a unique GMEM slot to defeat
    //       dead-code elimination.
    if (threadIdx.x < CHUNK_BYTES) {
        g_out[threadIdx.x] = smem[threadIdx.x];
    }
}
