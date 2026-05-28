// Runnable companion for Chapter 01 — TMA with SWIZZLE_128B.
//
// One CTA, 128 threads.  A single 2D TMA bulk load fetches the WHOLE
// 8x64 row-major BF16 tensor (1024 bytes) into shared memory, but this
// time the descriptor requests SWIZZLE_128B instead of SWIZZLE_NONE.
// The TMA engine permutes 16-byte (= 8 BF16) chunks as it writes SMEM.
//
// To make that permutation observable, the kernel then copies SMEM out
// to `g_out` in *linear physical order* — element i of SMEM goes to
// g_out[i].  So g_out is a faithful snapshot of the swizzled SMEM byte
// layout, which the host then compares against the un-swizzled input.
//
// Functionally: g_out[i] = (swizzled SMEM)[i]   for i in 0..ROWS*COLS-1.
//
// Compiled by main.py via nvcc -> cubin (cached on disk by mtime).

#include <cuda.h>          // CUtensorMap
#include <cuda_bf16.h>     // __nv_bfloat16
#include <cstdint>         // uint32_t, uint64_t

constexpr unsigned ROWS        = 8;
constexpr unsigned COLS        = 64;                  // BF16 elements per row
constexpr unsigned ELEM_BYTES  = 2;                   // BF16
constexpr unsigned TILE_BYTES  = ROWS * COLS * ELEM_BYTES;   // 1024


extern "C" __global__ void tma_swizzle_demo(
    const __grid_constant__ CUtensorMap tmap,
    __nv_bfloat16* __restrict__ g_out
) {
    // TMA requires 128-byte SMEM alignment.  With SWIZZLE_128B the engine
    // also treats the destination as a sequence of 128-byte rows, each
    // split into eight 16-byte chunks it may permute.
    //
    // IMPORTANT: 128B swizzle is keyed off the *absolute* SMEM offset of
    // each element, not the offset within our array.  We want the tile to
    // start at window offset 0 so the swizzle pattern is the canonical
    // "XOR chunk by row".  A static `__shared__ uint64_t mbar` would be
    // placed first and push our dynamic tile to offset 128 (= one chunk
    // row), shifting the whole pattern.  So we put the tile FIRST in
    // dynamic SMEM and carve the mbarrier out right after it.
    extern __shared__ __align__(128) __nv_bfloat16 smem[];   // tile @ offset 0
    uint64_t* mbar_ptr = reinterpret_cast<uint64_t*>(&smem[ROWS * COLS]);

    const uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(mbar_ptr);
    const uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem);

    // ── 1) Initialize the mbarrier.
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar_addr));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    // ── 2) One thread issues the TMA bulk load for the ENTIRE tile.
    //
    // boxDim = (COLS, ROWS) = (64, 8): the box covers all 8 rows at once,
    // so a single load fills SMEM.  coord (0, 0) = top-left of the tensor.
    if (threadIdx.x == 0) {
        const int coord_x = 0;   // column 0 (innermost)
        const int coord_y = 0;   // row 0
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];"
            :: "r"(smem_addr), "l"(&tmap), "r"(coord_x), "r"(coord_y), "r"(mbar_addr)
            : "memory");
        asm volatile(
            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
            :: "r"(mbar_addr), "r"(TILE_BYTES) : "memory");
    }

    // ── 3) All threads spin until the load lands.
    const uint32_t phase = 0;
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE_%=;\n\t"
        "bra WAIT_%=;\n\t"
        "DONE_%=:\n\t"
        "}"
        :: "r"(mbar_addr), "r"(phase) : "memory");

    // ── 4) Copy SMEM out in LINEAR PHYSICAL ORDER.  Reading smem[i] walks
    //       shared memory exactly as the TMA engine laid it out, so g_out
    //       becomes a snapshot of the swizzled layout.  128 threads, 512
    //       elements -> each thread copies 4.
    for (int i = threadIdx.x; i < ROWS * COLS; i += blockDim.x) {
        g_out[i] = smem[i];
    }
}
