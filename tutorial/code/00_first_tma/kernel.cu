// Runnable companion for Chapter 00 — A first TMA program.
//
// One CTA, 128 threads.  A single 2D TMA bulk load fetches one row
// (64 BF16 elements = 128 bytes) of an 8x64 row-major BF16 tensor
// into shared memory.  Each thread then reads one byte from SMEM
// and writes it to a unique GMEM slot (defeats DCE).
//
// Functionally: g_out[:128] = g_in[0, :] (the first row, as bytes).
//
// Style note: every PTX instruction is shown inline at its use site
// (no helper-function wrappers).  This keeps the entire kernel body
// readable top-to-bottom — what you see is what runs.
//
// Compiled and launched by main.py via cuda-python (NVRTC + driver API).

// NVRTC's default include path does not have libcu++ (`cuda/std/*`) or
// the host `<cstdint>` header.  Provide the integer typedefs we need
// locally so this file compiles standalone.
typedef unsigned char       uint8_t;
typedef unsigned int        uint32_t;
typedef unsigned long long  uint64_t;

// NVRTC also lacks the CUDA driver-API headers that declare CUtensorMap.
// Declare it locally as a 128-byte opaque struct (the actual layout is
// filled in host-side by cuTensorMapEncodeTiled; the kernel only ever
// takes &tmap to pass to the TMA instruction).
struct alignas(64) CUtensorMap {
    uint64_t opaque[16];   // 128 bytes
};

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
    // cp.async.bulk.tensor.2d takes a SMEM destination, a CUtensorMap
    // pointer + a coordinate pair ({coord_x, coord_y} for 2D), and the
    // mbarrier whose tx-count it will decrement when bytes land.
    //
    // mbarrier.arrive.expect_tx adds CHUNK_BYTES to the mbarrier's
    // tx_count and decrements arrival_count by 1.
    if (threadIdx.x == 0) {
        // Coordinate count matches the descriptor's rank (here, 2).
        // Values are in *elements* (not bytes), innermost-first:
        //   coord_x = column index (0 → start of row)
        //   coord_y = row index    (0 → first row)
        const int coord_x = 0;
        const int coord_y = 0;
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];"
            :: "r"(smem_addr), "l"(&tmap), "r"(coord_x), "r"(coord_y), "r"(mbar_addr)
            : "memory");
        asm volatile(
            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
            :: "r"(mbar_addr), "r"(CHUNK_BYTES) : "memory");
    }

    // ── 3) All threads spin on try_wait.parity until the load lands.
    //
    // try_wait.parity succeeds when the mbarrier's current parity bit
    // is the *opposite* of the operand.  After init, parity is 0; the
    // first completion flips it to 1.  So we pass `phase = 0` to mean
    // "block until parity is no longer 0 = no longer at init state."
    // A K-loop would `phase ^= 1` after each successful wait so the
    // software mirror chases the hardware's flipping parity.
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
