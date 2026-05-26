# A first TMA program

This chapter is hands-on.  By the end of it you will have a complete,
runnable B200 kernel that uses TMA to copy a chunk of bytes from
global memory into shared memory — and nothing else.  No matmul.  No
ring buffer.  No pipelining.  Just *one* TMA load and the minimum
scaffolding around it.

> 📁 **Runnable code:** [`tutorial/code/00_first_tma/`](https://github.com/tongzhou8086/mmcomposer/tree/master/tutorial/code/00_first_tma).
> The kernel and host launcher shown below are reproduced verbatim there.
> Run with `pip install -r ../requirements.txt && python main.py` on a
> Blackwell (sm_100a) GPU.

The point is to internalize each piece in isolation:

* Host side: how to build a `CUtensorMap` descriptor.
* Kernel side: how to issue the `cp.async.bulk.tensor.1d` instruction.
* mbarrier: how to wait for the load to complete.

Once these three are in muscle memory, the matmul chapters get to
focus on what they're actually about (`tcgen05.mma`, multi-stage
buffering, warp specialization) without having to teach TMA from
scratch every time.

## What we're building

A single-CTA kernel that:

1. TMA-loads `CHUNK_BYTES` bytes from `g_in[0 … CHUNK_BYTES)` into
   shared memory.
2. Has each thread read one byte from SMEM and write it to
   `g_out[threadIdx.x]`.

The "read back and write" part exists only to prevent the compiler
from optimizing the entire load away.  Functionally it's `g_out =
g_in` done through TMA + SMEM.

We'll use:

* **`CHUNK_BYTES = 128`** (fits in 128 threads × 1 byte each).
* **`uint8`** as the element type (no data-type conversions to worry
  about yet).
* **1D TMA** (the simplest variant — coords are a single 1D index).
* **One CTA**, **128 threads**.

## Step 1 — host side: the `CUtensorMap`

A `CUtensorMap` is a 128-byte opaque struct that describes a tensor's
layout to the TMA engine.  It's built once on the host via
`cuTensorMapEncodeTiled` and passed to the kernel as a
`__grid_constant__` parameter.

For 1D streaming the API is straightforward:

```cpp
#include <cuda.h>

constexpr size_t CHUNK_BYTES = 128;

CUtensorMap tmap;
uint64_t global_dim[1]      = { CHUNK_BYTES };   // total tensor length, in elements
uint32_t box_dim[1]         = { CHUNK_BYTES };   // per-load box length
uint32_t element_strides[1] = { 1 };

cuTensorMapEncodeTiled(
    &tmap,
    CU_TENSOR_MAP_DATA_TYPE_UINT8,   // 1-byte elements
    /*rank=*/             1,
    /*globalAddress=*/    g_in_ptr,  // raw uint8* to the start of the buffer
    /*globalDim=*/        global_dim,
    /*globalStrides=*/    nullptr,   // only rank − 1 entries needed; 0 here
    /*boxDim=*/           box_dim,
    /*elementStrides=*/   element_strides,
    /*interleave=*/       CU_TENSOR_MAP_INTERLEAVE_NONE,
    /*swizzle=*/          CU_TENSOR_MAP_SWIZZLE_NONE,
    /*l2Promotion=*/      CU_TENSOR_MAP_L2_PROMOTION_NONE,
    /*oobFill=*/          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
```

A few things to call out:

* **`globalDim`** is the logical shape of the entire tensor as the TMA
  engine sees it.  For 1D it's just one number — the total length in
  elements.
* **`globalStrides`** is the per-dimension stride array, *but only for
  the outer dimensions* — there are `rank − 1` of them.  For 1D that's
  zero entries, so we pass `nullptr`.
* **`boxDim`** is the per-load shape — how many elements a single
  TMA bulk fetches.  Each `cp.async.bulk.tensor` instruction grabs
  exactly one box.
* **`elementStrides`** lets you skip elements in any dimension.  Almost
  always `{1, 1, ...}`.
* **`swizzle`** is the SMEM-side layout the TMA engine will apply on
  arrival.  For 1D streaming we use `NONE`.  (Matmul will use `128B`
  — see the brief sidebar in Step 4.)
* **`oobFill`** controls what happens if a TMA coord goes past the
  global shape.  `NONE` means the caller is responsible for staying
  in bounds.

That's it for the host side.  Once `tmap` is built, it's passed to
the kernel by value — the kernel parameter is
`const __grid_constant__ CUtensorMap`.

## Step 2 — kernel side: the TMA instruction

The instruction is `cp.async.bulk.tensor.1d`.  Operands:

```
cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes
    [smem_dst], [tmap_ptr, {coord_x}], [mbar_addr];
```

* `smem_dst` — a 32-bit SMEM address obtained via
  `__cvta_generic_to_shared(...)`.
* `tmap_ptr` — a 64-bit generic pointer to the `CUtensorMap`.  Taking
  the address of the `__grid_constant__` parameter gives exactly this.
* `coord_x` — for 1D, a single 32-bit element index into the global
  tensor.  This is the *start* of the box we want to load; the TMA
  engine fetches `boxDim[0]` consecutive elements from there.
* `mbar_addr` — an SMEM address of the mbarrier that will receive the
  completion signal.

The `.shared::cta.global` part says "destination is local SMEM, source
is global memory."  The `.mbarrier::complete_tx::bytes` modifier says
"when this bulk lands, decrement the named mbarrier's tx-count by the
number of bytes transferred."

Wrapped as a C helper:

```cpp
__device__ __forceinline__ void tma_1d_load(
    uint32_t smem_dst, const void* tmap_ptr, int coord_x, uint32_t mbar_addr
) {
    asm volatile(
        "cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2}], [%3];"
        :: "r"(smem_dst), "l"(tmap_ptr), "r"(coord_x), "r"(mbar_addr)
        : "memory");
}
```

## Step 3 — the mbarrier handshake

We covered the mbarrier primitive conceptually in
[Part 1](../part1_gpu_arch/mbarrier).  Here we just *use* it.  Three
things happen, in this order:

1. **Init.**  One thread calls `mbarrier.init` to set the arrival
   count and start the parity bit at 0.
2. **Producer side.**  The same (or another) thread issues the TMA
   bulk *and* calls `mbarrier.arrive.expect_tx(N)`, telling the
   mbarrier "expect N bytes total."  The bulk's `complete_tx::bytes`
   modifier will decrement the tx-count as bytes arrive.
3. **Consumer side.**  All threads call `mbarrier.try_wait.parity`,
   which blocks until both the arrival count and tx-count reach zero.

In our minimal example, thread 0 does the init + issue + expect_tx
and all 128 threads do the wait.

### Init

```cpp
__shared__ uint64_t mbar;

if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&mbar)));
    asm volatile("fence.mbarrier_init.release.cluster;");
}
__syncthreads();   // make sure init is visible before any thread waits
```

The `fence.mbarrier_init.release.cluster` is required so the
asynchronous TMA engine (which lives in a separate proxy from regular
thread-issued ops) sees the mbarrier init before any subsequent
`complete_tx` touches it.  Without the fence, the load can in
principle arrive at the mbarrier before the mbarrier finishes
initializing, and the arrival gets dropped.

### Wait

```cpp
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
```

`phase` is the software-mirror of the mbarrier's parity bit (see the
[Part 1 mbarrier chapter](../part1_gpu_arch/mbarrier)).  For a
*single* load it's always 0; we'd only flip it across multiple
iterations.

## Step 4 — swizzling, briefly

We set `swizzle = CU_TENSOR_MAP_SWIZZLE_NONE` above.  For pure 1D
streaming of byte data, NONE is the right choice — there's nothing
to swizzle, and we don't have a downstream consumer (like
`tcgen05.mma`) that imposes a specific layout.

When we get to matmul, the consumer is `tcgen05.mma`, which **does**
impose a specific swizzled SMEM layout (128B swizzling).  The
beautiful thing is that TMA can do this layout for you: set
`swizzle = CU_TENSOR_MAP_SWIZZLE_128B` in the descriptor, and TMA
arranges the bytes in SMEM in the form the MMA expects.  We'll see
exactly how that works in chapter 01.

For this chapter, just know that swizzling is a *property of the
descriptor* — same instruction, same SMEM destination, different byte
arrangement.

## Step 5 — the complete kernel

Putting it all together:

```cpp
constexpr size_t CHUNK_BYTES = 128;

__global__ void tma_demo(
    const __grid_constant__ CUtensorMap tmap,
    uint8_t* __restrict__ g_out
) {
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ uint64_t mbar;
    const uint32_t mbar_addr = __cvta_generic_to_shared(&mbar);
    const uint32_t smem_addr = __cvta_generic_to_shared(smem);

    // 1) Initialize the mbarrier.
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar_addr));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    // 2) One thread issues the TMA load and declares the expected byte count.
    if (threadIdx.x == 0) {
        tma_1d_load(smem_addr, &tmap, /*coord_x=*/ 0, mbar_addr);
        asm volatile(
            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
            :: "r"(mbar_addr), "r"((unsigned)CHUNK_BYTES) : "memory");
    }

    // 3) All threads wait for the load to complete.
    mbarrier_wait(mbar_addr, /*phase=*/ 0);

    // 4) SMEM now contains CHUNK_BYTES of data.  Each thread reads one
    //    byte and writes it to a unique GMEM slot (defeats DCE).
    if (threadIdx.x < CHUNK_BYTES) {
        g_out[threadIdx.x] = smem[threadIdx.x];
    }
}
```

A few asides on the kernel:

* `extern __shared__ __align__(128) uint8_t smem[]` — TMA requires
  128-byte SMEM alignment.  We declare it explicitly.
* `__shared__ uint64_t mbar` lives in static SMEM; its address is
  converted to a 32-bit SMEM index via `__cvta_generic_to_shared`.
* The `const __grid_constant__ CUtensorMap` parameter is how the
  128-byte tensor map is passed by value.  The runtime copies the
  struct into the kernel's parameter area; `&tmap` gives a generic
  pointer the TMA instruction can use.

## Step 6 — host launch

```cpp
size_t shared_bytes = CHUNK_BYTES;
tma_demo<<<1, 128, shared_bytes>>>(tmap, g_out_device);
```

One CTA, 128 threads, `CHUNK_BYTES` of dynamic shared memory.  When
the kernel returns, `g_out_device` is a byte-for-byte copy of the
first `CHUNK_BYTES` of the input buffer.

## What to take away

You've now seen everything you need at the TMA instruction-and-API
level:

* A `CUtensorMap` describes the global tensor's shape, the per-load
  box shape, and the SMEM-side swizzle.
* `cp.async.bulk.tensor.1d` (and its 2D, 3D variants) issues a single
  bulk-load using the descriptor + a coordinate vector.
* An `mbarrier` is the completion-signal channel: init it, declare
  the expected bytes via `arrive.expect_tx`, wait via
  `try_wait.parity`.
* TMA can handle SMEM-side swizzling for you — useful when the
  downstream consumer (like `tcgen05.mma`) wants a specific layout.

Everything in this chapter generalizes directly to matmul: swap the
1D descriptor for a 2D one, change `SWIZZLE_NONE` to `SWIZZLE_128B`,
and feed the SMEM into a `tcgen05.mma` instruction instead of reading
it back to GMEM.  That's the next chapter.
