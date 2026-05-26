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
* Kernel side: how to issue the `cp.async.bulk.tensor.2d` instruction.
* mbarrier: how to wait for the load to complete.

Once these three are in muscle memory, the matmul chapters get to
focus on what they're actually about (`tcgen05.mma`, multi-stage
buffering, warp specialization) without having to teach TMA from
scratch every time.

## What we're building

A single-CTA kernel that:

1. TMA-loads **one row** (64 BF16 elements = 128 bytes) from a global
   8×64 BF16 tensor into shared memory.
2. Has threads 0..63 each read one BF16 element from SMEM and write
   it to `g_out[threadIdx.x]`.

The "read back and write" part exists only to prevent the compiler
from optimizing the entire load away.  Functionally it's
`g_out = first_row(g_in)` done through TMA + SMEM.

Both buffers are **torch BF16 tensors** on CUDA — torch handles
allocation and deallocation, we just pass `g_in.data_ptr()` and
`g_out.data_ptr()` to the kernel.

We'll use:

* **BF16** as the element type (same dtype matmul uses; 2 bytes per
  element).
* **8 × 64 row-major tensor** in global memory (1024 bytes total).
* **2D TMA** with box = (64 cols × 1 row) — one row per load.
* **`SWIZZLE_NONE`** so SMEM byte order matches global byte order
  (matmul will use `SWIZZLE_128B`; we'll get there in chapter 01).
* **One CTA**, **128 threads**.

> **Aside on 1D TMA.**  In principle TMA also supports `rank=1`
> descriptors and a `cp.async.bulk.tensor.1d` instruction, which
> would be conceptually simpler.  In practice, the driver rejects
> `rank=1` configurations we tried on B200 with
> `CUDA_ERROR_INVALID_VALUE` (likely some undocumented
> swizzle/dtype/dim interaction).  Going straight to 2D matches what
> matmul actually uses and avoids the dead-end.

## Step 1 — host side: the `CUtensorMap`

A `CUtensorMap` is a 128-byte opaque struct that describes a tensor's
layout to the TMA engine.  It's built once on the host via
`cuTensorMapEncodeTiled` from the CUDA driver API.

For this tutorial we wrap that C call in a small helper —
`encode_tensor_map(...)` in `cuda_utils.py` — which binds
`cuTensorMapEncodeTiled` from `libcuda.so` via `ctypes`.  We use the
helper instead of cuda-python's wrapper because the cuda-python
bindings for this function have shifted across versions; going
through `libcuda` directly is more stable and the call site reads
cleaner.  See [`tutorial/code/cuda_utils.py`](https://github.com/tongzhou8086/mmcomposer/tree/master/tutorial/code/cuda_utils.py)
for the binding.

The call site for our 8 × 64 BF16 tensor (where `g_in` is a torch
CUDA tensor):

```python
ROWS, COLS, ELEM_BYTES = 8, 64, 2  # BF16

g_in  = (torch.arange(ROWS * COLS, dtype=torch.float32) % 17.0).to(
    device="cuda", dtype=torch.bfloat16).view(ROWS, COLS)
g_out = torch.zeros(COLS, device="cuda", dtype=torch.bfloat16)

tmap = encode_tensor_map(
    dtype=TMA_BFLOAT16,
    rank=2,
    gptr=g_in.data_ptr(),              # torch tensor's device pointer
    global_dim=[COLS, ROWS],           # innermost first
    global_strides=[COLS * ELEM_BYTES],  # bytes per row
    box_dim=[COLS, 1],                 # per-load tile = one row
    element_strides=[1, 1],
    swizzle=TMA_SWIZZLE_NONE,
)
```

A few things to call out:

* **`rank`** is the number of dimensions of the tensor the descriptor
  describes.  `rank = 1` → 1D array; `rank = 2` → 2D matrix; up to
  `rank = 5`.  It's the standard tensor-rank usage (unrelated to
  linear-algebra matrix rank; they share a word but mean different
  things).
  Rank determines the shape of every other field below:

  | Array            | Entries        | Why                                                |
  |------------------|:--------------:|----------------------------------------------------|
  | `globalDim`      | `rank`         | one length per dim                                 |
  | `globalStrides`  | **`rank − 1`** | innermost stride is always 1 element (implicit)    |
  | `boxDim`         | `rank`         | the box has a size in every dim                    |
  | `elementStrides` | `rank`         | one stride per dim                                 |

  On the kernel side, `cp.async.bulk.tensor.<rank>d` takes exactly
  `rank` coordinates in its `{coord_x, ...}` vector.  Descriptor rank
  → kernel-side dimensionality.

* **`globalDim`** is the logical shape of the entire tensor as the
  TMA engine sees it.  One entry per dimension (`rank` entries
  total), listed **innermost first** — the first entry corresponds
  to the `x` coordinate you pass to the kernel-side instruction.

  | Tensor                  | `rank` | `globalDim`           | Note                              |
  |-------------------------|:------:|-----------------------|-----------------------------------|
  | 2D BF16 (8 × 64)        | 2      | `{64, 8}`             | our chapter                       |
  | 2D matrix `A: (M, K)` row-major | 2 | `{K, M}`             | K is contiguous → innermost       |
  | 3D, e.g. `[K/64, M, 64]`| 3      | `{64, M, K/64}`       | matmul slab trick (chapter 06)    |

* **`globalStrides`** is the per-dimension stride array, *but only for
  the outer dimensions* — there are `rank − 1` of them.  The
  innermost stride is always one element and is implicit.  Strides
  are in **bytes**, not elements.  For our 8 × 64 BF16 tensor the
  row stride is `64 × 2 = 128` bytes.
* **`boxDim`** is the per-load shape — how many elements a single
  TMA bulk fetches in each dimension.  Same dimensionality and
  ordering as `globalDim`.  Each `cp.async.bulk.tensor` instruction
  grabs exactly one box, starting at the coordinates you pass.

  The key relationship: `boxDim ≤ globalDim` along every dimension.

  - If `boxDim == globalDim` in every dim, **one TMA load covers the
    whole tensor** (single-shot transfer).
  - If `boxDim < globalDim` in some dim, **the tensor takes multiple
    loads to fully cover**, one per box at a different coordinate.
    Our chapter does one load that covers row 0 only; loading the
    other 7 rows would mean 7 more loads at coord `(0, 1), (0, 2),
    ... (0, 7)`.  In matmul kernels the K-loop iterates exactly
    this way, one box per K-tile.
* **`elementStrides`** lets you skip elements in any dimension.  Almost
  always `{1, 1, ...}`.
* **`swizzle`** is the SMEM-side layout the TMA engine will apply on
  arrival.  We use `NONE` here so SMEM bytes match global bytes
  one-to-one — easy to verify.  Matmul will use `128B`; see Step 4.
* **`oobFill`** (defaulted by the helper) controls what happens if a
  TMA coord goes past the global shape.  `NONE` (default) means the
  caller is responsible for staying in bounds.

### Aside — coming from numpy / PyTorch?

`globalDim` and `boxDim` are both *shape tuples* — they're not
opposites.  The distinction is *whose shape*:

| TMA descriptor | numpy / torch analogy                                  |
|----------------|--------------------------------------------------------|
| `globalDim`    | `whole_tensor.shape` — the entire tensor's shape       |
| `boxDim`       | the shape of one *tile / window* cut from it           |

A concrete framing: suppose you want to walk a 2D matrix one tile at
a time.

```python
A = torch.empty(M, K)              # the global tensor
tile = A[m0:m0+BM, k0:k0+BK]       # one box; tile.shape == (BM, BK)
```

* `A.shape`    ↔ `globalDim = {K, M}` (TMA uses innermost-first order)
* `tile.shape` ↔ `boxDim    = {BK, BM}`
* `(m0, k0)`   ↔ the `{coord_x, coord_y}` passed to the instruction

One TMA instruction is essentially a hardware-accelerated
`A[m0:m0+BM, k0:k0+BK]`, copied directly into SMEM.

That's it for the host side.  Once `tmap` is built, it's passed to
the kernel by value — the kernel parameter is
`const __grid_constant__ CUtensorMap`.

## Step 2 — kernel side: the TMA instruction

The instruction is `cp.async.bulk.tensor.2d`.  Operands:

```
cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes
    [smem_dst], [tmap_ptr, {coord_x, coord_y}], [mbar_addr];
```

* `smem_dst` — a 32-bit SMEM address obtained via
  `__cvta_generic_to_shared(...)`.
* `tmap_ptr` — a 64-bit generic pointer to the `CUtensorMap`.  Taking
  the address of the `__grid_constant__` parameter gives exactly this.
* `coord_x`, `coord_y` — 32-bit element indices into the global
  tensor, **innermost first**.  For our 2D descriptor:
  - `coord_x` is the column index (in BF16 elements)
  - `coord_y` is the row index
  - `(0, 0)` is "start of row 0"; the engine fetches `boxDim[0] = 64`
    elements from there, going `boxDim[1] = 1` row deep.
* `mbar_addr` — an SMEM address of the mbarrier that will receive the
  completion signal.

The `.shared::cta.global` part says "destination is local SMEM, source
is global memory."  The `.mbarrier::complete_tx::bytes` modifier says
"when this bulk lands, decrement the named mbarrier's tx-count by the
number of bytes transferred."

In CUDA C++ that's an `asm volatile` block (we keep every PTX
instruction inline at its use site — no helper-function wrappers,
so the kernel body reads top-to-bottom):

```cpp
asm volatile(
    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
    "[%0], [%1, {%2, %3}], [%4];"
    :: "r"(smem_addr), "l"(&tmap), "r"(coord_x), "r"(coord_y), "r"(mbar_addr)
    : "memory");
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

`try_wait.parity` is a single PTX instruction that returns a predicate
`P` — true once the mbarrier's current parity differs from the
operand.  Since we want to *block* until that happens, we wrap it in a
small spin loop entirely inside one `asm` block:

```cpp
const uint32_t phase = 0;
asm volatile(
    "{\n\t .reg .pred P;\n\t"
    "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
    "@P bra DONE_%=;\n\t"
    "bra WAIT_%=;\n\t"
    "DONE_%=:\n\t"
    "}"
    :: "r"(mbar_addr), "r"(phase) : "memory");
```

`phase` is the software-mirror of the mbarrier's parity bit (see the
[Part 1 mbarrier chapter](../part1_gpu_arch/mbarrier)).  The
instruction's semantics are: **succeeds when the mbarrier's current
parity bit is the *opposite* of the operand.**

For a single-shot load, the state trace is:

| Event                          | mbarrier parity | Operand we pass | Wait result   |
|--------------------------------|:---------------:|:---------------:|:-------------:|
| After `mbarrier.init`          | 0               | —               | not yet       |
| After our load completes       | **1** (flipped) | 0               | **succeeds**  |

So we pass `phase = 0` on the first wait because we want to block
until the parity is *no longer* 0 — i.e., until the load has
completed and the parity has flipped to 1.

If we did a second load on the same mbarrier (chapter 02 will), its
completion would flip parity back to 0, so we'd pass `phase = 1` for
that wait.  In a real K-loop that's `phase ^= 1` after each successful
wait — the software mirror chases the hardware's flipping bit.

The `%=` suffix on the labels (`WAIT_%=`, `DONE_%=`) tells the
compiler to make them unique per `asm` instantiation, so the same
loop can appear in multiple places in the same kernel without label
collisions.

## Step 4 — swizzling, briefly

We set `swizzle = TMA_SWIZZLE_NONE` above.  For a "just verify the
bytes arrived" demo, NONE is the right choice — SMEM bytes are laid
out in the same order as global bytes, so reading SMEM linearly
matches reading global memory linearly.

When we get to matmul, the consumer is `tcgen05.mma`, which **does**
impose a specific swizzled SMEM layout (128B swizzling).  The
beautiful thing is that TMA can do this layout for you: set
`swizzle = TMA_SWIZZLE_128B` in the descriptor, and TMA arranges the
bytes in SMEM in the form the MMA expects.  The trade-off is that
*reading SMEM linearly no longer reproduces the source bytes in
order* — they've been rearranged for the MMA's benefit.  We'll see
exactly how that works in chapter 01.

For this chapter, just know that swizzling is a *property of the
descriptor* — same instruction, same SMEM destination, different byte
arrangement.

## Step 5 — the complete kernel

Putting it all together — every PTX op is inline at its use site:

```cpp
#include <cuda.h>          // CUtensorMap
#include <cuda_bf16.h>     // __nv_bfloat16
#include <cstdint>         // uint32_t, uint64_t

constexpr unsigned COLS        = 64;
constexpr unsigned CHUNK_BYTES = COLS * 2;   // 64 BF16 elems = 128 bytes

extern "C" __global__ void tma_demo(
    const __grid_constant__ CUtensorMap tmap,
    __nv_bfloat16* __restrict__ g_out
) {
    extern __shared__ __align__(128) __nv_bfloat16 smem[];
    __shared__ uint64_t mbar;
    const uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(&mbar);
    const uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem);

    // 1) Initialize the mbarrier.
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar_addr));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    // 2) One thread issues the TMA load and declares the expected byte count.
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
            :: "r"(mbar_addr), "r"(CHUNK_BYTES) : "memory");
    }

    // 3) All threads wait for the load to complete.
    const uint32_t phase = 0;
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE_%=;\n\t"
        "bra WAIT_%=;\n\t"
        "DONE_%=:\n\t"
        "}"
        :: "r"(mbar_addr), "r"(phase) : "memory");

    // 4) SMEM now holds the first row's COLS BF16 elements.  Threads
    //    0..COLS-1 each write one element to g_out.
    if (threadIdx.x < COLS) {
        g_out[threadIdx.x] = smem[threadIdx.x];
    }
}
```

A few asides on the kernel:

* `extern __shared__ __align__(128) __nv_bfloat16 smem[]` — TMA
  requires 128-byte SMEM alignment.  We declare it explicitly.
* `__shared__ uint64_t mbar` lives in static SMEM; its address is
  converted to a 32-bit SMEM index via `__cvta_generic_to_shared`.
* The `const __grid_constant__ CUtensorMap` parameter is how the
  128-byte tensor map is passed by value.  The runtime copies the
  struct into the kernel's parameter area; `&tmap` gives a generic
  pointer the TMA instruction can use.
* The file is compiled with **nvcc** (not NVRTC), so we get the
  standard CUDA headers — `<cuda.h>` for `CUtensorMap`,
  `<cuda_bf16.h>` for `__nv_bfloat16`, `<cstdint>` for the integer
  typedefs.  The cubin is cached on disk next to `kernel.cu` and
  rebuilt only when the source is newer (`cuda_utils.compile_kernel`
  handles this).

## Step 6 — host launch

```python
launch(tma_demo,
       grid=(1, 1, 1),
       block=(THREADS_PER_CTA, 1, 1),
       shared=CHUNK_BYTES,
       args=[arg_tmap, arg_gout])
```

One CTA, 128 threads, `CHUNK_BYTES` of dynamic shared memory.  When
the kernel returns, `g_out` contains the first row of the input
tensor, byte-for-byte.

## What to take away

You've now seen everything you need at the TMA instruction-and-API
level:

* A `CUtensorMap` describes the global tensor's shape, the per-load
  box shape, and the SMEM-side swizzle.
* `cp.async.bulk.tensor.2d` (and its 1D / 3D variants) issues a
  single bulk-load using the descriptor + a coordinate vector.
* An `mbarrier` is the completion-signal channel: init it, declare
  the expected bytes via `arrive.expect_tx`, wait via
  `try_wait.parity`.
* TMA can handle SMEM-side swizzling for you — useful when the
  downstream consumer (like `tcgen05.mma`) wants a specific layout.

Everything in this chapter generalizes directly to matmul: keep the
2D descriptor, change `SWIZZLE_NONE` to `SWIZZLE_128B`, bump `boxDim`
to a real tile size like `(64, 128)`, and feed the SMEM into a
`tcgen05.mma` instruction instead of reading it back to GMEM.  That's
the next chapter.
