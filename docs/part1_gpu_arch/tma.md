# TMA — Tensor Memory Accelerator

> **Status:** stub — content TBD.

TMA is the hardware engine on Hopper/Blackwell that copies tiles between
global memory and shared memory using a **tensor map** (a 128-byte
descriptor that encodes the layout of the operand once, and is then
reused across every load).  It replaces the older `cp.async` family in
modern matmul kernels.

## Why TMA exists

* **Address calculation is offloaded.**  Pre-TMA, every thread computed
  its own address into A or B.  TMA does this in hardware from the
  descriptor + a small coordinate vector.
* **Bulk transfers, not per-thread.**  A single TMA instruction issues
  one bulk move — typically tens of KB at a time.  No need for an
  entire warp to participate.
* **Hardware-managed sync.**  TMA integrates with `mbarrier` via the
  `mbarrier::complete_tx::bytes` modifier; the consumer waits on the
  mbarrier, not on each thread's individual completion.

## The tensor map (`CUtensorMap`)

A 128-byte opaque struct built host-side via `cuTensorMapEncodeTiled`.
Passed to the kernel as a `__grid_constant__` parameter.  Encodes:

* Data type (`CU_TENSOR_MAP_DATA_TYPE_BFLOAT16`, ...)
* Global pointer + shape + strides
* Box dimensions (the tile shape TMA will copy)
* Element strides (usually all 1)
* Swizzle mode (`SWIZZLE_NONE`, `SWIZZLE_32B`, `SWIZZLE_64B`, `SWIZZLE_128B`)
* Out-of-bounds fill mode
* L2 promotion policy

TODO: walk through a concrete `init_tmap_2d_128B(...)` and a 3D one.

## The bulk-tensor copy instructions

The PTX instruction family is `cp.async.bulk.tensor.{1d,2d,3d,4d,5d}`.
Operands and modifiers:

```
cp.async.bulk.tensor.<rank>d
    .shared::{cta|cluster}.global
    .mbarrier::complete_tx::bytes
    [.cta_group::N]
    [.multicast::cluster]
    [.L2::cache_hint]
    [smem_dst], [tmap_ptr, {coords...}], [mbar_addr];
```

* `.shared::cta` vs `.shared::cluster` — destination address space.  Use
  `.shared::cluster` when the destination is on a peer CTA via the
  cluster-shared addressing convention.
* `.mbarrier::complete_tx::bytes` — every bulk auto-decrements the named
  mbarrier's tx-count by the number of bytes it transferred.
* `.cta_group::N` — *required* in cluster mode so peer-CTA bulks
  correctly bookkeep tx-count on the cluster-wide mbarrier.  Easy to
  miss — and missing it produces a silent hang, not an error.
* `.multicast::cluster` — broadcasts the bulk to multiple CTAs in the
  cluster via a CTA mask.

TODO: a short example of each form.

## Coordinates and box dimensions

The coords passed to the instruction are *not* byte offsets — they are
indices into the descriptor's logical shape.  Box dims are encoded
once in the descriptor; the coords pick *which* box at runtime.

TODO: worked example with a (M, K) row-major A and a (BM, BK) box.

## Swizzling

128B swizzle is what tcgen05 expects.  Lower-byte swizzles exist
(`SWIZZLE_32B`, `SWIZZLE_64B`) but matmul kernels almost always pick
128B.  The swizzle pattern is encoded in the descriptor and applied
automatically on the destination side — the consumer (tcgen05.mma)
just sees a swizzle-aware SMEM layout.

TODO: diagram of the 128B swizzle pattern.

## Common pitfalls

* **Missing `.cta_group::2` in cluster mode.**  Peer-CTA bulks silently
  fail to advance tx-count → hard hang.  Easiest mistake to make when
  porting a non-cluster kernel to cluster mode.
* **Tx-count mismatch.**  The mbarrier expects N bytes total; if the
  sum of bulks doesn't equal N, you either hang (under) or proceed
  before data lands (over).  `mbarrier.arrive.expect_tx(N)` is what
  declares the expected total.
* **Misaligned box dims.**  Inner-dim box width must align with the
  swizzle.  At 128B swizzle with BF16, the inner box dim is always 64
  elements (= 128 bytes).
* **Stale descriptor.**  TMA descriptors are device-side constants;
  changing them per-launch is fine, but a stale pointer in the
  descriptor caches incorrectly.  Recompute the descriptor when A/B
  data pointers change.
