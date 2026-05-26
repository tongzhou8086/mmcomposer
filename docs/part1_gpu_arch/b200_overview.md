# B200 — Hardware Overview

A focused summary of the B200 facts that matter for GEMM optimization.
Not a comprehensive architecture document — see NVIDIA's whitepaper for
that.  This chapter lists the numbers and structural decisions that we
will rely on throughout Part 2.

## Topology and scale

- **SMs**: 148 per GPU
- **Compute cluster**: SMs are arranged into pairs that can form 2-CTA
  thread block clusters via `__cluster_dims__(2, 1, 1)`.  This is the
  hardware unit that enables the `cta_group::2` MMA variant.
- **Threads per SM**: up to 2048
- **Warps per SM**: up to 64
- **Blocks per SM**: up to 32

## Peak rates (BF16)

- TODO: peak tensor-core BF16 throughput per SM
- TODO: peak tensor-core BF16 throughput per GPU
- TODO: HBM3e bandwidth
- TODO: L2 bandwidth (segmented L2, per-side / aggregate)

## Memory hierarchy

- **HBM3e** — TODO: total capacity and aggregate bandwidth
- **L2 cache** — 132 MB total, arranged as two segments.  Cross-segment
  reads are not free; tile-rasterization order can matter when working
  sets approach 132 MB.
- **Shared memory (SMEM)** — up to ~228 KB per CTA dynamic SMEM (opt-in
  via `cudaFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)`).  This is
  what gates the maximum pipeline depth `NS`.
- **Tensor Memory (TMEM)** — new on Blackwell.  128 lanes × 512 columns
  per SM, addressed via `tcgen05.{alloc, ld, st, mma}`.  See Part 2,
  chapter 6.

## Tensor cores: tcgen05 vs mma.sync

B200 supports both:

- **`tcgen05.mma`** — Blackwell-native async MMA via TMEM.  Higher
  throughput; the primary path covered in this book.
- **`mma.sync`** — legacy synchronous MMA (sm_75+).  Still functional on
  B200.  Used in the early chapters of Part 2 as the simpler starting
  point before introducing tcgen05.

## TMA and other async data movement

- **`cp.async.bulk.tensor.{1d,2d,3d,4d,5d}`** — TMA bulk loads with
  hardware descriptors.  See Part 2, chapters 4 and 8.
- **`cp.async.ca`** — pre-TMA async copy (sm_80+).  Used in chapter 3
  before introducing TMA.
- **Mbarriers** (`mbarrier.{init, arrive, arrive.expect_tx,
  try_wait.parity}`) — the synchronization primitive that ties async
  ops to their consumers.  Threaded through Parts 4 onward.

## What to keep in mind throughout

- The **per-SM SMEM budget (~228 KB)** sets the ceiling on how deep the
  TMA pipeline can be staged.  Many of the wins in Part 2 come from
  reducing per-stage SMEM so more stages fit.
- The **132 MB L2** is large but not unlimited.  At K ≈ 16384 with
  BF16, an A-stripe of 16 cluster-rows is already ~67 MB.  Walk order
  becomes load-bearing for large shapes.
- **Async issue is cheap; sync is expensive.**  Almost every Part 2
  optimization is either reducing sync cost or hiding latency.
