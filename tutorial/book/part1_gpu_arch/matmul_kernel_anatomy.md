# Anatomy of a matmul kernel

At a high level, every modern matmul kernel on a GPU is split into
three phases that execute in order:

* **Prologue** — one-time setup before the K-loop.  Compute the
  per-CTA tile offsets `(off_m, off_n)`, derive shared-memory addresses
  for the A and B staging slots, and apply any
  CTA-swizzle to the block-id → tile mapping.

* **Main K-loop** — a loop over the K-dimension tiles.  Each iteration
  loads a pair of A and B tiles from global memory and accumulates
  their product into the running C tile.  How fetch and compute are
  scheduled relative to each other (sequential, overlapped, multi-stage
  pipelined) is one of the central optimization choices — see the next
  section.

* **Epilogue** — drain the in-flight MMAs if any, optionally apply a fused
  elementwise function, and write the accumulated C tile back to
  global memory.

The main K-loop is where the kernel spends almost all of its time, and
where every performance lever lives.

## Tile fetch and MMA compute

The K-loop does two things — fetch and compute — and both want to be
**asynchronous** on modern hardware, so they can overlap.

* **Tile fetch.**  Explicitly async on Ampere and later: `cp.async`
  (Ampere) or `cp.async.bulk.tensor` (Hopper TMA, Blackwell TMA).
  Even pre-Ampere kernels achieved a weaker form of overlap, since
  the warp scheduler could keep issuing independent global loads.

* **MMA compute.**  Sync vs async depends on the instruction:
  * `mma.sync` — synchronous; supported on Volta through Blackwell.
    The issuing warp stalls until the MMA retires.
  * `wgmma.mma_async` — async; Hopper-native, warpgroup-issued,
    drained via a fence.
  * `tcgen05.mma` — async; Blackwell-native, one-thread-issued,
    accumulates into TMEM, drained via `tcgen05.commit`.

## The key optimization: overlap

To approach peak throughput, you almost always need to **overlap the
tile fetch of iteration *k+1* with the MMA compute of iteration *k***.
The standard mechanism is **SMEM multi-stage buffering**: allocate `NS`
slots in shared memory and run them as a ring buffer.  At any moment,
up to `NS − 1` tile fetches can be in flight while one slot is being
consumed by MMA; when MMA finishes consuming slot `i`, that slot
becomes available for the next fetch.

```
                iter k-1     iter k       iter k+1     iter k+2
Tile fetch:    [ fetch ]   [ fetch  ]   [ fetch   ]   [ fetch ]
MMA compute:       ...    [ compute ]   [ compute ]   ...
                              ↑ overlapped with the next fetch
```

(How "tile fetch" and "MMA compute" are mapped onto threads/warps
varies by architecture — they can run in the same warp interleaved,
or in dedicated warps via warp specialization.  The pipeline shape is
the same either way.)

Almost every Part 2 optimization is *either* an improvement to one end
of this pipeline (faster fetch, faster MMA), *or* a richer way to keep
the two ends overlapped (multi-stage buffering, warp specialization,
2-CTA cluster MMA).
