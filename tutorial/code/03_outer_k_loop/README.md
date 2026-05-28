# Chapter 03 — outer K-loop (runnable)

Companion code for [Chapter 03 in the
book](../../book/part2_optimization_ladder/03_outer_k_loop).  Same
single-CTA matmul as chapter 02, but K is now a runtime parameter
streamed through the SMEM tile in chunks of BK = 64.

## What it computes

```
C[M, N] = A[M, K] @ B[K, N]   with M = 128, N = 256, K = 512
```

One CTA, single-stage SMEM, no pipelining.  The outer loop runs
`K / BK = 8` iterations; each iteration TMAs the next `(BM × BK)` /
`(BN × BK)` slab into the same SMEM tile and accumulates into the same
TMEM accumulator with 4 back-to-back `tcgen05.mma` calls.

The three things that change vs chapter 02:

* **Mbarrier phase alternates** — `mbarrier_wait_phase(mb, k_iter & 1)`
  every iteration (the parity bit flips on each completion).
* **Accumulate predicate** — `P = false` only for `k_iter == 0 && kk
  == 0`; every other MMA accumulates.
* **TMA coordinate** — `x = k_iter * BK` advances the K-window each
  iter; `y = 0` (we still own a single M-block).

## Files

- `kernel.cu` — the kernel.  Same wrappers + epilogue as chapter 02;
  the body is wrapped in a K-loop and `mbarrier_wait` takes a phase.
- `main.py` — host launcher.  K is now passed as a kernel argument and
  the TMA box is `(BK, BM)` / `(BK, BN)` so the K-window slides
  iteration-by-iteration.
- `../cuda_utils.py` — shared plumbing (unchanged).

## Run

```bash
pip install -r ../requirements.txt
python main.py
```

Expected output (on a B200 / sm_100a):

```
M = 128, N = 256, K = 512   (8 outer K-iters of BK = 64)
  max |C - C_ref|     = ...
  |C_ref|_max         = ...
  max relative error  = < 0.1 %
✓ matches PyTorch reference
```

## What this does *not* do yet

- **Single SMEM stage.**  TMA and MMA strictly alternate — the wait on
  `mma_done_mb` at the bottom of each iter blocks the *next* TMA from
  starting.  Roughly half utilization.  Chapter 04 adds multi-stage
  buffering so the two run concurrently.
- **Single CTA.**  The grid is `(1, 1, 1)`, so the kernel still only
  computes one `(BM × BN) = (128 × 256)` output tile.  Full M × N
  output needs a grid of CTAs (next-after-04 chapter).
- **Direct (uncoalesced) writeback.**  Same as chapter 02.

## Requires

- NVIDIA Blackwell (sm_100a / B200).
- CUDA toolkit ≥ 13.0, Python `cuda-python` ≥ 12.0, `torch`.
