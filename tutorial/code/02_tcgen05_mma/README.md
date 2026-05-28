# Chapter 02 — first `tcgen05.mma` (runnable)

Companion code for [Chapter 02 in the
book](../../book/part2_optimization_ladder/02_tcgen05_mma).  The
minimal kernel that exercises every piece chapter 02 introduced:
TMEM alloc, SWIZZLE_128B TMA loads, the matrix descriptor + `idesc`,
`tcgen05.mma`, the commit/wait handshake, and `tcgen05.ld` for
readback.

## What it computes

```
C[M, N] = A[M, K] @ B[K, N]      with M = 128, N = 256, K = 64
```

One CTA, no pipelining, no outer K-loop.  Tile sizes are picked to
match the MMA atom as tightly as the swizzle allows: K = 64 is the
smallest K that exercises 128B swizzle on both operands (K = 64
BF16 = 128 bytes = one swizzle row).  Within that single SMEM tile,
four back-to-back `tcgen05.mma` calls cover K = 64 in K = 16 strides.

## Files

- `kernel.cu` — the kernel.  All PTX inline, every step from chapter 02
  labelled (TMEM alloc → TMA → fence → 4 MMAs → commit → wait →
  `tcgen05.ld` → GMEM store).
- `main.py` — host launcher: builds the two `CUtensorMap`s, transposes
  `B` on the host so its SMEM lands in N-major-with-K-inner form,
  bumps the dynamic-SMEM cap (48 KB tile crosses the default), runs
  the kernel, verifies against `(A.float() @ B.float()).to(bf16)`.
- `../cuda_utils.py` — shared plumbing (CUDA init, nvcc compile,
  launch, `encode_tensor_map`).

## Run

```bash
pip install -r ../requirements.txt
python main.py
```

Expected output (on a B200 / sm_100a):

```
M = 128, N = 256, K = 64   (BF16 × BF16 → BF16, FP32 accumulate)
  max |C - C_ref|     = ...
  |C_ref|_max          = ...
  max relative error   = ...
✓ matches PyTorch reference
```

## What this does *not* do yet

Deliberately, to keep the code short:

- **No outer K-loop.**  K = 64 fits in one SMEM tile, so the kernel
  runs one stage.  Real matmuls iterate K = `BK, 2·BK, ...` and stream
  fresh tiles into a ring buffer; that's a later chapter.
- **No pipelining.**  The TMA load and the MMA strictly sequence
  through one mbar each.  Multi-stage buffering comes later.
- **Direct (uncoalesced) TMEM → GMEM writeback.**  Each lane reads its
  8 BF16 accumulators and stores them at `C[my_row][n..n+7]`, which
  scatters across 32 cache lines per warp.  SMEM-staged coalescing
  is a separate optimization chapter.
- **Single CTA.**  The grid is `(1, 1, 1)`.  Real kernels launch
  `(M/BM × N/BN)` CTAs to cover the full problem.

The point is to see the MMA chain work end-to-end against PyTorch
before adding the production machinery on top.

## Requires

- NVIDIA Blackwell (sm_100a / B200).
- CUDA toolkit ≥ 13.0 (`tcgen05.*` PTX), Python `cuda-python` ≥ 12.0,
  `torch` (see `../requirements.txt`).
