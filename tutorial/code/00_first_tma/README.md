# Chapter 00 — A first TMA program (runnable)

Companion code for [Chapter 00 in the
book](../../book/part2_optimization_ladder/00_first_tma).  Builds a
1D `CUtensorMap`, launches a single-CTA kernel that issues one TMA
load of 128 bytes from GMEM into SMEM, and verifies the round-trip.

## Files

- `kernel.cu` — the kernel source.  Compiled at runtime via NVRTC.
- `main.py` — cuda-python host-side launcher: builds the descriptor,
  launches the kernel, verifies the result.
- `../cuda_utils.py` — shared plumbing (CUDA init, NVRTC compile,
  cuLaunchKernel argument packing, error checking).  Lives one
  directory up so other chapters reuse it; this `main.py` imports
  from there.

## Run

From this directory:

```bash
pip install -r ../requirements.txt
python main.py
```

Expected output (on a B200 / sm_100a):

```
✓ TMA load verified: 128 bytes copied correctly via TMA.
  g_in [first 8 bytes]: [0 1 2 3 4 5 6 7]
  g_out[first 8 bytes]: [0 1 2 3 4 5 6 7]
```

## Requires

- NVIDIA GPU with sm_90 or newer (Hopper or Blackwell).  This example
  is targeted at **sm_100a** (B200); on Hopper (sm_90a) it should work
  too — adjust the `arch` string in `main.py` if NVRTC complains.
- CUDA toolkit ≥ 12.0.
- Python package `cuda-python` ≥ 12.0 (see `../requirements.txt`).
