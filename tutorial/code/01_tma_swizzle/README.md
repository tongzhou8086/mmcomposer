# Chapter 01 — TMA with SWIZZLE_128B (runnable)

Companion code for [Chapter 01 in the
book](../../book/part2_optimization_ladder/01_tma_swizzle).  Loads the
whole 8×64 BF16 tile in one TMA bulk with `swizzle=128B`, copies SMEM
back out in linear physical order, and prints the result so the chunk
permutation is visible directly.

Differs from chapter 00 in three lines: `swizzle=TMA_SWIZZLE_128B`,
`box_dim=[COLS, ROWS]` (whole tile), and a structured input
(`row*10 + chunk_index`) chosen to make the permutation legible.

## Files

- `kernel.cu` — the kernel.  Tile placed first in dynamic SMEM (window
  offset 0) so the swizzle pattern is the canonical `chunk XOR (row%8)`.
- `main.py` — host launcher: builds the descriptor, launches, prints
  input vs swizzled output, and verifies the XOR-by-row rule.
- `../cuda_utils.py` — shared plumbing (CUDA init, nvcc compile, launch,
  `encode_tensor_map`).

## Run

```bash
pip install -r ../requirements.txt
python main.py
```

Expected output (on a B200 / sm_100a) ends with:

```
OUTPUT g_out (raw SWIZZLE_128B SMEM layout):
  row 0: [ 0,  1,  2,  3,  4,  5,  6,  7]
  row 1: [11, 10, 13, 12, 15, 14, 17, 16]
  ...
  row 7: [77, 76, 75, 74, 73, 72, 71, 70]

✓ Layout matches the 128B swizzle rule:  physical_chunk = logical_chunk XOR (row % 8)
```

## Requires

- NVIDIA Blackwell (sm_100a / B200).  Should also work on Hopper
  (sm_90a) — adjust the arch if needed.
- CUDA toolkit ≥ 12.0, Python `cuda-python` ≥ 12.0, `torch` (see
  `../requirements.txt`).
