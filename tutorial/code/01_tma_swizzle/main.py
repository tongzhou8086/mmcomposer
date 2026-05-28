"""Runnable companion for Chapter 01 — TMA with SWIZZLE_128B.

Builds a 2D BF16 CUtensorMap for an 8x64 tensor with swizzle=128B,
loads the whole tile into SMEM in one TMA bulk, copies SMEM back out
in linear physical order, and prints the result next to the input so
the chunk permutation SWIZZLE_128B applies is visible directly.

Differs from chapter 00 in exactly three lines:
  * swizzle=TMA_SWIZZLE_128B          (was TMA_SWIZZLE_NONE)
  * box_dim=[COLS, ROWS]              (was [COLS, 1] — load all rows)
  * structured input (see below)      (was torch.randn)

Run:
    pip install -r ../requirements.txt
    python main.py
"""

import os
import sys
import ctypes

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cuda_utils import (
    cu, init_cuda, compile_kernel, launch,
    encode_tensor_map, TMA_BFLOAT16, TMA_SWIZZLE_128B,
)

from cuda.bindings import driver


# ── Constants matching kernel.cu ────────────────────────────────────────────
ROWS, COLS, ELEM_BYTES = 8, 64, 2           # 8 x 64 BF16 = 1024 bytes
TILE_BYTES             = ROWS * COLS * ELEM_BYTES
THREADS_PER_CTA        = 128
CHUNK                  = 8                   # 16-byte swizzle chunk = 8 BF16
HERE = os.path.dirname(os.path.abspath(__file__))


# ── 1. Init CUDA + compile kernel.cu via nvcc ──────────────────────────────
device, ctx = init_cuda()
module, fns = compile_kernel(os.path.join(HERE, "kernel.cu"),
                             device,
                             kernels=["tma_swizzle_demo"])
tma_swizzle_demo = fns["tma_swizzle_demo"]


# ── 2. Allocate device buffers via torch ───────────────────────────────────
#
# Structured input instead of randn: every element of a 16-byte chunk
# (8 consecutive BF16 values) is set to the same integer  row*10 + chunk.
# SWIZZLE_128B only ever permutes whole chunks, never elements within a
# chunk, so chunk-constant values make the permutation legible with no
# intra-chunk noise.  Values stay <= 77, exactly representable in BF16.
row_idx   = torch.arange(ROWS, device="cuda").view(ROWS, 1)
chunk_idx = (torch.arange(COLS, device="cuda") // CHUNK).view(1, COLS)
g_in  = (row_idx * 10 + chunk_idx).to(torch.bfloat16)        # (ROWS, COLS)
g_out = torch.zeros(ROWS * COLS, dtype=torch.bfloat16, device="cuda")


# ── 3. Build the 2D BF16 CUtensorMap with SWIZZLE_128B ─────────────────────
tmap = encode_tensor_map(
    dtype=TMA_BFLOAT16,
    rank=2,
    gptr=g_in.data_ptr(),
    global_dim=[COLS, ROWS],                # innermost first
    global_strides=[COLS * ELEM_BYTES],     # bytes per row
    box_dim=[COLS, ROWS],                   # whole tile in one load
    element_strides=[1, 1],
    swizzle=TMA_SWIZZLE_128B,               # <-- the subject of this chapter
)


# ── 4. Launch ──────────────────────────────────────────────────────────────
arg_tmap = (ctypes.c_byte * 128).from_buffer_copy(tmap.tobytes())
arg_gout = ctypes.c_void_p(g_out.data_ptr())

launch(tma_swizzle_demo,
       grid=(1, 1, 1),
       block=(THREADS_PER_CTA, 1, 1),
       shared=TILE_BYTES + 8,          # tile (1024B) + mbarrier (8B), carved in-kernel
       args=[arg_tmap, arg_gout])


# ── 5. Show the effect ──────────────────────────────────────────────────────
g_out = g_out.view(ROWS, COLS)

# One representative value per 8-BF16 chunk (every CHUNK-th column).
def chunk_reps(t):
    return t[:, ::CHUNK].to(torch.int32).cpu().tolist()

print("Each value below is  row*10 + chunk_index  (constant within a chunk).")
print("Columns are the 8 chunks of each 128-byte row.\n")

print("INPUT  g_in  (natural, un-swizzled chunk order):")
for r, row in enumerate(chunk_reps(g_in)):
    print(f"  row {r}: {row}")

print("\nOUTPUT g_out (raw SWIZZLE_128B SMEM layout):")
for r, row in enumerate(chunk_reps(g_out)):
    print(f"  row {r}: {row}")

# Verify the layout is exactly chunk-XOR-by-row: physical chunk pc of row r
# holds logical chunk (pc XOR r).
expected = torch.empty_like(g_in)
for r in range(ROWS):
    for pc in range(COLS // CHUNK):
        lc = pc ^ (r % 8)
        expected[r, pc * CHUNK:(pc + 1) * CHUNK] = g_in[r, lc * CHUNK:(lc + 1) * CHUNK]

if torch.equal(g_out, expected):
    print("\n✓ Layout matches the 128B swizzle rule:  physical_chunk = logical_chunk XOR (row % 8)")
else:
    print("\n✗ Layout does NOT match the expected XOR-by-row rule.")
    sys.exit(1)


# ── 6. Cleanup ─────────────────────────────────────────────────────────────
cu(driver.cuModuleUnload(module))
cu(driver.cuDevicePrimaryCtxRelease(device))
