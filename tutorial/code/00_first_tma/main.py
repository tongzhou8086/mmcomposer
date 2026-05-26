"""Runnable companion for Chapter 00 — A first TMA program.

Builds the host-side CUtensorMap, compiles kernel.cu via NVRTC,
launches it on a single CTA with 128 threads, and verifies that
g_out == g_in[:CHUNK_BYTES].

Generic cuda-python plumbing (init, NVRTC, launch, memcpy) lives in
`../cuda_utils.py`.

Run:
    pip install -r ../requirements.txt
    python main.py
"""

import os
import sys
import ctypes
import numpy as np


# Pull in shared cuda-python plumbing from the parent tutorial/code/ dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cuda_utils import (
    cu, init_cuda, compile_kernel, launch, htod, dtoh,
    encode_tensor_map, TMA_BFLOAT16, TMA_SWIZZLE_NONE,
)

from cuda.bindings import driver


# ── Constants matching kernel.cu ────────────────────────────────────────────
# We work in BF16 (2 bytes per element) and use 2D TMA — the same shape
# matmul will use later.  Tensor is 8 rows × 64 cols (1024 bytes total);
# one TMA load fetches one row (64 elems = 128 bytes), matching the
# 128B swizzle.
ROWS            = 8
COLS            = 64      # inner dim, in BF16 elements (= 128 bytes per row)
ELEM_BYTES      = 2       # BF16
TOTAL_BYTES     = ROWS * COLS * ELEM_BYTES   # 1024
CHUNK_BYTES     = COLS * ELEM_BYTES          # 128 bytes per TMA load
THREADS_PER_CTA = 128
HERE = os.path.dirname(os.path.abspath(__file__))


# ── 1. Init CUDA + compile kernel.cu via NVRTC ──────────────────────────────
device, ctx = init_cuda()
module, fns = compile_kernel(os.path.join(HERE, "kernel.cu"),
                             device,
                             kernels=["tma_demo"])
tma_demo = fns["tma_demo"]


# ── 2. Allocate device buffers ──────────────────────────────────────────────
# Input is TOTAL_BYTES long; we'll TMA-load just the first CHUNK_BYTES
# (= the first row of the 2D tensor).
g_in_host = (np.arange(TOTAL_BYTES, dtype=np.uint32) % 256).astype(np.uint8)
g_in_d  = htod(g_in_host)
g_out_d = cu(driver.cuMemAlloc(CHUNK_BYTES))


# ── 3. Build the 1D CUtensorMap ─────────────────────────────────────────────
#
# This is the chapter's actual subject matter.  Everything above is
# generic CUDA bootstrap; everything below the launch is verification.
#
# globalDim: total tensor shape, innermost-first.  `rank` entries.
# boxDim:    per-load shape, same ordering.  `rank` entries.
# Here both are [128] — one TMA load covers the entire 128-byte buffer.
# globalStrides: outer-dim strides in BYTES.  rank - 1 entries.
# For 1D that's 0 entries → omit / pass None.
#
# encode_tensor_map (defined in cuda_utils) wraps libcuda's
# cuTensorMapEncodeTiled directly via ctypes — cleaner across
# cuda-python versions, no typed-element wrapping required.
# Returns a numpy uint8 array of 128 bytes (the opaque descriptor).
# 2D BF16 descriptor: tensor is (ROWS, COLS) row-major, box is one row.
# Innermost-first ordering: dim[0] is COLS, dim[1] is ROWS.
# globalStrides has 1 entry (= rank - 1) — the BYTE stride between
# successive rows.
tmap = encode_tensor_map(
    dtype=TMA_BFLOAT16,
    rank=2,
    gptr=int(g_in_d),
    global_dim=[COLS, ROWS],                # innermost first
    global_strides=[COLS * ELEM_BYTES],     # bytes per row
    box_dim=[COLS, 1],                      # one row per load
    element_strides=[1, 1],
    swizzle=TMA_SWIZZLE_NONE,   # linear byte order — easiest to verify
)


# ── 4. Launch ───────────────────────────────────────────────────────────────
#
# Kernel signature is (CUtensorMap by-value, uint8_t* g_out).
# By-value structs are passed as a ctypes byte-array of the right size;
# pointers as ctypes.c_void_p.

arg_tmap = (ctypes.c_byte * 128).from_buffer_copy(tmap.tobytes())
arg_gout = ctypes.c_void_p(int(g_out_d))

launch(tma_demo,
       grid=(1, 1, 1),
       block=(THREADS_PER_CTA, 1, 1),
       shared=CHUNK_BYTES,
       args=[arg_tmap, arg_gout])


# ── 5. Copy back + verify ───────────────────────────────────────────────────
g_out_host = dtoh(g_out_d, CHUNK_BYTES, np.uint8)

if np.array_equal(g_out_host, g_in_host[:CHUNK_BYTES]):
    print(f"✓ TMA load verified: {CHUNK_BYTES} bytes copied correctly via TMA.")
    print(f"  g_in [first 8 bytes]: {g_in_host[:8]}")
    print(f"  g_out[first 8 bytes]: {g_out_host[:8]}")
else:
    print("✗ MISMATCH:")
    print(f"  g_in [first 16]: {g_in_host[:16]}")
    print(f"  g_out[first 16]: {g_out_host[:16]}")
    sys.exit(1)


# ── 6. Cleanup ──────────────────────────────────────────────────────────────
cu(driver.cuMemFree(g_in_d))
cu(driver.cuMemFree(g_out_d))
cu(driver.cuModuleUnload(module))
cu(driver.cuDevicePrimaryCtxRelease(device))
