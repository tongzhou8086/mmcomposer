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
from cuda_utils import cu, init_cuda, compile_kernel, launch, htod, dtoh

from cuda.bindings import driver


# ── Constants matching kernel.cu ────────────────────────────────────────────
CHUNK_BYTES = 128
THREADS_PER_CTA = 128
HERE = os.path.dirname(os.path.abspath(__file__))


# ── 1. Init CUDA + compile kernel.cu via NVRTC ──────────────────────────────
device, ctx = init_cuda()
module, fns = compile_kernel(os.path.join(HERE, "kernel.cu"),
                             device,
                             kernels=["tma_demo"])
tma_demo = fns["tma_demo"]


# ── 2. Allocate device buffers ──────────────────────────────────────────────
g_in_host = np.arange(CHUNK_BYTES, dtype=np.uint8)        # 0, 1, ..., 127
g_in_d  = htod(g_in_host)
g_out_d = cu(driver.cuMemAlloc(CHUNK_BYTES))


# ── 3. Build the 1D CUtensorMap ─────────────────────────────────────────────
#
# This is the chapter's actual subject matter.  Everything above is
# generic CUDA bootstrap; everything below the launch is verification.
#
# globalDim: total tensor shape, innermost-first.  rank entries.
# boxDim:    per-load shape, same ordering.  rank entries.
# Here both are {128} — one TMA load covers the entire 128-byte buffer.
global_dim      = (ctypes.c_uint64 * 1)(CHUNK_BYTES)
box_dim         = (ctypes.c_uint32 * 1)(CHUNK_BYTES)
element_strides = (ctypes.c_uint32 * 1)(1)
# globalStrides: outer-dim strides in BYTES, rank - 1 entries.
# For 1D that's 0 entries → pass NULL.

tmap = driver.CUtensorMap()
cu(driver.cuTensorMapEncodeTiled(
    tmap,
    driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    1,                                                            # rank = number of dims (1D here)
    int(g_in_d),                                                  # global address
    global_dim,
    None,                                                         # globalStrides (NULL for 1D)
    box_dim,
    element_strides,
    driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
    driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
    driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
))


# ── 4. Launch ───────────────────────────────────────────────────────────────
#
# Kernel signature is (CUtensorMap by-value, uint8_t* g_out).
# By-value structs are passed as a ctypes byte-array of the right size;
# pointers as ctypes.c_void_p.

tmap_bytes = bytes(tmap)
arg_tmap = (ctypes.c_byte * len(tmap_bytes)).from_buffer_copy(tmap_bytes)
arg_gout = ctypes.c_void_p(int(g_out_d))

launch(tma_demo,
       grid=(1, 1, 1),
       block=(THREADS_PER_CTA, 1, 1),
       shared=CHUNK_BYTES,
       args=[arg_tmap, arg_gout])


# ── 5. Copy back + verify ───────────────────────────────────────────────────
g_out_host = dtoh(g_out_d, CHUNK_BYTES, np.uint8)

if np.array_equal(g_out_host, g_in_host):
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
cu(driver.cuCtxDestroy(ctx))
