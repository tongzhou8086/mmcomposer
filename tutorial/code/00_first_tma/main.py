"""Runnable companion for Chapter 00 — A first TMA program.

Builds a 2D BF16 CUtensorMap for an 8×64 tensor in CUDA memory,
compiles kernel.cu via nvcc, launches the kernel, and verifies that
the first row was copied correctly.

Generic cuda-python plumbing (init, nvcc compile, launch) lives in
`../cuda_utils.py`.

Run:
    pip install -r ../requirements.txt
    python main.py
"""

import os
import sys
import ctypes

import torch

# Pull in shared cuda-python plumbing from the parent tutorial/code/ dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cuda_utils import (
    cu, init_cuda, compile_kernel, launch,
    encode_tensor_map, TMA_BFLOAT16, TMA_SWIZZLE_NONE,
)

from cuda.bindings import driver


# ── Constants matching kernel.cu ────────────────────────────────────────────
ROWS, COLS, ELEM_BYTES = 8, 64, 2           # 8 × 64 BF16 = 1024 bytes
CHUNK_BYTES            = COLS * ELEM_BYTES  # 128 bytes per TMA load (one row)
THREADS_PER_CTA        = 128
HERE = os.path.dirname(os.path.abspath(__file__))


# ── 1. Init CUDA + compile kernel.cu via nvcc ──────────────────────────────
device, ctx = init_cuda()
module, fns = compile_kernel(os.path.join(HERE, "kernel.cu"),
                             device,
                             kernels=["tma_demo"])
tma_demo = fns["tma_demo"]


# ── 2. Allocate device buffers via torch ───────────────────────────────────
# Cycle values 0..16 through the tensor so the verification has something
# non-trivial to compare against; BF16 represents these exactly.
g_in  = (torch.arange(ROWS * COLS, dtype=torch.float32) % 17.0).to(
    device="cuda", dtype=torch.bfloat16).view(ROWS, COLS)
g_out = torch.zeros(COLS, device="cuda", dtype=torch.bfloat16)


# ── 3. Build the 2D BF16 CUtensorMap ───────────────────────────────────────
#
# This is the chapter's actual subject matter.  Everything above is
# generic CUDA bootstrap; everything below the launch is verification.
#
# globalDim: total tensor shape, innermost-first.   rank entries.
# boxDim:    per-load shape, same ordering.         rank entries.
# globalStrides: outer-dim strides in BYTES.        rank - 1 entries.
tmap = encode_tensor_map(
    dtype=TMA_BFLOAT16,
    rank=2,
    gptr=g_in.data_ptr(),                   # torch tensor's device pointer
    global_dim=[COLS, ROWS],                # innermost first
    global_strides=[COLS * ELEM_BYTES],     # bytes per row
    box_dim=[COLS, 1],                      # one row per load
    element_strides=[1, 1],
    swizzle=TMA_SWIZZLE_NONE,               # linear byte order — easiest to verify
)


# ── 4. Launch ──────────────────────────────────────────────────────────────
#
# Kernel signature is (CUtensorMap by-value, __nv_bfloat16* g_out).
# By-value structs are passed as a ctypes byte-array of the right size;
# pointers as ctypes.c_void_p.

arg_tmap = (ctypes.c_byte * 128).from_buffer_copy(tmap.tobytes())
arg_gout = ctypes.c_void_p(g_out.data_ptr())

launch(tma_demo,
       grid=(1, 1, 1),
       block=(THREADS_PER_CTA, 1, 1),
       shared=CHUNK_BYTES,
       args=[arg_tmap, arg_gout])


# ── 5. Verify ──────────────────────────────────────────────────────────────
expected = g_in[0]
if torch.equal(g_out, expected):
    print(f"✓ TMA load verified: {COLS} BF16 elements copied correctly via TMA.")
    print(f"  g_in[0, :8]:  {g_in[0, :8].cpu().tolist()}")
    print(f"  g_out[:8]:    {g_out[:8].cpu().tolist()}")
else:
    print("✗ MISMATCH:")
    print(f"  g_in[0, :8]:  {g_in[0, :8].cpu().tolist()}")
    print(f"  g_out[:8]:    {g_out[:8].cpu().tolist()}")
    sys.exit(1)


# ── 6. Cleanup ─────────────────────────────────────────────────────────────
# Torch handles tensor deallocation automatically on scope exit.
cu(driver.cuModuleUnload(module))
cu(driver.cuDevicePrimaryCtxRelease(device))
