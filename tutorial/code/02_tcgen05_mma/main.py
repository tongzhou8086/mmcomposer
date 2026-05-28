"""Runnable companion for Chapter 02 — first tcgen05.mma.

Builds a single-tile BF16 matmul on B200:

    C[M, N] = A[M, K] @ B[K, N],   M = 128, N = 256, K = 64

One CTA, no pipelining.  Exercises the full Blackwell async-MMA chain:
TMEM alloc → TMA bulk loads (SWIZZLE_128B) → tcgen05.mma × 4 → commit →
mbarrier wait → tcgen05.ld → GMEM store.  Verifies against PyTorch.

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


# ── Problem shape (must match kernel.cu's constants) ───────────────────────
M, N, K          = 128, 256, 64
ELEM_BYTES       = 2
THREADS_PER_CTA  = 128
A_SMEM_BYTES     = M * K * ELEM_BYTES        # 16 KB
B_SMEM_BYTES     = N * K * ELEM_BYTES        # 32 KB
TILE_BYTES       = A_SMEM_BYTES + B_SMEM_BYTES   # 48 KB
HERE             = os.path.dirname(os.path.abspath(__file__))


# ── 1. Init CUDA + compile kernel.cu via nvcc ──────────────────────────────
device, ctx = init_cuda()
module, fns = compile_kernel(os.path.join(HERE, "kernel.cu"),
                             device,
                             kernels=["tcgen05_demo"])
kernel = fns["tcgen05_demo"]

# 48 KB of dynamic SMEM crosses the default per-CTA cap on most archs;
# request the larger budget explicitly.  +1 KB of slack absorbs the
# __align__(1024) padding on the tile base.
SHARED_BYTES = TILE_BYTES + 1024
cu(driver.cuFuncSetAttribute(
    kernel,
    driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    SHARED_BYTES))


# ── 2. Inputs + reference ──────────────────────────────────────────────────
torch.manual_seed(0)
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

# Reference computed in FP32 for a fair BF16 comparison.
C_ref = (A.float() @ B.float()).to(torch.bfloat16)

# B is transposed on the host so its SMEM layout comes out
# [N rows][K cols, innermost] — matching the matrix descriptor's
# expectation when idesc bit 16 = 0.  Cost: a one-time copy.
B_t = B.t().contiguous()      # (N, K) row-major


# ── 3. TMA descriptors (SWIZZLE_128B, whole tile in one bulk) ──────────────
A_tmap = encode_tensor_map(
    dtype=TMA_BFLOAT16,
    rank=2,
    gptr=A.data_ptr(),
    global_dim=[K, M],                  # innermost first
    global_strides=[K * ELEM_BYTES],    # bytes per M-row
    box_dim=[K, M],                     # whole A tile in one load
    element_strides=[1, 1],
    swizzle=TMA_SWIZZLE_128B,
)
B_tmap = encode_tensor_map(
    dtype=TMA_BFLOAT16,
    rank=2,
    gptr=B_t.data_ptr(),
    global_dim=[K, N],
    global_strides=[K * ELEM_BYTES],
    box_dim=[K, N],                     # whole B^T tile in one load
    element_strides=[1, 1],
    swizzle=TMA_SWIZZLE_128B,
)


# ── 4. Launch ──────────────────────────────────────────────────────────────
arg_a = (ctypes.c_byte * 128).from_buffer_copy(A_tmap.tobytes())
arg_b = (ctypes.c_byte * 128).from_buffer_copy(B_tmap.tobytes())
arg_c = ctypes.c_void_p(C.data_ptr())

launch(kernel,
       grid=(1, 1, 1),
       block=(THREADS_PER_CTA, 1, 1),
       shared=SHARED_BYTES,
       args=[arg_a, arg_b, arg_c])


# ── 5. Verify ──────────────────────────────────────────────────────────────
C_f   = C.float()
ref_f = C_ref.float()
max_abs_err = (C_f - ref_f).abs().max().item()
max_ref     = ref_f.abs().max().item()
rel         = max_abs_err / max(max_ref, 1e-8)

print(f"M = {M}, N = {N}, K = {K}   (BF16 × BF16 → BF16, FP32 accumulate)")
print(f"  max |C - C_ref|     = {max_abs_err:.4f}")
print(f"  |C_ref|_max          = {max_ref:.4f}")
print(f"  max relative error   = {rel:.4%}")

# BF16 accumulating into F32 gives a few ULPs of slop after K = 64 adds —
# allow up to ~2% relative.
if torch.allclose(C_f, ref_f, rtol=2e-2, atol=2e-2):
    print("✓ matches PyTorch reference")
else:
    print("✗ MISMATCH")
    print(f"  C[0, :8]:     {C[0, :8].cpu().tolist()}")
    print(f"  C_ref[0, :8]: {C_ref[0, :8].cpu().tolist()}")
    sys.exit(1)


# ── 6. Cleanup ─────────────────────────────────────────────────────────────
cu(driver.cuModuleUnload(module))
cu(driver.cuDevicePrimaryCtxRelease(device))
