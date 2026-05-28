"""Runnable companion for Chapter 03 — outer K-loop.

Same single-CTA matmul as chapter 02, but K is now a runtime parameter
streamed through the SMEM tile in chunks of BK = 64.  Default problem:

    C[M, N] = A[M, K] @ B[K, N],   M = 128, N = 256, K = 512

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


# ── Problem shape ──────────────────────────────────────────────────────────
BM, BN, BK       = 128, 256, 64
M, N, K          = BM, BN, 512           # K must be a multiple of BK
assert K % BK == 0, "K must be a multiple of BK"

ELEM_BYTES       = 2
THREADS_PER_CTA  = 128
A_SMEM_BYTES     = BM * BK * ELEM_BYTES
B_SMEM_BYTES     = BN * BK * ELEM_BYTES
TILE_BYTES       = A_SMEM_BYTES + B_SMEM_BYTES         # 48 KB
HERE             = os.path.dirname(os.path.abspath(__file__))


# ── 1. Init CUDA + compile kernel.cu ───────────────────────────────────────
device, ctx = init_cuda()
module, fns = compile_kernel(os.path.join(HERE, "kernel.cu"),
                             device,
                             kernels=["matmul_k_loop"])
kernel = fns["matmul_k_loop"]

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

C_ref = (A.float() @ B.float()).to(torch.bfloat16)

# B transposed so SMEM lands N-major-with-K-inner (see ch02).
B_t = B.t().contiguous()                              # (N, K) row-major


# ── 3. TMA descriptors — note global_dim now uses the FULL K ───────────────
#
# Inner (K) box size = BK = 64 BF16 = 128 bytes (one swizzle row),
# outer box size = BM or BN.  The kernel calls TMA with coord
# (k_iter * BK, 0) to step through K.
A_tmap = encode_tensor_map(
    dtype=TMA_BFLOAT16,
    rank=2,
    gptr=A.data_ptr(),
    global_dim=[K, M],
    global_strides=[K * ELEM_BYTES],
    box_dim=[BK, BM],                                 # per-tile box
    element_strides=[1, 1],
    swizzle=TMA_SWIZZLE_128B,
)
B_tmap = encode_tensor_map(
    dtype=TMA_BFLOAT16,
    rank=2,
    gptr=B_t.data_ptr(),
    global_dim=[K, N],
    global_strides=[K * ELEM_BYTES],
    box_dim=[BK, BN],
    element_strides=[1, 1],
    swizzle=TMA_SWIZZLE_128B,
)


# ── 4. Launch ──────────────────────────────────────────────────────────────
arg_a = (ctypes.c_byte * 128).from_buffer_copy(A_tmap.tobytes())
arg_b = (ctypes.c_byte * 128).from_buffer_copy(B_tmap.tobytes())
arg_c = ctypes.c_void_p(C.data_ptr())
arg_K = ctypes.c_int(K)

launch(kernel,
       grid=(1, 1, 1),
       block=(THREADS_PER_CTA, 1, 1),
       shared=SHARED_BYTES,
       args=[arg_a, arg_b, arg_c, arg_K])


# ── 5. Verify ──────────────────────────────────────────────────────────────
C_f   = C.float()
ref_f = C_ref.float()
max_abs_err = (C_f - ref_f).abs().max().item()
max_ref     = ref_f.abs().max().item()
rel         = max_abs_err / max(max_ref, 1e-8)

print(f"M = {M}, N = {N}, K = {K}   ({K // BK} outer K-iters of BK = {BK})")
print(f"  max |C - C_ref|     = {max_abs_err:.4f}")
print(f"  |C_ref|_max          = {max_ref:.4f}")
print(f"  max relative error   = {rel:.4%}")

# At K = 512 the BF16 inputs accumulate K rounding errors in FP32 — a
# few percent of the largest output magnitude is normal.
if torch.allclose(C_f, ref_f, rtol=5e-2, atol=5e-2):
    print("✓ matches PyTorch reference")
else:
    print("✗ MISMATCH")
    print(f"  C[0, :8]:     {C[0, :8].cpu().tolist()}")
    print(f"  C_ref[0, :8]: {C_ref[0, :8].cpu().tolist()}")
    sys.exit(1)


# ── 6. Cleanup ─────────────────────────────────────────────────────────────
cu(driver.cuModuleUnload(module))
cu(driver.cuDevicePrimaryCtxRelease(device))
