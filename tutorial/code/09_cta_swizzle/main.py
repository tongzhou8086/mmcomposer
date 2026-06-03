"""Runnable companion for Chapter 09 — CTA swizzling.

Holds NS = 5 (ch08's best at 4096³) fixed and sweeps GROUP_SIZE_M ∈
{1, 4, 8, 16}, where GSM = 1 reproduces ch08's no-swizzle walk.  All
runs at M = N = K = 4096 — see the README for the rationale for
fixing a single shape.
"""

import os
import sys
import ctypes

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cuda_utils import (
    cu, init_cuda, compile_kernel, launch, time_kernel_us,
    encode_tensor_map, TMA_BFLOAT16, TMA_SWIZZLE_128B,
)

from cuda.bindings import driver


BM, BN, BK    = 128, 256, 64
CTA_GROUP     = 2
BN_LOCAL      = BN // CTA_GROUP
ELEM_BYTES    = 2
THREADS       = 128
NS            = 5
GSM_SWEEP     = [1, 4, 8, 16]
M, N, K       = 8192, 8192, 8192

A_SLOT_BYTES  = BM       * BK * ELEM_BYTES
B_SLOT_BYTES  = BN_LOCAL * BK * ELEM_BYTES
SLOT_BYTES    = A_SLOT_BYTES + B_SLOT_BYTES
BN_PAD        = BN + 8
EPI_STAGING   = BM * BN_PAD * ELEM_BYTES
SHARED_BYTES  = max(NS * SLOT_BYTES, EPI_STAGING) + 1024


device, ctx = init_cuda()
module, fns = compile_kernel(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.cu"),
    device,
    kernels=[f"matmul_swizzle_ns{NS}_gsm{g}" for g in GSM_SWEEP])
kernels = {g: fns[f"matmul_swizzle_ns{NS}_gsm{g}"] for g in GSM_SWEEP}

for kern in kernels.values():
    cu(driver.cuFuncSetAttribute(
        kern,
        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        SHARED_BYTES))


torch.manual_seed(0)
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

A_tmap = encode_tensor_map(
    dtype=TMA_BFLOAT16, rank=2, gptr=A.data_ptr(),
    global_dim=[K, M], global_strides=[K * ELEM_BYTES],
    box_dim=[BK, BM], element_strides=[1, 1], swizzle=TMA_SWIZZLE_128B)
B_tmap = encode_tensor_map(
    dtype=TMA_BFLOAT16, rank=2, gptr=B.data_ptr(),
    global_dim=[N, K], global_strides=[N * ELEM_BYTES],
    box_dim=[64, BK], element_strides=[1, 1], swizzle=TMA_SWIZZLE_128B)

arg_a = (ctypes.c_byte * 128).from_buffer_copy(A_tmap.tobytes())
arg_b = (ctypes.c_byte * 128).from_buffer_copy(B_tmap.tobytes())
arg_c = ctypes.c_void_p(C.data_ptr())
arg_M, arg_N, arg_K = ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)
args = [arg_a, arg_b, arg_c, arg_M, arg_N, arg_K]

grid_m_clusters = M // (CTA_GROUP * BM)
grid_n          = N // BN
grid = (grid_m_clusters * grid_n * CTA_GROUP, 1, 1)


def time_kernel(kernel):
    """Median per-call time (µs) via the shared do_bench wrapper."""
    return time_kernel_us(lambda: launch(
        kernel, grid=grid, block=(THREADS, 1, 1),
        shared=SHARED_BYTES, args=args, sync=False))


# Correctness check on GSM = 1
C.zero_()
launch(kernels[1], grid=grid, block=(THREADS, 1, 1),
       shared=SHARED_BYTES, args=args)
C_ref = (A.float() @ B.float()).to(torch.bfloat16)
rel = (C.float() - C_ref.float()).abs().max().item() / C_ref.float().abs().max().item()
ok = "✓" if rel < 5e-2 else "✗"

flops = 2.0 * M * N * K
print(f"{ok}  M=N=K={M}   NS={NS}   grid={grid_m_clusters}×{grid_n} clusters   rel err={rel:.2%}\n")
print(f"     {'GSM':>4}   {'walk':<32}   us/call   TFLOPS")
print(f"     {'─'*4}   {'─'*32}   ─────────   ───────")
walks = {
    1:  "= ch08 (N-fast within M-row)",
    4:  "M-fast within chunks of 4",
    8:  "M-fast within chunks of 8",
    16: "M-fast within chunks of 16",
}
for g in GSM_SWEEP:
    us = time_kernel(kernels[g])
    tf = flops / (us * 1e-6) / 1e12
    print(f"     {g:>4}   {walks[g]:<32}   {us:>9.1f}   {tf:>7.1f}")


cu(driver.cuModuleUnload(module))
cu(driver.cuDevicePrimaryCtxRelease(device))
