"""Runnable companion for Chapter 11 — hoist descriptor builds above mbar wait.

Compiles ch09's kernel (descriptor builds inside the post-wait loop)
and ch11's kernel (descriptor builds hoisted above the wait), and
times them at the same shape so the head-to-head is direct.
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

HERE      = os.path.dirname(os.path.abspath(__file__))
CH09_DIR  = os.path.normpath(os.path.join(HERE, "..", "09_cta_swizzle"))


device, ctx = init_cuda()

mod09, fns09 = compile_kernel(
    os.path.join(CH09_DIR, "kernel.cu"),
    device,
    kernels=[f"matmul_swizzle_ns{NS}_gsm{g}" for g in GSM_SWEEP])
k09 = {g: fns09[f"matmul_swizzle_ns{NS}_gsm{g}"] for g in GSM_SWEEP}

mod10, fns10 = compile_kernel(
    os.path.join(HERE, "kernel.cu"),
    device,
    kernels=[f"matmul_hoistdesc_ns{NS}_gsm{g}" for g in GSM_SWEEP])
k10 = {g: fns10[f"matmul_hoistdesc_ns{NS}_gsm{g}"] for g in GSM_SWEEP}

for kern in [*k09.values(), *k10.values()]:
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


def time_kernel(kernel, iters=200, warmup=20):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        launch(kernel, grid=grid, block=(THREADS, 1, 1),
               shared=SHARED_BYTES, args=args)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        launch(kernel, grid=grid, block=(THREADS, 1, 1),
               shared=SHARED_BYTES, args=args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3


# Correctness check: ch11 GSM=8
C.zero_()
launch(k10[8], grid=grid, block=(THREADS, 1, 1),
       shared=SHARED_BYTES, args=args)
C_ref = (A.float() @ B.float()).to(torch.bfloat16)
rel = (C.float() - C_ref.float()).abs().max().item() / C_ref.float().abs().max().item()
ok = "✓" if rel < 5e-2 else "✗"

flops = 2.0 * M * N * K
print(f"{ok}  M=N=K={M}   NS={NS}   rel err={rel:.2%}\n")
print(f"     {'GSM':>4}   {'ch09':>9}   {'ch11 (hoist)':>14}   {'speedup':>8}")
print(f"     {'─'*4}   {'─'*9}   {'─'*14}   {'─'*8}")
for g in GSM_SWEEP:
    us_09 = time_kernel(k09[g])
    us_10 = time_kernel(k10[g])
    tf_09 = flops / (us_09 * 1e-6) / 1e12
    tf_10 = flops / (us_10 * 1e-6) / 1e12
    print(f"     {g:>4}   {tf_09:>9.1f}   {tf_10:>14.1f}   {us_09/us_10:>7.3f}x")


cu(driver.cuModuleUnload(mod10))
cu(driver.cuModuleUnload(mod09))
cu(driver.cuDevicePrimaryCtxRelease(device))
