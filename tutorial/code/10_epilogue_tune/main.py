"""Runnable companion for Chapter 10 — tuning the epilogue.

Sweeps NUM_WARPS ∈ {4, 8} × LD_X ∈ {8, 16, 32, 64} = 8 configs at
M = N = K = 8192, with NS = 5, GSM = 8 fixed (best from ch09).
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
NS            = 5
GSM           = 8
M, N, K       = 8192, 8192, 8192

NW_SWEEP      = [4, 8]
LDX_SWEEP     = [8, 16, 32, 64]

A_SLOT_BYTES  = BM       * BK * ELEM_BYTES
B_SLOT_BYTES  = BN_LOCAL * BK * ELEM_BYTES
SLOT_BYTES    = A_SLOT_BYTES + B_SLOT_BYTES
BN_PAD        = BN + 8
EPI_STAGING   = BM * BN_PAD * ELEM_BYTES
SHARED_BYTES  = max(NS * SLOT_BYTES, EPI_STAGING) + 1024

HERE = os.path.dirname(os.path.abspath(__file__))


device, ctx = init_cuda()

module, fns = compile_kernel(
    os.path.join(HERE, "kernel.cu"),
    device,
    kernels=[f"matmul_epi_nw{nw}_ldx{ldx}"
             for nw in NW_SWEEP for ldx in LDX_SWEEP])
kernels = {(nw, ldx): fns[f"matmul_epi_nw{nw}_ldx{ldx}"]
           for nw in NW_SWEEP for ldx in LDX_SWEEP}

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


def time_kernel(kernel, threads, iters=200, warmup=20):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    block = (threads, 1, 1)
    for _ in range(warmup):
        launch(kernel, grid=grid, block=block,
               shared=SHARED_BYTES, args=args)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        launch(kernel, grid=grid, block=block,
               shared=SHARED_BYTES, args=args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3


# Correctness check: pick (8 warps, x16) as a non-trivial config.
C.zero_()
launch(kernels[(8, 16)], grid=grid, block=(8 * 32, 1, 1),
       shared=SHARED_BYTES, args=args)
C_ref = (A.float() @ B.float()).to(torch.bfloat16)
rel = (C.float() - C_ref.float()).abs().max().item() / C_ref.float().abs().max().item()
ok = "✓" if rel < 5e-2 else "✗"

flops = 2.0 * M * N * K
print(f"{ok}  M=N=K={M}   NS={NS}, GSM={GSM}   rel err={rel:.2%}")
print(f"     (4-warp & 8-warp × LD_X ∈ {{8, 16, 32, 64}}, head-to-head TFLOPS)\n")

# Header
hdr = "     LD_X →   " + "   ".join(f"{ldx:>7}" for ldx in LDX_SWEEP)
print(hdr)
print("     ─────────" + "   ".join([f"{'─'*7}"] * len(LDX_SWEEP)))
for nw in NW_SWEEP:
    threads = nw * 32
    row = [f"{nw} warps  "]
    for ldx in LDX_SWEEP:
        us = time_kernel(kernels[(nw, ldx)], threads=threads)
        tf = flops / (us * 1e-6) / 1e12
        row.append(f"{tf:>7.1f}")
    print("     " + "   ".join(row))


cu(driver.cuModuleUnload(module))
cu(driver.cuDevicePrimaryCtxRelease(device))
