"""Runnable companion for Chapter 07 — coalesced SMEM-staged epilogue.

Compiles both this kernel and chapter 06's (direct-writeback) kernel,
runs both on three shapes, and reports the speedup.
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


BM, BN, BK   = 128, 256, 64
NS           = 2
ELEM_BYTES   = 2
THREADS      = 128
SLOT_BYTES   = BM * BK * ELEM_BYTES + BN * BK * ELEM_BYTES
SHARED_BYTES = NS * SLOT_BYTES + 1024
HERE         = os.path.dirname(os.path.abspath(__file__))
CH06_DIR     = os.path.normpath(os.path.join(HERE, "..", "06_k_major_b"))


device, ctx = init_cuda()

mod07, fns07 = compile_kernel(os.path.join(HERE, "kernel.cu"),
                              device, kernels=["matmul_coalesced_epilogue"])
k07 = fns07["matmul_coalesced_epilogue"]

mod06, fns06 = compile_kernel(os.path.join(CH06_DIR, "kernel.cu"),
                              device, kernels=["matmul_k_major_b"])
k06 = fns06["matmul_k_major_b"]

for kernel in (k06, k07):
    cu(driver.cuFuncSetAttribute(
        kernel,
        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        SHARED_BYTES))


def setup(M, N, K):
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

    grid_m, grid_n = M // BM, N // BN
    grid = (grid_m * grid_n, 1, 1)
    args = [arg_a, arg_b, arg_c, arg_M, arg_N, arg_K]
    return A, B, C, grid, args


def time_kernel(kernel, grid, args):
    """Median per-call time (µs) via the shared do_bench wrapper."""
    return time_kernel_us(lambda: launch(
        kernel, grid=grid, block=(THREADS, 1, 1),
        shared=SHARED_BYTES, args=args, sync=False))


for (M, N, K) in [(2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192)]:
    A, B, C, grid, args = setup(M, N, K)

    # Correctness — run ch07, check against PyTorch
    C.zero_()
    launch(k07, grid=grid, block=(THREADS,1,1),
           shared=SHARED_BYTES, args=args)
    C_ref = (A.float() @ B.float()).to(torch.bfloat16)
    rel = (C.float() - C_ref.float()).abs().max().item() / C_ref.float().abs().max().item()
    ok = "✓" if rel < 5e-2 else "✗"

    us_06 = time_kernel(k06, grid, args)
    us_07 = time_kernel(k07, grid, args)
    flops = 2.0 * M * N * K
    tflops_06 = flops / (us_06 * 1e-6) / 1e12
    tflops_07 = flops / (us_07 * 1e-6) / 1e12

    print(f"{ok}  M=N=K={M:>4}   grid={M//BM}×{N//BN}={(M//BM)*(N//BN)} CTAs   rel err={rel:.2%}")
    print(f"     ch06 (direct):     {us_06:7.1f} us/call   {tflops_06:6.1f} TFLOPS")
    print(f"     ch07 (coalesced):  {us_07:7.1f} us/call   {tflops_07:6.1f} TFLOPS")
    print(f"     speedup ch07/ch06: {us_06 / us_07:.2f}x")
    print()


cu(driver.cuModuleUnload(mod07))
cu(driver.cuModuleUnload(mod06))
cu(driver.cuDevicePrimaryCtxRelease(device))
