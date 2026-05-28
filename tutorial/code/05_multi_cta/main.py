"""Runnable companion for Chapter 05 — multi-CTA grid mapping.

Same warp-specialized multi-stage kernel as chapter 04, now tiled
across the whole M × N output by a grid of (M/BM) × (N/BN) CTAs.
Times the kernel at two problem sizes and compares against PyTorch
(which calls cuBLAS).
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


# ── Compile-time shapes (match kernel.cu) ──────────────────────────────────
BM, BN, BK   = 128, 256, 64
NS           = 2
ELEM_BYTES   = 2
THREADS      = 128
SLOT_BYTES   = BM * BK * ELEM_BYTES + BN * BK * ELEM_BYTES
SHARED_BYTES = NS * SLOT_BYTES + 1024
HERE         = os.path.dirname(os.path.abspath(__file__))


# ── 1. Init + compile ──────────────────────────────────────────────────────
device, ctx = init_cuda()
module, fns = compile_kernel(os.path.join(HERE, "kernel.cu"),
                             device, kernels=["matmul_multi_cta"])
kernel = fns["matmul_multi_cta"]

cu(driver.cuFuncSetAttribute(
    kernel,
    driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    SHARED_BYTES))


def run_and_time(M, N, K, iters=200, warmup=20):
    assert M % BM == 0 and N % BN == 0 and K % BK == 0, \
        "M, N, K must be multiples of (BM, BN, BK) = (128, 256, 64)"

    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    B_t = B.t().contiguous()

    A_tmap = encode_tensor_map(
        dtype=TMA_BFLOAT16, rank=2, gptr=A.data_ptr(),
        global_dim=[K, M], global_strides=[K * ELEM_BYTES],
        box_dim=[BK, BM], element_strides=[1, 1], swizzle=TMA_SWIZZLE_128B)
    B_tmap = encode_tensor_map(
        dtype=TMA_BFLOAT16, rank=2, gptr=B_t.data_ptr(),
        global_dim=[K, N], global_strides=[K * ELEM_BYTES],
        box_dim=[BK, BN], element_strides=[1, 1], swizzle=TMA_SWIZZLE_128B)

    arg_a = (ctypes.c_byte * 128).from_buffer_copy(A_tmap.tobytes())
    arg_b = (ctypes.c_byte * 128).from_buffer_copy(B_tmap.tobytes())
    arg_c = ctypes.c_void_p(C.data_ptr())
    arg_M, arg_N, arg_K = ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)

    grid_m, grid_n = M // BM, N // BN
    grid = (grid_m * grid_n, 1, 1)
    args = [arg_a, arg_b, arg_c, arg_M, arg_N, arg_K]

    # Correctness check
    launch(kernel, grid=grid, block=(THREADS, 1, 1),
           shared=SHARED_BYTES, args=args)

    C_ref = (A.float() @ B.float()).to(torch.bfloat16)
    rel = (C.float() - C_ref.float()).abs().max().item() / C_ref.float().abs().max().item()

    # Time this kernel
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        launch(kernel, grid=grid, block=(THREADS,1,1),
               shared=SHARED_BYTES, args=args)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        launch(kernel, grid=grid, block=(THREADS,1,1),
               shared=SHARED_BYTES, args=args)
    end.record()
    torch.cuda.synchronize()
    us_ours = start.elapsed_time(end) / iters * 1e3

    # Time PyTorch (cuBLAS) for the same problem
    for _ in range(warmup):
        _ = A @ B
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        C_pt = A @ B
    end.record()
    torch.cuda.synchronize()
    us_pt = start.elapsed_time(end) / iters * 1e3

    flops = 2.0 * M * N * K
    tflops_ours = flops / (us_ours * 1e-6) / 1e12
    tflops_pt   = flops / (us_pt   * 1e-6) / 1e12

    return {
        "grid": (grid_m, grid_n),
        "rel":  rel,
        "us_ours": us_ours, "tflops_ours": tflops_ours,
        "us_pt":   us_pt,   "tflops_pt":   tflops_pt,
    }


# ── 2. Run two problem sizes ───────────────────────────────────────────────
for (M, N, K) in [(2048, 2048, 2048), (4096, 4096, 4096)]:
    r = run_and_time(M, N, K)
    ok = "✓" if r["rel"] < 5e-2 else "✗"
    print(f"{ok}  M=N=K={M:>4}   grid={r['grid'][0]}×{r['grid'][1]}={r['grid'][0]*r['grid'][1]} CTAs   rel err={r['rel']:.2%}")
    print(f"     ours:     {r['us_ours']:7.1f} us/call   {r['tflops_ours']:6.1f} TFLOPS")
    print(f"     PyTorch:  {r['us_pt']:7.1f} us/call   {r['tflops_pt']:6.1f} TFLOPS")
    print(f"     ours / PyTorch = {r['tflops_ours']/r['tflops_pt']:.1%}")
    print()


# ── 3. Cleanup ─────────────────────────────────────────────────────────────
cu(driver.cuModuleUnload(module))
cu(driver.cuDevicePrimaryCtxRelease(device))
