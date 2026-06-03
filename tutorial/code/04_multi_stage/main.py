"""Runnable companion for Chapter 04 — multi-stage buffering.

Single-CTA matmul with NUM_STAGES = 2 (ring buffer in SMEM).  TMA warp
and MMA warp run independent K-loops; both stay busy concurrently.

Verifies against PyTorch and times the kernel, also compiles and times
chapter 03's single-stage kernel for a head-to-head TFLOPS comparison.

Problem default:
    C[M, N] = A[M, K] @ B[K, N],   M = 128, N = 256, K = 4096
"""

import os
import sys
import ctypes
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cuda_utils import (
    cu, init_cuda, compile_kernel, launch, time_kernel_us,
    encode_tensor_map, TMA_BFLOAT16, TMA_SWIZZLE_128B,
)

from cuda.bindings import driver


# ── Problem shape ──────────────────────────────────────────────────────────
BM, BN, BK = 128, 256, 64
NS         = 2
M, N, K    = BM, BN, 4096

ELEM_BYTES = 2
THREADS    = 128
SLOT_BYTES = BM * BK * ELEM_BYTES + BN * BK * ELEM_BYTES   # 48 KB per slot
HERE       = os.path.dirname(os.path.abspath(__file__))
CH03_DIR   = os.path.normpath(os.path.join(HERE, "..", "03_outer_k_loop"))


# ── 1. Init CUDA + compile both kernels ────────────────────────────────────
device, ctx = init_cuda()

mod04, fns04 = compile_kernel(os.path.join(HERE, "kernel.cu"),
                              device, kernels=["matmul_multi_stage"])
k04 = fns04["matmul_multi_stage"]

mod03, fns03 = compile_kernel(os.path.join(CH03_DIR, "kernel.cu"),
                              device, kernels=["matmul_k_loop"])
k03 = fns03["matmul_k_loop"]

SHARED_04 = NS * SLOT_BYTES + 1024
SHARED_03 = SLOT_BYTES + 1024
cu(driver.cuFuncSetAttribute(
    k04, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    SHARED_04))
cu(driver.cuFuncSetAttribute(
    k03, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    SHARED_03))


# ── 2. Inputs + reference ──────────────────────────────────────────────────
torch.manual_seed(0)
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

C_ref = (A.float() @ B.float()).to(torch.bfloat16)

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
arg_K = ctypes.c_int(K)


# ── 3. Correctness check on ch04 ───────────────────────────────────────────
launch(k04, grid=(1, 1, 1), block=(THREADS, 1, 1),
       shared=SHARED_04, args=[arg_a, arg_b, arg_c, arg_K])

C_f, ref_f = C.float(), C_ref.float()
max_abs_err = (C_f - ref_f).abs().max().item()
max_ref     = ref_f.abs().max().item()
rel         = max_abs_err / max(max_ref, 1e-8)
print(f"M = {M}, N = {N}, K = {K}   ({K // BK} K-iters, NS = {NS})")
print(f"  max |C - C_ref|      = {max_abs_err:.4f}")
print(f"  |C_ref|_max          = {max_ref:.4f}")
print(f"  max relative error   = {rel:.4%}")

# BF16 accum-in-FP32 noise at K=4096 with Gaussian inputs sits around
# 1-3% relative; 5% is generous.
if torch.allclose(C_f, ref_f, rtol=5e-2, atol=5e-2):
    print("✓ matches PyTorch reference\n")
else:
    print("✗ MISMATCH\n")
    print(f"  C[0, :8]:     {C[0, :8].cpu().tolist()}")
    print(f"  C_ref[0, :8]: {C_ref[0, :8].cpu().tolist()}")
    sys.exit(1)


# ── 4. Timing: ch03 vs ch04 ────────────────────────────────────────────────
# Use the shared do_bench-based timer.  It flushes L2 between samples and
# picks an adaptive iter count to fit a 200 ms rep window — so fast and
# slow kernels both get well-sampled measurements under cold-cache
# conditions (the same harness ch12's autotuner uses).
args = [arg_a, arg_b, arg_c, arg_K]
us_03 = time_kernel_us(lambda: launch(
    k03, grid=(1,1,1), block=(THREADS,1,1),
    shared=SHARED_03, args=args, sync=False))
us_04 = time_kernel_us(lambda: launch(
    k04, grid=(1,1,1), block=(THREADS,1,1),
    shared=SHARED_04, args=args, sync=False))

# FLOPs: 2 * M * N * K (one MAC = 2 FLOPs).
flops = 2.0 * M * N * K
tflops_03 = flops / (us_03 * 1e-6) / 1e12
tflops_04 = flops / (us_04 * 1e-6) / 1e12

print("Timing  (single CTA, median via triton.testing.do_bench):")
print(f"  ch03 (NS = 1, no overlap):  {us_03:7.2f} us/call   {tflops_03:6.1f} TFLOPS")
print(f"  ch04 (NS = {NS}, overlap):       {us_04:7.2f} us/call   {tflops_04:6.1f} TFLOPS")
print(f"  speedup ch04 / ch03:        {us_03 / us_04:.2f}x")


# ── 5. Cleanup ─────────────────────────────────────────────────────────────
cu(driver.cuModuleUnload(mod04))
cu(driver.cuModuleUnload(mod03))
cu(driver.cuDevicePrimaryCtxRelease(device))
