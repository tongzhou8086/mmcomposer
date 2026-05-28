"""Runnable companion for Chapter 08 — 2-CTA cluster MMA + NS sweep.

Compiles ch07's single-CTA kernel and ch08's 2-CTA cluster kernels at
NS = 2, 3, 4, 5, 6, 7.  Runs each on three problem sizes and prints
the head-to-head TFLOPS table.
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


BM, BN, BK   = 128, 256, 64
CTA_GROUP    = 2
BN_LOCAL     = BN // CTA_GROUP             # 128
ELEM_BYTES   = 2
THREADS      = 128
A_SLOT_BYTES = BM       * BK * ELEM_BYTES   # 16 KB
B_SLOT_BYTES = BN_LOCAL * BK * ELEM_BYTES   # 16 KB
SLOT_BYTES   = A_SLOT_BYTES + B_SLOT_BYTES  # 32 KB / slot / CTA
NS_SWEEP     = [2, 3, 4, 5, 6, 7]
HERE         = os.path.dirname(os.path.abspath(__file__))
CH07_DIR     = os.path.normpath(os.path.join(HERE, "..", "07_coalesced_epilogue"))


device, ctx = init_cuda()

# Compile ch07 (single-CTA, NS=2) for the baseline column.
mod07, fns07 = compile_kernel(os.path.join(CH07_DIR, "kernel.cu"),
                              device, kernels=["matmul_coalesced_epilogue"])
k07 = fns07["matmul_coalesced_epilogue"]
ch07_slot = BM * BK * ELEM_BYTES + BN * BK * ELEM_BYTES                 # 48 KB
ch07_shared = 2 * ch07_slot + 1024
cu(driver.cuFuncSetAttribute(
    k07,
    driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    ch07_shared))

# Compile ch08 with all NS variants in one go.
mod08, fns08 = compile_kernel(
    os.path.join(HERE, "kernel.cu"),
    device,
    kernels=[f"matmul_cluster_ns{ns}" for ns in NS_SWEEP])
k08 = {ns: fns08[f"matmul_cluster_ns{ns}"] for ns in NS_SWEEP}

# The dynamic SMEM has two uses across the kernel's lifetime:
#   * during the K-loop:  NS slots × SLOT_BYTES per slot (A + B halves)
#   * during the epilogue: a [BM][BN+8] BF16 staging buffer (~66 KB)
# In ch07's single-CTA kernel the K-loop's per-CTA budget (96 KB at
# NS=2) was always bigger than the 66 KB staging, so we never had to
# think about it.  In ch08 the per-CTA SLOT_BYTES shrinks (B is split
# across the cluster), so at NS=2 the K-loop only needs 64 KB — less
# than the staging.  Take the max so the allocation covers both phases.
BN_PAD             = 256 + 8
EPI_STAGING_BYTES  = BM * BN_PAD * ELEM_BYTES        # 67584 B

def shared_for(ns):
    return max(ns * SLOT_BYTES, EPI_STAGING_BYTES) + 1024

for ns, kern in k08.items():
    cu(driver.cuFuncSetAttribute(
        kern,
        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_for(ns)))


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

    args = [arg_a, arg_b, arg_c, arg_M, arg_N, arg_K]
    return A, B, C, args


def time_kernel(kernel, grid, args, shared_bytes, iters=200, warmup=20):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        launch(kernel, grid=grid, block=(THREADS, 1, 1),
               shared=shared_bytes, args=args)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        launch(kernel, grid=grid, block=(THREADS, 1, 1),
               shared=shared_bytes, args=args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3


for (M, N, K) in [(2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192)]:
    assert M % (CTA_GROUP * BM) == 0, "M must be a multiple of 2*BM = 256 for the 2-CTA cluster"
    assert N % BN == 0
    assert K % BK == 0

    A, B, C, args = setup(M, N, K)
    flops = 2.0 * M * N * K

    # Correctness check on ch08 NS=2.
    grid_07 = ((M // BM) * (N // BN), 1, 1)
    grid_08 = ((M // (CTA_GROUP * BM)) * (N // BN) * CTA_GROUP, 1, 1)
    C.zero_()
    launch(k08[2], grid=grid_08, block=(THREADS, 1, 1),
           shared=shared_for(2), args=args)
    C_ref = (A.float() @ B.float()).to(torch.bfloat16)
    rel = (C.float() - C_ref.float()).abs().max().item() / C_ref.float().abs().max().item()
    ok = "✓" if rel < 5e-2 else "✗"

    # Timing
    us_07 = time_kernel(k07, grid_07, args, ch07_shared)
    tf_07 = flops / (us_07 * 1e-6) / 1e12

    print(f"{ok}  M=N=K={M:<5}  rel err={rel:.2%}")
    print(f"     {'config':<28} {'SMEM':>8}    us/call   TFLOPS")
    print(f"     {'─'*28} {'─'*8}    {'─'*7}   {'─'*6}")
    print(f"     {'ch07 NS=2 (single-CTA)':<28} {ch07_shared//1024:>5} KB   "
          f"{us_07:7.1f}   {tf_07:6.1f}")
    for ns in NS_SWEEP:
        shared = shared_for(ns)
        us = time_kernel(k08[ns], grid_08, args, shared)
        tf = flops / (us * 1e-6) / 1e12
        label = f"ch08 NS={ns} (2-CTA cluster)"
        print(f"     {label:<28} {shared//1024:>5} KB   {us:7.1f}   {tf:6.1f}")
    print()


cu(driver.cuModuleUnload(mod08))
cu(driver.cuModuleUnload(mod07))
cu(driver.cuDevicePrimaryCtxRelease(device))
