# ─────────────────────────────────────────────────────────────────────
# Tier 1 — baseline: multi-stage SMEM ring + synchronous MMA (no warp
# specialization), single-CTA, generalized variable-warp epilogue, with
# CTA-swizzle (GROUP_SIZE_M) as a tunable.
#
# This launcher is a fragment: mmcomposer prepends the self-contained
# runtime preamble (CUDA driver plumbing) above it, so the file you
# downloaded runs on its own.  Just:  python <this file>.py
# Requires: torch, numpy, cuda-python (`cuda.bindings`), and `nvcc` on PATH.
# ─────────────────────────────────────────────────────────────────────

import os
import ctypes

import torch
from cuda.bindings import driver


# ── User-tunable constants (mirror kernel.cu — mmcomposer keeps in sync) ──
BM, BN, BK   = 128, 256, 64
NS           = 2
GROUP_SIZE_M = 8
NUM_WARPS    = 4

ELEM_BYTES   = 2
THREADS      = NUM_WARPS * 32
SLOT_BYTES   = BM * BK * ELEM_BYTES + BN * BK * ELEM_BYTES
EPI_BYTES    = BM * (BN + 8) * ELEM_BYTES
# K-loop ring (NS slots) and epilogue staging share the same dynamic
# SMEM region but are time-disjoint — alloc the max of the two.
SHARED_BYTES = max(NS * SLOT_BYTES, EPI_BYTES) + 1024
HERE         = os.path.dirname(os.path.abspath(__file__))


device, ctx = init_cuda()

mod, fns = compile_kernel(os.path.join(HERE, "kernel.cu"),
                          device, kernels=["matmul_dbuf"])
kernel = fns["matmul_dbuf"]

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
    # Store-side descriptor: BM×BN tile of C (M,N) row-major.  SWIZZLE_NONE
    # matches the dense SMEM staging the epilogue writes.
    C_tmap = encode_tensor_map(
        dtype=TMA_BFLOAT16, rank=2, gptr=C.data_ptr(),
        global_dim=[N, M], global_strides=[N * ELEM_BYTES],
        box_dim=[BN, BM], element_strides=[1, 1], swizzle=TMA_SWIZZLE_NONE)

    arg_a = (ctypes.c_byte * 128).from_buffer_copy(A_tmap.tobytes())
    arg_b = (ctypes.c_byte * 128).from_buffer_copy(B_tmap.tobytes())
    arg_c_tmap = (ctypes.c_byte * 128).from_buffer_copy(C_tmap.tobytes())
    arg_c = ctypes.c_void_p(C.data_ptr())
    args = [arg_a, arg_b, arg_c_tmap, arg_c,
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)]

    grid = (M // BM * N // BN, 1, 1)
    return A, B, C, grid, args


for (M, N, K) in [(2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192)]:
    A, B, C, grid, args = setup(M, N, K)
    flops = 2.0 * M * N * K

    C.zero_()
    launch(kernel, grid=grid, block=(THREADS, 1, 1),
           shared=SHARED_BYTES, args=args)
    C_ref = (A.float() @ B.float()).to(torch.bfloat16)
    rel = (C.float() - C_ref.float()).abs().max().item() / C_ref.float().abs().max().item()
    ok = "OK" if rel < 5e-2 else "FAIL"

    us = time_kernel_us(lambda: launch(
        kernel, grid=grid, block=(THREADS, 1, 1),
        shared=SHARED_BYTES, args=args, sync=False))
    tf = flops / (us * 1e-6) / 1e12

    print(f"{ok}  M=N=K={M:>5}   grid={M//BM}x{N//BN}={(M//BM)*(N//BN)} CTAs   rel err={rel:.2%}")
    print(f"     BM={BM} BN={BN} BK={BK} NS={NS} GSM={GROUP_SIZE_M} NW={NUM_WARPS}   "
          f"{us:7.1f} us/call   {tf:6.1f} TFLOPS")
    print()


cu(driver.cuModuleUnload(mod))
cu(driver.cuDevicePrimaryCtxRelease(device))
