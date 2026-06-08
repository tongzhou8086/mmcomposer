# ─────────────────────────────────────────────────────────────────────
# Tier 2 — multi-stage SMEM ring + warp-specialized async MMA (dedicated
# TMA + MMA warps), single-CTA, generalized variable-warp epilogue, with
# CTA-swizzle (GROUP_SIZE_M) as a tunable.
#
# This launcher is a fragment: mmcomposer prepends the self-contained
# runtime preamble above it, so the downloaded file runs on its own:
#     python <this file>.py
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
TMA_STORE    = 0
PERSISTENT   = 0    # 1 → launch grid = #SMs; each CTA walks many tiles

ELEM_BYTES   = 2
THREADS      = NUM_WARPS * 32
SLOT_BYTES   = BM * BK * ELEM_BYTES + BN * BK * ELEM_BYTES
EPI_BYTES    = BM * (BN if TMA_STORE else BN + 8) * ELEM_BYTES
SHARED_BYTES = max(NS * SLOT_BYTES, EPI_BYTES) + 1024
HERE         = os.path.dirname(os.path.abspath(__file__))


device, ctx = init_cuda()

# Persistent grid launches one CTA per SM; the kernel's tile loop then
# walks a strided run of output tiles.  Query the SM count up front.
NUM_SMS = cu(driver.cuDeviceGetAttribute(
    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device))

mod, fns = compile_kernel(os.path.join(HERE, "kernel.cu"),
                          device, kernels=["matmul_coalesced_epilogue"])
kernel = fns["matmul_coalesced_epilogue"]

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

    num_tiles = (M // BM) * (N // BN)
    grid = (NUM_SMS if PERSISTENT else num_tiles, 1, 1)
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

    sched = f"persistent grid={grid[0]} CTAs" if PERSISTENT else f"grid={M//BM}x{N//BN}={(M//BM)*(N//BN)} CTAs"
    print(f"{ok}  M=N=K={M:>5}   {sched}   rel err={rel:.2%}")
    print(f"     BM={BM} BN={BN} BK={BK} NS={NS} GSM={GROUP_SIZE_M} NW={NUM_WARPS} PERSISTENT={PERSISTENT}   "
          f"{us:7.1f} us/call   {tf:6.1f} TFLOPS")
    print()


cu(driver.cuModuleUnload(mod))
cu(driver.cuDevicePrimaryCtxRelease(device))
