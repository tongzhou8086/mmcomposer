"""Runnable companion for Chapter 13 — TMA store epilogue.

Single-config launcher (NS=4, GSM=8, NW=8, LDX=8) — no autotune.
Head-to-head against ch12's best autotuned config at the same shape,
across the same 11-shape sweep, so the perf delta from the structural
change is direct.
"""

import os
import sys
import time
import ctypes

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cuda_utils import (
    cu, init_cuda, compile_kernel, launch, time_kernel_us,
    encode_tensor_map, TMA_BFLOAT16, TMA_SWIZZLE_128B, TMA_SWIZZLE_NONE,
)

from cuda.bindings import driver


BM, BN, BK    = 128, 256, 64
CTA_GROUP     = 2
BN_LOCAL      = BN // CTA_GROUP
WARP_SIZE     = 32
ELEM_BYTES    = 2

# Fixed for ch13.
NS_13         = 4
GSM_13        = 8
NW_13         = 8

A_SLOT_BYTES  = BM       * BK * ELEM_BYTES
B_SLOT_BYTES  = BN_LOCAL * BK * ELEM_BYTES
SLOT_BYTES    = A_SLOT_BYTES + B_SLOT_BYTES
# Ch13 C_sh is tightly-packed (no +8 padding) so TMA can read it.
C_SH_BYTES_13 = BM * BN * ELEM_BYTES
SHARED_13     = max(NS_13 * SLOT_BYTES, C_SH_BYTES_13) + 1024

SHAPES = list(range(2048, 12288 + 1, 1024))

HERE     = os.path.dirname(os.path.abspath(__file__))
CH12_DIR = os.path.normpath(os.path.join(HERE, "..", "12_autotune"))

device, ctx = init_cuda()


# ── Compile ch13 (single launcher) ──────────────────────────────────
print(f"Compiling ch13 kernel ... ", end="", flush=True)
t0 = time.time()
mod13, fns13 = compile_kernel(
    os.path.join(HERE, "kernel.cu"),
    device,
    kernels=[f"matmul_tmast_ns{NS_13}_gsm{GSM_13}_nw{NW_13}_ldx8"])
k13 = fns13[f"matmul_tmast_ns{NS_13}_gsm{GSM_13}_nw{NW_13}_ldx8"]
cu(driver.cuFuncSetAttribute(
    k13,
    driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    SHARED_13))
print(f"done in {time.time() - t0:.1f}s")


# ── Compile ch12 (NS × GSM cross at NW=8, LDX=8) ───────────────────
NS_SWEEP_12  = [3, 4, 5, 6, 7]
GSM_SWEEP_12 = [1, 4, 8, 16]
def ch12_kname(ns, gsm): return f"matmul_tune_ns{ns}_gsm{gsm}_nw8_ldx8"

print(f"Compiling ch12 sweep ({len(NS_SWEEP_12)*len(GSM_SWEEP_12)} variants) ... ",
      end="", flush=True)
t0 = time.time()
mod12, fns12 = compile_kernel(
    os.path.join(CH12_DIR, "kernel.cu"),
    device,
    kernels=[ch12_kname(ns, gsm) for ns in NS_SWEEP_12 for gsm in GSM_SWEEP_12])
k12 = {(ns, gsm): fns12[ch12_kname(ns, gsm)]
       for ns in NS_SWEEP_12 for gsm in GSM_SWEEP_12}

BN_PAD_12 = BN + 8
C_SH_BYTES_12 = BM * BN_PAD_12 * ELEM_BYTES
def shared_for_ch12(ns):
    return max(ns * SLOT_BYTES, C_SH_BYTES_12) + 1024
for (ns, gsm), kern in k12.items():
    cu(driver.cuFuncSetAttribute(
        kern,
        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_for_ch12(ns)))
print(f"done in {time.time() - t0:.1f}s")


# ── Per-shape setup ─────────────────────────────────────────────────
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
    # Ch13 needs a C tensormap.  box = (BN, BM), SWIZZLE_NONE so the
    # SMEM source can be tightly-packed BM × BN bf16.
    C_tmap = encode_tensor_map(
        dtype=TMA_BFLOAT16, rank=2, gptr=C.data_ptr(),
        global_dim=[N, M], global_strides=[N * ELEM_BYTES],
        box_dim=[BN, BM], element_strides=[1, 1], swizzle=TMA_SWIZZLE_NONE)
    return A, B, C, A_tmap, B_tmap, C_tmap


def args_ch12(A_tmap, B_tmap, C, M, N, K):
    arg_a = (ctypes.c_byte * 128).from_buffer_copy(A_tmap.tobytes())
    arg_b = (ctypes.c_byte * 128).from_buffer_copy(B_tmap.tobytes())
    arg_c = ctypes.c_void_p(C.data_ptr())
    return [arg_a, arg_b, arg_c,
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)]


def args_ch13(A_tmap, B_tmap, C_tmap, M, N, K):
    arg_a = (ctypes.c_byte * 128).from_buffer_copy(A_tmap.tobytes())
    arg_b = (ctypes.c_byte * 128).from_buffer_copy(B_tmap.tobytes())
    arg_c = (ctypes.c_byte * 128).from_buffer_copy(C_tmap.tobytes())
    return [arg_a, arg_b, arg_c,
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)]


def grid_for(M, N):
    grid_m_clusters = M // (CTA_GROUP * BM)
    grid_n          = N // BN
    return (grid_m_clusters * grid_n * CTA_GROUP, 1, 1)


def autotune_ch12(M, N, K, args12, g):
    grid_m_clusters = M // (CTA_GROUP * BM)
    best_us, best_cfg = float("inf"), None
    for (ns, gsm), kern in k12.items():
        if gsm > grid_m_clusters:
            continue
        us = time_kernel_us(lambda K_=kern, ns_=ns: launch(
            K_, grid=g, block=(NW_13 * WARP_SIZE, 1, 1),
            shared=shared_for_ch12(ns_), args=args12, sync=False),
            warmup_ms=20, rep_ms=200)
        if us < best_us:
            best_us, best_cfg = us, (ns, gsm)
    return best_us, best_cfg


# ── Run the sweep ───────────────────────────────────────────────────
print()
print(f"  {'shape':<8}  {'ch12 (NS,GSM)':>14}  {'ch12 TF':>8}  "
      f"{'ch13 TF':>8}  {'delta':>7}  {'ratio':>6}  ok?")
print(f"  {'─'*8}  {'─'*14}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*6}  ──")

for sz in SHAPES:
    M = N = K = sz
    A, B, C, A_tmap, B_tmap, C_tmap = setup(M, N, K)
    flops = 2.0 * M * N * K
    g = grid_for(M, N)

    a12 = args_ch12(A_tmap, B_tmap, C, M, N, K)
    a13 = args_ch13(A_tmap, B_tmap, C_tmap, M, N, K)

    # Correctness: run ch13, compare vs PyTorch.
    C.zero_()
    launch(k13, grid=g, block=(NW_13 * WARP_SIZE, 1, 1),
           shared=SHARED_13, args=a13)
    C_ref = (A.float() @ B.float()).to(torch.bfloat16)
    rel = (C.float() - C_ref.float()).abs().max().item() / C_ref.float().abs().max().item()
    ok = "✓" if rel < 5e-2 else "✗"

    # Time both.
    us12, cfg12 = autotune_ch12(M, N, K, a12, g)
    us13 = time_kernel_us(lambda: launch(
        k13, grid=g, block=(NW_13 * WARP_SIZE, 1, 1),
        shared=SHARED_13, args=a13, sync=False),
        warmup_ms=50, rep_ms=500)

    tf12 = flops / (us12 * 1e-6) / 1e12
    tf13 = flops / (us13 * 1e-6) / 1e12
    delta = tf13 - tf12
    ratio = tf13 / tf12
    cfg_str = f"({cfg12[0]},{cfg12[1]})"
    print(f"  {M}^3   {cfg_str:>14}  {tf12:>8.1f}  {tf13:>8.1f}  "
          f"{delta:>+7.1f}  {ratio:>5.0%}  {ok}")

print()
cu(driver.cuModuleUnload(mod13))
cu(driver.cuModuleUnload(mod12))
cu(driver.cuDevicePrimaryCtxRelease(device))
