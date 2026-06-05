"""Bench ch12 (hand-CUDA) + Triton (canonical) + cuBLAS at one FFN-like shape.

The square-shape sweep (main.py) tests M=N=K ∈ {2K..12K}.  This script
times the same three contenders at a non-square shape close to the
FFN forward matmul: M=11264, K=3584, N=28672.

  M = 11264  — closest cluster-MMA-divisible (44 pairs) to 11136
  K = 3584   — model hidden dim
  N = 28672  — 2 × FFN hidden (the packed [left|gate] width)

All timings use `triton.testing.do_bench` with L2-flush between reps,
so cuBLAS doesn't get warm-cache reuse it wouldn't see in a real
workload.  ch12 autotunes (NS, GSM) at this shape first.
"""

from __future__ import annotations

import os
import sys
import time
import ctypes

import torch
import triton.testing as tt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cuda_utils import (
    cu, init_cuda, compile_kernel, launch,
    encode_tensor_map, TMA_BFLOAT16, TMA_SWIZZLE_128B,
)
from cuda.bindings import driver

HERE     = os.path.dirname(os.path.abspath(__file__))
CH12_DIR = os.path.normpath(os.path.join(HERE, "..", "12_autotune"))

BM, BN, BK    = 128, 256, 64
CTA_GROUP     = 2
BN_LOCAL      = BN // CTA_GROUP
WARP_SIZE     = 32
ELEM_BYTES    = 2

A_SLOT_BYTES  = BM       * BK * ELEM_BYTES
B_SLOT_BYTES  = BN_LOCAL * BK * ELEM_BYTES
SLOT_BYTES    = A_SLOT_BYTES + B_SLOT_BYTES
BN_PAD        = BN + 8
C_SH_BYTES_12 = BM * BN_PAD * ELEM_BYTES

M, K, N = 22272, 3584, 28672   # M = 11136 * 2, divisible by 256


def main():
    device, ctx = init_cuda()
    flops = 2.0 * M * N * K

    NS_SWEEP_12  = [3, 4, 5, 6, 7]
    GSM_SWEEP_12 = [1, 4, 8, 16]
    def ch12_kname(ns, gsm): return f"matmul_tune_ns{ns}_gsm{gsm}_nw8_ldx8"

    print(f"shape  M={M}  K={K}  N={N}  bf16")
    print(f"FLOPs  = {flops / 1e12:.2f} TFLOP")
    print("Compiling ch12 sweep ... ", end="", flush=True)
    t0 = time.time()
    mod12, fns12 = compile_kernel(
        os.path.join(CH12_DIR, "kernel.cu"), device,
        kernels=[ch12_kname(ns, gsm) for ns in NS_SWEEP_12 for gsm in GSM_SWEEP_12])
    k12 = {(ns, gsm): fns12[ch12_kname(ns, gsm)]
           for ns in NS_SWEEP_12 for gsm in GSM_SWEEP_12}
    def shared_for_ch12(ns):
        return max(ns * SLOT_BYTES, C_SH_BYTES_12) + 1024
    for (ns, _), kern in k12.items():
        cu(driver.cuFuncSetAttribute(
            kern,
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shared_for_ch12(ns)))
    print(f"done in {time.time() - t0:.1f}s")

    sys.path.insert(0, HERE)
    from kernel import triton_matmul

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
    args12 = [arg_a, arg_b, arg_c,
              ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)]
    grid_m_clusters = M // (CTA_GROUP * BM)
    grid_n          = N // BN
    grid12 = (grid_m_clusters * grid_n * CTA_GROUP, 1, 1)

    print("\nAutotuning ch12 at this shape (do_bench)...")
    best_ms, best_cfg = float("inf"), None
    for (ns, gsm), kern in k12.items():
        if gsm > grid_m_clusters:
            continue
        ms = tt.do_bench(
            lambda K_=kern, ns_=ns: launch(
                K_, grid=grid12, block=(8 * WARP_SIZE, 1, 1),
                shared=shared_for_ch12(ns_), args=args12, sync=False),
            warmup=50, rep=200, return_mode="median")
        if ms < best_ms:
            best_ms, best_cfg = ms, (ns, gsm)
    ns_w, gsm_w = best_cfg
    kern12 = k12[best_cfg]
    print(f"  best ch12 cfg: NS={ns_w}, GSM={gsm_w}  ({best_ms:.3f} ms)")

    # Correctness for the winner.
    C.zero_()
    launch(kern12, grid=grid12, block=(8 * WARP_SIZE, 1, 1),
           shared=shared_for_ch12(ns_w), args=args12)
    torch.cuda.synchronize()
    C_ref = (A.float() @ B.float()).to(torch.bfloat16)
    rel12 = (C.float() - C_ref.float()).abs().max().item() / C_ref.float().abs().max().item()
    print(f"  ch12 correctness  rel_max={rel12:.3e}  {'OK' if rel12 <= 5e-2 else 'FAIL'}")

    # Triton: first call triggers autotune (cached afterwards).
    triton_matmul(A, B, C)
    torch.cuda.synchronize()
    rel_tri = (C.float() - C_ref.float()).abs().max().item() / C_ref.float().abs().max().item()
    print(f"  triton correctness rel_max={rel_tri:.3e}  {'OK' if rel_tri <= 5e-2 else 'FAIL'}")

    # ── do_bench all three ──
    print("\n  variant            ms     TFLOPS    /cublas")
    cases = [
        ("cublas",    lambda: torch.matmul(A, B)),
        ("triton-mm", lambda: triton_matmul(A, B, C)),
        ("ch12",      lambda: launch(
            kern12, grid=grid12, block=(8 * WARP_SIZE, 1, 1),
            shared=shared_for_ch12(ns_w), args=args12, sync=False)),
    ]
    times_ms = {}
    for name, fn in cases:
        ms = tt.do_bench(fn, warmup=50, rep=500, return_mode="median")
        times_ms[name] = ms
        tf = flops / (ms * 1e-3) / 1e12
        print(f"  {name:<14s} {ms:>8.3f}   {tf:>7.1f}")
    print("\n  variant            ms     TFLOPS    /cublas")
    for name in ("cublas", "triton-mm", "ch12"):
        ms = times_ms[name]
        tf = flops / (ms * 1e-3) / 1e12
        ratio = times_ms["cublas"] / ms
        print(f"  {name:<14s} {ms:>8.3f}   {tf:>7.1f}    {ratio:>5.3f}x")

    cu(driver.cuModuleUnload(mod12))
    cu(driver.cuDevicePrimaryCtxRelease(device))


if __name__ == "__main__":
    main()
