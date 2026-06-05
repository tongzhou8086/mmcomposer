"""Runnable companion for Chapter 13 — Triton baseline.

3-way head-to-head at every shape in the standard 11-shape sweep:

  ch12   our hand-written CUDA, autotuned over (NS, GSM).
  Triton this chapter's kernel, autotuned over (BLOCK_M, BLOCK_N, GSM, NS, NW).
  cuBLAS PyTorch's `A @ B`.

The point of the chapter is to put ours next to Triton — a canonical
B200 kernel that uses the same primitives we taught — and see how
close the ladder lands.  cuBLAS is the absolute reference but Triton
is the more interesting one: it's what an idiomatic person would
write, and a fair "did we learn the right things" target.
"""

import os
import sys
import time
import ctypes

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cuda_utils import (
    cu, init_cuda, compile_kernel, launch, time_kernel_us,
    encode_tensor_map, TMA_BFLOAT16, TMA_SWIZZLE_128B,
)

from cuda.bindings import driver

# Local imports (after init_cuda is set up — Triton lazily picks up
# the current CUDA context).
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

SHAPES = list(range(2048, 12288 + 1, 1024))


def main():
    device, ctx = init_cuda()

    # ── Compile ch12's NS × GSM sweep (NW=8, LDX=8) ─────────────────
    NS_SWEEP_12  = [3, 4, 5, 6, 7]
    GSM_SWEEP_12 = [1, 4, 8, 16]
    def ch12_kname(ns, gsm): return f"matmul_tune_ns{ns}_gsm{gsm}_nw8_ldx8"

    print("Compiling ch12 sweep ... ", end="", flush=True)
    t0 = time.time()
    mod12, fns12 = compile_kernel(
        os.path.join(CH12_DIR, "kernel.cu"),
        device,
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

    # ── Lazy-import the Triton kernel (Triton picks up current ctx) ─
    sys.path.insert(0, HERE)
    from kernel import triton_matmul

    # ── 3-way sweep ─────────────────────────────────────────────────
    print()
    print(f"  {'shape':<8}  {'ch12 cfg':>10}  {'ch12 TF':>8}  "
          f"{'Triton TF':>10}  {'cuBLAS TF':>10}  "
          f"{'ch12/cuBLAS':>11}  {'Triton/cuBLAS':>13}  ok?")
    print("  " + "─"*98)

    for sz in SHAPES:
        M = N = K = sz
        torch.manual_seed(0)
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        flops = 2.0 * M * N * K

        # ── ch12 setup ──
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

        # ── ch12 autotune at this shape ──
        best_us, best_cfg = float("inf"), None
        for (ns, gsm), kern in k12.items():
            if gsm > grid_m_clusters:
                continue
            us = time_kernel_us(lambda K_=kern, ns_=ns: launch(
                K_, grid=grid12, block=(8 * WARP_SIZE, 1, 1),
                shared=shared_for_ch12(ns_), args=args12, sync=False),
                warmup_ms=20, rep_ms=200)
            if us < best_us:
                best_us, best_cfg = us, (ns, gsm)
        kern12 = k12[best_cfg]
        ns_w, gsm_w = best_cfg

        # ── Final timed runs ──
        us12 = time_kernel_us(lambda: launch(
            kern12, grid=grid12, block=(8 * WARP_SIZE, 1, 1),
            shared=shared_for_ch12(ns_w), args=args12, sync=False),
            warmup_ms=50, rep_ms=500)

        # Triton: first call triggers autotune; subsequent calls cached.
        triton_matmul(A, B, C)
        torch.cuda.synchronize()
        us_tri = time_kernel_us(lambda: triton_matmul(A, B, C),
                                warmup_ms=50, rep_ms=500)

        us_pt  = time_kernel_us(lambda: torch.matmul(A, B),
                                warmup_ms=50, rep_ms=500)

        # ── Correctness ──
        C.zero_()
        launch(kern12, grid=grid12, block=(8 * WARP_SIZE, 1, 1),
               shared=shared_for_ch12(ns_w), args=args12)
        C_ref = (A.float() @ B.float()).to(torch.bfloat16)
        rel12 = (C.float() - C_ref.float()).abs().max().item() / max(C_ref.float().abs().max().item(), 1e-9)
        C.zero_()
        triton_matmul(A, B, C)
        torch.cuda.synchronize()
        rel_tri = (C.float() - C_ref.float()).abs().max().item() / max(C_ref.float().abs().max().item(), 1e-9)
        ok = "✓" if rel12 < 5e-2 and rel_tri < 5e-2 else "✗"

        tf12 = flops / (us12   * 1e-6) / 1e12
        tft  = flops / (us_tri * 1e-6) / 1e12
        tfb  = flops / (us_pt  * 1e-6) / 1e12

        print(f"  {M}^3   ({ns_w:>2d},{gsm_w:>2d})    {tf12:>8.1f}    "
              f"{tft:>10.1f}    {tfb:>10.1f}      "
              f"{tf12/tfb:>8.0%}        {tft/tfb:>8.0%}      {ok}")

    cu(driver.cuModuleUnload(mod12))
    cu(driver.cuDevicePrimaryCtxRelease(device))


if __name__ == "__main__":
    main()
