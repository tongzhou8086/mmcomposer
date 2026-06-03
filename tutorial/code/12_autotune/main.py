"""Runnable companion for Chapter 12 — autotuning capstone.

Compiles all 160 kernel variants (NS × GSM × NUM_WARPS × LD_X) and
autotunes per problem shape.  Sweeps shapes M = N = K ∈ {2048, 3072,
…, 16384}.

Uses **median** of per-batch timings rather than arithmetic mean,
which is more stable against transient noise (scheduler hiccups,
contention, etc.) — important when the autotuner's ranking decisions
hinge on differences of just a few percent.
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

BM, BN, BK    = 128, 256, 64
CTA_GROUP     = 2
BN_LOCAL      = BN // CTA_GROUP
WARP_SIZE     = 32
ELEM_BYTES    = 2
NS_SWEEP      = [3, 4, 5, 6, 7]
GSM_SWEEP     = [1, 4, 8, 16]
NW_SWEEP      = [8]      # NW=4 dropped — NW=8 wins more consistently and
LDX_SWEEP     = [8, 16, 32, 64]
SHAPES        = list(range(2048, 12288 + 1, 1024))   # 11 shapes

A_SLOT_BYTES  = BM       * BK * ELEM_BYTES
B_SLOT_BYTES  = BN_LOCAL * BK * ELEM_BYTES
SLOT_BYTES    = A_SLOT_BYTES + B_SLOT_BYTES
BN_PAD        = BN + 8
EPI_STAGING   = BM * BN_PAD * ELEM_BYTES

HERE = os.path.dirname(os.path.abspath(__file__))


def shared_for(ns):
    return max(ns * SLOT_BYTES, EPI_STAGING) + 1024


# ── 1. Init + compile all 160 variants ─────────────────────────────────────
device, ctx = init_cuda()

def kname(ns, gsm, nw, ldx):
    return f"matmul_tune_ns{ns}_gsm{gsm}_nw{nw}_ldx{ldx}"

CONFIGS = [(ns, gsm, nw, ldx)
           for ns  in NS_SWEEP
           for gsm in GSM_SWEEP
           for nw  in NW_SWEEP
           for ldx in LDX_SWEEP]
NUM_CONFIGS = len(CONFIGS)

print(f"Compiling {NUM_CONFIGS} variants (cold compile takes a few minutes)... ",
      end="", flush=True)
t0 = time.time()
module, fns = compile_kernel(
    os.path.join(HERE, "kernel.cu"),
    device,
    kernels=[kname(*cfg) for cfg in CONFIGS])
print(f"done in {time.time() - t0:.1f}s")

kernels = {cfg: fns[kname(*cfg)] for cfg in CONFIGS}

# Per-NS dynamic SMEM cap.
for cfg, kern in kernels.items():
    ns = cfg[0]
    cu(driver.cuFuncSetAttribute(
        kern,
        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_for(ns)))


# ── 2. Timing — shared do_bench wrapper from cuda_utils ──────────────────
#
# Every chapter in the tutorial uses cuda_utils.time_kernel_us for kernel
# timing (a thin wrapper around triton.testing.do_bench).  This chapter is
# no exception — both the autotuner's per-config ranking and the final
# reported numbers go through the same harness.
def time_my_kernel(kern, threads, sh, args, grid, warmup_ms=20, rep_ms=200):
    return time_kernel_us(
        lambda: launch(kern, grid=grid, block=(threads, 1, 1),
                       shared=sh, args=args, sync=False),
        warmup_ms=warmup_ms, rep_ms=rep_ms,
    )


# ── 3. The Autotuner ───────────────────────────────────────────────────────
class Autotuner:
    def __init__(self, kernels):
        self.kernels = kernels
        self.cache   = {}      # (M, N, K) -> cfg

    def pick(self, M, N, K, args, grid,
             warmup_ms=20, rep_ms=200):
        key = (M, N, K)
        if key in self.cache:
            return self.kernels[self.cache[key]], self.cache[key]

        # Skip configs that the kernel's GSM-clamp would collapse to a
        # smaller-GSM variant we're already timing.  Without this prune
        # we get pure-noise ranking between equivalent variants at
        # small shapes (e.g. GSM=16 ≡ GSM=8 at M=2048).
        grid_m_clusters = M // (CTA_GROUP * BM)

        best_us, best_cfg = float("inf"), None
        for cfg, kern in self.kernels.items():
            ns, gsm, nw, _ldx = cfg
            if gsm > grid_m_clusters:
                continue
            us = time_my_kernel(kern, nw * WARP_SIZE, shared_for(ns),
                                args, grid,
                                warmup_ms=warmup_ms, rep_ms=rep_ms)
            if us < best_us:
                best_us, best_cfg = us, cfg

        self.cache[key] = best_cfg
        return self.kernels[best_cfg], best_cfg


# ── 4. Per-shape setup ─────────────────────────────────────────────────────
def setup(M, N, K):
    assert M % (CTA_GROUP * BM) == 0
    assert N % BN == 0
    assert K % BK == 0
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
    arg_M = ctypes.c_int(M); arg_N = ctypes.c_int(N); arg_K = ctypes.c_int(K)
    args = [arg_a, arg_b, arg_c, arg_M, arg_N, arg_K]
    grid = ((M // (CTA_GROUP * BM)) * (N // BN) * CTA_GROUP, 1, 1)
    return A, B, C, args, grid


def time_pytorch_us(A, B, warmup_ms=20, rep_ms=200):
    """PyTorch (cuBLAS) baseline via the same do_bench harness."""
    return time_kernel_us(lambda: A @ B,
                          warmup_ms=warmup_ms, rep_ms=rep_ms)


# ── 5. Sweep all shapes ────────────────────────────────────────────────────
tuner = Autotuner(kernels)

print(f"\nAutotuning {len(SHAPES)} shapes vs. PyTorch (cuBLAS):\n")
print(f"  {'shape':<8}  {'best (NS, GSM, NW, LDX)':<26}  "
      f"{'ours':>6}  {'cuBLAS':>7}  {'ratio*':>6}")
print(f"  {'─'*8}  {'─'*26}  {'─'*6}  {'─'*7}  {'─'*6}")

results = []
for sz in SHAPES:
    M = N = K = sz
    A, B, C, args, grid = setup(M, N, K)

    kern, cfg = tuner.pick(M, N, K, args, grid)
    ns, gsm, nw, ldx = cfg

    # Correctness check
    C.zero_()
    launch(kern, grid=grid, block=(nw * WARP_SIZE, 1, 1),
           shared=shared_for(ns), args=args)
    C_ref = (A.float() @ B.float()).to(torch.bfloat16)
    rel = (C.float() - C_ref.float()).abs().max().item() / C_ref.float().abs().max().item()

    # Final timings — same do_bench harness as the autotuner uses for
    # ranking, just with a bit more rep budget so the reported number
    # is more stable.
    us_ours = time_my_kernel(kern, nw * WARP_SIZE, shared_for(ns),
                             args, grid, warmup_ms=50, rep_ms=500)
    us_pt   = time_pytorch_us(A, B, warmup_ms=50, rep_ms=500)
    flops   = 2.0 * M * N * K
    tf_ours = flops / (us_ours * 1e-6) / 1e12
    tf_pt   = flops / (us_pt   * 1e-6) / 1e12

    ratio   = tf_ours / tf_pt
    cfg_str = f"({ns}, {gsm}, {nw}, {ldx})"
    flag = "✓" if rel < 1e-1 else "✗"
    print(f"  {sz}^3   {cfg_str:<26}  "
          f"{tf_ours:>6.0f}  {tf_pt:>7.1f}  {flag} {ratio:>4.0%}")

    results.append((sz, ns, gsm, nw, ldx, rel, us_ours, tf_ours, us_pt, tf_pt))

print("  * ratio = ours / cuBLAS")
print()


cu(driver.cuModuleUnload(module))
cu(driver.cuDevicePrimaryCtxRelease(device))
