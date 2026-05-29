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
    cu, init_cuda, compile_kernel, launch,
    encode_tensor_map, TMA_BFLOAT16, TMA_SWIZZLE_128B,
)

from cuda.bindings import driver

# Optional: import the b42_gsm production kernel from mymatmul for a
# third comparison column.  Diagnostic / debugging use — not part of
# the chapter's user-facing story.  If pycuda or the mymatmul tree
# isn't available the b42 column is just skipped silently.
try:
    sys.path.insert(0, "/data/home/tong/projects/mymatmul")
    from mymatmul.gpu.blackwell.matmul_b42_gsm import matmul_b42_gsm  # noqa
    HAS_B42 = True
except Exception as _e:
    HAS_B42 = False
    print(f"(b42 unavailable: {_e})")


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


# ── 2. L2 invalidation + median-of-batches timing ──────────────────────────
#
# B200 L2 cache is 132 MB.  Allocate a 256-MB scratch buffer once and
# write through it before each timed batch to evict whatever the
# previous batch left in L2.  Without this, configs that happen to
# benefit from cross-launch L2 reuse get an inflated reading, biasing
# the tuner toward configs that look good in steady-state but are
# slower for the more realistic "first call" scenario.  The
# invalidation runs once per *batch*, not per launch — within a batch
# the L2 warms up normally, which is what kernels see in practice.
L2_FLUSH_BYTES = 256 * 1024 * 1024
_l2_scratch    = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda")

def invalidate_l2():
    _l2_scratch.zero_()

def time_median(kern, threads, sh, args, grid, n_batches=5, iters=5):
    """Return median per-call time (µs).

    Records n_batches independent timed segments, each averaging
    `iters` launches.  Returns the median of those n_batches numbers.
    Using median (not mean) discards single-batch noise — important
    when ranking variants whose true gaps are a few percent.  L2 is
    flushed before each batch so warmth from the previous batch
    doesn't bias the timing.
    """
    block = (threads, 1, 1)
    # warmup
    for _ in range(2):
        launch(kern, grid=grid, block=block, shared=sh, args=args, sync=False)
    torch.cuda.synchronize()
    times_us = []
    for _ in range(n_batches):
        invalidate_l2()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            # sync=False: queue launches back-to-back so we measure
            # device throughput, not host launch-sync round trips.
            # At small shapes (~5 µs kernels), per-launch sync would
            # double or triple the apparent runtime.
            launch(kern, grid=grid, block=block, shared=sh, args=args, sync=False)
        end.record()
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) / iters * 1e3)
    times_us.sort()
    return times_us[len(times_us) // 2]


# ── 3. The Autotuner ───────────────────────────────────────────────────────
class Autotuner:
    def __init__(self, kernels):
        self.kernels = kernels
        self.cache   = {}     # (M, N, K) -> cfg

    def pick(self, M, N, K, args, grid,
             tune_batches=7, tune_iters=50):
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
            us = time_median(kern, nw * WARP_SIZE, shared_for(ns),
                             args, grid,
                             n_batches=tune_batches, iters=tune_iters)
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


def time_pytorch_median(A, B, n_batches=11, iters=20):
    """PyTorch (cuBLAS) baseline, same median + L2-flush methodology."""
    for _ in range(2):
        _ = A @ B
    torch.cuda.synchronize()
    times_us = []
    for _ in range(n_batches):
        invalidate_l2()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _ = A @ B
        end.record()
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) / iters * 1e3)
    times_us.sort()
    return times_us[len(times_us) // 2]


def time_b42_median(A, B, n_batches=11, iters=20):
    """mymatmul/b42_gsm production kernel, same methodology.

    First call autotunes b42's own (BN, BK, NS, GSM) sweep over its
    5 base configs × 4 GSMs.  Subsequent calls dispatch to the cached
    winner.  We then time those dispatches the same way we time
    cuBLAS / our autotuned kernel.
    """
    # warmup + autotune b42's internal cache for this shape
    for _ in range(2):
        _ = matmul_b42_gsm(A, B)
    torch.cuda.synchronize()
    times_us = []
    for _ in range(n_batches):
        invalidate_l2()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _ = matmul_b42_gsm(A, B)
        end.record()
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) / iters * 1e3)
    times_us.sort()
    return times_us[len(times_us) // 2]


# ── 5. Sweep all shapes ────────────────────────────────────────────────────
tuner = Autotuner(kernels)

b42_col = "  b42  " if HAS_B42 else ""
b42_sep = "  ─────" if HAS_B42 else ""

print(f"\nAutotuning {len(SHAPES)} shapes vs. PyTorch (cuBLAS)"
      + (" and mymatmul/b42_gsm" if HAS_B42 else "") + ":\n")
print(f"  {'shape':<8}  {'best (NS, GSM, NW, LDX)':<25}  "
      f"{'ours':>7}  {'cuBLAS':>7}{b42_col}  {'ratio*':>6}")
print(f"  {'─'*8}  {'─'*25}  {'─'*7}  {'─'*7}{b42_sep}  {'─'*6}")

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

    # Final, well-warmed median timings with L2 flush.
    us_ours = time_median(kern, nw * WARP_SIZE, shared_for(ns),
                          args, grid, n_batches=11, iters=50)
    us_pt   = time_pytorch_median(A, B)
    flops   = 2.0 * M * N * K
    tf_ours = flops / (us_ours * 1e-6) / 1e12
    tf_pt   = flops / (us_pt   * 1e-6) / 1e12

    us_b42 = None
    tf_b42 = None
    b42_field = ""
    if HAS_B42:
        try:
            us_b42 = time_b42_median(A, B)
            tf_b42 = flops / (us_b42 * 1e-6) / 1e12
            b42_field = f"  {tf_b42:>5.0f}"
        except Exception as e:
            b42_field = f"  err: {type(e).__name__}"

    ratio   = tf_ours / tf_pt
    cfg_str = f"({ns}, {gsm}, {nw}, {ldx})"
    flag = "✓" if rel < 1e-1 else "✗"
    print(f"  {sz}^3   {cfg_str:<25}  "
          f"{tf_ours:>7.1f}  {tf_pt:>7.1f}{b42_field}  {flag} {ratio:>4.0%}")

    results.append((sz, ns, gsm, nw, ldx, rel, us_ours, tf_ours, us_pt, tf_pt, us_b42, tf_b42))

print("  * ratio = ours / cuBLAS")
print()


cu(driver.cuModuleUnload(module))
cu(driver.cuDevicePrimaryCtxRelease(device))
