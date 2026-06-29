"""Benchmark mmc.matmul vs torch (cuBLAS), two ways.

  1) GPU-only kernel time -- triton.testing.do_bench(warmup=1000ms, rep=1000ms,
     median).  Pure device time; excludes Python/CPU dispatch.
  2) End-to-end wall-clock per call -- median of 100 iterations, each timed with
     torch.cuda.synchronize() on both sides, so it INCLUDES CPU launch overhead.

Comparing (2)-(1) for mmc vs torch shows whether mmc's host side adds overhead.

    python examples/benchmark.py [M [N [K]]]      # default 4096^3

Requires a B200, nvcc, and a CUDA-enabled PyTorch + triton.  If the shape hasn't
been tuned yet, the first get_tuned_kernel call auto-tunes it once.
"""
import statistics
import sys
import time

import torch
from triton.testing import do_bench

import mmcomposer as mmc


def end_to_end_ms(fn, iters=100):
    """Median full wall-clock per call (CPU dispatch + GPU + sync) over `iters`."""
    fn()
    torch.cuda.synchronize()                       # warm
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main() -> int:
    vals = [int(x) for x in sys.argv[1:4]] or [4096]
    M = vals[0]
    N = vals[1] if len(vals) > 1 else M
    K = vals[2] if len(vals) > 2 else M
    if not torch.cuda.is_available():
        print("no CUDA device -- run on a GPU node", file=sys.stderr)
        return 2

    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    c = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")    # mmc output (reused)
    ct = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")   # torch output (reused)
    flops = 2.0 * M * N * K

    def tflops(ms):
        return flops / (ms * 1e-3) / 1e12

    gemm = mmc.get_tuned_kernel(a, b)              # tunes once if cold, else loads cache

    ref = a.float() @ b.float()
    rel = ((gemm(a, b, c).float() - ref).norm() / ref.norm()).item()
    print(f"\nshape {M}x{N}x{K} bf16   rel_err(mmc vs fp32) = {rel:.2e}\n", flush=True)

    mmc_fn = lambda: gemm(a, b, c, sync=False)            # noqa: E731  (do_bench syncs)
    torch_fn = lambda: torch.mm(a, b, out=ct)            # noqa: E731

    print("== GPU-only kernel time (triton do_bench, warmup=1000ms rep=1000ms, median) ==", flush=True)
    g_mmc = do_bench(mmc_fn, warmup=1000, rep=1000, return_mode="median")
    g_tor = do_bench(torch_fn, warmup=1000, rep=1000, return_mode="median")
    print(f"  mmc    {g_mmc:8.3f} ms   {tflops(g_mmc):7.0f} TFLOPS", flush=True)
    print(f"  torch  {g_tor:8.3f} ms   {tflops(g_tor):7.0f} TFLOPS   (mmc/torch = {g_mmc / g_tor:.3f})\n",
          flush=True)

    print("== End-to-end, kernel callable (median 100 iters, reused output, incl. CPU+sync) ==", flush=True)
    e_mmc = end_to_end_ms(mmc_fn, 100)
    e_tor = end_to_end_ms(torch_fn, 100)
    print(f"  mmc gemm()         {e_mmc:8.3f} ms", flush=True)
    print(f"  torch.mm(out=)     {e_tor:8.3f} ms   (mmc/torch = {e_mmc / e_tor:.3f})\n", flush=True)

    # Full public API path: per call does validate + key + dict lookup AND allocates
    # a fresh output -- the realistic torch-like usage. torch.matmul allocates too.
    print("== End-to-end, full public API (median 100 iters, fresh output alloc each call) ==", flush=True)
    api_mmc = end_to_end_ms(lambda: mmc.matmul(a, b), 100)            # noqa: E731
    api_tor = end_to_end_ms(lambda: torch.matmul(a, b), 100)          # noqa: E731
    print(f"  mmc.matmul(a,b)    {api_mmc:8.3f} ms", flush=True)
    print(f"  torch.matmul(a,b)  {api_tor:8.3f} ms   (mmc/torch = {api_mmc / api_tor:.3f})\n", flush=True)

    print("== implied per-call host overhead (end-to-end callable - do_bench) ==", flush=True)
    print(f"  mmc    {e_mmc - g_mmc:8.3f} ms", flush=True)
    print(f"  torch  {e_tor - g_tor:8.3f} ms", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
