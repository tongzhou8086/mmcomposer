"""Basic `mmc.matmul` usage example.

    import mmcomposer as mmc
    c = mmc.matmul(a, b)        # c = a @ b, best-known kernel for this shape

On the first call for a new shape, mmc auto-tunes once (~100 s) and caches the
winner to disk; later calls (and future sessions) load it instantly.  The call
is asynchronous on torch's current stream, just like `torch.matmul` -- the result
is ordered before any following torch op, and a host read (`.item()`/`.cpu()`)
syncs as usual.

Two APIs:
    c    = mmc.matmul(a, b)             # one-shot, torch-like
    gemm = mmc.get_tuned_kernel(a, b)   # reusable callable for repeated calls

Inputs are bf16, row-major contiguous; M and N multiples of 256, K a multiple of
64; B200 (sm_100a).  Requires nvcc on first tune.

    python examples/matmul.py [M [N [K]]]      # default 4096x4096x4096
"""
import sys

import torch

import mmcomposer as mmc


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

    # one-shot (auto-tunes + caches on the first call for a new shape)
    c = mmc.matmul(a, b)
    ref = a.float() @ b.float()
    rel = ((c.float() - ref).norm() / ref.norm()).item()
    print(f"shape M={M} N={N} K={K}   mmc.matmul rel_err (vs fp32) = {rel:.2e}")

    # reusable callable + reuse an out= buffer (host overhead ~0, async)
    gemm = mmc.get_tuned_kernel(a, b)
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    out_t = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    # GPU kernel time (triton do_bench: warmup 1000 ms, rep 1000 ms, median)
    from triton.testing import do_bench
    flops = 2.0 * M * N * K
    g = do_bench(lambda: gemm(a, b, out, sync=False), warmup=1000, rep=1000,
                 return_mode="median")
    t = do_bench(lambda: torch.mm(a, b, out=out_t), warmup=1000, rep=1000,
                 return_mode="median")
    print(f"  mmc    {g:8.3f} ms   {flops / (g * 1e-3) / 1e12:7.0f} TFLOPS")
    print(f"  torch  {t:8.3f} ms   {flops / (t * 1e-3) / 1e12:7.0f} TFLOPS"
          f"   (mmc/torch = {g / t:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
