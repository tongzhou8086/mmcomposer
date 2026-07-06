"""Quickstart: mmcomposer as a drop-in matmul on Blackwell (B200).

    python examples/quickstart.py             # default 4096 x 4096 x 4096
    python examples/quickstart.py 8192        # square 8192
    python examples/quickstart.py 4096 4096 768   # M N K

A = [M, K], B = [K, N]; bf16, M arbitrary, N a multiple of 8, K a multiple of 64.
"""
import sys
import torch
import mmcomposer as mmc
from triton.testing import do_bench


# Shape from the command line: no args -> 4096 cube; one arg N -> square N;
# three args -> M N K.
args = sys.argv[1:]
if len(args) == 0:
    M = N = K = 4096
elif len(args) == 1:
    M = N = K = int(args[0])
elif len(args) == 3:
    M, N, K = (int(x) for x in args)
else:
    sys.exit("usage: quickstart.py [N | M N K]")

a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
b = torch.randn(K, N, dtype=a.dtype, device=a.device)

# The first call for a new shape auto-tunes once (cached to disk, reused in
# future sessions); later calls load instantly.  Async on torch's current
# stream, just like torch.matmul.
c = mmc.matmul(a, b)

# Correctness vs torch.  These are bf16, so use bf16-scale tolerances
# (rtol ~2e-2, atol ~1e-1) -- the fp32 default 1e-5 would spuriously fail.
ref = a @ b
ok = torch.allclose(c, ref, rtol=2e-2, atol=1e-1)
print(f"shape M={M} N={N} K={K}   allclose vs torch = {ok}")
assert ok

# GPU kernel time (triton do_bench: warmup 1000 ms, rep 1000 ms, median).
flops = 2.0 * M * N * K
g = do_bench(lambda: mmc.matmul(a, b), warmup=1000, rep=1000, return_mode="median")
t = do_bench(lambda: a @ b, warmup=1000, rep=1000, return_mode="median")
print(f"mmc    {g:8.3f} ms   {flops / (g * 1e-3) / 1e12:7.0f} TFLOPS")
print(f"torch  {t:8.3f} ms   {flops / (t * 1e-3) / 1e12:7.0f} TFLOPS"
      f"   (mmc/torch = {g / t:.3f})")
