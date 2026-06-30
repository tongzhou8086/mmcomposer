"""Quickstart: a fused elementwise epilogue on mmc.matmul (Blackwell B200).

Pass `epilogue=` a one-in/one-out function (lambda or def) written with the
builtins in `mmcomposer.epilogue`; it is fused into the GEMM epilogue and applied
to every output element in fp32, before the bf16 store -- one kernel, no extra
activation pass.  See mmcomposer/EPILOGUE.md for the language.

    pip install -e .
    python examples/quickstart_epilogue.py                   # default FFN shape 32768x4608x768
    python examples/quickstart_epilogue.py 8192              # square 8192
    python examples/quickstart_epilogue.py 4096 4096 768      # M N K

Fusing the activation matters most on memory-bound shapes (small K, large M x N),
e.g. an FFN projection -- there torch pays a full extra GMEM round trip for the
separate activation kernel, which fusion removes.  On compute-bound squares the
GEMM dominates and the gap is small.
"""
import sys

import torch
import torch.nn.functional as F
from triton.testing import do_bench

import mmcomposer as mmc
from mmcomposer.epilogue import sigmoid

# Define the epilogue ONCE and reuse the object -> fast path (not re-traced per
# call).  A `def` works the same way:  def silu(x): return x * sigmoid(x)
silu = lambda x: x * sigmoid(x)            # noqa: E731  (SiLU / swish)

# Shape from the command line: no args -> 4096 cube; one arg N -> square N;
# three args -> M N K.
args = sys.argv[1:]
if len(args) == 0:
    M, N, K = 32768, 4608, 768          # an FFN projection (memory-bound: small K)
elif len(args) == 1:
    M = N = K = int(args[0])
elif len(args) == 3:
    M, N, K = (int(x) for x in args)
else:
    sys.exit("usage: quickstart_epilogue.py [N | M N K]")

a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

# Fused matmul + SiLU in a single kernel.  First call for a new shape auto-tunes
# the GEMM once; first call for a new epilogue compiles one fused cubin (both
# cached to disk).  Async on torch's current stream, like mmc.matmul.
c = mmc.matmul(a, b, epilogue=silu)

# Correctness vs torch: silu(a @ b).  bf16, so bf16-scale tolerances.
ref = F.silu(a @ b)
ok = torch.allclose(c, ref, rtol=2e-2, atol=1e-1)
print(f"shape M={M} N={N} K={K}   SiLU epilogue allclose vs torch = {ok}")
assert ok

# GPU kernel time (triton do_bench: warmup 1000 ms, rep 1000 ms, median):
# one fused kernel vs torch doing matmul + a separate activation kernel.
out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
flops = 2.0 * M * N * K
g = do_bench(lambda: mmc.matmul(a, b, epilogue=silu, out=out, sync=False),
             warmup=1000, rep=1000, return_mode="median")
t = do_bench(lambda: F.silu(torch.mm(a, b)),
             warmup=1000, rep=1000, return_mode="median")
print(f"mmc fused (matmul+SiLU)   {g:8.3f} ms   {flops / (g * 1e-3) / 1e12:7.0f} TFLOPS")
print(f"torch (matmul then SiLU)  {t:8.3f} ms   {flops / (t * 1e-3) / 1e12:7.0f} TFLOPS"
      f"   (mmc/torch = {g / t:.3f})")
