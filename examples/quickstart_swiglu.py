"""Quickstart: fused GEMM + SwiGLU (dual-B) on Blackwell (B200).

    python examples/quickstart_swiglu.py                  # default FFN shape 32768x4608x768
    python examples/quickstart_swiglu.py 8192             # square 8192
    python examples/quickstart_swiglu.py 32768 4608 768   # M N K

A = [M, K]; packed projection weight B = [K, N], split by column views into
B_left, B_gate = [K, N/2]; bf16, M and N multiples of 256, K a multiple of 64.
One fused launch returns:

    c = [M, N]      packed wide GEMM, [left | gate] per BN=256 tile
    d = [M, N/2]    left * silu(gate)   <- the SwiGLU activation
"""
import sys

import torch
import torch.nn.functional as F
from triton.testing import do_bench

import mmcomposer as mmc

# Shape from the command line: no args -> 4096 cube; one arg N -> square N;
# three args -> M N K.
args = sys.argv[1:]
if len(args) == 0:
    M, N, K = 32768, 4608, 768          # an FFN shape: M tokens, N=4608, K=768
elif len(args) == 1:
    M = N = K = int(args[0])
elif len(args) == 3:
    M, N, K = (int(x) for x in args)
else:
    sys.exit("usage: quickstart_swiglu.py [N | M N K]")

H = N // 2
a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
b_left = b[:, :H]   # "up" projection, column view into packed B
b_gate = b[:, H:]   # "gate" projection, column view into packed B

# One fused launch -> packed projections (c) + SwiGLU activation (d).
# Compiles once per machine, then async on torch's current stream.
c, d = mmc.matmul_swiglu_dual_b_ns6_s2(a, b_left, b_gate)

# Correctness vs torch (bf16 tolerances): d == (a @ b_left) * silu(a @ b_gate).
d_ref = (a @ b_left) * F.silu(a @ b_gate)
ok = torch.allclose(d, d_ref, rtol=2e-2, atol=1e-1)
print(f"shape M={M} N={N} K={K}   D allclose vs torch = {ok}")
assert ok

# GPU kernel time (triton do_bench: warmup 1000 ms, rep 1000 ms, median):
# the fused kernel vs torch doing the same SwiGLU eagerly (two GEMMs + gate).
flops = 2.0 * M * N * K
g = do_bench(lambda: mmc.matmul_swiglu_dual_b_ns6_s2(a, b_left, b_gate, c=c, d=d,
                                                     sync=False),
             warmup=1000, rep=1000, return_mode="median")
t = do_bench(lambda: (a @ b_left) * F.silu(a @ b_gate),
             warmup=1000, rep=1000, return_mode="median")
print(f"mmc fused    {g:8.3f} ms   {flops / (g * 1e-3) / 1e12:7.0f} TFLOPS")
print(f"torch eager  {t:8.3f} ms   {flops / (t * 1e-3) / 1e12:7.0f} TFLOPS"
      f"   (mmc/torch = {g / t:.3f})")
