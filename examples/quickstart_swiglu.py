"""Quickstart: fused GEMM + SwiGLU (dual-B) on Blackwell (B200).

    python examples/quickstart_swiglu.py                  # default FFN shape 30000x4608x768
    python examples/quickstart_swiglu.py 8192             # square 8192
    python examples/quickstart_swiglu.py 30000 4608 768   # M N K

A = [M, K]; packed projection weight B = [K, N], split by column views into
B_left, B_gate = [K, N/2]; bf16, M arbitrary (ragged token counts welcome),
N a multiple of 256, K a multiple of 64.  A ragged M -- the usual case, since M
is the token count -- is handled by a ceil-div grid + TMA out-of-bounds clipping.
One fused launch returns:

    c = [M, N]      wide GEMM [ left | gate ] = x @ [B_left | B_gate]  (the preact)
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
    M, N, K = 30000, 4608, 768          # ragged token count: M=30000 is NOT a
                                        # multiple of 256 (30000 % 256 = 48)
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

# One fused launch -> combined projection (c) + SwiGLU activation (d).
# Compiles once per machine, then async on torch's current stream.
c, d = mmc.matmul_swiglu_dual_b_ns6_s2(a, b_left, b_gate)

# Correctness vs torch (bf16 tolerances).
# C is the combined preactivation x @ [B_left | B_gate] -- exactly what you'd save
# for a backward pass -- so check BOTH outputs, not just D:
c_ref = torch.cat([a @ b_left, a @ b_gate], dim=1)      # == x @ W1.t()
d_ref = (a @ b_left) * F.silu(a @ b_gate)
ok_c = torch.allclose(c, c_ref, rtol=2e-2, atol=1e-1)
ok_d = torch.allclose(d, d_ref, rtol=2e-2, atol=1e-1)
ragged = f"  (M ragged: M % 256 = {M % 256})" if M % 256 else ""
print(f"shape M={M} N={N} K={K}   C allclose = {ok_c}   D allclose = {ok_d}{ragged}")
assert ok_c and ok_d

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
