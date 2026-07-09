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

# One fused launch.  Hopper currently supports inference-style no-preact output;
# Blackwell currently supports the training-style path that also stores preact.
cap_major = torch.cuda.get_device_capability()[0]
store_preact = cap_major == 10
if cap_major not in (9, 10):
    raise RuntimeError(
        f"expected Hopper or Blackwell CUDA device, got capability {torch.cuda.get_device_capability()}")

if store_preact:
    c, d = mmc.matmul_swiglu_dual_b(a, b_left, b_gate, store_preact=True)
else:
    c = None
    d = mmc.matmul_swiglu_dual_b(a, b_left, b_gate)

# Correctness vs torch (bf16 tolerances).
left_ref = a @ b_left
gate_ref = a @ b_gate
d_ref = left_ref * F.silu(gate_ref)
ok_d = torch.allclose(d, d_ref, rtol=2e-2, atol=1e-1)
if c is not None:
    c_ref = torch.cat([left_ref, gate_ref], dim=1)      # == x @ W1.t()
    ok_c = torch.allclose(c, c_ref, rtol=2e-2, atol=1e-1)
    c_status = str(ok_c)
else:
    ok_c = True
    c_status = "skipped"
ragged = f"  (M ragged: M % 128 = {M % 128})" if M % 128 else ""
mode = "store_preact" if store_preact else "no_preact"
print(f"shape M={M} N={N} K={K}   mode={mode}   C allclose = {c_status}   "
      f"D allclose = {ok_d}{ragged}")
assert ok_c and ok_d

# GPU kernel time (triton do_bench: warmup 1000 ms, rep 1000 ms, median):
# the fused kernel vs torch doing the same SwiGLU eagerly (two GEMMs + gate).
flops = 2.0 * M * N * K
if store_preact:
    g = do_bench(lambda: mmc.matmul_swiglu_dual_b(a, b_left, b_gate,
                                                 store_preact=True, preact=c, out=d,
                                                 sync=False),
                 warmup=1000, rep=1000, return_mode="median")
else:
    g = do_bench(lambda: mmc.matmul_swiglu_dual_b(a, b_left, b_gate, out=d,
                                                 sync=False),
                 warmup=1000, rep=1000, return_mode="median")
t = do_bench(lambda: (a @ b_left) * F.silu(a @ b_gate),
             warmup=1000, rep=1000, return_mode="median")
print(f"mmc fused    {g:8.3f} ms   {flops / (g * 1e-3) / 1e12:7.0f} TFLOPS")
print(f"torch eager  {t:8.3f} ms   {flops / (t * 1e-3) / 1e12:7.0f} TFLOPS"
      f"   (mmc/torch = {g / t:.3f})")
