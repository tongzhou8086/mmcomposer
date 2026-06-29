"""Fused GEMM + SwiGLU (dual-B) usage example.

The fused kernel takes the activation `a` and the TWO projection weight matrices
of a SwiGLU FFN as separate tensors, and computes both projections plus the
SwiGLU gate in one launch:

    a       [M, K]      input activations            (bf16, row-major)
    b_left  [K, N/2]    "up"   projection weights    (bf16, row-major)
    b_gate  [K, N/2]    "gate" projection weights    (bf16, row-major)

      c, d = mmc.matmul_swiglu_dual_b_ns6_s2(a, b_left, b_gate)

    c       [M, N]      packed wide GEMM, [left128 | gate128] per BN=256 tile
    d       [M, N/2]    left * silu(gate)   <- the SwiGLU activation you want

`d` is the FFN activation; `c` is the raw packed projection output the kernel
also stores (handy for a backward pass).  Like `mmc.matmul`, the call is async
on torch's current stream; pass `c=`/`d=` to reuse buffers or `sync=True` to block.

Shape rules: M and N multiples of 256, K a multiple of 64 (N = 2 * b_left's
second dim).  Requires a B200 + nvcc; the kernel compiles once per machine.

    python examples/swiglu_dual_b.py [M [N [K]]]      # default 4096x4096x4096
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
    b_left = torch.randn(K, N // 2, dtype=torch.bfloat16, device="cuda")   # up
    b_gate = torch.randn(K, N // 2, dtype=torch.bfloat16, device="cuda")   # gate

    # one fused launch -> packed projections (c) + SwiGLU activation (d)
    c, d = mmc.matmul_swiglu_dual_b_ns6_s2(a, b_left, b_gate)
    print(f"shape M={M} N={N} K={K}   c{tuple(c.shape)}  d{tuple(d.shape)}")

    # reference: d = (a@b_left) * silu(a@b_gate)
    left = (a.float() @ b_left.float())
    gate = (a.float() @ b_gate.float())
    d_ref = left * torch.nn.functional.silu(gate)
    rel = ((d.float() - d_ref).norm() / d_ref.norm()).item()
    print(f"SwiGLU activation d rel_err (vs fp32 torch) = {rel:.2e}")

    # GPU kernel time (triton do_bench: warmup 1000 ms, rep 1000 ms, median).
    # Compare against the equivalent packed wide GEMM in cuBLAS (same FLOPs,
    # no SwiGLU fusion): torch.mm(a, b_packed) with b_packed = [b_left | b_gate].
    from triton.testing import do_bench
    flops = 2.0 * M * N * K                      # two M x (N/2) x K GEMMs
    b_packed = torch.empty(K, N, dtype=torch.bfloat16, device="cuda")
    b_packed.view(K, N // 256, 256)[:, :, :128] = b_left.view(K, N // 256, 128)
    b_packed.view(K, N // 256, 256)[:, :, 128:] = b_gate.view(K, N // 256, 128)

    g = do_bench(lambda: mmc.matmul_swiglu_dual_b_ns6_s2(a, b_left, b_gate, c=c, d=d,
                                                         sync=False),
                 warmup=1000, rep=1000, return_mode="median")
    t = do_bench(lambda: torch.mm(a, b_packed, out=c), warmup=1000, rep=1000,
                 return_mode="median")
    print(f"  fused swiglu  {g:8.3f} ms   {flops / (g * 1e-3) / 1e12:7.0f} TFLOPS")
    print(f"  cuBLAS GEMM   {t:8.3f} ms   {flops / (t * 1e-3) / 1e12:7.0f} TFLOPS"
          f"   (the GEMM alone, no gate)")
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
