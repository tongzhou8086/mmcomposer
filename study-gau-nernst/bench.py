"""Bench gau-nernst v7c inside mmcomposer, with cuBLAS in the same script for a
fair %-of-cuBLAS ratio.  Mirrors their data layout (A=(M,K), B=(N,K).T).

Run on a B200 node:
    srun ... python study-gau-nernst/bench.py [MxNxK ...]
Default shapes: 4096 (sanity) + 32768x4608x768 (our low-K shape).
"""
import sys, torch
from pathlib import Path
import torch.utils.cpp_extension as ext
import triton.testing

HERE = Path(__file__).resolve().parent
ext.load("gn_v7", sources=[str(HERE / "matmul_v7.cu"), str(HERE / "binding.cpp")],
         extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"],
         extra_ldflags=["-lcuda"], is_python_module=False, verbose=False)
ops = torch.ops.gn_matmul

def bench(fn, A, B):
    # Triton do_bench: warmup / rep are in MILLISECONDS; returns median latency (ms).
    return triton.testing.do_bench(lambda: fn(A, B), warmup=1000, rep=1000)

def run(M, N, K):
    scale = K ** -0.5
    A = torch.randn(M, K, device='cuda').mul(scale).bfloat16()
    B = torch.randn(N, K, device='cuda').mul(scale).bfloat16().T   # (K,N) view, K-major storage
    ref = torch.mm(A.float(), B.float()).bfloat16()
    flops = 2 * M * N * K
    print(f"\n=== {M}x{N}x{K} ===")
    res = {}
    for name, fn in [("cuBLAS", torch.mm), ("v7c", ops.matmul_v7c)]:
        out = fn(A, B)
        rel = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
        ms = bench(fn, A, B)
        res[name] = flops / (ms * 1e-3) / 1e12
        print(f"  {name:7}: {ms*1000:8.1f} us   {res[name]:7.0f} TFLOPS   rel_err={rel:.2e}")
    print(f"  v7c = {res['v7c']/res['cuBLAS']*100:.0f}% of cuBLAS")

shapes = sys.argv[1:] or ["4096", "32768x4608x768"]
for s in shapes:
    M, N, K = (int(x) for x in s.split("x")) if "x" in s else (int(s),) * 3
    run(M, N, K)
