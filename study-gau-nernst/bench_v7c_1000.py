import torch
from pathlib import Path
import torch.utils.cpp_extension as ext
import triton.testing

HERE = Path(__file__).resolve().parent
ext.load("gn_v7", sources=[str(HERE / "matmul_v7.cu"), str(HERE / "binding.cpp")],
         extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"],
         extra_ldflags=["-lcuda"], is_python_module=False, verbose=False)
ops = torch.ops.gn_matmul

M, N, K = 32768, 4608, 768
scale = K ** -0.5
A = torch.randn(M, K, device="cuda").mul(scale).bfloat16()
B = torch.randn(N, K, device="cuda").mul(scale).bfloat16().T
ref = torch.mm(A.float(), B.float()).bfloat16()
flops = 2 * M * N * K

for name, fn in [("cuBLAS", torch.mm), ("v7c", ops.matmul_v7c)]:
    out = fn(A, B)
    rel = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
    ms = triton.testing.do_bench(lambda: fn(A, B), warmup=1000, rep=1000)
    tflops = flops / (ms * 1e-3) / 1e12
    print(f"{name}: {ms * 1000:.1f} us {tflops:.0f} TFLOPS rel_err={rel:.2e}")
