// Minimal torch binding for gau-nernst v7 (just v7a/b/c), so we can build only
// matmul_v7.cu without the v0-v6 sources.  C-ABI matches matmul_v7.cu.
#include <torch/library.h>
#include <ATen/ATen.h>
#include <cuda_bf16.h>

typedef void MatmulFn(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K);

MatmulFn matmul_v7a;
MatmulFn matmul_v7b;
MatmulFn matmul_v7c;

template <MatmulFn fn>
at::Tensor matmul(const at::Tensor& A, const at::Tensor& B) {
  int M = A.size(0), K = A.size(1), N = B.size(1);
  auto C = at::empty({M, N}, A.options());
  fn(reinterpret_cast<nv_bfloat16*>(A.data_ptr()),
     reinterpret_cast<nv_bfloat16*>(B.data_ptr()),
     reinterpret_cast<nv_bfloat16*>(C.data_ptr()), M, N, K);
  return C;
}

TORCH_LIBRARY(gn_matmul, m) {
  m.def("matmul_v7a(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v7a>);
  m.def("matmul_v7b(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v7b>);
  m.def("matmul_v7c(Tensor A, Tensor B) -> Tensor", &matmul<matmul_v7c>);
}
