"""Fused GEMM + SwiGLU (dual-B) --- a fixed, hand-tuned Blackwell kernel.

Unlike `matmul` (a codegen + autotune sweep), this wraps a single pre-written
study kernel, `fused_matmul_swiglu_out_fast_dual_b_ns6_s2.cu` (config baked in:
BM128/BN256/BK64, NS=6, 2-CTA cluster, 2-stage TMA store).  Two B halves in,
two outputs out:

    A[M, K]            bf16, row-major
    B_left[K, N/2]     bf16, row-major or row-major column view
    B_gate[K, N/2]     bf16, row-major or row-major column view
  ->
    C[M, N]            packed wide GEMM, per BN=256 tile as [left128 | gate128]
    D[M, N/2]          left * silu(gate)   (the SwiGLU activation)

It compiles the cubin once per machine (cached under the artifact cache) and,
like `runtime.kernel`, caches the per-(shape, buffers) launch state and launches
asynchronously on torch's current stream.

Public API:
    validate(a, b_left, b_gate) -> (M, N, K)      # pure, no GPU
    kernel() -> callable  k(a, b_left, b_gate[, c, d]) -> (c, d)
"""
from __future__ import annotations

import ctypes
import pathlib

from . import cache as kcache
from . import compiler
from . import runtime

# ---- baked-in config (must match the .cu constexprs) ----------------------
SYMBOL = "matmul_cluster"
BM, BN, BK = 128, 256, 64
NS = 6
NUM_WARPS = 4
CTA_GROUP = 2
STORE_N = 64
TMA_STORE_STAGES = 2
ELEM_BYTES = 2          # bf16
LAUNCH_THREADS = (NUM_WARPS + 4) * 32   # 256

_SRC = pathlib.Path(__file__).resolve().parent / "kernels" / "swiglu" / \
    "fused_matmul_swiglu_out_fast_dual_b_ns6_s2.cu"


# ---- validation (pure) ----------------------------------------------------
def _check_dense_or_column_view(name, t, *, elem_bytes=ELEM_BYTES):
    if t.stride(1) != 1:
        raise ValueError(f"{name} must have unit column stride")
    if t.stride(0) < t.shape[1]:
        raise ValueError(f"{name} row stride {t.stride(0)} is too small for width {t.shape[1]}")
    if t.stride(0) <= 0:
        raise ValueError(f"{name} must have positive row stride")
    if t.data_ptr() % 16:
        raise ValueError(f"{name} data pointer must be 16-byte aligned for TMA")
    if (t.stride(0) * elem_bytes) % 16:
        raise ValueError(f"{name} row stride must be 16-byte aligned for TMA")


def validate(a, b_left, b_gate):
    """Check dtype/layout/shape and return (M, N, K).  N is the *packed* width
    (= 2 * b_left.shape[1])."""
    import torch
    for name, t in (("a", a), ("b_left", b_left), ("b_gate", b_gate)):
        if t.dtype != torch.bfloat16:
            raise TypeError(f"swiglu supports bf16 only ({name} is {t.dtype})")
        if t.dim() != 2:
            raise ValueError(f"{name} must be 2-D, got {t.dim()}-D")
    if not a.is_contiguous():
        raise ValueError("a must be row-major contiguous")
    M, Ka = a.shape
    Kl, Hl = b_left.shape
    Kg, Hg = b_gate.shape
    if not (Ka == Kl == Kg):
        raise ValueError(f"K disagrees: a {tuple(a.shape)}, b_left {tuple(b_left.shape)}, "
                         f"b_gate {tuple(b_gate.shape)}")
    if Hl != Hg:
        raise ValueError(f"b_left and b_gate must share N/2: {Hl} vs {Hg}")
    _check_dense_or_column_view("b_left", b_left)
    _check_dense_or_column_view("b_gate", b_gate)
    N = 2 * Hl
    errs = []
    if M % (CTA_GROUP * BM):
        errs.append(f"M={M} must be a multiple of {CTA_GROUP * BM}")
    if N % BN:
        errs.append(f"N={N} (=2*{Hl}) must be a multiple of {BN}")
    if Ka % BK:
        errs.append(f"K={Ka} must be a multiple of {BK}")
    if errs:
        raise ValueError("unsupported shape for the swiglu kernel: " + "; ".join(errs))
    return M, N, Ka


# ---- cubin (compile once, cached) -----------------------------------------
_CUBIN: dict = {}   # arch -> cubin path


def _ensure_cubin(arch: str) -> str:
    """Compile the shipped .cu into the artifact cache once; return cubin path."""
    if arch not in _CUBIN:
        build = kcache.cache_root() / "build" / arch / "swiglu"
        build.mkdir(parents=True, exist_ok=True)
        dst = build / _SRC.name
        src_text = _SRC.read_text()
        # copy into the writable build dir if missing/stale (install dir is data)
        if not dst.exists() or dst.read_text() != src_text:
            dst.write_text(src_text)
        _CUBIN[arch] = compiler.compile_one(str(dst), arch=arch)
    return _CUBIN[arch]


# ---- launch geometry (pure) -----------------------------------------------
def launch_dims(num_sms: int):
    a_slot = BM * BK * ELEM_BYTES
    b_slot = (BN // CTA_GROUP) * BK * ELEM_BYTES
    compute_ring = NS * (a_slot + b_slot)
    epilogue_ring = BM * STORE_N * ELEM_BYTES * TMA_STORE_STAGES
    shared = compute_ring + epilogue_ring + 1024
    grid = (num_sms - num_sms % CTA_GROUP, 1, 1)
    block = (LAUNCH_THREADS, 1, 1)
    return grid, block, shared


def _descriptors(a, b_left, b_gate, c, d, M, N, K):
    """A, B_left, B_gate, C, D TMA descriptors (np arrays)."""
    rt, _ = runtime._backends()
    H = N // 2
    bl_stride = int(b_left.stride(0)) * ELEM_BYTES
    bg_stride = int(b_gate.stride(0)) * ELEM_BYTES
    A = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=a.data_ptr(),
                             global_dim=[K, M], global_strides=[K * ELEM_BYTES],
                             box_dim=[BK, BM], element_strides=[1, 1],
                             swizzle=rt.TMA_SWIZZLE_128B)
    BL = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=b_left.data_ptr(),
                              global_dim=[H, K], global_strides=[bl_stride],
                              box_dim=[STORE_N, BK], element_strides=[1, 1],
                              swizzle=rt.TMA_SWIZZLE_128B)
    BG = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=b_gate.data_ptr(),
                              global_dim=[H, K], global_strides=[bg_stride],
                              box_dim=[STORE_N, BK], element_strides=[1, 1],
                              swizzle=rt.TMA_SWIZZLE_128B)
    C = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=c.data_ptr(),
                             global_dim=[N, M], global_strides=[N * ELEM_BYTES],
                             box_dim=[STORE_N, BM], element_strides=[1, 1],
                             swizzle=rt.TMA_SWIZZLE_128B)
    D = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=d.data_ptr(),
                             global_dim=[H, M], global_strides=[H * ELEM_BYTES],
                             box_dim=[STORE_N, BM], element_strides=[1, 1],
                             swizzle=rt.TMA_SWIZZLE_128B)
    return A, BL, BG, C, D


def _prepare(a, b_left, b_gate, c, d, M, N, K, cubin_path):
    rt, driver = runtime._backends()
    _, num_sms = runtime._ensure_cuda()
    _, fn = runtime._load(cubin_path, SYMBOL)
    grid, block, shared = launch_dims(num_sms)
    rt.cu(driver.cuFuncSetAttribute(
        fn, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared))
    maps = _descriptors(a, b_left, b_gate, c, d, M, N, K)
    args = [(ctypes.c_byte * 128).from_buffer_copy(m.tobytes()) for m in maps]
    args += [ctypes.c_void_p(c.data_ptr()),
             ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)]
    return fn, grid, block, shared, args


def kernel():
    """Return a callable ``k(a, b_left, b_gate, c=None, d=None) -> (c, d)``.

    Compiles the cubin once per machine, caches the per-(shape, buffers) launch
    state, and launches asynchronously on torch's current stream (like
    `runtime.kernel`).  Pass `c`/`d` to reuse output buffers, else fresh bf16
    outputs are allocated (C is [M, N]; D is [M, N/2])."""
    import torch
    state: dict = {}

    def call(a, b_left, b_gate, c=None, d=None, *, sync=False, stream=None):
        M, N, K = validate(a, b_left, b_gate)
        H = N // 2
        if c is None:
            c = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
        if d is None:
            d = torch.empty(M, H, dtype=torch.bfloat16, device=a.device)
        if stream is None:
            stream = torch.cuda.current_stream(a.device).cuda_stream
        cubin = _ensure_cubin(compiler.DEFAULT_ARCH)
        skey = (M, N, K, a.data_ptr(), b_left.data_ptr(), b_gate.data_ptr(),
                b_left.stride(), b_gate.stride(), c.data_ptr(), d.data_ptr())
        st = state.get(skey)
        if st is None:
            st = _prepare(a, b_left, b_gate, c, d, M, N, K, cubin)
            state[skey] = st
        rt, _ = runtime._backends()
        fn, grid, block, shared, args = st
        rt.launch(fn, grid=grid, block=block, shared=shared, args=args,
                  stream=stream, sync=sync)
        return c, d

    return call
