"""Fixed Hopper dual-B SwiGLU kernel used by ``mmc.matmul_swiglu_dual_b``.

This is the inference-oriented path for Hopper: it computes

    D = (A @ B_left) * silu(A @ B_gate)

without storing the packed preactivation ``[A @ B_left | A @ B_gate]``.  The
kernel is based on the fixed Hopper WS GEMM pipeline: BM128 / internal BN256 /
BK64 / WG2 / NS4 / GM8 / 2 TMA-store stages.  The internal BN256 accumulator is
interpreted as ``[left 128 | gate 128]`` and the epilogue writes only ``D[M,H]``.
"""
from __future__ import annotations

import ctypes
import pathlib

from . import cache as kcache
from . import compiler
from . import hopper as _hopper
from . import runtime
from . import swiglu as _swiglu

ARCH = "sm_90a"
SYMBOL = "matmul_hopper_swiglu_dual_b_bm128_bn256_bk64_wg2_ns4_gm8"

BM, BN, BK = 128, 256, 64
OUT_N = BN // 2
NWG, NS = 2, 4
STORE_N = 64
TMA_STORE_STAGES = 2
ELEM_BYTES = 2
THREADS = NWG * 128 + 32
SHARED_BYTES = (
    NS * BM * BK + NS * BK * BN + TMA_STORE_STAGES * BM * STORE_N
) * ELEM_BYTES

_SRC = pathlib.Path(__file__).resolve().parent / "kernels" / "hopper" / \
    "hopper_swiglu_dual_b_kernel.cu"

_CUBIN: str | None = None


def validate(a, b_left, b_gate):
    """Check dtype/layout/shape and return (M, H, K)."""
    M, N, K = _swiglu.validate(a, b_left, b_gate)
    H = N // 2
    for name, t in (("a", a), ("b_left", b_left), ("b_gate", b_gate)):
        if not getattr(t, "is_cuda", False):
            raise ValueError(f"{name} must be a CUDA tensor")
        if t.device != a.device:
            raise ValueError(f"{name} must be on {a.device}, got {t.device}")
    errs = []
    if H % OUT_N:
        errs.append(f"H={H} must be a multiple of {OUT_N}")
    if K // BK < NS:
        errs.append(f"K={K} must be at least {NS * BK} for the fixed NS{NS} pipeline")
    if errs:
        raise ValueError("unsupported shape for Hopper fixed SwiGLU: " + "; ".join(errs))
    return M, H, K


def _ensure_cubin() -> str:
    """Compile the shipped Hopper SwiGLU .cu into the artifact cache once."""
    global _CUBIN
    if _CUBIN is None:
        build = kcache.cache_root() / "build" / ARCH / "hopper"
        build.mkdir(parents=True, exist_ok=True)
        dst = build / _SRC.name
        src_text = _SRC.read_text()
        if not dst.exists() or dst.read_text() != src_text:
            dst.write_text(src_text)
        _CUBIN = compiler.compile_one(str(dst), arch=ARCH, extra_opts=["-DLB_MIN_BLOCKS=1"])
    return _CUBIN


def launch_dims(M: int, H: int, K: int, num_sms: int):
    total_tiles = ((M + BM - 1) // BM) * (H // OUT_N)
    grid = (min(num_sms, total_tiles), 1, 1)
    block = (THREADS, 1, 1)
    return grid, block, SHARED_BYTES


def _descriptors(a, b_left, b_gate, d, M, H, K):
    rt, _ = runtime._backends()
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
    D = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=d.data_ptr(),
                             global_dim=[H, M], global_strides=[H * ELEM_BYTES],
                             box_dim=[STORE_N, BM], element_strides=[1, 1],
                             swizzle=rt.TMA_SWIZZLE_128B)
    return A, BL, BG, D


def _prepare(a, b_left, b_gate, d, M, H, K, cubin_path, device_index):
    rt, driver = runtime._backends()
    num_sms = _hopper._ensure_device_context(device_index)
    _, fn = _hopper._load_for_device(cubin_path, SYMBOL, device_index)
    grid, block, shared = launch_dims(M, H, K, num_sms)
    rt.cu(driver.cuFuncSetAttribute(
        fn, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared))
    maps = _descriptors(a, b_left, b_gate, d, M, H, K)
    args = [(ctypes.c_byte * 128).from_buffer_copy(m.tobytes()) for m in maps]
    args += [ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(H)]
    return fn, grid, block, shared, args


def kernel():
    """Return a callable ``k(a, b_left, b_gate, d=None) -> d`` for Hopper SwiGLU."""
    import torch

    state: dict = {}

    def call(a, b_left, b_gate, d=None, *, sync=False, stream=None):
        M, H, K = validate(a, b_left, b_gate)
        device_index = _hopper._device_index(a.device)
        if not _hopper.is_hopper_device(a.device):
            raise RuntimeError(f"fixed Hopper SwiGLU requires sm_90, got {a.device}")
        if d is None:
            d = torch.empty(M, H, dtype=torch.bfloat16, device=a.device)
        else:
            if d.dtype != torch.bfloat16 or tuple(d.shape) != (M, H) or not d.is_cuda:
                raise ValueError(f"out must be a CUDA bf16 tensor of shape ({M}, {H})")
            if d.device != a.device:
                raise ValueError(f"out must be on {a.device}, got {d.device}")
            if not d.is_contiguous():
                raise ValueError("out must be row-major contiguous")
        if stream is None:
            stream = torch.cuda.current_stream(a.device).cuda_stream
        cubin = _ensure_cubin()
        skey = (device_index, M, H, K, a.data_ptr(), b_left.data_ptr(), b_gate.data_ptr(),
                b_left.stride(), b_gate.stride(), d.data_ptr())
        st = state.get(skey)
        if st is None:
            st = _prepare(a, b_left, b_gate, d, M, H, K, cubin, device_index)
            state[skey] = st
        else:
            _hopper._ensure_device_context(device_index)
        rt, _ = runtime._backends()
        fn, grid, block, shared, args = st
        rt.launch(fn, grid=grid, block=block, shared=shared, args=args,
                  stream=stream, sync=sync)
        return d

    return call
