"""Fixed Hopper dual-B SwiGLU kernels used by ``mmc.matmul_swiglu_dual_b``.

Both Hopper paths use the same fixed WS pipeline: BM128 / internal BN256 / BK64 /
WG2 / NS4 / 2 TMA-store stages.  The default grouping is GM8; experimental
non-default GM values dispatch to a runtime-GM no-preact entry point.  The
internal BN256 accumulator is interpreted as ``[left 128 | gate 128]``.

``kernel()`` computes and stores only ``D = left * silu(gate)``.
``kernel_store_preact()`` stores both packed preactivation ``C = [left | gate]``
and ``D``.
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
DEFAULT_GM = 8
SYMBOL_NO_PREACT_GM8 = "matmul_hopper_swiglu_dual_b_bm128_bn256_bk64_wg2_ns4_gm8"
SYMBOL_STORE_PREACT_GM8 = (
    "matmul_hopper_swiglu_dual_b_store_preact_bm128_bn256_bk64_wg2_ns4_gm8"
)
SYMBOL_NO_PREACT_RUNTIME_GM = (
    "matmul_hopper_swiglu_dual_b_bm128_bn256_bk64_wg2_ns4_runtime_gm"
)

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

_KERNEL_DIR = pathlib.Path(__file__).resolve().parent / "kernels" / "hopper"
_SRC = _KERNEL_DIR / "hopper_swiglu_dual_b_kernel.cu"
_PACKAGED_CUBIN = _KERNEL_DIR / "hopper_swiglu_dual_b_kernel_sm_90a.cubin"
# The bundled cubin is built with CUDA 12.8.  Newer drivers are backward
# compatible; older drivers can still use the nvcc fallback if available.
_PACKAGED_CUBIN_MIN_DRIVER = 12080

_CUBIN: str | None = None


def _normalize_gm(gm: int) -> int:
    gm = int(gm)
    if gm < 1:
        raise ValueError(f"experimental Hopper SwiGLU gm must be positive, got {gm}")
    return gm


def _uses_runtime_gm(gm: int) -> bool:
    return gm != DEFAULT_GM


def _symbol_no_preact(gm: int) -> str:
    return SYMBOL_NO_PREACT_RUNTIME_GM if _uses_runtime_gm(gm) else SYMBOL_NO_PREACT_GM8


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


def _cuda_driver_version() -> int:
    """Return the CUDA driver API version, for example 12080 for CUDA 12.8."""
    rt, driver = runtime._backends()
    return int(rt.cu(driver.cuDriverGetVersion()))


def _compile_cubin_fallback(reason: str | None) -> str:
    build = kcache.cache_root() / "build" / ARCH / "hopper"
    build.mkdir(parents=True, exist_ok=True)
    dst = build / _SRC.name
    src_text = _SRC.read_text()
    if not dst.exists() or dst.read_text() != src_text:
        dst.write_text(src_text)
    try:
        return compiler.compile_one(str(dst), arch=ARCH, extra_opts=["-DLB_MIN_BLOCKS=1"])
    except Exception as exc:
        if reason:
            raise RuntimeError(
                "Unable to use the packaged Hopper SwiGLU cubin "
                f"({reason}), and the nvcc fallback failed."
            ) from exc
        raise


def _ensure_cubin() -> str:
    """Return a usable Hopper SwiGLU cubin, preferring the packaged binary."""
    global _CUBIN
    if _CUBIN is None:
        reason = None
        if _PACKAGED_CUBIN.exists():
            driver_version = _cuda_driver_version()
            if driver_version >= _PACKAGED_CUBIN_MIN_DRIVER:
                _CUBIN = str(_PACKAGED_CUBIN)
                return _CUBIN
            reason = (
                f"CUDA driver API version {driver_version} is older than "
                f"{_PACKAGED_CUBIN_MIN_DRIVER} required by the bundled CUDA 12.8 cubin"
            )
        else:
            reason = f"packaged cubin not found at {_PACKAGED_CUBIN}"
        _CUBIN = _compile_cubin_fallback(reason)
    return _CUBIN


def launch_dims(M: int, H: int, K: int, num_sms: int):
    total_tiles = ((M + BM - 1) // BM) * (H // OUT_N)
    grid = (min(num_sms, total_tiles), 1, 1)
    block = (THREADS, 1, 1)
    return grid, block, SHARED_BYTES


def _common_descriptors(a, b_left, b_gate, M, H, K):
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
    return A, BL, BG


def _d_descriptor(d, M, H):
    rt, _ = runtime._backends()
    return rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=d.data_ptr(),
                                global_dim=[H, M], global_strides=[H * ELEM_BYTES],
                                box_dim=[STORE_N, BM], element_strides=[1, 1],
                                swizzle=rt.TMA_SWIZZLE_128B)


def _c_descriptor(c, M, H):
    rt, _ = runtime._backends()
    N = 2 * H
    return rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=c.data_ptr(),
                                global_dim=[N, M], global_strides=[N * ELEM_BYTES],
                                box_dim=[STORE_N, BM], element_strides=[1, 1],
                                swizzle=rt.TMA_SWIZZLE_128B)


def _prepare_no_preact(a, b_left, b_gate, d, M, H, K, cubin_path, device_index,
                         gm: int = DEFAULT_GM):
    gm = _normalize_gm(gm)
    rt, driver = runtime._backends()
    num_sms = _hopper._ensure_device_context(device_index)
    _, fn = _hopper._load_for_device(cubin_path, _symbol_no_preact(gm), device_index)
    grid, block, shared = launch_dims(M, H, K, num_sms)
    rt.cu(driver.cuFuncSetAttribute(
        fn, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared))
    maps = (*_common_descriptors(a, b_left, b_gate, M, H, K), _d_descriptor(d, M, H))
    args = [(ctypes.c_byte * 128).from_buffer_copy(m.tobytes()) for m in maps]
    args += [ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(H)]
    if _uses_runtime_gm(gm):
        args.append(ctypes.c_int(gm))
    return fn, grid, block, shared, args


def _prepare_store_preact(a, b_left, b_gate, c, d, M, H, K, cubin_path, device_index):
    rt, driver = runtime._backends()
    num_sms = _hopper._ensure_device_context(device_index)
    _, fn = _hopper._load_for_device(cubin_path, SYMBOL_STORE_PREACT_GM8, device_index)
    grid, block, shared = launch_dims(M, H, K, num_sms)
    rt.cu(driver.cuFuncSetAttribute(
        fn, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared))
    maps = (*_common_descriptors(a, b_left, b_gate, M, H, K),
            _c_descriptor(c, M, H), _d_descriptor(d, M, H))
    args = [(ctypes.c_byte * 128).from_buffer_copy(m.tobytes()) for m in maps]
    args += [ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(H)]
    return fn, grid, block, shared, args


def _check_d(d, M, H, device):
    import torch

    if d.dtype != torch.bfloat16 or tuple(d.shape) != (M, H) or not d.is_cuda:
        raise ValueError(f"out must be a CUDA bf16 tensor of shape ({M}, {H})")
    if d.device != device:
        raise ValueError(f"out must be on {device}, got {d.device}")
    if not d.is_contiguous():
        raise ValueError("out must be row-major contiguous")


def _check_c(c, M, H, device):
    import torch

    N = 2 * H
    if c.dtype != torch.bfloat16 or tuple(c.shape) != (M, N) or not c.is_cuda:
        raise ValueError(f"preact must be a CUDA bf16 tensor of shape ({M}, {N})")
    if c.device != device:
        raise ValueError(f"preact must be on {device}, got {c.device}")
    if not c.is_contiguous():
        raise ValueError("preact must be row-major contiguous")


def kernel(gm: int = DEFAULT_GM):
    """Return a callable ``k(a, b_left, b_gate, d=None) -> d`` for Hopper SwiGLU."""
    import torch

    gm = _normalize_gm(gm)
    state: dict = {}

    def call(a, b_left, b_gate, d=None, *, sync=False, stream=None):
        M, H, K = validate(a, b_left, b_gate)
        device_index = _hopper._device_index(a.device)
        if not _hopper.is_hopper_device(a.device):
            raise RuntimeError(f"fixed Hopper SwiGLU requires sm_90, got {a.device}")
        if d is None:
            d = torch.empty(M, H, dtype=torch.bfloat16, device=a.device)
        else:
            _check_d(d, M, H, a.device)
        if stream is None:
            stream = torch.cuda.current_stream(a.device).cuda_stream
        cubin = _ensure_cubin()
        skey = (gm, device_index, M, H, K, a.data_ptr(), b_left.data_ptr(), b_gate.data_ptr(),
                b_left.stride(), b_gate.stride(), d.data_ptr())
        st = state.get(skey)
        if st is None:
            st = _prepare_no_preact(a, b_left, b_gate, d, M, H, K, cubin, device_index, gm)
            state[skey] = st
        else:
            _hopper._ensure_device_context(device_index)
        rt, _ = runtime._backends()
        fn, grid, block, shared, args = st
        rt.launch(fn, grid=grid, block=block, shared=shared, args=args,
                  stream=stream, sync=sync)
        return d

    return call


def kernel_store_preact():
    """Return a callable ``k(a, b_left, b_gate, c=None, d=None) -> (c, d)``."""
    import torch

    state: dict = {}

    def call(a, b_left, b_gate, c=None, d=None, *, sync=False, stream=None):
        M, H, K = validate(a, b_left, b_gate)
        device_index = _hopper._device_index(a.device)
        if not _hopper.is_hopper_device(a.device):
            raise RuntimeError(f"fixed Hopper SwiGLU requires sm_90, got {a.device}")
        if c is None:
            c = torch.empty(M, 2 * H, dtype=torch.bfloat16, device=a.device)
        else:
            _check_c(c, M, H, a.device)
        if d is None:
            d = torch.empty(M, H, dtype=torch.bfloat16, device=a.device)
        else:
            _check_d(d, M, H, a.device)
        if stream is None:
            stream = torch.cuda.current_stream(a.device).cuda_stream
        cubin = _ensure_cubin()
        skey = (device_index, M, H, K, a.data_ptr(), b_left.data_ptr(), b_gate.data_ptr(),
                b_left.stride(), b_gate.stride(), c.data_ptr(), d.data_ptr())
        st = state.get(skey)
        if st is None:
            st = _prepare_store_preact(a, b_left, b_gate, c, d, M, H, K, cubin, device_index)
            state[skey] = st
        else:
            _hopper._ensure_device_context(device_index)
        rt, _ = runtime._backends()
        fn, grid, block, shared, args = st
        rt.launch(fn, grid=grid, block=block, shared=shared, args=args,
                  stream=stream, sync=sync)
        return c, d

    return call
