"""Fixed Hopper GEMM kernel used by ``mmc.matmul`` on sm_90 GPUs.

This wraps the study kernel ``hopper_ws_tma_load_store_kernel.cu`` with the
fixed configuration we want to expose first:

    BM128 / BN256 / BK64 / WG2 / NS4 / 2 TMA-store stages

The default grouping is GM8 and binds the original static-GM symbol.  Experimental
non-default GM values bind a runtime-GM entry point.  The wrapper is intentionally
narrow: plain bf16 GEMM only, no autotuning, and no epilogue fusion yet.  It
compiles once into the MMComposer artifact cache and then launches asynchronously
on PyTorch's current stream.
"""
from __future__ import annotations

import ctypes
import pathlib

from . import cache as kcache
from . import compiler
from . import runtime

ARCH = "sm_90a"
DEFAULT_GM = 8
SYMBOL_GM8 = "matmul_hopper_ws_tma_load_store_bm128_bn256_bk64_wg2_ns4_gm8"
SYMBOL_RUNTIME_GM = "matmul_hopper_ws_tma_load_store_bm128_bn256_bk64_wg2_ns4_runtime_gm"

BM, BN, BK = 128, 256, 64
NWG, NS = 2, 4
STORE_N = 64
TMA_STORE_STAGES = 2
ELEM_BYTES = 2
THREADS = NWG * 128 + 32
SHARED_BYTES = (
    NS * BM * BK + NS * BK * BN + TMA_STORE_STAGES * BM * STORE_N
) * ELEM_BYTES

_SRC = pathlib.Path(__file__).resolve().parent / "kernels" / "hopper" / \
    "hopper_ws_tma_load_store_kernel.cu"

_CUBIN: str | None = None
_MODULE_CACHE: dict[tuple[int, str, str], tuple[object, object]] = {}
_DEVICE_CONTEXTS: dict[int, tuple[object, object]] = {}
_SM_COUNTS: dict[int, int] = {}


def _normalize_gm(gm: int) -> int:
    gm = int(gm)
    if gm < 1:
        raise ValueError(f"experimental Hopper matmul gm must be positive, got {gm}")
    return gm


def _uses_runtime_gm(gm: int) -> bool:
    return gm != DEFAULT_GM


def _symbol(gm: int) -> str:
    return SYMBOL_RUNTIME_GM if _uses_runtime_gm(gm) else SYMBOL_GM8


def is_hopper_device(device=None) -> bool:
    """Return True when the active CUDA device is Hopper-class sm_90."""
    import torch

    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major == 9


def validate(a, b):
    """Check dtype/layout/shape and return (M, N, K)."""
    import torch

    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"Hopper matmul supports bf16 inputs only (got {a.dtype}, {b.dtype})")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"expected 2-D matrices, got {a.dim()}-D and {b.dim()}-D")
    M, Ka = a.shape
    Kb, N = b.shape
    if Ka != Kb:
        raise ValueError(f"inner dims disagree: {tuple(a.shape)} @ {tuple(b.shape)}")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Hopper matmul inputs must be CUDA tensors")
    if a.device != b.device:
        raise ValueError(f"inputs must be on the same CUDA device ({a.device} vs {b.device})")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("inputs must be row-major contiguous (A: MxK, B: KxN)")
    errs = []
    if M < 1:
        errs.append(f"M={M} must be positive")
    if N % BN:
        errs.append(f"N={N} must be a multiple of {BN}")
    if Ka % BK:
        errs.append(f"K={Ka} must be a multiple of {BK}")
    if Ka // BK < NS:
        errs.append(f"K={Ka} must be at least {NS * BK} for the fixed NS{NS} pipeline")
    if errs:
        raise ValueError("unsupported shape for Hopper fixed WS matmul: " + "; ".join(errs))
    return M, N, Ka


def _device_index(device) -> int:
    import torch

    return torch.cuda.current_device() if device.index is None else device.index


def _ensure_device_context(device_index: int) -> int:
    """Make the CUDA primary context for ``device_index`` current and return SMs."""
    rt, driver = runtime._backends()
    if device_index not in _DEVICE_CONTEXTS:
        rt.cu(driver.cuInit(0))
        dev = rt.cu(driver.cuDeviceGet(device_index))
        ctx = rt.cu(driver.cuDevicePrimaryCtxRetain(dev))
        _DEVICE_CONTEXTS[device_index] = (dev, ctx)
        _SM_COUNTS[device_index] = rt.cu(driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev))
    _, ctx = _DEVICE_CONTEXTS[device_index]
    rt.cu(driver.cuCtxSetCurrent(ctx))
    return _SM_COUNTS[device_index]


def _load_for_device(cubin_path: str, symbol: str, device_index: int):
    rt, driver = runtime._backends()
    _ensure_device_context(device_index)
    key = (device_index, cubin_path, symbol)
    if key not in _MODULE_CACHE:
        with open(cubin_path, "rb") as f:
            data = f.read()
        mod = rt.cu(driver.cuModuleLoadData(data))
        fn = rt.cu(driver.cuModuleGetFunction(mod, symbol.encode()))
        _MODULE_CACHE[key] = (mod, fn)
    return _MODULE_CACHE[key]


def _ensure_cubin() -> str:
    """Compile the shipped Hopper .cu into the artifact cache once."""
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


def launch_dims(M: int, N: int, K: int, num_sms: int):
    total_tiles = ((M + BM - 1) // BM) * (N // BN)
    grid = (min(num_sms, total_tiles), 1, 1)
    block = (THREADS, 1, 1)
    return grid, block, SHARED_BYTES


def _descriptors(a, b, c, M, N, K):
    rt, _ = runtime._backends()
    A = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=a.data_ptr(),
                             global_dim=[K, M], global_strides=[K * ELEM_BYTES],
                             box_dim=[BK, BM], element_strides=[1, 1],
                             swizzle=rt.TMA_SWIZZLE_128B)
    B = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=b.data_ptr(),
                             global_dim=[N, K], global_strides=[N * ELEM_BYTES],
                             box_dim=[STORE_N, BK], element_strides=[1, 1],
                             swizzle=rt.TMA_SWIZZLE_128B)
    C = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=c.data_ptr(),
                             global_dim=[N, M], global_strides=[N * ELEM_BYTES],
                             box_dim=[STORE_N, BM], element_strides=[1, 1],
                             swizzle=rt.TMA_SWIZZLE_128B)
    return A, B, C


def _prepare(a, b, c, M, N, K, cubin_path, device_index, gm: int = DEFAULT_GM):
    gm = _normalize_gm(gm)
    rt, driver = runtime._backends()
    num_sms = _ensure_device_context(device_index)
    _, fn = _load_for_device(cubin_path, _symbol(gm), device_index)
    grid, block, shared = launch_dims(M, N, K, num_sms)
    rt.cu(driver.cuFuncSetAttribute(
        fn, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared))
    maps = _descriptors(a, b, c, M, N, K)
    args = [(ctypes.c_byte * 128).from_buffer_copy(m.tobytes()) for m in maps]
    args += [ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(N)]
    if _uses_runtime_gm(gm):
        args.append(ctypes.c_int(gm))
    return fn, grid, block, shared, args


def kernel(gm: int = DEFAULT_GM):
    """Return a callable ``k(a, b, c=None) -> c`` for Hopper WS GEMM."""
    import torch

    gm = _normalize_gm(gm)
    state: dict = {}

    def call(a, b, c=None, *, sync=False, stream=None):
        M, N, K = validate(a, b)
        device_index = _device_index(a.device)
        if not is_hopper_device(a.device):
            raise RuntimeError(f"fixed Hopper matmul requires sm_90, got {a.device}")
        if c is None:
            c = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
        else:
            if c.dtype != torch.bfloat16 or tuple(c.shape) != (M, N) or not c.is_cuda:
                raise ValueError(f"out must be a CUDA bf16 tensor of shape ({M}, {N})")
            if c.device != a.device:
                raise ValueError(f"out must be on {a.device}, got {c.device}")
            if not c.is_contiguous():
                raise ValueError("out must be row-major contiguous")
        if stream is None:
            stream = torch.cuda.current_stream(a.device).cuda_stream
        cubin = _ensure_cubin()
        skey = (gm, device_index, M, N, K, a.data_ptr(), b.data_ptr(), c.data_ptr())
        st = state.get(skey)
        if st is None:
            st = _prepare(a, b, c, M, N, K, cubin, device_index, gm)
            state[skey] = st
        else:
            _ensure_device_context(device_index)
        rt, _ = runtime._backends()
        fn, grid, block, shared, args = st
        rt.launch(fn, grid=grid, block=block, shared=shared, args=args,
                  stream=stream, sync=sync)
        return c

    return call
