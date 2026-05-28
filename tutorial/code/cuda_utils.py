"""Shared cuda-python plumbing for the tutorial examples.

Hides the per-call `(err, *rest)` tuple-unpacking, the NVRTC compile
dance, and the cuLaunchKernel argument-packing so each chapter's
`main.py` can focus on the actual CUDA concepts being taught.

Used by `<chapter>/main.py` via:

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cuda_utils import init_cuda, compile_kernel, launch, htod, dtoh, cu
"""

import ctypes
import os
import subprocess
import sys

import numpy as np
from cuda.bindings import driver, nvrtc, runtime


# ── ctypes binding for cuTensorMapEncodeTiled ───────────────────────────────
#
# cuda-python's wrapper for this function has shifted across versions
# (strict typed scalars, evolving return-value conventions).  Calling
# the C entry point directly via ctypes is more stable and the
# descriptor-build site reads much more cleanly.  See cuda_utils.py
# docstring + chapter 00 for context.

_libcuda = ctypes.CDLL("libcuda.so", mode=ctypes.RTLD_GLOBAL)
_cuTensorMapEncodeTiled = _libcuda.cuTensorMapEncodeTiled
_cuTensorMapEncodeTiled.restype = ctypes.c_int
_cuTensorMapEncodeTiled.argtypes = [
    ctypes.c_void_p,                  # CUtensorMap* (output)
    ctypes.c_int,                     # dtype
    ctypes.c_uint32,                  # rank
    ctypes.c_void_p,                  # globalAddress
    ctypes.POINTER(ctypes.c_uint64),  # globalDim
    ctypes.POINTER(ctypes.c_uint64),  # globalStrides
    ctypes.POINTER(ctypes.c_uint32),  # boxDim
    ctypes.POINTER(ctypes.c_uint32),  # elementStrides
    ctypes.c_int,                     # interleave
    ctypes.c_int,                     # swizzle
    ctypes.c_int,                     # l2 promotion
    ctypes.c_int,                     # oob fill
]


# Mirrors the CUtensorMap* enums from cuda.h.  Numeric values stable.
TMA_UINT8       = 0
TMA_UINT16      = 1
TMA_UINT32      = 2
TMA_INT32       = 3
TMA_UINT64      = 4
TMA_INT64       = 5
TMA_FLOAT16     = 6
TMA_FLOAT32     = 7
TMA_FLOAT64     = 8
TMA_BFLOAT16    = 9

TMA_INTERLEAVE_NONE = 0
TMA_INTERLEAVE_16B  = 1
TMA_INTERLEAVE_32B  = 2

TMA_SWIZZLE_NONE = 0
TMA_SWIZZLE_32B  = 1
TMA_SWIZZLE_64B  = 2
TMA_SWIZZLE_128B = 3

TMA_L2_NONE = 0
TMA_L2_64B  = 1
TMA_L2_128B = 2
TMA_L2_256B = 3

TMA_OOB_NONE = 0
TMA_OOB_NAN_REQUEST_ZERO_FMA = 1


def encode_tensor_map(
    *,
    dtype: int,
    rank: int,
    gptr: int,
    global_dim,
    box_dim,
    element_strides,
    global_strides=None,
    interleave: int = TMA_INTERLEAVE_NONE,
    swizzle: int = TMA_SWIZZLE_NONE,
    l2_promotion: int = TMA_L2_NONE,
    oob_fill: int = TMA_OOB_NONE,
) -> np.ndarray:
    """Build a 128-byte CUtensorMap via libcuda's cuTensorMapEncodeTiled.

    All shape/stride args are Python sequences of ints:

      global_dim       — length `rank`, innermost-first
      global_strides   — length `rank - 1`, in BYTES, outer-dim strides;
                         pass None or [] for 1D
      box_dim          — length `rank`, innermost-first
      element_strides  — length `rank`, typically all 1s

    Returns a numpy uint8 array of length 128 holding the descriptor.
    Pass it to a kernel as a by-value 128-byte struct argument.
    """
    if global_strides is None:
        global_strides = []
    assert len(global_dim)      == rank,     f"global_dim must have {rank} entries"
    assert len(box_dim)         == rank,     f"box_dim must have {rank} entries"
    assert len(element_strides) == rank,     f"element_strides must have {rank} entries"
    assert len(global_strides)  == rank - 1, f"global_strides must have {rank - 1} entries"

    tmap = np.zeros(128, dtype=np.uint8)
    gdim_arr = (ctypes.c_uint64 * rank)(*global_dim)
    bdim_arr = (ctypes.c_uint32 * rank)(*box_dim)
    estr_arr = (ctypes.c_uint32 * rank)(*element_strides)
    gstr_arr = ((ctypes.c_uint64 * (rank - 1))(*global_strides)
                if rank > 1 else None)

    err = _cuTensorMapEncodeTiled(
        tmap.ctypes.data,
        dtype,
        rank,
        gptr,
        gdim_arr,
        gstr_arr,
        bdim_arr,
        estr_arr,
        interleave,
        swizzle,
        l2_promotion,
        oob_fill,
    )
    if err != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: CUresult={err}")
    return tmap


# ── Error checking ──────────────────────────────────────────────────────────

def cu(result):
    """Unwrap a cuda-python `(err, *rest)` return tuple, raising on error.

    cuda-python's driver / nvrtc / runtime bindings all return a leading
    error code followed by any out-parameters.  This helper checks the
    error and returns the rest (a single value if just one, else a
    tuple, or None if no out-params).
    """
    err, *rest = result
    if isinstance(err, driver.CUresult):
        if err != driver.CUresult.CUDA_SUCCESS:
            _, name = driver.cuGetErrorName(err)
            raise RuntimeError(f"CUDA driver error: {name.decode()}")
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"NVRTC error: {err}")
    elif isinstance(err, runtime.cudaError_t):
        if err != runtime.cudaError_t.cudaSuccess:
            _, name = runtime.cudaGetErrorName(err)
            raise RuntimeError(f"CUDA runtime error: {name.decode()}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")
    if not rest:
        return None
    if len(rest) == 1:
        return rest[0]
    return tuple(rest)


# ── Device + context ────────────────────────────────────────────────────────

def init_cuda(device_id: int = 0):
    """Init the CUDA driver, pick a device, and bring up its primary context.

    Returns (device, ctx).  We use the **primary context** rather than
    `cuCtxCreate` because (a) its signature has shifted across recent
    cuda-python versions, and (b) the primary context plays nicely with
    anything else using CUDA in the same process (PyTorch, etc.).

    Caller is responsible for releasing the primary context with
    `cu(driver.cuDevicePrimaryCtxRelease(device))` at the end.
    """
    cu(driver.cuInit(0))
    device = cu(driver.cuDeviceGet(device_id))
    ctx = cu(driver.cuDevicePrimaryCtxRetain(device))
    cu(driver.cuCtxSetCurrent(ctx))
    return device, ctx


def compute_arch(device) -> str:
    """Return e.g. 'sm_100a' for the current device (suitable for NVRTC)."""
    major = cu(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device))
    minor = cu(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device))
    return f"sm_{major}{minor}a"


# ── nvcc compile (with mtime-based cubin cache) ─────────────────────────────

def compile_kernel(src_path: str, device, kernels: list, extra_opts: list = None):
    """Compile a .cu file via `nvcc --cubin` and resolve named kernels.

    The cubin is cached on disk next to the .cu file (suffix
    `_sm_XYZa.cubin`).  Re-compile happens only when the .cu's mtime
    is newer than the cubin's — so repeated runs are fast.

    nvcc is used (rather than NVRTC) so the kernel can include
    standard CUDA headers (`<cuda.h>`, `<cuda_bf16.h>`, `<cstdint>`,
    etc.) without manual workarounds.

    Returns (module, {kernel_name: CUfunction}).
    """
    arch = compute_arch(device)
    cubin_path = src_path[:-3] + f"_{arch}.cubin"

    needs_rebuild = (not os.path.exists(cubin_path)
                     or os.path.getmtime(src_path) > os.path.getmtime(cubin_path))
    if needs_rebuild:
        print(f"[nvcc] compiling {os.path.basename(src_path)} → "
              f"{os.path.basename(cubin_path)} ... ",
              end="", flush=True)
        nvcc = os.environ.get("NVCC", "nvcc")
        cmd = [nvcc, f"-arch={arch}", "-O3", "--std=c++17", "--cubin"]
        if extra_opts:
            cmd.extend(extra_opts)
        cmd += [src_path, "-o", cubin_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write(r.stderr)
            raise RuntimeError(f"nvcc failed (exit {r.returncode})")
        print("done", flush=True)

    with open(cubin_path, "rb") as f:
        cubin = f.read()
    module = cu(driver.cuModuleLoadData(cubin))
    fns = {k: cu(driver.cuModuleGetFunction(module, k.encode()))
           for k in kernels}
    return module, fns


# ── Memory I/O ──────────────────────────────────────────────────────────────

def htod(host: np.ndarray) -> int:
    """Allocate device memory matching `host.nbytes` and copy host → device."""
    d = cu(driver.cuMemAlloc(host.nbytes))
    cu(driver.cuMemcpyHtoD(d, host.ctypes.data, host.nbytes))
    return d


def dtoh(d: int, nbytes: int, dtype) -> np.ndarray:
    """Allocate a host ndarray of `dtype` and copy `nbytes` from `d`."""
    out = np.empty(nbytes // np.dtype(dtype).itemsize, dtype=dtype)
    cu(driver.cuMemcpyDtoH(out.ctypes.data, d, nbytes))
    return out


# ── Launch ──────────────────────────────────────────────────────────────────

def launch(kernel, *, grid, block, shared: int, args: list, stream: int = 0,
           sync: bool = True):
    """Launch a kernel.

    `args` is a list of ctypes objects (one per kernel parameter).
    For by-value structs, pass a `(ctypes.c_byte * N).from_buffer_copy(bytes)`.
    For pointers, pass a `ctypes.c_void_p(int_address)`.

    `sync=True` (default) blocks the host until the kernel completes —
    convenient for chapter examples that read C right after launching.
    Timing harnesses should pass `sync=False` and synchronize once at
    the end of a batch; otherwise the per-launch sync (~5–10 µs round
    trip) inflates small-shape timings significantly.
    """
    arg_ptrs = (ctypes.c_void_p * len(args))(
        *[ctypes.addressof(a) for a in args]
    )
    cu(driver.cuLaunchKernel(
        kernel,
        *grid,
        *block,
        shared,
        stream,
        arg_ptrs,
        0,    # extra (unused)
    ))
    if sync:
        cu(driver.cuCtxSynchronize())
