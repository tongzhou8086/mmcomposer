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
import sys

import numpy as np
from cuda.bindings import driver, nvrtc, runtime


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
    """Init the CUDA driver, pick a device, create a context.

    Returns (device, ctx).  Caller is responsible for ctx cleanup
    via `cu(driver.cuCtxDestroy(ctx))` at the end.
    """
    cu(driver.cuInit(0))
    device = cu(driver.cuDeviceGet(device_id))
    ctx = cu(driver.cuCtxCreate(0, device))
    return device, ctx


def compute_arch(device) -> str:
    """Return e.g. 'sm_100a' for the current device (suitable for NVRTC)."""
    major = cu(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device))
    minor = cu(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device))
    return f"sm_{major}{minor}a"


# ── NVRTC compile ───────────────────────────────────────────────────────────

def compile_kernel(src_path: str, device, kernels: list, extra_opts: list = None):
    """Compile a .cu file via NVRTC and resolve named kernels.

    Returns (module, {kernel_name: CUfunction}).
    """
    with open(src_path, "rb") as f:
        src = f.read()
    prog = cu(nvrtc.nvrtcCreateProgram(src, src_path.encode(), 0, [], []))
    arch = compute_arch(device)
    opts = [f"--gpu-architecture={arch}".encode(),
            b"-std=c++17",
            b"-default-device"]
    if extra_opts:
        opts.extend(o.encode() if isinstance(o, str) else o for o in extra_opts)
    err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        log_size = cu(nvrtc.nvrtcGetProgramLogSize(prog))
        log = bytearray(log_size)
        cu(nvrtc.nvrtcGetProgramLog(prog, log))
        sys.stderr.write(log.decode(errors="replace"))
        raise RuntimeError("NVRTC compile failed")
    cubin_size = cu(nvrtc.nvrtcGetCUBINSize(prog))
    cubin = bytearray(cubin_size)
    cu(nvrtc.nvrtcGetCUBIN(prog, cubin))
    module = cu(driver.cuModuleLoadData(bytes(cubin)))
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

def launch(kernel, *, grid, block, shared: int, args: list, stream: int = 0):
    """Launch a kernel.

    `args` is a list of ctypes objects (one per kernel parameter).
    For by-value structs, pass a `(ctypes.c_byte * N).from_buffer_copy(bytes)`.
    For pointers, pass a `ctypes.c_void_p(int_address)`.
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
    cu(driver.cuCtxSynchronize())
