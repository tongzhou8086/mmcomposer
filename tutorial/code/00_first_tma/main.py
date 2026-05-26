"""Runnable companion for Chapter 00 — A first TMA program.

Builds the host-side CUtensorMap, compiles kernel.cu via NVRTC,
launches it on a single CTA with 128 threads, and verifies that
g_out == g_in[:CHUNK_BYTES].

Run:
    pip install -r ../requirements.txt
    python main.py
"""

import os
import sys
import ctypes
import numpy as np

from cuda.bindings import driver, nvrtc, runtime


# ── Constants matching kernel.cu ────────────────────────────────────────────
CHUNK_BYTES = 128
THREADS_PER_CTA = 128


# ── Tiny error-checking helper ──────────────────────────────────────────────
def cu(result):
    """Unwrap a cuda-python return tuple.

    cuda-python's driver/nvrtc bindings return (err, *rest) tuples.
    This helper checks err and returns *rest, or raises.
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
    if len(rest) == 1:
        return rest[0]
    return tuple(rest) if rest else None


# ── 1. Init CUDA + create a context ─────────────────────────────────────────
cu(driver.cuInit(0))
device = cu(driver.cuDeviceGet(0))
ctx = cu(driver.cuCtxCreate(0, device))

# Detect compute capability for NVRTC arch flag.
major = cu(driver.cuDeviceGetAttribute(
    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device))
minor = cu(driver.cuDeviceGetAttribute(
    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device))
arch = f"sm_{major}{minor}a"  # e.g. sm_100a for B200

if major < 10:
    print(f"WARNING: detected {arch}; TMA bulk-tensor is sm_90+, "
          f"and this example targets B200 (sm_100a).", file=sys.stderr)


# ── 2. Compile kernel.cu via NVRTC ──────────────────────────────────────────
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "kernel.cu"), "rb") as f:
    src = f.read()

prog = cu(nvrtc.nvrtcCreateProgram(src, b"kernel.cu", 0, [], []))
opts = [f"--gpu-architecture={arch}".encode(),
        b"-std=c++17",
        b"-default-device"]
opts_c = (ctypes.c_char_p * len(opts))(*opts)
# nvrtcCompileProgram is special: pass options as a list directly.
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
kernel = cu(driver.cuModuleGetFunction(module, b"tma_demo"))


# ── 3. Allocate device buffers ──────────────────────────────────────────────
g_in_host = np.arange(CHUNK_BYTES, dtype=np.uint8)  # 0, 1, 2, ..., 127
g_in_d  = cu(driver.cuMemAlloc(CHUNK_BYTES))
g_out_d = cu(driver.cuMemAlloc(CHUNK_BYTES))
cu(driver.cuMemcpyHtoD(g_in_d, g_in_host.ctypes.data, CHUNK_BYTES))


# ── 4. Build the 1D CUtensorMap ─────────────────────────────────────────────
#
# Describes g_in_d as a 1D array of CHUNK_BYTES uint8 elements, with a
# per-load box covering all CHUNK_BYTES.  No swizzling, no L2 promotion,
# no out-of-bounds fill.

global_dim      = (ctypes.c_uint64 * 1)(CHUNK_BYTES)
box_dim         = (ctypes.c_uint32 * 1)(CHUNK_BYTES)
element_strides = (ctypes.c_uint32 * 1)(1)
# globalStrides is rank - 1 entries; for 1D that's 0 entries → pass NULL.

tmap = driver.CUtensorMap()
cu(driver.cuTensorMapEncodeTiled(
    tmap,
    driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    1,                                                            # rank
    int(g_in_d),                                                  # global address
    global_dim,
    None,                                                         # globalStrides
    box_dim,
    element_strides,
    driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
    driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
    driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
))


# ── 5. Launch ───────────────────────────────────────────────────────────────
#
# The kernel takes the CUtensorMap by VALUE (it's a 128-byte struct),
# so we pass the struct's bytes as the first kernel-argument.

# Pack args: (tmap by-value, g_out pointer).
# cuda-python's launch path wants a list of ctypes pointers.
tmap_bytes = bytes(tmap)
arg_tmap = (ctypes.c_byte * len(tmap_bytes)).from_buffer_copy(tmap_bytes)
arg_gout = ctypes.c_void_p(int(g_out_d))

args = [arg_tmap, arg_gout]
arg_ptrs = (ctypes.c_void_p * len(args))(
    *[ctypes.addressof(a) if isinstance(a, ctypes.Array) else ctypes.addressof(a) for a in args]
)

shared_bytes = CHUNK_BYTES
cu(driver.cuLaunchKernel(
    kernel,
    1, 1, 1,                              # grid (CTAs)
    THREADS_PER_CTA, 1, 1,                # block (threads/CTA)
    shared_bytes,                         # dynamic SMEM bytes
    0,                                    # stream
    arg_ptrs,                             # kernel params
    0,                                    # extra (unused)
))
cu(driver.cuCtxSynchronize())


# ── 6. Copy back + verify ───────────────────────────────────────────────────
g_out_host = np.empty(CHUNK_BYTES, dtype=np.uint8)
cu(driver.cuMemcpyDtoH(g_out_host.ctypes.data, g_out_d, CHUNK_BYTES))

if np.array_equal(g_out_host, g_in_host):
    print(f"✓ TMA load verified: {CHUNK_BYTES} bytes copied correctly via TMA.")
    print(f"  g_in [first 8 bytes]: {g_in_host[:8]}")
    print(f"  g_out[first 8 bytes]: {g_out_host[:8]}")
else:
    print(f"✗ MISMATCH:")
    print(f"  g_in [first 16]: {g_in_host[:16]}")
    print(f"  g_out[first 16]: {g_out_host[:16]}")
    sys.exit(1)


# ── 7. Cleanup ──────────────────────────────────────────────────────────────
cu(driver.cuMemFree(g_in_d))
cu(driver.cuMemFree(g_out_d))
cu(driver.cuModuleUnload(module))
cu(driver.cuCtxDestroy(ctx))
