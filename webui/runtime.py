"""Kernel execution --- the `runtime` module from DESIGN.md.

Turns "a compiled cubin + a config + tensors" into a GPU launch.  Imported and
called (this is the execution counterpart of the generated standalone host).  It
does NOT compile (that's the `compile` module) and does NOT generate code; it
assumes the cubin already exists.

Public API:
    config_from_combo(tier, k) -> config      # merge enumerate's (tier, knobs)
    launch_dims(config, M, N, K)              -> (grid, block, shared)   # pure, no GPU
    setup_and_launch(config, M, N, K, a, b, c, cubin_path) -> c
    kernel(config, cubin_path) -> callable    # k(a, b[, c]) -> c, with caches

The kernel() callable caches the loaded module per process and (re)builds the
small per-launch state; the cubin is loaded from disk once.
"""
from __future__ import annotations

import ctypes
import pathlib
import sys

_KERNELS = pathlib.Path(__file__).resolve().parent / "kernels"
if str(_KERNELS) not in sys.path:
    sys.path.insert(0, str(_KERNELS))

STORE_N = 64        # fixed TMA-store chunk width
ELEM_BYTES = 2      # bf16

# Low-level driver primitives (_runtime) + cuda-python are imported LAZILY so
# this module -- and the pure launch_dims -- import fine on a machine with no GPU
# / no libcuda.  They're only needed once we actually load or launch a kernel.
_RT = None
_DRIVER = None
_device = None
_num_sms = None


def _backends():
    global _RT, _DRIVER
    if _RT is None:
        import _runtime as rt
        from cuda.bindings import driver
        _RT, _DRIVER = rt, driver
    return _RT, _DRIVER


def _ensure_cuda():
    global _device, _num_sms
    rt, driver = _backends()
    if _device is None:
        _device, _ = rt.init_cuda()
        _num_sms = rt.cu(driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, _device))
    return _device, _num_sms


def config_from_combo(tier, k) -> dict:
    """Merge enumerate's (tier, knobs) into a flat config the runtime understands."""
    return {**k, "symbol": tier["symbol"], "cluster": tier["cluster"]}


# ---- launch geometry (pure; mirrors gpu_codegen_driver.launch_spec) --------
def launch_dims(config, M, N, K, num_sms=None):
    """Return (grid, block, shared_bytes).  Pure / no GPU for the non-persistent
    path; `num_sms` is only consulted for persistent grids (defaults to the
    queried SM count if CUDA is up)."""
    c = config
    cta_group = 2 if c["cluster"] else 1
    bn_local = c["bn"] // cta_group
    a_slot = c["bm"] * c["bk"] * 2
    b_slot = bn_local * c["bk"] * 2
    slot = a_slot + b_slot
    if c.get("overlap", 0) and c.get("tma_pipelined", 0):
        epi = c["bm"] * 64 * 2 * c.get("tma_store_stages", 2)
    elif c.get("overlap", 0) and c["cluster"] and c.get("split_epilogue", 0):
        epi = c["bm"] * (c["bn"] // 2 + 8) * 2
    else:
        epi = c["bm"] * (c["bn"] + 8) * 2
    shared = ((c["ns"] * slot + epi) if c.get("overlap", 0)
              else max(c["ns"] * slot, epi)) + 1024
    block = (((c["nw"] + 4) * 32 if c.get("overlap", 0) else c["nw"] * 32), 1, 1)
    if c.get("persistent") and num_sms:
        grid = (num_sms - num_sms % cta_group, 1, 1)
    elif c["cluster"]:
        grid = ((M // (cta_group * c["bm"])) * (N // c["bn"]) * cta_group, 1, 1)
    else:
        grid = ((M // c["bm"]) * (N // c["bn"]), 1, 1)
    return grid, block, shared


# ---- module load cache -----------------------------------------------------
_MODULE_CACHE: dict = {}   # (cubin_path, symbol) -> (module, function)


def _load(cubin_path, symbol):
    rt, driver = _backends()
    key = (cubin_path, symbol)
    if key not in _MODULE_CACHE:
        with open(cubin_path, "rb") as f:
            data = f.read()
        mod = rt.cu(driver.cuModuleLoadData(data))
        fn = rt.cu(driver.cuModuleGetFunction(mod, symbol.encode()))
        _MODULE_CACHE[key] = (mod, fn)
    return _MODULE_CACHE[key]


def _descriptors(config, M, N, K, a, b, c):
    """Build the A/B/C TMA descriptors (np arrays).  A & B K-major, C row-major."""
    rt, _ = _backends()
    bm, bn, bk = config["bm"], config["bn"], config["bk"]
    A = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=a.data_ptr(),
                             global_dim=[K, M], global_strides=[K * ELEM_BYTES],
                             box_dim=[bk, bm], element_strides=[1, 1],
                             swizzle=rt.TMA_SWIZZLE_128B)
    B = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=b.data_ptr(),
                             global_dim=[N, K], global_strides=[N * ELEM_BYTES],
                             box_dim=[64, bk], element_strides=[1, 1],
                             swizzle=rt.TMA_SWIZZLE_128B)
    if config.get("tma_pipelined", 0):
        C = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=c.data_ptr(),
                                 global_dim=[N, M], global_strides=[N * ELEM_BYTES],
                                 box_dim=[STORE_N, bm], element_strides=[1, 1],
                                 swizzle=rt.TMA_SWIZZLE_128B)
    else:
        C = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=c.data_ptr(),
                                 global_dim=[N, M], global_strides=[N * ELEM_BYTES],
                                 box_dim=[bn, bm], element_strides=[1, 1],
                                 swizzle=rt.TMA_SWIZZLE_NONE)
    return A, B, C


def setup_and_launch(config, M, N, K, a, b, c, cubin_path, *, sync=True):
    """Load (cached), build descriptors, set SMEM attr, and launch.  Returns c."""
    rt, driver = _backends()
    _, num_sms = _ensure_cuda()
    _, fn = _load(cubin_path, config["symbol"])
    grid, block, shared = launch_dims(config, M, N, K, num_sms=num_sms)
    rt.cu(driver.cuFuncSetAttribute(
        fn, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared))
    A, B, C = _descriptors(config, M, N, K, a, b, c)
    args = [(ctypes.c_byte * 128).from_buffer_copy(d.tobytes()) for d in (A, B, C)]
    args += [ctypes.c_void_p(c.data_ptr()),
             ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)]
    rt.launch(fn, grid=grid, block=block, shared=shared, args=args, sync=sync)
    return c


def kernel(config, cubin_path):
    """Return a callable ``k(a, b, c=None) -> c`` for this (already-compiled)
    config.  The module is loaded once (cached); pass `c` to reuse an output
    buffer, else a fresh bf16 output is allocated each call."""
    import torch

    def call(a, b, c=None, *, sync=True):
        M, Ka = a.shape
        Kb, N = b.shape
        assert Ka == Kb, f"inner dims disagree: {a.shape} @ {b.shape}"
        if c is None:
            c = torch.zeros(M, N, dtype=torch.bfloat16, device=a.device)
        return setup_and_launch(config, M, N, Ka, a, b, c, cubin_path, sync=sync)

    return call
