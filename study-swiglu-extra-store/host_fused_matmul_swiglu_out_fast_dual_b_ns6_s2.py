#!/usr/bin/env python3
"""Clean host launcher for fused_matmul_swiglu_out_fast_dual_b_ns6_s2.cu.

This is the shareable dual-B fused GEMM+SwiGLU study kernel:

  C[M, N]     = packed wide GEMM output, per BN=256 tile as [left128 | gate128]
  D[M, N / 2] = left * silu(gate)

The kernel loads left and gate from two separate B tensors:

  B_left[K, N / 2]
  B_gate[K, N / 2]

Run from the repo root on a B200 allocation, for example:

  srunpy study-swiglu-extra-store/host_fused_matmul_swiglu_out_fast_dual_b_ns6_s2.py
"""

from __future__ import annotations

import argparse
import ctypes
import os
import pathlib
import subprocess

import numpy as np
import torch
from cuda.bindings import driver


HERE = pathlib.Path(__file__).resolve().parent
KERNEL_SOURCE = HERE / "fused_matmul_swiglu_out_fast_dual_b_ns6_s2.cu"
KERNEL_SYMBOL = "matmul_cluster"

# Must match the constexprs in fused_matmul_swiglu_out_fast_dual_b_ns6_s2.cu.
BM = 128
BN = 256
BK = 64
NS = 6
NUM_WARPS = 4
CTA_GROUP = 2
STORE_N = 64
TMA_STORE_STAGES = 2
BF16_BYTES = 2


# Minimal standalone CUDA-driver/TMA helpers. These mirror the small subset of
# webui/kernels/_runtime.py used by the MMComposer generated host scripts.
TMA_BFLOAT16 = 9
TMA_INTERLEAVE_NONE = 0
TMA_SWIZZLE_128B = 3
TMA_L2_NONE = 0
TMA_OOB_NONE = 0

_cu_tensor_map_encode_tiled = None


def cu_tensor_map_encode_tiled():
    global _cu_tensor_map_encode_tiled
    if _cu_tensor_map_encode_tiled is None:
        last_error = None
        for soname in ("libcuda.so", "libcuda.so.1"):
            try:
                libcuda = ctypes.CDLL(soname, mode=ctypes.RTLD_GLOBAL)
                break
            except OSError as exc:
                last_error = exc
        else:
            raise RuntimeError(
                "Could not load libcuda.so. Run this script inside a CUDA GPU "
                "allocation/environment."
            ) from last_error

        fn = libcuda.cuTensorMapEncodeTiled
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        _cu_tensor_map_encode_tiled = fn
    return _cu_tensor_map_encode_tiled


def cu(result):
    err, *rest = result
    if err != driver.CUresult.CUDA_SUCCESS:
        _, name = driver.cuGetErrorName(err)
        if isinstance(name, bytes):
            name = name.decode()
        raise RuntimeError(f"CUDA driver error: {name}")
    if not rest:
        return None
    if len(rest) == 1:
        return rest[0]
    return tuple(rest)


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
    swizzle: int = TMA_SWIZZLE_128B,
    l2_promotion: int = TMA_L2_NONE,
    oob_fill: int = TMA_OOB_NONE,
) -> np.ndarray:
    if global_strides is None:
        global_strides = []
    if len(global_dim) != rank:
        raise ValueError(f"global_dim must have {rank} entries")
    if len(box_dim) != rank:
        raise ValueError(f"box_dim must have {rank} entries")
    if len(element_strides) != rank:
        raise ValueError(f"element_strides must have {rank} entries")
    if len(global_strides) != rank - 1:
        raise ValueError(f"global_strides must have {rank - 1} entries")

    tmap = np.zeros(128, dtype=np.uint8)
    gdim_arr = (ctypes.c_uint64 * rank)(*global_dim)
    bdim_arr = (ctypes.c_uint32 * rank)(*box_dim)
    estr_arr = (ctypes.c_uint32 * rank)(*element_strides)
    gstr_arr = (
        (ctypes.c_uint64 * (rank - 1))(*global_strides)
        if rank > 1 else None
    )
    err = cu_tensor_map_encode_tiled()(
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


def init_cuda(device_id: int):
    cu(driver.cuInit(0))
    device = cu(driver.cuDeviceGet(device_id))
    ctx = cu(driver.cuDevicePrimaryCtxRetain(device))
    cu(driver.cuCtxSetCurrent(ctx))
    return device, ctx


def compute_arch(device) -> str:
    major = cu(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        )
    )
    minor = cu(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device,
        )
    )
    return f"sm_{major}{minor}a"


def compile_kernel(src_path: pathlib.Path, device, kernels: list[str]):
    arch = compute_arch(device)
    cubin_path = src_path.with_name(src_path.stem + f"_{arch}.cubin")
    needs_rebuild = (
        not cubin_path.exists()
        or src_path.stat().st_mtime > cubin_path.stat().st_mtime
    )
    if needs_rebuild:
        nvcc = os.environ.get("NVCC", "nvcc")
        cmd = [
            nvcc,
            f"-arch={arch}",
            "-O3",
            "--std=c++17",
            "--cubin",
            str(src_path),
            "-o",
            str(cubin_path),
        ]
        print(f"[nvcc] compiling {src_path.name} -> {cubin_path.name} ... ", end="", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"nvcc failed:\n{proc.stderr}")
        print("done", flush=True)

    cubin = cubin_path.read_bytes()
    module = cu(driver.cuModuleLoadData(cubin))
    fns = {name: cu(driver.cuModuleGetFunction(module, name.encode())) for name in kernels}
    return module, fns


def launch(kernel, *, grid, block, shared: int, args: list, stream: int = 0, sync: bool = True):
    arg_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.addressof(arg) for arg in args])
    cu(driver.cuLaunchKernel(kernel, *grid, *block, shared, stream, arg_ptrs, 0))
    if sync:
        cu(driver.cuCtxSynchronize())


def time_kernel_us(call_fn, warmup_ms: int, rep_ms: int) -> float:
    import triton.testing

    ms_med, _, _ = triton.testing.do_bench(
        call_fn,
        warmup=warmup_ms,
        rep=rep_ms,
        quantiles=(0.5, 0.0, 1.0),
    )
    return ms_med * 1000.0


def parse_shape(spec: str) -> tuple[int, int, int]:
    parts = spec.lower().replace(",", "x").split("x")
    if len(parts) == 1:
        n = int(parts[0])
        return n, n, n
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "shape must be N or MxNxK, e.g. 4096 or 32768x4608x768"
        )
    return tuple(int(x) for x in parts)  # type: ignore[return-value]


def validate_shape(m: int, n: int, k: int) -> None:
    problems = []
    if m % (CTA_GROUP * BM):
        problems.append(f"M must be divisible by {CTA_GROUP * BM}")
    if n % BN:
        problems.append(f"N must be divisible by {BN}")
    if k % BK:
        problems.append(f"K must be divisible by {BK}")
    if n % 2:
        problems.append("N must be even for the dual-B split")
    if problems:
        raise ValueError(f"unsupported shape {m}x{n}x{k}: " + "; ".join(problems))


def shared_bytes() -> int:
    bn_local = BN // CTA_GROUP
    a_slot = BM * BK * BF16_BYTES
    b_slot = bn_local * BK * BF16_BYTES
    compute_ring = NS * (a_slot + b_slot)
    epilogue_ring = BM * STORE_N * BF16_BYTES * TMA_STORE_STAGES
    return compute_ring + epilogue_ring + 1024


def launch_spec(device) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
    num_sms = cu(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            device,
        )
    )
    grid_x = num_sms - (num_sms % CTA_GROUP)
    block_x = (NUM_WARPS + 4) * 32
    return (grid_x, 1, 1), (block_x, 1, 1), shared_bytes()


def tensor_map_arg(tmap):
    return (ctypes.c_byte * 128).from_buffer_copy(tmap.tobytes())


def build_packed_b(b_left: torch.Tensor, b_gate: torch.Tensor, n: int) -> torch.Tensor:
    """Interleave separate B halves into the packed BN-tiled layout."""
    k = b_left.shape[0]
    b_packed = torch.empty((k, n), dtype=torch.bfloat16, device=b_left.device)
    packed = b_packed.view(k, n // BN, BN)
    packed[:, :, : BN // 2] = b_left.view(k, n // BN, BN // 2)
    packed[:, :, BN // 2 :] = b_gate.view(k, n // BN, BN // 2)
    return b_packed


def build_inputs(m: int, n: int, k: int, seed: int, need_reference: bool):
    torch.manual_seed(seed)
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b_left = torch.randn((k, n // 2), dtype=torch.bfloat16, device="cuda")
    b_gate = torch.randn((k, n // 2), dtype=torch.bfloat16, device="cuda")
    c = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    d = torch.empty((m, n // 2), dtype=torch.bfloat16, device="cuda")

    b_packed = None
    c_ref = None
    d_ref = None
    if need_reference:
        b_packed = build_packed_b(b_left, b_gate, n)
        left_ref = torch.mm(a, b_left)
        gate_ref = torch.mm(a, b_gate)
        c_ref = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        c_view = c_ref.view(m, n // BN, BN)
        c_view[:, :, : BN // 2] = left_ref.view(m, n // BN, BN // 2)
        c_view[:, :, BN // 2 :] = gate_ref.view(m, n // BN, BN // 2)
        d_ref = (
            left_ref.float()
            * (gate_ref.float() * torch.sigmoid(gate_ref.float()))
        ).to(torch.bfloat16)
    return a, b_left, b_gate, c, d, b_packed, c_ref, d_ref


def encode_maps(a, b_left, b_gate, c, d, m: int, n: int, k: int):
    a_tmap = encode_tensor_map(
        dtype=TMA_BFLOAT16,
        rank=2,
        gptr=a.data_ptr(),
        global_dim=[k, m],
        global_strides=[k * BF16_BYTES],
        box_dim=[BK, BM],
        element_strides=[1, 1],
        swizzle=TMA_SWIZZLE_128B,
    )
    b_left_tmap = encode_tensor_map(
        dtype=TMA_BFLOAT16,
        rank=2,
        gptr=b_left.data_ptr(),
        global_dim=[n // 2, k],
        global_strides=[(n // 2) * BF16_BYTES],
        box_dim=[STORE_N, BK],
        element_strides=[1, 1],
        swizzle=TMA_SWIZZLE_128B,
    )
    b_gate_tmap = encode_tensor_map(
        dtype=TMA_BFLOAT16,
        rank=2,
        gptr=b_gate.data_ptr(),
        global_dim=[n // 2, k],
        global_strides=[(n // 2) * BF16_BYTES],
        box_dim=[STORE_N, BK],
        element_strides=[1, 1],
        swizzle=TMA_SWIZZLE_128B,
    )
    c_tmap = encode_tensor_map(
        dtype=TMA_BFLOAT16,
        rank=2,
        gptr=c.data_ptr(),
        global_dim=[n, m],
        global_strides=[n * BF16_BYTES],
        box_dim=[STORE_N, BM],
        element_strides=[1, 1],
        swizzle=TMA_SWIZZLE_128B,
    )
    d_tmap = encode_tensor_map(
        dtype=TMA_BFLOAT16,
        rank=2,
        gptr=d.data_ptr(),
        global_dim=[n // 2, m],
        global_strides=[(n // 2) * BF16_BYTES],
        box_dim=[STORE_N, BM],
        element_strides=[1, 1],
        swizzle=TMA_SWIZZLE_128B,
    )
    return a_tmap, b_left_tmap, b_gate_tmap, c_tmap, d_tmap


def rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (got.float() - ref.float()).abs().max().item()
    denom = ref.float().abs().max().item()
    return diff if denom == 0.0 else diff / denom


def tflops(m: int, n: int, k: int, us: float) -> float:
    return (2.0 * m * n * k) / (us * 1e-6) / 1e12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shape", nargs="?", type=parse_shape, default=parse_shape("32768x4608x768"))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-ms", type=int, default=1000)
    ap.add_argument("--rep-ms", type=int, default=1000)
    ap.add_argument("--no-check", action="store_true", help="skip reference tensors and correctness check")
    ap.add_argument("--no-bench", action="store_true", help="skip kernel timing")
    ap.add_argument("--no-cublas", action="store_true", help="skip packed wide GEMM cuBLAS timing")
    args = ap.parse_args()

    m, n, k = args.shape
    validate_shape(m, n, k)
    if not KERNEL_SOURCE.exists():
        raise FileNotFoundError(KERNEL_SOURCE)

    need_reference = (not args.no_check) or ((not args.no_bench) and (not args.no_cublas))
    device, _ctx = init_cuda(args.device)
    arch = compute_arch(device)
    grid, block, shared = launch_spec(device)
    max_shared = cu(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            device,
        )
    )

    print(f"kernel: {KERNEL_SOURCE.name}")
    print(f"shape:  M={m} N={n} K={k}")
    print(f"config: BM={BM} BN={BN} BK={BK} NS={NS} TMA_STORE_STAGES={TMA_STORE_STAGES}")
    print(f"launch: grid={grid} block={block} shared={shared} bytes (max opt-in {max_shared})")
    print(f"arch:   {arch}")

    module = None
    try:
        module, kernels = compile_kernel(KERNEL_SOURCE, device, [KERNEL_SYMBOL])
        kernel = kernels[KERNEL_SYMBOL]
        cu(
            driver.cuFuncSetAttribute(
                kernel,
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared,
            )
        )

        a, b_left, b_gate, c, d, b_packed, c_ref, d_ref = build_inputs(
            m, n, k, args.seed, need_reference
        )
        maps = encode_maps(a, b_left, b_gate, c, d, m, n, k)
        launch_args = [
            tensor_map_arg(maps[0]),
            tensor_map_arg(maps[1]),
            tensor_map_arg(maps[2]),
            tensor_map_arg(maps[3]),
            tensor_map_arg(maps[4]),
            ctypes.c_void_p(c.data_ptr()),
            ctypes.c_int(m),
            ctypes.c_int(n),
            ctypes.c_int(k),
        ]

        c.zero_()
        d.zero_()
        torch.cuda.synchronize()
        # Descriptors and ctypes argument wrappers are built once above and
        # reused for all launches. The timed loop below does not re-encode TMA
        # tensor maps.
        launch(kernel, grid=grid, block=block, shared=shared, args=launch_args)

        if not args.no_check:
            assert c_ref is not None and d_ref is not None
            c_rel = rel_err(c, c_ref)
            d_rel = rel_err(d, d_ref)
            print(f"check:  C rel_err={c_rel:.3e}  D rel_err={d_rel:.3e}")
            if c_rel >= 5e-2 or d_rel >= 5e-2:
                print("check:  FAILED")
                return 1
            print("check:  OK")

        if not args.no_bench:
            us = time_kernel_us(
                lambda: launch(
                    kernel, grid=grid, block=block, shared=shared,
                    args=launch_args, sync=False,
                ),
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
            )
            print(f"kernel: {us:.3f} us, {tflops(m, n, k, us):.1f} TFLOPS")

            if not args.no_cublas:
                if b_packed is None:
                    b_packed = build_packed_b(b_left, b_gate, n)
                cublas_us = time_kernel_us(
                    lambda: torch.mm(a, b_packed),
                    warmup_ms=args.warmup_ms,
                    rep_ms=args.rep_ms,
                )
                print(f"cuBLAS: {cublas_us:.3f} us, {tflops(m, n, k, cublas_us):.1f} TFLOPS")
    finally:
        if module is not None:
            try:
                cu(driver.cuModuleUnload(module))
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
