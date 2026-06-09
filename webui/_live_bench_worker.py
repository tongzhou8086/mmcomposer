"""Live-benchmark worker — runs on a B200 compute node (invoked via srun by
live_bench.py).  Compiles an already-rendered kernel.cu, runs it + cuBLAS at
one (M,N,K), and writes {us, tflops, cublas_tflops, vs_cublas, rel_err} JSON.

Decoupled from mvp_core on purpose: the app (jump node, no GPU) renders the
kernel; this worker only needs torch + cuda-python + nvcc.
"""
import os, sys, json, argparse, ctypes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "kernels"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)      # path to rendered kernel.cu
    ap.add_argument("--symbol", required=True)       # entry function name
    ap.add_argument("--out", required=True)          # result json path
    ap.add_argument("--cluster", type=int, default=0)
    ap.add_argument("--persistent", type=int, default=0)
    ap.add_argument("--overlap", type=int, default=0)
    for k in ("bm", "bn", "bk", "ns", "nw", "tma_store"):
        ap.add_argument(f"--{k}", type=int, required=True)
    ap.add_argument("-M", type=int, required=True)
    ap.add_argument("-N", type=int, required=True)
    ap.add_argument("-K", type=int, required=True)
    a = ap.parse_args()

    res = {"ok": False, "error": None}
    try:
        import torch
        import _runtime as rt
        from cuda.bindings import driver

        device, ctx = rt.init_cuda()
        num_sms = rt.cu(driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device))

        mod, fns = rt.compile_kernel(a.kernel, device, kernels=[a.symbol])
        kernel = fns[a.symbol]

        M, N, K = a.M, a.N, a.K
        cta_group = 2 if a.cluster else 1
        bn_local = a.bn // cta_group
        slot = a.bm * a.bk * 2 + bn_local * a.bk * 2
        epi = a.bm * (a.bn if a.tma_store else a.bn + 8) * 2
        shared = (a.ns * slot + epi if a.overlap else max(a.ns * slot, epi)) + 1024
        block = (a.nw * 32, 1, 1)
        if a.persistent:
            grid = (num_sms - num_sms % cta_group, 1, 1)
        elif a.cluster:
            grid = ((M // (cta_group * a.bm)) * (N // a.bn) * cta_group, 1, 1)
        else:
            grid = ((M // a.bm) * (N // a.bn), 1, 1)

        rt.cu(driver.cuFuncSetAttribute(
            kernel, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared))

        torch.manual_seed(0)
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        A_t = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=A.data_ptr(),
            global_dim=[K, M], global_strides=[K * 2], box_dim=[a.bk, a.bm],
            element_strides=[1, 1], swizzle=rt.TMA_SWIZZLE_128B)
        B_t = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=B.data_ptr(),
            global_dim=[N, K], global_strides=[N * 2], box_dim=[64, a.bk],
            element_strides=[1, 1], swizzle=rt.TMA_SWIZZLE_128B)
        C_t = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=C.data_ptr(),
            global_dim=[N, M], global_strides=[N * 2], box_dim=[a.bn, a.bm],
            element_strides=[1, 1], swizzle=rt.TMA_SWIZZLE_NONE)
        args = [(ctypes.c_byte * 128).from_buffer_copy(x.tobytes()) for x in (A_t, B_t, C_t)] + \
               [ctypes.c_void_p(C.data_ptr()), ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)]

        C.zero_()
        rt.launch(kernel, grid=grid, block=block, shared=shared, args=args)
        ref = (A.float() @ B.float()).to(torch.bfloat16)
        rel = (C.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()

        # Longer do_bench window for a robust per-click number: do_bench picks
        # its own iter count to fill rep_ms, flushes L2 between reps, and returns
        # the median — so a longer rep is just more samples in one call (cleaner
        # than an outer median).  Intra-allocation jitter (esp. cuBLAS) settles;
        # inter-allocation boost-clock draw still applies (fresh B200 per srun).
        us = rt.time_kernel_us(lambda: rt.launch(kernel, grid=grid, block=block,
                                                 shared=shared, args=args, sync=False),
                               warmup_ms=50, rep_ms=500)
        cub_us = rt.time_kernel_us(lambda: torch.mm(A, B), warmup_ms=50, rep_ms=500)
        flops = 2.0 * M * N * K
        res.update(ok=(rel < 5e-2), rel_err=rel, us=us, tflops=flops / (us * 1e-6) / 1e12,
                   cublas_us=cub_us, cublas_tflops=flops / (cub_us * 1e-6) / 1e12,
                   num_sms=num_sms, grid=grid[0])
        res["vs_cublas"] = res["tflops"] / res["cublas_tflops"] if res["cublas_tflops"] else None
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"

    with open(a.out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
