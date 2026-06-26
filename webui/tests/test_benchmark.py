#!/usr/bin/env python3
"""Unit tests for the benchmark module (webui/benchmark.py).

CPU layer: pure helpers (gemm_flops, tflops_from_us, rel_error).
GPU layer: time a real op via do_bench and check sane TFLOPS; skipped without CUDA.

Run:  python webui/tests/test_benchmark.py   (or pytest)
"""
import math
import pathlib
import sys

WEBUI = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEBUI))
sys.path.insert(0, str(WEBUI.parent))  # repo root for mmcomposer
sys.path.insert(0, str(WEBUI / "kernels"))

from mmcomposer import benchmark as bench


def test_gemm_flops():
    assert bench.gemm_flops(128, 256, 64) == 2.0 * 128 * 256 * 64


def test_tflops_from_us():
    # 2e9 flops in 1000 us = 2e9 / 1e-3 / 1e12 = 2.0 TFLOP/s
    assert math.isclose(bench.tflops_from_us(2e9, 1000.0), 2.0, rel_tol=1e-9)
    # inverse relationship: half the time -> double the TFLOPS
    assert math.isclose(bench.tflops_from_us(2e9, 500.0), 4.0, rel_tol=1e-9)


def test_rel_error():
    import torch
    x = torch.randn(64, 64)
    assert bench.rel_error(x, x) == 0.0
    assert bench.rel_error(x + 1.0, x) > 0.0


def test_benchmark_times_cublas_with_sane_tflops():
    import torch
    if not torch.cuda.is_available():
        print("    SKIP (no CUDA)")
        return
    M = N = K = 2048
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    r = bench.benchmark(lambda: torch.mm(a, b), flops=bench.gemm_flops(M, N, K))
    print(f"    cuBLAS {M}^3: {r.latency_us:.1f} us, {r.tflops:.0f} TFLOPS")
    assert r.latency_us > 0
    assert r.tflops > 100        # B200 bf16 cuBLAS is many hundreds of TFLOPS
    # robust median variant returns a comparable number
    rm = bench.benchmark_median(lambda: torch.mm(a, b),
                                flops=bench.gemm_flops(M, N, K), samples=3)
    assert rm.tflops > 100


def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
