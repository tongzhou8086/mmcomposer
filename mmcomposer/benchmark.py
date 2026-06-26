"""Benchmarking --- the `benchmark` module from DESIGN.md.

A pure ``do_bench``-style timer over a **callable**.  It knows nothing about
kernels, so it times anything: a generated kernel's entry point, cuBLAS
(``torch.mm``), ``torch.matmul``, etc.  Compilation happens elsewhere (the
`compile`/`runtime` modules); this only measures.

Correctness is kept SEPARATE from timing: `rel_error(out, ref)` is the companion
check the orchestrator runs to decide whether a combo counts.

Public API:
    gemm_flops(M, N, K) -> float                         # pure
    tflops_from_us(flops, us) -> float                   # pure
    rel_error(out, ref) -> float                         # torch, no GPU needed
    time_us(fn, warmup_ms, rep_ms) -> float              # GPU (do_bench)
    benchmark(fn, *, flops=None, ...) -> BenchResult     # GPU
    benchmark_median(fn, *, flops=None, samples=3, ...) -> BenchResult   # GPU, robust
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

DEFAULT_WARMUP_MS = 300
DEFAULT_REP_MS = 200

# _runtime (which dlopens libcuda at import) is loaded lazily, so this module and
# its pure helpers import fine on a GPU-less machine.
_RT = None


def _rt():
    global _RT
    if _RT is None:
        import pathlib
        import sys
        k = pathlib.Path(__file__).resolve().parent / "kernels"
        if str(k) not in sys.path:
            sys.path.insert(0, str(k))
        import _runtime
        _RT = _runtime
    return _RT


@dataclass
class BenchResult:
    latency_us: float
    tflops: float | None = None


# ---- pure arithmetic (CPU-testable) ---------------------------------------
def gemm_flops(M: int, N: int, K: int) -> float:
    """FLOPs of an M×N×K matmul (2 per multiply-add)."""
    return 2.0 * M * N * K


def tflops_from_us(flops: float, us: float) -> float:
    """Achieved TFLOP/s given total flops and per-call time in microseconds."""
    return flops / (us * 1e-6) / 1e12


def rel_error(out, ref) -> float:
    """Relative L2 error ``||out - ref|| / ||ref||``, computed in fp32."""
    o, r = out.float(), ref.float()
    return ((o - r).norm() / r.norm()).item()


# ---- timing (GPU; one op per call) ----------------------------------------
def time_us(fn, warmup_ms: int = DEFAULT_WARMUP_MS, rep_ms: int = DEFAULT_REP_MS) -> float:
    """Median per-call time of `fn` in microseconds, via do_bench (L2-flushed).

    `fn` must perform exactly one op per call; for a kernel launch, do the launch
    with ``sync=False`` and let do_bench handle synchronization.
    """
    return _rt().time_kernel_us(fn, warmup_ms=warmup_ms, rep_ms=rep_ms)


def benchmark(fn, *, flops: float | None = None,
              warmup_ms: int = DEFAULT_WARMUP_MS,
              rep_ms: int = DEFAULT_REP_MS) -> BenchResult:
    """Time `fn` once (median over a do_bench window); attach TFLOP/s if `flops`."""
    us = time_us(fn, warmup_ms, rep_ms)
    return BenchResult(latency_us=us,
                       tflops=tflops_from_us(flops, us) if flops else None)


def benchmark_median(fn, *, flops: float | None = None, samples: int = 3,
                     warmup_samples: int = 1,
                     warmup_ms: int = DEFAULT_WARMUP_MS,
                     rep_ms: int = DEFAULT_REP_MS) -> BenchResult:
    """Robust timing: discard `warmup_samples` do_bench samples (boost-clock
    outliers), then take the median of `samples`.  Useful for the reference."""
    for _ in range(warmup_samples):
        time_us(fn, warmup_ms, rep_ms)
    us = statistics.median(time_us(fn, warmup_ms, rep_ms) for _ in range(samples))
    return BenchResult(latency_us=us,
                       tflops=tflops_from_us(flops, us) if flops else None)
