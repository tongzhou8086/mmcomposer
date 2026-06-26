"""MMComposer -- a CUDA matmul kernel generator + autotuner for Blackwell (B200).

    import mmcomposer as mmc
    c = mmc.matmul(a, b)               # auto-tunes + caches on first sight of a shape
    gemm = mmc.get_tuned_kernel(a, b)  # reusable callable for that shape
    mmc.tune(M, N, K)                  # explicit offline pre-tune

The whole core lives in this package -- a self-contained, modular pipeline
(see mmcomposer/DESIGN.md):

    enumerate (combos) -> codegen -> compile -> runtime -> benchmark -> cache
                                                          (leaderboard renders)

`autotune.tune` orchestrates the sweep in-process; `mmc.matmul`/`get_tuned_kernel`
serve the best cached kernel (auto-tuning on a cold shape).
"""
from . import (compiler, cache, leaderboard, mvp_core, combos,  # noqa: F401
               runtime, benchmark, codegen, autotune)
from .mmc import matmul, get_tuned_kernel, tune  # noqa: F401

__all__ = ["matmul", "get_tuned_kernel", "tune",
           "combos", "compiler", "runtime", "benchmark",
           "cache", "leaderboard", "autotune", "mvp_core", "codegen"]
