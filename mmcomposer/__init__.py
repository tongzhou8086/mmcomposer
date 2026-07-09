"""MMComposer -- a CUDA matmul kernel generator + autotuner for Blackwell (B200).

    import mmcomposer as mmc
    c = mmc.matmul(a, b)               # auto-tunes + caches on first sight of a shape
    gemm = mmc.get_tuned_kernel(a, b)  # reusable callable for that shape
    mmc.tune(M, N, K)                  # explicit offline pre-tune

The whole core lives in this package -- a self-contained, modular pipeline
(see mmcomposer/DESIGN.md):

    enumerate (combos) -> codegen -> compile -> runtime -> benchmark -> cache
                                                          (leaderboard renders)

The leaf modules are imported eagerly; the API (matmul/get_tuned_kernel/tune) and
the autotune orchestrator are exposed lazily via module __getattr__, so importing
the package stays light and `python -m mmcomposer.autotune` runs without a runpy
double-import warning.
"""
import importlib as _importlib

from . import (compiler, cache, leaderboard, mvp_core, combos,  # noqa: F401
               runtime, benchmark, codegen, swiglu, hopper, hopper_swiglu, epilogue)

# defined in mmcomposer.mmc
_API = {"matmul", "get_tuned_kernel", "get_epilogue_kernel", "tune",
        "matmul_swiglu_dual_b", "matmul_swiglu_dual_b_ns6_s2"}

__all__ = ["matmul", "get_tuned_kernel", "get_epilogue_kernel", "tune",
           "matmul_swiglu_dual_b", "matmul_swiglu_dual_b_ns6_s2",
           "combos", "compiler", "runtime", "benchmark", "cache", "leaderboard",
           "autotune", "autotune_isolated", "mvp_core", "codegen", "swiglu",
           "hopper", "hopper_swiglu", "epilogue"]


def __getattr__(name):
    if name in _API:
        return getattr(_importlib.import_module(".mmc", __name__), name)
    if name == "autotune":
        return _importlib.import_module(".autotune", __name__)
    if name == "autotune_isolated":
        return _importlib.import_module(".autotune_isolated", __name__)
    raise AttributeError(f"module 'mmcomposer' has no attribute {name!r}")
