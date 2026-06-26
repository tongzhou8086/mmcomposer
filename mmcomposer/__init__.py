"""MMComposer -- a CUDA matmul kernel generator + autotuner for Blackwell (B200).

    import mmcomposer as mmc
    c = mmc.matmul(a, b)               # auto-tunes + caches on first sight of a shape
    gemm = mmc.get_tuned_kernel(a, b)  # reusable callable for that shape
    mmc.tune(M, N, K)                  # explicit offline pre-tune

Stage-B migration (see mmcomposer/DESIGN.md): modules are being relocated from
``webui/`` into this package one chunk at a time.  Already relocated leaves are
imported directly; the rest of the API/leaf modules are exposed lazily (via
module __getattr__) so that importing a single relocated submodule does NOT pull
in the whole API -- which would re-enter the webui shims and deadlock.
"""
import importlib as _importlib
import pathlib as _pathlib
import sys as _sys

_WEBUI = _pathlib.Path(__file__).resolve().parent.parent / "webui"
for _p in (_WEBUI, _WEBUI / "kernels"):
    _ps = str(_p)
    if _ps not in _sys.path:
        _sys.path.insert(0, _ps)

# Relocated into the package: import eagerly (all package-internal, CPU-safe;
# the libcuda-touching bits in runtime/benchmark are imported lazily on use).
from . import (compiler, cache, leaderboard, mvp_core, combos,  # noqa: E402,F401
               runtime, benchmark, codegen)

# Still under webui/ (relocated in Chunk 3): exposed lazily so importing a single
# mmcomposer.<leaf> doesn't drag in the API and re-enter the webui shims.
_API = {"matmul", "get_tuned_kernel", "tune"}        # live in webui/mmc.py

__all__ = ["matmul", "get_tuned_kernel", "tune",
           "combos", "compiler", "runtime", "benchmark",
           "cache", "leaderboard", "autotune", "mvp_core", "codegen"]


def __getattr__(name):
    if name in _API:
        return getattr(_importlib.import_module("mmc"), name)
    if name == "autotune":          # still under webui/ (Chunk 3)
        return _importlib.import_module("autotune")
    raise AttributeError(f"module 'mmcomposer' has no attribute {name!r}")
