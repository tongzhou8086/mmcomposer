"""mmc -- the public Python API for MMComposer (see mmcomposer/DESIGN.md).

    import mmc
    c = mmc.matmul(a, b)               # auto-tunes + caches on first sight of a shape
    gemm = mmc.get_tuned_kernel(a, b)  # reusable callable for that shape
    mmc.tune(M, N, K)                  # explicit offline pre-tune

Thin glue over the leaves + the autotune orchestrator:
  get_tuned_kernel = cache.best -> (hit: build callable) | (miss: autotune.tune -> build)
  matmul           = get_tuned_kernel(a, b)(a, b)

v0 constraints (the kernels' limits): bf16 in/out, A (M,K) & B (K,N) row-major
contiguous, C row-major; M and N multiples of 256, K a multiple of 64; B200
(sm_100a).  Unsupported inputs raise a clear error.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # webui/

import mvp_core as mc        # noqa: E402
import compiler              # noqa: E402
import runtime               # noqa: E402
import cache as kcache       # noqa: E402
import autotune              # noqa: E402

DEFAULT_DTYPE = "bf16"
DEFAULT_ARCH = kcache.DEFAULT_ARCH

# in-process kernel-callable cache, keyed by shape_key (avoids re-render/compile)
_KERNELS: dict = {}


# ---- input validation -----------------------------------------------------
def _shape_dtype(a, b):
    """Validate dtype/layout/shape (no device check) and return (M, N, K)."""
    import torch
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"mmc supports bf16 inputs only (got {a.dtype}, {b.dtype})")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"expected 2-D matrices, got {a.dim()}-D and {b.dim()}-D")
    M, Ka = a.shape
    Kb, N = b.shape
    if Ka != Kb:
        raise ValueError(f"inner dims disagree: {tuple(a.shape)} @ {tuple(b.shape)}")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("inputs must be row-major contiguous (A: M×K, B: K×N)")
    errs = []
    if M % 256:
        errs.append(f"M={M} must be a multiple of 256")
    if N % 256:
        errs.append(f"N={N} must be a multiple of 256")
    if Ka % 64:
        errs.append(f"K={Ka} must be a multiple of 64")
    if errs:
        raise ValueError("unsupported shape for v0 kernels: " + "; ".join(errs))
    return M, N, Ka


def _validate(a, b):
    if not (a.is_cuda and b.is_cuda):
        raise ValueError("mmc inputs must be CUDA tensors")
    return _shape_dtype(a, b)


# ---- build a callable from a stored config --------------------------------
def _tier_for(config):
    for t in mc.TIER_MAP.values():
        if t and t["dir"] == config["dir"] and t["cluster"] == config["cluster"]:
            return t
    raise RuntimeError(f"no tier for config dir={config.get('dir')} cluster={config.get('cluster')}")


def _build(config):
    """Render -> compile -> bind: return a runtime.kernel callable for `config`."""
    tier = _tier_for(config)
    build_root = kcache.cache_root() / "build" / DEFAULT_ARCH
    src = autotune._render(tier, config, build_root)
    cubin = compiler.compile_one(src)
    return runtime.kernel(config, cubin)


# ---- public API -----------------------------------------------------------
def tune(M, N, K, *, dtype=DEFAULT_DTYPE, scope="production", **kw) -> dict:
    """Pre-tune a shape (offline): sweep the scope, write the winner to the cache.
    Returns autotune.tune's summary dict."""
    tier_dirs, filters = autotune.scope_to_dirs_filters(scope)
    return autotune.tune(M, N, K, tier_dirs=tier_dirs, filters=filters,
                         dtype=dtype, **kw)


def get_tuned_kernel(a, b, *, tune_if_missing=True):
    """Return a callable ``k(a, b) -> c`` running the best-known kernel for this
    shape.  Reuses the cached config/cubin; auto-tunes on a cold shape (unless
    `tune_if_missing=False`, which then raises)."""
    M, N, K = _validate(a, b)
    key = kcache.shape_key(M, N, K, DEFAULT_DTYPE, DEFAULT_ARCH)
    if key in _KERNELS:
        return _KERNELS[key]
    rec = kcache.best(key)
    if rec is None:
        if not tune_if_missing:
            raise RuntimeError(
                f"no tuned config for {key}; run mmc.tune({M}, {N}, {K}) first "
                f"or call with tune_if_missing=True")
        summary = tune(M, N, K)
        rec = kcache.best(key)
        if rec is None:
            raise RuntimeError(f"tuning produced no valid config for {key}: "
                               f"{summary.get('error')}")
    fn = _build(rec["config"])
    _KERNELS[key] = fn
    return fn


def matmul(a, b, *, tune_if_missing=True):
    """``c = a @ b`` with the best-known MMComposer kernel for this shape
    (auto-tunes + caches on the first call for a new shape)."""
    return get_tuned_kernel(a, b, tune_if_missing=tune_if_missing)(a, b)
