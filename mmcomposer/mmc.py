"""mmc -- the public Python API for MMComposer (see mmcomposer/DESIGN.md).

    import mmc
    c = mmc.matmul(a, b)               # Hopper fixed WS or Blackwell auto-tuned kernel
    gemm = mmc.get_tuned_kernel(a, b)  # reusable callable for that shape
    mmc.tune(M, N, K)                  # explicit offline pre-tune

Thin glue over the leaves + the autotune orchestrator:
  get_tuned_kernel = cache.best -> (hit: build callable) | (miss: autotune.tune -> build)
  matmul           = get_tuned_kernel(a, b)(a, b)

v0 constraints: bf16 in/out, A (M,K) and B (K,N) row-major contiguous, C row-major.
On Hopper (sm_90a), plain GEMM uses the fixed WS BM128/BN256/BK64/WG2/NS4/GM8
kernel.  On Blackwell (sm_100a), plain GEMM uses the existing autotuned
generated-kernel path with arbitrary M, N a multiple of 8 for TMA stride
alignment, and K a multiple of 64.  Unsupported inputs raise a clear error.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import time
import weakref

from . import mvp_core as mc        # noqa: F401
from . import compiler
from . import runtime
from . import cache as kcache
from . import autotune
from . import autotune_isolated
from . import epilogue as epi
from . import hopper as _hopper
from . import swiglu as _swiglu

DEFAULT_DTYPE = "bf16"
DEFAULT_ARCH = kcache.DEFAULT_ARCH

# in-process kernel-callable cache, keyed by shape_key (avoids re-render/compile)
_KERNELS: dict = {}
_EPI_KERNELS: dict = {}        # (shape_key, epilogue_digest) -> callable
# Memoize trace+lower by the epilogue callable so a *reused* epilogue object is
# not re-traced every call (a fresh inline lambda each call still re-traces --
# define the epilogue once and reuse it in hot loops).
_TRACE_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_HOPPER_FIXED_WS = None        # fixed Hopper WS kernel callable (lazy, shape-agnostic)
_SWIGLU_DUAL_B_NS6_S2 = None   # the fixed swiglu kernel callable (lazy, shape-agnostic)


def _trace(epilogue):
    """Return (cuda_expr, digest) for an epilogue callable, memoized by object."""
    td = _TRACE_CACHE.get(epilogue)
    if td is None:
        cuda = epi.to_cuda(epilogue)
        tag = hashlib.sha1(cuda.encode()).hexdigest()[:10]
        try:
            _TRACE_CACHE[epilogue] = (cuda, tag)
        except TypeError:                 # not weak-referenceable -> just don't cache
            pass
        return cuda, tag
    return td


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
    # M is arbitrary (it is never a TMA stride -- only a row count, so edge tiles
    # are handled by ceil-div grid + TMA out-of-bounds clipping).  N must stay a
    # multiple of 8: the row-major B/C/aux row stride is N*2 bytes and TMA requires
    # 16-byte-aligned strides.  K stays a multiple of BK=64 (no partial-K path).
    errs = []
    if M < 1:
        errs.append(f"M={M} must be positive")
    if N % 8:
        errs.append(f"N={N} must be a multiple of 8 (TMA 16-byte stride alignment)")
    if Ka % 64:
        errs.append(f"K={Ka} must be a multiple of 64")
    if errs:
        raise ValueError("unsupported shape: " + "; ".join(errs))
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


def _build_epilogue(config, cuda_expr, n_extra=0):
    """Render -> compile -> bind for `config` with an elementwise epilogue spliced
    in.  Uses the same `autotune._render` path (and build tag) as the sweep, so the
    winner's already-compiled fused cubin is reused (compile_one skips nvcc).
    `n_extra` = number of extra epilogue inputs (phase 2)."""
    tier = _tier_for(config)
    build_root = kcache.cache_root() / "build" / DEFAULT_ARCH
    src = autotune._render(tier, config, build_root, epilogue=cuda_expr, n_extra=n_extra)
    cubin = compiler.compile_one(src)
    return runtime.kernel(config, cubin)


# ---- one-time auto-tune progress (printed only on a cold shape) -----------
def _autotune_progress():
    """An autotune `on_event` callback that prints concise tuning progress to
    stderr (throttled).  Fresh per tune so its throttle state is isolated."""
    st = {"t": 0.0}

    def cb(key, phase, **kw):
        if phase == "enumerate":
            print(f"[mmcomposer]   {kw['n_valid']} candidate configs to evaluate",
                  file=sys.stderr, flush=True)
        elif phase == "compiled":
            print(f"[mmcomposer]   compiled {kw['n_compiled']}/{kw['n_valid']} kernels; "
                  f"benchmarking on the GPU...", file=sys.stderr, flush=True)
        elif phase == "benchmark":
            done, total = kw.get("done"), kw.get("total")
            now = time.monotonic()
            if total and (done >= total or now - st["t"] >= 2.0):
                st["t"] = now
                print(f"[mmcomposer]   benchmarked {done}/{total} ({100 * done // total}%)",
                      file=sys.stderr, flush=True)

    return cb


# ---- public API -----------------------------------------------------------
def tune(M, N, K, *, dtype=DEFAULT_DTYPE, scope="production", isolated=False, **kw) -> dict:
    """Pre-tune a shape (offline): sweep the scope, write the winner to the cache.
    Returns the autotune summary dict.  Pass ``isolated=True`` to benchmark each
    candidate in a fresh child process."""
    tier_dirs, filters = autotune.scope_to_dirs_filters(scope)
    tuner = autotune_isolated if isolated else autotune
    return tuner.tune(M, N, K, tier_dirs=tier_dirs, filters=filters,
                      dtype=dtype, **kw)


def get_tuned_kernel(a, b, *, tune_if_missing=True, autotune_isolated=False):
    """Return a callable ``k(a, b) -> c`` running the best-known kernel for this
    shape.  Reuses the cached config/cubin; auto-tunes on a cold shape (unless
    `tune_if_missing=False`, which then raises).  ``autotune_isolated=True`` only
    affects cold-shape tuning; cache hits stay cache hits."""
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
        # Cold shape -> auto-tune once.  This message (and the progress below)
        # appears ONLY when tuning actually runs; warm/cached calls are silent.
        print(f"[mmcomposer] no tuned kernel for {M}x{N}x{K} {DEFAULT_DTYPE} on "
              f"{DEFAULT_ARCH} -- auto-tuning now (one-time per machine; cached to "
              f"{kcache.cache_root()} and reused in future sessions)",
              file=sys.stderr, flush=True)
        t0 = time.monotonic()
        summary = tune(M, N, K, isolated=autotune_isolated, on_event=_autotune_progress())
        rec = kcache.best(key)
        if rec is None:
            raise RuntimeError(f"tuning produced no valid config for {key}: "
                               f"{summary.get('error')}")
        print(f"[mmcomposer] auto-tune complete in {time.monotonic() - t0:.0f}s: "
              f"best {rec['tflops']:.0f} TFLOPS ({rec['vs_cublas']:.0%} of cuBLAS) -- cached.",
              file=sys.stderr, flush=True)
    fn = _build(rec["config"])
    _KERNELS[key] = fn
    return fn


def get_epilogue_kernel(a, b, epilogue, *, tune_if_missing=True,
                        autotune_isolated=False):
    """Like get_tuned_kernel, but fuses an elementwise epilogue (see
    mmcomposer/epilogue.py) onto each output element.  `epilogue` is a one-in/
    one-out callable (lambda or def) over the epilogue DSL.

    The fused kernel is a tuned **variant**: keyed by (shape, epilogue digest), it
    auto-tunes the GEMM *with the epilogue spliced in* on first use (so the winning
    config is the best one for the fused kernel, not the plain GEMM), and caches it.
    Extra-input epilogues use a constrained stable in-process sweep by default;
    pass ``autotune_isolated=True`` to run the full isolated sweep for debugging."""
    M, N, K = _validate(a, b)
    cuda, tag = _trace(epilogue)             # trace + lower (memoized by object)
    n_extra = epi.n_inputs(epilogue)
    key = kcache.shape_key(M, N, K, DEFAULT_DTYPE, DEFAULT_ARCH, epi=tag)
    if key in _EPI_KERNELS:
        return _EPI_KERNELS[key]

    rec = kcache.best(key)
    if rec is None:
        if not tune_if_missing:
            raise RuntimeError(
                f"no tuned config for {key}; run with tune_if_missing=True")
        print(f"[mmcomposer] no tuned kernel for {M}x{N}x{K} {DEFAULT_DTYPE} on "
              f"{DEFAULT_ARCH} with this epilogue -- auto-tuning the fused variant now "
              f"(one-time per machine; cached to {kcache.cache_root()} and reused later)",
              file=sys.stderr, flush=True)
        t0 = time.monotonic()
        summary = tune(M, N, K, isolated=autotune_isolated,
                       epilogue=cuda, epi_tag=tag, n_extra=n_extra,
                       ref_fn=epi.to_torch(epilogue), on_event=_autotune_progress())
        rec = kcache.best(key)
        if rec is None:
            raise RuntimeError(f"tuning produced no valid config for {key}: "
                               f"{summary.get('error')}")
        print(f"[mmcomposer] auto-tune complete in {time.monotonic() - t0:.0f}s: "
              f"best {rec['tflops']:.0f} TFLOPS ({rec['vs_cublas']:.0%} of cuBLAS) -- cached.",
              file=sys.stderr, flush=True)
    fn = _build_epilogue(rec["config"], cuda, n_extra)
    _EPI_KERNELS[key] = fn
    return fn


def matmul(a, b, *, out=None, sync=False, tune_if_missing=True, epilogue=None,
           aux=None, autotune_isolated=False):
    """``c = a @ b`` with the best-known MMComposer kernel for this shape
    (auto-tunes + caches on the first call for a new shape).

    Asynchronous like ``torch.matmul``: enqueues on torch's current stream and
    returns immediately (the result is ordered before following torch ops).  Pass
    ``out=`` to reuse an output buffer, or ``sync=True`` to block until done.

    Pass ``epilogue=`` (a callable over the epilogue DSL, see mmcomposer/epilogue.py)
    to fuse an elementwise op onto each output element, e.g.
    ``mmc.matmul(a, b, epilogue=lambda x: x * sigmoid(x))`` for SiLU.  A multi-arg
    epilogue takes extra same-shape ``[M,N]`` inputs via ``aux=[c, ...]``, e.g.
    ``mmc.matmul(a, b, epilogue=lambda x, c: x * c, aux=[c])`` for ``(a@b)*c``.
    Pass ``autotune_isolated=True`` to use process-isolated autotuning on a cold
    cache miss."""
    if epilogue is None:
        if getattr(a, "is_cuda", False) and _hopper.is_hopper_device(a.device):
            global _HOPPER_FIXED_WS
            if _HOPPER_FIXED_WS is None:
                _HOPPER_FIXED_WS = _hopper.kernel()
            return _HOPPER_FIXED_WS(a, b, out, sync=sync)
        return get_tuned_kernel(a, b, tune_if_missing=tune_if_missing,
                                autotune_isolated=autotune_isolated)(a, b, out, sync=sync)
    if getattr(a, "is_cuda", False) and _hopper.is_hopper_device(a.device):
        raise NotImplementedError(
            "Hopper mmc.matmul currently supports plain GEMM only; epilogue fusion "
            "still uses the Blackwell codegen path and has not been ported to sm_90.")
    import torch
    aux = tuple(aux or ())
    M, N, _ = _validate(a, b)
    want = epi.n_inputs(epilogue)
    if len(aux) != want:
        raise ValueError(f"epilogue takes {want} extra input(s); got aux of length {len(aux)}")
    for i, t in enumerate(aux):
        if t.dtype != torch.bfloat16 or tuple(t.shape) != (M, N) or not t.is_contiguous():
            raise ValueError(f"aux[{i}] must be a contiguous bf16 tensor of shape ({M}, {N})")
    return get_epilogue_kernel(a, b, epilogue,
                               tune_if_missing=tune_if_missing,
                               autotune_isolated=autotune_isolated)(a, b, out, aux=aux, sync=sync)


def matmul_swiglu_dual_b_ns6_s2(a, b_left, b_gate, *, c=None, d=None, sync=False):
    """Fused GEMM + SwiGLU with two B halves (the fixed ``..._dual_b_ns6_s2``
    Blackwell kernel; config baked in, no autotune).

        A[M, K], B_left[K, N/2], B_gate[K, N/2]  ->  C[M, N], D[M, N/2]

    where ``left = A @ B_left``, ``gate = A @ B_gate``, ``C`` is the packed wide
    GEMM ([left | gate] per BN=256 tile) and ``D = left * silu(gate)`` is the
    SwiGLU activation.  Returns ``(c, d)``.  Compiles once per machine (cached),
    then async on torch's current stream like ``matmul``; pass ``c=``/``d=`` to
    reuse buffers or ``sync=True`` to block."""
    global _SWIGLU_DUAL_B_NS6_S2
    if _SWIGLU_DUAL_B_NS6_S2 is None:
        _SWIGLU_DUAL_B_NS6_S2 = _swiglu.kernel()
    return _SWIGLU_DUAL_B_NS6_S2(a, b_left, b_gate, c=c, d=d, sync=sync)
