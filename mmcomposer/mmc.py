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
from . import hopper_swiglu as _hopper_swiglu
from . import swiglu as _swiglu
from . import torch_ops

DEFAULT_DTYPE = "bf16"
DEFAULT_ARCH = kcache.DEFAULT_ARCH

# in-process kernel-callable cache, keyed by shape_key (avoids re-render/compile)
_KERNELS: dict = {}
_EPI_KERNELS: dict = {}        # (shape_key, epilogue_digest) -> callable
# Memoize trace+lower by the epilogue callable so a *reused* epilogue object is
# not re-traced every call (a fresh inline lambda each call still re-traces --
# define the epilogue once and reuse it in hot loops).
_TRACE_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_HOPPER_FIXED_WS: dict[int, object] = {}  # Hopper WS matmul by GM
_SWIGLU_DUAL_B_NS6_S2 = None   # the fixed swiglu kernel callable (lazy, shape-agnostic)
_HOPPER_SWIGLU_DUAL_B: dict[int, object] = {}  # Hopper no-preact SwiGLU by GM
_HOPPER_SWIGLU_DUAL_B_STORE_PREACT = None  # fixed Hopper preact-storing callable


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


# ---- launch helpers (shared by the public wrappers and the custom ops) ----
# These do device dispatch + kernel-callable caching and launch *asynchronously*
# on torch's current stream (host sync, if any, is the caller's job).  They are
# what the ``mmc::*`` custom ops call, so the op body reuses the exact same path
# as the eager fallback.  Inputs are assumed already validated by the wrapper.
def _launch_matmul(a, b, out, gm=8):
    """Plain GEMM into `out` (allocated if None); return the output tensor."""
    gm = _hopper._normalize_gm(gm)
    if getattr(a, "is_cuda", False) and _hopper.is_hopper_device(a.device):
        global _HOPPER_FIXED_WS
        fn = _HOPPER_FIXED_WS.get(gm)
        if fn is None:
            fn = _hopper.kernel(gm=gm)
            _HOPPER_FIXED_WS[gm] = fn
        return fn(a, b, out, sync=False)
    if gm != _hopper.DEFAULT_GM:
        raise ValueError("gm is an experimental Hopper-only matmul knob")
    return get_tuned_kernel(a, b)(a, b, out, sync=False)


def _launch_swiglu_d(a, b_left, b_gate, out, gm=8):
    """SwiGLU dual-B returning only D into `out` (allocated if None); return D.
    Hopper only -- Blackwell has no no-preact kernel yet."""
    gm = _hopper_swiglu._normalize_gm(gm)
    if _swiglu._is_blackwell_device(a.device):
        if gm != _hopper_swiglu.DEFAULT_GM:
            raise ValueError("gm is an experimental Hopper-only SwiGLU knob")
        raise NotImplementedError(
            "Blackwell matmul_swiglu_dual_b currently uses the fixed ns6/s2 "
            "kernel, which always stores preact; store_preact=False needs a "
            "separate no-preact kernel.")
    if _hopper.is_hopper_device(a.device):
        global _HOPPER_SWIGLU_DUAL_B
        fn = _HOPPER_SWIGLU_DUAL_B.get(gm)
        if fn is None:
            fn = _hopper_swiglu.kernel(gm=gm)
            _HOPPER_SWIGLU_DUAL_B[gm] = fn
        return fn(a, b_left, b_gate, d=out, sync=False)
    raise NotImplementedError(
        "matmul_swiglu_dual_b currently supports CUDA Hopper/Blackwell devices only")


def _launch_swiglu_cd(a, b_left, b_gate, preact, out):
    """SwiGLU dual-B returning (C preact, D), each allocated if None; return
    ``(c, d)``.  Blackwell ns6/s2, or the Hopper preact-storing kernel."""
    if _swiglu._is_blackwell_device(a.device):
        global _SWIGLU_DUAL_B_NS6_S2
        if _SWIGLU_DUAL_B_NS6_S2 is None:
            _SWIGLU_DUAL_B_NS6_S2 = _swiglu.kernel()
        return _SWIGLU_DUAL_B_NS6_S2(a, b_left, b_gate, c=preact, d=out, sync=False)
    if _hopper.is_hopper_device(a.device):
        global _HOPPER_SWIGLU_DUAL_B_STORE_PREACT
        if _HOPPER_SWIGLU_DUAL_B_STORE_PREACT is None:
            _HOPPER_SWIGLU_DUAL_B_STORE_PREACT = _hopper_swiglu.kernel_store_preact()
        return _HOPPER_SWIGLU_DUAL_B_STORE_PREACT(
            a, b_left, b_gate, c=preact, d=out, sync=False)
    raise NotImplementedError(
        "matmul_swiglu_dual_b currently supports CUDA Hopper/Blackwell devices only")


def _is_compiling() -> bool:
    """True while Dynamo is tracing (so we skip host syncs inside a graph)."""
    import torch
    fn = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    try:
        return bool(fn()) if fn else False
    except Exception:
        return False


def _check_matmul_out(out, M, N, device):
    import torch
    if out.dtype != torch.bfloat16 or tuple(out.shape) != (M, N):
        raise ValueError(f"out must be a bf16 tensor of shape ({M}, {N})")
    if not out.is_cuda or out.device != device:
        raise ValueError(f"out must be a CUDA tensor on {device}")
    if not out.is_contiguous():
        raise ValueError("out must be row-major contiguous")


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
           aux=None, autotune_isolated=False, gm=8):
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
    cache miss.

    The plain (``epilogue=None``) path runs through the ``mmc::matmul`` torch
    custom op when available (torch >= 2.4), so the default path is traceable by
    ``torch.compile`` and differentiable; ``autotune_isolated`` there falls back
    to the default in-process tuner.  ``gm`` is an experimental Hopper-only
    grouping knob for plain GEMM; the default ``gm=8`` preserves the original
    custom-op path, while non-default GM values are eager-only for now.  The
    ``epilogue=`` path stays eager (an epilogue is a Python callable and cannot
    be a custom-op argument)."""
    import torch
    gm = _hopper._normalize_gm(gm)
    if epilogue is None:
        M, N, K = _validate(a, b)
        is_hopper = _hopper.is_hopper_device(a.device)
        if gm != _hopper.DEFAULT_GM and not is_hopper:
            raise ValueError("gm is an experimental Hopper-only matmul knob")
        if gm != _hopper.DEFAULT_GM and _is_compiling():
            raise NotImplementedError(
                "non-default gm for Hopper matmul is an eager-only experimental knob"
            )
        if not tune_if_missing and not is_hopper:
            key = kcache.shape_key(M, N, K, DEFAULT_DTYPE, DEFAULT_ARCH)
            if key not in _KERNELS and kcache.best(key) is None:
                raise RuntimeError(
                    f"no tuned config for {key}; run mmc.tune({M}, {N}, {K}) first "
                    f"or call with tune_if_missing=True")
        if autotune_isolated and not _is_compiling() and not is_hopper:
            # Warm the isolated-tuned kernel into the in-process cache so the op
            # body's get_tuned_kernel() reuses it instead of tuning in-process.
            get_tuned_kernel(a, b, tune_if_missing=tune_if_missing,
                             autotune_isolated=True)
        if gm == _hopper.DEFAULT_GM and torch_ops.ENABLED:
            if out is None:
                c = torch.ops.mmc.matmul(a, b)
            else:
                _check_matmul_out(out, M, N, a.device)
                torch.ops.mmc.matmul_out(a, b, out)
                c = out
        else:
            c = _launch_matmul(a, b, out, gm=gm)
        if sync and not _is_compiling():
            torch.cuda.current_stream(a.device).synchronize()
        return c
    if gm != _hopper.DEFAULT_GM:
        raise ValueError("gm is supported only for plain Hopper matmul")
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


def _validate_swiglu_dual_b_cuda(a, b_left, b_gate):
    M, N, K = _swiglu.validate(a, b_left, b_gate)
    tensors = (("a", a), ("b_left", b_left), ("b_gate", b_gate))
    for name, t in tensors:
        if not getattr(t, "is_cuda", False):
            raise ValueError(f"{name} must be a CUDA tensor")
        if t.device != a.device:
            raise ValueError(f"{name} must be on {a.device}, got {t.device}")
    return M, N, K


def _check_swiglu_output(name, t, shape, device):
    import torch
    if t is None:
        return
    if t.dtype != torch.bfloat16 or tuple(t.shape) != tuple(shape):
        raise ValueError(f"{name} must be a bf16 tensor of shape {tuple(shape)}")
    if not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if t.device != device:
        raise ValueError(f"{name} must be on {device}, got {t.device}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be row-major contiguous")


def matmul_swiglu_dual_b(a, b_left, b_gate, *, store_preact=False, preact=None,
                         out=None, sync=False, gm=8):
    """Fused ``left * silu(gate)`` from two B halves.

    ``A[M,K]``, ``B_left[K,H]`` and ``B_gate[K,H]`` produce the SwiGLU output
    ``D[M,H]``.  By default this is the inference-oriented form and returns only
    ``D``.  Pass ``store_preact=True`` for training-style use; then the API also
    stores and returns the packed preactivation ``C[M,2H] = [A@B_left | A@B_gate]``
    as ``(C, D)``.

    Current backend coverage:
      * Blackwell: ``store_preact=True`` dispatches to the fixed ns6/s2 kernel.
      * Hopper: both ``store_preact=False`` and ``store_preact=True`` dispatch to
        fixed WS kernels.  ``gm`` is an experimental Hopper-only grouping knob
        for the inference/no-preact path; the default ``gm=8`` preserves the
        original public kernel.

    Runs through the ``mmc::swiglu_dual_b[_preact]`` torch custom ops when
    available (torch >= 2.4), so the default path is traceable by
    ``torch.compile``.  The ``store_preact=True`` form with no user buffers is
    differentiable (``register_autograd``) -- the training path; passing
    ``preact=``/``out=`` buffers uses the in-place op variant (no autograd).
    Non-default ``gm`` values are eager-only for now."""
    import torch
    gm = _hopper_swiglu._normalize_gm(gm)
    if store_preact and gm != _hopper_swiglu.DEFAULT_GM:
        raise ValueError(
            "gm is currently supported only for Hopper SwiGLU inference "
            "(store_preact=False)"
        )
    if not store_preact and preact is not None:
        raise ValueError("preact may only be provided when store_preact=True")
    if gm != _hopper_swiglu.DEFAULT_GM and _is_compiling():
        raise NotImplementedError(
            "non-default gm for Hopper SwiGLU is an eager-only experimental knob"
        )

    # ``_swiglu.validate`` probes ``.data_ptr()`` (TMA 16-byte alignment), which
    # is not available on FakeTensors while Dynamo traces -- so skip the eager
    # validation under compile and derive shapes from ``.shape`` only; the op
    # body re-validates on real tensors at execution.
    if _is_compiling():
        H = b_left.shape[1]
        M, N = a.shape[0], 2 * H
    else:
        M, N, _ = _validate_swiglu_dual_b_cuda(a, b_left, b_gate)
        H = N // 2
        _check_swiglu_output("out", out, (M, H), a.device)
        if store_preact:
            _check_swiglu_output("preact", preact, (M, N), a.device)

    if not store_preact:
        if gm == _hopper_swiglu.DEFAULT_GM and torch_ops.ENABLED:
            if out is None:
                d = torch.ops.mmc.swiglu_dual_b(a, b_left, b_gate)
            else:
                torch.ops.mmc.swiglu_dual_b_out(a, b_left, b_gate, out)
                d = out
        else:
            d = _launch_swiglu_d(a, b_left, b_gate, out, gm=gm)
        if sync and not _is_compiling():
            torch.cuda.current_stream(a.device).synchronize()
        return d

    # store_preact=True -> (C preact, D)
    if torch_ops.ENABLED:
        if preact is None and out is None:
            c, d = torch.ops.mmc.swiglu_dual_b_preact(a, b_left, b_gate)
        else:
            # Buffer(s) supplied: allocate the missing one and fill both in place.
            if preact is None:
                preact = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
            if out is None:
                out = torch.empty(M, H, dtype=torch.bfloat16, device=a.device)
            torch.ops.mmc.swiglu_dual_b_preact_out(a, b_left, b_gate, preact, out)
            c, d = preact, out
    else:
        c, d = _launch_swiglu_cd(a, b_left, b_gate, preact, out)
    if sync and not _is_compiling():
        torch.cuda.current_stream(a.device).synchronize()
    return c, d


def matmul_swiglu_dual_b_ns6_s2(a, b_left, b_gate, *, c=None, d=None, sync=False):
    """Kernel-specific Blackwell fixed-config SwiGLU entry point.

    This is the ns6/s2 implementation used by ``matmul_swiglu_dual_b`` when
    ``store_preact=True`` on Blackwell.  It always writes both ``C[M,N]``
    preactivation and ``D[M,N/2]`` activation and returns ``(c, d)``.
    """
    global _SWIGLU_DUAL_B_NS6_S2
    if _SWIGLU_DUAL_B_NS6_S2 is None:
        _SWIGLU_DUAL_B_NS6_S2 = _swiglu.kernel()
    return _SWIGLU_DUAL_B_NS6_S2(a, b_left, b_gate, c=c, d=d, sync=sync)
