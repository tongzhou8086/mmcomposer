#!/usr/bin/env python3
"""autotune.py -- in-process terminal timing autotune.

Sweeps a timing subset of valid knob combinations for a shape on the LOCAL GPU
and prints the top configs by measured TFLOPS, with a live leaderboard.  The
whole pipeline runs in-process from the modular leaves (see mmcomposer/DESIGN.md):

    enumerate -> codegen -> compile -> runtime -> verify + benchmark -> cache
                                                    (leaderboard polls the cache)

No srun / subprocess -- getting onto a GPU is the caller's concern.  The `tune()`
function is the shared orchestrator used by this CLI and (later) the mmc package.

Usage (on a machine with a B200):
    python -m mmcomposer.autotune 8192                 # square 8192^3
    python -m mmcomposer.autotune 32768x4608x768       # rectangular MxNxK
    python -m mmcomposer.autotune 8192 --scope full --top 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

from . import mvp_core as mc        # noqa: F401
from . import combos
from . import compiler
from . import runtime
from . import benchmark as bench
from . import cache as kcache
from . import leaderboard as lb

CORRECT_TOL = 5e-2


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def parse_shape(tok: str):
    """'8192' -> (8192,8192,8192); '32768x4608x768' -> (32768,4608,768)."""
    tok = tok.lower().strip()
    if "x" in tok:
        M, N, K = (int(v) for v in tok.split("x"))
        return M, N, K
    s = int(tok)
    return s, s, s


def parse_int_csv(spec):
    if spec is None:
        return None
    return [int(x) for x in spec.split(",") if x.strip()]


def with_filter_override(filters, key, spec):
    vals = parse_int_csv(spec)
    if vals is not None:
        filters[key] = vals


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _fits(tier, k, M, N, K) -> bool:
    """Whether a (shape-agnostic) combo's tile actually divides this shape.  The
    tile columns BN must divide N, the per-cluster rows (2*BM for a 2-CTA cluster,
    else BM) must divide M, and BK must divide K -- otherwise the grid/tiling is
    wrong and the launch fails (CUDA_ERROR_LAUNCH_FAILED)."""
    cta = 2 if tier["cluster"] else 1
    return (M % (cta * k["bm"]) == 0) and (N % k["bn"] == 0) and (K % k["bk"] == 0)


def _render(tier, k, build_root, epilogue=None, n_extra=0) -> str:
    """Render kernel.cu for (tier, k) into a tagged build dir; return its path.

    `epilogue` (a CUDA fp32 expression) is spliced in and folded into the build
    tag, so a fused variant gets its own cubin (no collision with the plain one).
    `n_extra` = number of extra epilogue inputs (phase 2)."""
    sig = {**k, "dir": tier["dir"], "cluster": tier["cluster"]}
    if epilogue:
        sig["epilogue"] = epilogue
    tag = hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:16]
    d = build_root / tag
    d.mkdir(parents=True, exist_ok=True)
    src = d / "kernel.cu"
    src.write_text(mc.render_kernel(
        tier, k["bm"], k["bn"], k["bk"], k["ns"], k["gsm"], k["nw"],
        ld_width=k.get("ld_width", 8), overlap=k.get("overlap", 0),
        split_epilogue=k.get("split_epilogue", 0), l1_no_alloc=k.get("l1_no_alloc", 0),
        tma_pipelined=k.get("tma_pipelined", 0),
        tma_store_stages=k.get("tma_store_stages", 2),
        single_tmem=k.get("single_tmem", 0), epilogue=epilogue, n_extra=n_extra))
    return str(src)


def _record_config(tier, k) -> dict:
    """Stored config: everything needed to re-render, re-launch, and display."""
    cfg = dict(k)
    cfg.update(dir=tier["dir"], symbol=tier["symbol"], cluster=tier["cluster"],
               ws=(tier["dir"] != "tier1_baseline"))
    return cfg


def scope_to_dirs_filters(scope: str = "production"):
    """Map a scope name to (tier_dirs, filters).  Shared by the CLI and mmc.tune.

    production -- warp-spec-on, BN=256/512, NS>=3, 2-CTA, single-TMEM only at BN512,
                  and persistent + overlapped + TMA-pipelined epilogue pinned on
                  (split/L1-no-alloc off).  These last knobs are pinned because
                  across 20 recorded shapes the winning config *always* used
                  PERS=1/OV=1/TMA=1 and never SPLIT=1/L1NA=1; pinning them cuts the
                  sweep ~73% (738 -> 198 combos).  Pass --scope full or the
                  per-knob CLI overrides to sweep them anyway.
    full       -- every tier/combo (incl. warp-spec off and BN=64); very large.
    """
    ws_dirs = list(dict.fromkeys(t["dir"] for k, t in mc.TIER_MAP.items() if t and k[0]))
    all_dirs = list(dict.fromkeys(t["dir"] for t in mc.TIER_MAP.values() if t))
    if scope == "production":
        return ws_dirs, {"bn": [256, 512], "ns": [x for x in mc.NS_OPTS if x >= 3],
                         "two_cta": [1], "tma_store_stages": [1, 2],
                         "single_tmem_policy": "bn512-only",
                         "persistent": [1], "overlap": [1], "tma_pipelined": [1],
                         "split_epilogue": [0], "l1_no_alloc": [0]}
    return all_dirs, {"single_tmem_policy": "all"}


def tune(M, N, K, *, tier_dirs, filters, dtype="bf16", arch=kcache.DEFAULT_ARCH,
         tol=CORRECT_TOL, warmup_ms=None, rep_ms=None,
         cublas_samples=3, cublas_warmup_samples=1, fresh=True,
         cache_obj=None, on_event=None,
         epilogue=None, epi_tag=None, ref_fn=None, n_extra=0) -> dict:
    """Run an in-process timing sweep on the local GPU.  Streams each correct
    combo's result into the cache; returns a summary dict
    {ok, key, cublas_tflops, best, n_valid, n_compiled, n_correct, error}.

    For a fused **epilogue variant**, pass `epilogue` (the CUDA fp32 expression),
    `epi_tag` (its digest, for the cache key), and `ref_fn` (a callable applying
    the same op to the fp32 reference tensor, e.g. epilogue.to_torch(fn)); every
    candidate is then compiled + benchmarked with the epilogue spliced in and
    verified against `ref_fn(a @ b)`."""
    import torch

    kc = cache_obj if cache_obj is not None else kcache.Cache()
    key = kcache.shape_key(M, N, K, dtype, arch, epi=epi_tag)
    if fresh:
        kc.clear(key)
    bench_kw = {}
    if warmup_ms is not None:
        bench_kw["warmup_ms"] = warmup_ms
    if rep_ms is not None:
        bench_kw["rep_ms"] = rep_ms

    def emit(phase, **kw):
        if on_event:
            on_event(key=key, phase=phase, **kw)

    # 1. enumerate, then drop combos whose tile doesn't divide THIS shape.
    # valid_combos is shape-agnostic (one cubin serves all shapes), so e.g. a
    # BN=512 config is enumerated even when N=2304 (512 does not divide 2304).
    # Launching it gives a wrong grid -> CUDA_ERROR_LAUNCH_FAILED, which poisons
    # the context and kills the whole sweep -- so filter by fit up front.
    combo_list = [(tier, k) for (tier, k) in combos.valid_combos(tier_dirs, filters)
                  if _fits(tier, k, M, N, K)]
    n_valid = len(combo_list)
    emit("enumerate", n_valid=n_valid)

    # 2. render (codegen) + 3. compile (parallel, CPU)
    build_root = kcache.cache_root() / "build" / arch
    builds = [(tier, k, _render(tier, k, build_root, epilogue=epilogue, n_extra=n_extra))
              for tier, k in combo_list]
    emit("compile", done=0, total=n_valid)
    comp = compiler.compile_many([src for (_, _, src) in builds])
    ok = [(tier, k, src) for (tier, k, src) in builds if comp[src].ok]
    n_compiled = len(ok)
    emit("compiled", n_compiled=n_compiled, n_valid=n_valid)

    # tensors + cuBLAS reference
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    # extra epilogue inputs (phase 2): random same-shape [M,N] tiles to tune/verify with
    aux = [torch.randn(M, N, dtype=torch.bfloat16, device="cuda") for _ in range(n_extra)]
    ref = a.float() @ b.float()
    if ref_fn is not None:                 # fused epilogue: reference applies the op too
        ref = ref_fn(ref, *[t.float() for t in aux])
    flops = bench.gemm_flops(M, N, K)
    cub = bench.benchmark_median(lambda: torch.mm(a, b), flops=flops,
                                 samples=cublas_samples,
                                 warmup_samples=cublas_warmup_samples, **bench_kw).tflops
    emit("cublas", cublas_tflops=cub, total=n_compiled)

    # 4. verify + benchmark each (serial on the GPU), streaming to the cache
    n_correct = 0
    for i, (tier, k, src) in enumerate(ok, 1):
        cfg = runtime.config_from_combo(tier, k)
        gemm = runtime.kernel(cfg, comp[src].cubin)
        c = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        try:
            gemm(a, b, c, aux=aux)             # one launch -> verify
            rel = bench.rel_error(c, ref)
        except Exception:                      # noqa: BLE001  (bad launch -> skip combo)
            emit("benchmark", done=i, total=n_compiled)
            continue
        if rel >= tol:
            emit("benchmark", done=i, total=n_compiled)
            continue
        r = bench.benchmark(lambda: gemm(a, b, c, aux=aux, sync=False), flops=flops, **bench_kw)
        kc.put(key, {"config": _record_config(tier, k),
                     "tflops": r.tflops,
                     "vs_cublas": (r.tflops / cub) if cub else None,
                     "rel_err": rel})
        n_correct += 1
        emit("benchmark", done=i, total=n_compiled, cublas_tflops=cub)

    best = kc.best(key)
    return {"ok": best is not None, "key": key, "cublas_tflops": cub, "best": best,
            "n_valid": n_valid, "n_compiled": n_compiled, "n_correct": n_correct,
            "error": None if best else "no correct combos measured for this shape"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="In-process autotune: sweep valid combos on the local GPU, print the top configs.")
    ap.add_argument("shape", help="square 'S' or rectangular 'MxNxK' (e.g. 8192 or 32768x4608x768)")
    ap.add_argument("--scope", choices=["production", "full"], default="production",
                    help="production (default): warp-spec-on, BN=256/512, NS>=3; full: all combos")
    ap.add_argument("--top", type=int, default=10, help="how many top configs to show (default 10)")
    ap.add_argument("--live-interval", type=float, default=3.0,
                    help="seconds between live leaderboard updates (default 3)")
    ap.add_argument("--live-redraw", choices=["auto", "always", "never"], default="always",
                    help="redraw the leaderboard in place (default always; "
                         "use 'never' for piped/captured output)")
    ap.add_argument("--warmup-ms", type=int, default=None, help="do_bench warmup window (ms)")
    ap.add_argument("--rep-ms", type=int, default=None, help="do_bench repetition window (ms)")
    ap.add_argument("--cublas-samples", type=int, default=3, help="measured cuBLAS samples (median)")
    ap.add_argument("--cublas-warmup-samples", type=int, default=1, help="throwaway cuBLAS samples")
    ap.add_argument("--bn", default=None, help="override BN list, e.g. 128,256")
    ap.add_argument("--ns", default=None, help="override NS list")
    ap.add_argument("--gsm", default=None, help="override GROUP_SIZE_M list")
    ap.add_argument("--nw", default=None, help="override NUM_WARPS list")
    ap.add_argument("--persistent", default=None, help="override PERSISTENT list")
    ap.add_argument("--two-cta", dest="two_cta", default=None, help="override TWO_CTA list")
    ap.add_argument("--overlap", default=None, help="override EPILOGUE_OVERLAP list")
    ap.add_argument("--split-epilogue", dest="split_epilogue", default=None)
    ap.add_argument("--l1-no-alloc", dest="l1_no_alloc", default=None)
    ap.add_argument("--tma-pipelined", dest="tma_pipelined", default=None)
    ap.add_argument("--tma-store-stages", dest="tma_store_stages", default=None)
    ap.add_argument("--single-tmem", dest="single_tmem", default=None)
    ap.add_argument("--single-tmem-policy", dest="single_tmem_policy",
                    choices=["all", "bn512-only"], default=None)
    args = ap.parse_args()
    M, N, K = parse_shape(args.shape)

    tier_dirs, filters = scope_to_dirs_filters(args.scope)
    for fkey, spec in (("bn", args.bn), ("ns", args.ns), ("gsm", args.gsm), ("nw", args.nw),
                       ("persistent", args.persistent), ("two_cta", args.two_cta),
                       ("overlap", args.overlap), ("split_epilogue", args.split_epilogue),
                       ("l1_no_alloc", args.l1_no_alloc), ("tma_pipelined", args.tma_pipelined),
                       ("tma_store_stages", args.tma_store_stages), ("single_tmem", args.single_tmem)):
        with_filter_override(filters, fkey, spec)
    if args.single_tmem_policy is not None:
        filters["single_tmem_policy"] = args.single_tmem_policy

    print(f"# autotune {M}x{N}x{K}  scope={args.scope}  (timing on the local GPU; "
          f"this can take a while)", flush=True)
    print(f"# filters={filters}", flush=True)

    kc = kcache.Cache()
    disp = lb.LiveDisplay(redraw=args.live_redraw)
    state = {"cub": None, "last": 0.0}

    def on_event(key, phase, **kw):
        if kw.get("cublas_tflops") is not None:
            state["cub"] = kw["cublas_tflops"]
        now = time.monotonic()
        force = phase in ("cublas", "compiled")
        if not force and (now - state["last"] < args.live_interval):
            return
        state["last"] = now
        disp.update(lb.render(kc.top_n(key, args.top), (M, N, K),
                              cublas_tflops=state["cub"], n_combos=len(kc.get(key)),
                              top=args.top, title="# live autotune leaderboard",
                              done=kw.get("done"), total=kw.get("total")))

    summary = tune(M, N, K, tier_dirs=tier_dirs, filters=filters,
                   warmup_ms=args.warmup_ms, rep_ms=args.rep_ms,
                   cublas_samples=args.cublas_samples,
                   cublas_warmup_samples=args.cublas_warmup_samples,
                   cache_obj=kc, on_event=on_event)

    if not summary["ok"]:
        print(f"\nFAILED: {summary['error']} "
              f"(valid={summary['n_valid']}, compiled={summary['n_compiled']})")
        return 1
    disp.update(lb.render(kc.get(summary["key"]), (M, N, K),
                          cublas_tflops=summary["cublas_tflops"],
                          n_combos=summary["n_correct"], top=args.top,
                          title="# final autotune leaderboard"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
