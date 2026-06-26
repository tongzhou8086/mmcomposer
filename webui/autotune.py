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
    python webui/autotune.py 8192                 # square 8192^3
    python webui/autotune.py 32768x4608x768       # rectangular MxNxK
    python webui/autotune.py 8192 --scope full --top 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # webui/

import mvp_core as mc        # noqa: E402
import combos                # noqa: E402
import compiler              # noqa: E402
import runtime               # noqa: E402
import benchmark as bench    # noqa: E402
import cache as kcache       # noqa: E402
import leaderboard as lb     # noqa: E402

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
def _render(tier, k, build_root) -> str:
    """Render kernel.cu for (tier, k) into a tagged build dir; return its path."""
    sig = {**k, "dir": tier["dir"], "cluster": tier["cluster"]}
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
        single_tmem=k.get("single_tmem", 0)))
    return str(src)


def _record_config(tier, k) -> dict:
    """Stored config: everything needed to re-render, re-launch, and display."""
    cfg = dict(k)
    cfg.update(dir=tier["dir"], symbol=tier["symbol"], cluster=tier["cluster"],
               ws=(tier["dir"] != "tier1_baseline"))
    return cfg


def tune(M, N, K, *, tier_dirs, filters, dtype="bf16", arch=kcache.DEFAULT_ARCH,
         tol=CORRECT_TOL, warmup_ms=None, rep_ms=None,
         cublas_samples=3, cublas_warmup_samples=1, fresh=True,
         cache_obj=None, on_event=None) -> dict:
    """Run an in-process timing sweep on the local GPU.  Streams each correct
    combo's result into the cache; returns a summary dict
    {ok, key, cublas_tflops, best, n_valid, n_compiled, n_correct, error}."""
    import torch

    kc = cache_obj if cache_obj is not None else kcache.Cache()
    key = kcache.shape_key(M, N, K, dtype, arch)
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

    # 1. enumerate
    combo_list = list(combos.valid_combos(tier_dirs, filters))
    n_valid = len(combo_list)
    emit("enumerate", n_valid=n_valid)

    # 2. render (codegen) + 3. compile (parallel, CPU)
    build_root = kcache.cache_root() / "build" / arch
    builds = [(tier, k, _render(tier, k, build_root)) for tier, k in combo_list]
    emit("compile", done=0, total=n_valid)
    comp = compiler.compile_many([src for (_, _, src) in builds])
    ok = [(tier, k, src) for (tier, k, src) in builds if comp[src].ok]
    n_compiled = len(ok)
    emit("compiled", n_compiled=n_compiled, n_valid=n_valid)

    # tensors + cuBLAS reference
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    ref = a.float() @ b.float()
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
            gemm(a, b, c)                      # one launch -> verify
            rel = bench.rel_error(c, ref)
        except Exception:                      # noqa: BLE001  (bad launch -> skip combo)
            emit("benchmark", done=i, total=n_compiled)
            continue
        if rel >= tol:
            emit("benchmark", done=i, total=n_compiled)
            continue
        r = bench.benchmark(lambda: gemm(a, b, c, sync=False), flops=flops, **bench_kw)
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
    ap.add_argument("--live-redraw", choices=["auto", "always", "never"], default="auto")
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

    ws_dirs = list(dict.fromkeys(t["dir"] for k, t in mc.TIER_MAP.items() if t and k[0]))
    all_dirs = list(dict.fromkeys(t["dir"] for t in mc.TIER_MAP.values() if t))
    production = args.scope == "production"
    tier_dirs = ws_dirs if production else all_dirs
    filters = {}
    if production:
        filters["bn"] = [256, 512]
        filters["ns"] = [x for x in mc.NS_OPTS if x >= 3]
        filters["two_cta"] = [1]
        filters["tma_store_stages"] = [1, 2]
        filters["single_tmem_policy"] = "bn512-only"
    else:
        filters["single_tmem_policy"] = "all"
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
