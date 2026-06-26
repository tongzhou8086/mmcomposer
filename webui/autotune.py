#!/usr/bin/env python3
"""autotune.py — terminal timing autotune.

Sweeps a timing-oriented subset of valid knob combinations for a shape on a
B200 and prints the top configs by measured TFLOPS.  Renders, compiles and benchmarks each combo on the local GPU, with a live
terminal leaderboard while results stream into the jsonl file.

Full compile/run coverage belongs to the test harness:
    python webui/tests/gpu_codegen_driver.py --mode correctness

Writes a throwaway matrix under tests/_scratch/; the committed
`kernels/compat_matrix.json` is never touched.

Usage (from repo root, on a machine with a B200):
    python webui/autotune.py 8192                # square 8192^3
    python webui/autotune.py 32768x4608x768      # rectangular MxNxK
    python webui/autotune.py 8192 --scope full --top 20

Scopes (mirror the UI radio):
    production (default) — warp-spec-on combos with the practical timing
                           policy BN=256/512, NS>=3.
    full                 — every combo, incl. warp-spec-off and BN=64; still
                           timed, so this can be very expensive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # webui/

import mvp_core as mc       # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
SCRATCH = HERE / "tests" / "_scratch" / "autotune"
DRIVER = HERE / "tests" / "gpu_codegen_driver.py"

FILTER_CLI = (
    ("bn", "--bn"),
    ("ns", "--ns"),
    ("gsm", "--gsm"),
    ("nw", "--nw"),
    ("persistent", "--persistent"),
    ("two_cta", "--two-cta"),
    ("overlap", "--overlap"),
    ("split_epilogue", "--split-epilogue"),
    ("l1_no_alloc", "--l1-no-alloc"),
    ("tma_pipelined", "--tma-pipelined"),
    ("tma_store_stages", "--tma-store-stages"),
    ("single_tmem", "--single-tmem"),
    ("single_tmem_policy", "--single-tmem-policy"),
)


# ---------------------------------------------------------------------------
# Local timing-sweep orchestration.  Drives tests/gpu_codegen_driver.py in
# --mode perf as a local subprocess on THIS machine's GPU (getting onto a GPU
# is the caller's concern), streams results into
# a per-run jsonl, and ranks them by measured TFLOPS.
# ---------------------------------------------------------------------------
def _normal_filters(filters=None, bn_opts=None):
    out = {}
    for k, v in (filters or {}).items():
        if v is None:
            continue
        out[k] = list(v) if isinstance(v, (list, tuple, set)) else v
    if bn_opts is not None and "bn" not in out:
        out["bn"] = list(bn_opts)
    out.setdefault("single_tmem_policy", "bn512-only")
    return out


def _filter_args(filters):
    args = []
    for key, flag in FILTER_CLI:
        vals = (filters or {}).get(key)
        if vals is not None:
            if isinstance(vals, (list, tuple, set)):
                vals = ",".join(str(v) for v in vals)
            args += [flag, str(vals)]
    return args


def _filter_sig(filters):
    filters = {k: (list(v) if isinstance(v, (list, tuple, set)) else v)
               for k, v in (filters or {}).items() if v is not None}
    return json.dumps(filters, sort_keys=True, separators=(",", ":"))


def _autotune_cmd(tier_dirs, M, N, K, out_matrix, filters=None, *, mode="perf",
                  bench_warmup_ms: int | None = None,
                  bench_rep_ms: int | None = None,
                  cublas_samples: int | None = None,
                  cublas_warmup_samples: int | None = None):
    """Build the local driver command (runs on this machine's GPU)."""
    py = os.environ.get("MMCOMPOSER_PY", sys.executable)
    cmd = [py, str(DRIVER), "--perf-shapes", f"{M}x{N}x{K}",
           "--tiers", ",".join(tier_dirs), "--invalid-sample", "0",
           "--mode", mode, "--compat-out", str(out_matrix)]
    if bench_warmup_ms is not None:
        cmd += ["--bench-warmup-ms", str(bench_warmup_ms)]
    if bench_rep_ms is not None:
        cmd += ["--bench-rep-ms", str(bench_rep_ms)]
    if cublas_warmup_samples is not None:
        cmd += ["--cublas-warmup-samples", str(cublas_warmup_samples)]
    if cublas_samples is not None:
        cmd += ["--cublas-samples", str(cublas_samples)]
    cmd += _filter_args(filters)
    return cmd


def _out_matrix(tier_dirs, M, N, K, filters=None):
    tag_src = ",".join(tier_dirs) + f"|{M}x{N}x{K}|{_filter_sig(filters)}"
    tag = hashlib.sha1(tag_src.encode()).hexdigest()[:16]
    return SCRATCH / f"autotune_{tag}.json"


def _rank_matrix(out_matrix, M, N, K) -> dict:
    """Parse a driver matrix file into ranked results for shape (M,N,K)."""
    out_matrix = pathlib.Path(out_matrix)
    if not out_matrix.exists():
        return {"ok": False, "error": "sweep produced no matrix (driver failed?)"}
    try:
        matrix = json.loads(out_matrix.read_text())
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"could not parse sweep output: {e}"}
    key = mc.shape_key(M, N, K)
    cub = (matrix.get("cublas_tflops") or {}).get(key)
    results = []
    for e in matrix.get("entries", []):
        if not e.get("correct"):
            continue
        p = (e.get("perf") or {}).get(key)
        if p and p.get("tflops"):
            results.append({**{k: e[k] for k in ("tier", "bm", "bn", "bk", "ns", "gsm",
                                                 "nw", "persistent")},
                            "two_cta": e.get("two_cta", 0),
                            "ld_width": e.get("ld_width", 8), "overlap": e.get("overlap", 0),
                            "split_epilogue": e.get("split_epilogue", 0),
                            "l1_no_alloc": e.get("l1_no_alloc", 0),
                            "tma_pipelined": e.get("tma_pipelined", 0),
                            "tma_store_stages": e.get("tma_store_stages", 2),
                            "single_tmem": e.get("single_tmem", 0),
                            "tflops": p["tflops"], "vs_cublas": p.get("vs_cublas"),
                            "rel_err": p.get("rel_err")})
    results.sort(key=lambda r: r["tflops"], reverse=True)
    return {"ok": bool(results), "cublas_tflops": cub, "results": results,
            "n_combos": len(results),
            "error": None if results else "no correct combos measured for this shape"}


def autotune_start(tier_dirs, M, N, K, filters=None, bn_opts=None,
                   bench_warmup_ms: int | None = None,
                   bench_rep_ms: int | None = None,
                   cublas_samples: int | None = None,
                   cublas_warmup_samples: int | None = None) -> dict:
    """Launch the local sweep in the BACKGROUND (non-blocking) so we can poll a
    live leaderboard.  Per-run jsonl (+ sidecars) keeps concurrent sweeps from
    clobbering each other.  Returns a job dict for autotune_poll/collect."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    filters = _normal_filters(filters, bn_opts)
    out_matrix = _out_matrix(tier_dirs, M, N, K, filters)
    jsonl = pathlib.Path(str(out_matrix)[:-5] + ".jsonl")   # autotune_<tag>.jsonl
    n_valid = pathlib.Path(str(jsonl) + ".nvalid")
    cublas = pathlib.Path(str(jsonl) + ".cublas")
    progress = pathlib.Path(str(jsonl) + ".progress")
    for f in (out_matrix, jsonl, n_valid, cublas, progress):  # clean slate for fresh progress
        try:
            f.unlink()
        except FileNotFoundError:
            pass
    cmd = _autotune_cmd(tier_dirs, M, N, K, out_matrix, filters, mode="perf",
                        bench_warmup_ms=bench_warmup_ms,
                        bench_rep_ms=bench_rep_ms,
                        cublas_samples=cublas_samples,
                        cublas_warmup_samples=cublas_warmup_samples) + ["--jsonl", str(jsonl)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    return {"proc": proc, "out_matrix": str(out_matrix),
            "jsonl": str(jsonl), "n_valid": str(n_valid),
            "progress": str(progress), "M": M, "N": N, "K": K,
            "filters": filters, "bench_warmup_ms": bench_warmup_ms,
            "bench_rep_ms": bench_rep_ms, "cublas_samples": cublas_samples,
            "cublas_warmup_samples": cublas_warmup_samples}


def autotune_poll(job) -> tuple:
    """Return (done, total, finished).  total is None until the driver has
    enumerated combos; done counts result lines (0 during the compile phase)."""
    finished = job["proc"].poll() is not None
    total = None
    try:
        total = int(pathlib.Path(job["n_valid"]).read_text().strip())
    except Exception:
        pass
    done = 0
    try:
        with open(job["jsonl"]) as f:
            done = sum(1 for _ in f)
    except FileNotFoundError:
        pass
    return done, total, finished


def autotune_progress(job) -> dict:
    """Return best-effort driver phase progress from the per-run sidecar."""
    try:
        return json.loads(pathlib.Path(job["progress"]).read_text())
    except Exception:
        return {}


def autotune_partial(job) -> dict:
    """Rank the results streamed into the jsonl SO FAR, for a live leaderboard.
    Same shape as _rank_matrix; ok=False (quietly) until the first combo lands."""
    M, N, K = job["M"], job["N"], job["K"]
    key = mc.shape_key(M, N, K)
    cub = None
    try:
        cub = json.loads(pathlib.Path(str(job["jsonl"]) + ".cublas").read_text()).get(key)
    except Exception:
        pass
    results = []
    try:
        with open(job["jsonl"]) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue          # half-written trailing line -- skip
                if not e.get("correct"):
                    continue
                p = (e.get("perf") or {}).get(key)
                if not (p and p.get("tflops")):
                    continue
                tf = p["tflops"]
                results.append({**{kk: e.get(kk) for kk in ("tier", "bm", "bn", "bk", "ns",
                                                             "gsm", "nw", "persistent")},
                                "two_cta": e.get("two_cta", 0),
                                "ld_width": e.get("ld_width", 8), "overlap": e.get("overlap", 0),
                                "split_epilogue": e.get("split_epilogue", 0),
                                "l1_no_alloc": e.get("l1_no_alloc", 0),
                                "tma_pipelined": e.get("tma_pipelined", 0),
                                "tma_store_stages": e.get("tma_store_stages", 2),
                                "single_tmem": e.get("single_tmem", 0),
                                "tflops": tf, "rel_err": p.get("rel_err"),
                                "vs_cublas": (tf / cub) if cub else None})
    except FileNotFoundError:
        pass
    results.sort(key=lambda r: r["tflops"], reverse=True)
    return {"ok": bool(results), "cublas_tflops": cub, "results": results,
            "n_combos": len(results), "error": None}


def autotune_collect(job) -> dict:
    """Parse the finished sweep's matrix into ranked results."""
    res = _rank_matrix(job["out_matrix"], job["M"], job["N"], job["K"])
    if not res.get("ok") and not res.get("stderr"):
        try:
            res["stderr"] = (job["proc"].stderr.read() or "")[-1200:] if job["proc"].stderr else ""
        except Exception:
            pass
    return res


def parse_shape(tok: str) -> tuple[int, int, int]:
    """'8192' -> (8192,8192,8192); '32768x4608x768' -> (32768,4608,768)."""
    tok = tok.lower().strip()
    if "x" in tok:
        M, N, K = (int(v) for v in tok.split("x"))
        return M, N, K
    s = int(tok)
    return s, s, s


def parse_int_csv(spec: str | None) -> list[int] | None:
    if spec is None:
        return None
    return [int(x) for x in spec.split(",") if x.strip()]


def with_filter_override(filters: dict[str, list[int]], key: str, spec: str | None) -> None:
    vals = parse_int_csv(spec)
    if vals is not None:
        filters[key] = vals


def format_progress_bar(done: int, total: int | None, *, width: int = 36) -> str:
    if total and total > 0:
        done = min(done, total)
        frac = done / total
        filled = min(width, int(round(frac * width)))
        bar = "#" * filled + "-" * (width - filled)
        return f"[{bar}] {done}/{total} ({frac * 100:5.1f}%)"
    return f"[{'?' * width}] {done} measured"


def progress_lines(done: int | None, total: int | None, progress: dict | None) -> list[str]:
    if done is None:
        return []
    p = progress or {}
    phase = p.get("phase")
    phase_done = p.get("done")
    phase_total = p.get("total")
    msg = p.get("message") or phase
    if phase and phase not in {"benchmarking", "collecting", "done"}:
        lines = [format_progress_bar(int(phase_done or 0), phase_total)]
        lines.append(f"phase: {msg}")
        if total:
            lines.append(f"measured combos: {done}/{total}")
        return lines
    lines = [format_progress_bar(done, total)]
    if msg and phase in {"benchmarking", "collecting", "done"}:
        lines.append(f"phase: {msg}")
    return lines


def render_leaderboard(res: dict, M: int, N: int, K: int, *, top: int,
                       title: str, done: int | None = None,
                       total: int | None = None,
                       progress: dict | None = None) -> str:
    cub = res.get("cublas_tflops")
    rows = res.get("results", [])[:top]
    lines = [title]
    lines.extend(progress_lines(done, total, progress))
    lines.append(f"cuBLAS reference: {cub:.0f} TFLOPS" if cub else "cuBLAS reference: n/a")
    lines.append(f"Top {len(rows)} of {res.get('n_combos', len(rows))} measured combos at "
                 f"{M}x{N}x{K}, by TFLOPS:")
    lines.append("")

    hdr = (f"{'#':>2}  {'TFLOPS':>7}  {'%cuBLAS':>7}  {'WS':>3} {'2CTA':>4}  "
           f"{'BN':>3} {'NS':>2} {'GSM':>3} {'NW':>2}  {'PERS':>4} "
           f"{'OV':>2} {'SPLIT':>5} {'L1NA':>4} {'TMA':>3} {'TMS':>3}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for i, r in enumerate(rows, 1):
        ws  = "on" if mc.toggles_for_dir(r["tier"])[0] else "off"
        cta = "on" if r.get("two_cta") else "off"
        vsc = f"{r['vs_cublas'] * 100:.0f}%" if r.get("vs_cublas") else "-"
        lines.append(f"{i:>2}  {r['tflops']:>7.0f}  {vsc:>7}  {ws:>3} {cta:>4}  "
                     f"{r['bn']:>3} {r['ns']:>2} {r['gsm']:>3} {r['nw']:>2}  "
                     f"{r['persistent']:>4} "
                     f"{r.get('overlap', 0):>2} {r.get('split_epilogue', 0):>5} "
                     f"{r.get('l1_no_alloc', 0):>4} {r.get('tma_pipelined', 0):>3} "
                     f"{r.get('tma_store_stages', 2):>3}")
    return "\n".join(lines) + "\n"


def print_leaderboard(res: dict, M: int, N: int, K: int, *, top: int,
                      title: str, done: int | None = None,
                      total: int | None = None,
                      progress: dict | None = None) -> None:
    print(render_leaderboard(res, M, N, K, top=top, title=title,
                             done=done, total=total, progress=progress),
          end="", flush=True)


def emit_live_block(block: str, state: dict, *, redraw: bool) -> None:
    if redraw and state.get("lines"):
        sys.stdout.write(f"\x1b[{state['lines']}F\x1b[J")
    sys.stdout.write(block)
    sys.stdout.flush()
    state["lines"] = len(block.splitlines())


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Terminal autotune: sweep valid knob combos on a B200, print the top configs.")
    ap.add_argument("shape", help="square 'S' or rectangular 'MxNxK' (e.g. 8192 or 32768x4608x768)")
    ap.add_argument("--scope", choices=["production", "full"], default="production",
                    help="production (default): warp-spec-on, BN=256/512, NS>=3; full: timed all-combo sweep")
    ap.add_argument("--top", type=int, default=10, help="how many top configs to print (default 10)")
    ap.add_argument("--timeout", type=int, default=3600, help="sweep timeout in seconds (default 3600)")
    ap.add_argument("--live-interval", type=float, default=3.0,
                    help="seconds between live leaderboard updates (default 3)")
    ap.add_argument("--live-redraw", choices=["auto", "always", "never"], default="auto",
                    help="update the same terminal table in place; default auto uses redraw only on a TTY")
    ap.add_argument("--warmup-ms", type=int, default=None,
                    help="override do_bench warmup window in ms for this run, e.g. 100")
    ap.add_argument("--rep-ms", type=int, default=None,
                    help="override do_bench repetition window in ms for this run, e.g. 200")
    ap.add_argument("--cublas-samples", type=int, default=None,
                    help="override measured cuBLAS samples; median is used, e.g. 10")
    ap.add_argument("--cublas-warmup-samples", type=int, default=None,
                    help="override throwaway cuBLAS samples before measured samples")
    ap.add_argument("--bn", default=None, help="override BN list, e.g. 128,256")
    ap.add_argument("--ns", default=None, help="override NS list, e.g. 3,4,5,6,7")
    ap.add_argument("--gsm", default=None, help="override GROUP_SIZE_M list")
    ap.add_argument("--nw", default=None, help="override NUM_WARPS list")
    ap.add_argument("--persistent", default=None, help="override PERSISTENT list")
    ap.add_argument("--two-cta", dest="two_cta", default=None,
                    help="override TWO_CTA list, e.g. 1 or 0,1")
    ap.add_argument("--overlap", default=None, help="override EPILOGUE_OVERLAP list")
    ap.add_argument("--split-epilogue", dest="split_epilogue", default=None,
                    help="override EPILOGUE_SPLIT list")
    ap.add_argument("--l1-no-alloc", dest="l1_no_alloc", default=None,
                    help="override EPILOGUE_L1_NO_ALLOC list")
    ap.add_argument("--tma-pipelined", dest="tma_pipelined", default=None,
                    help="override EPILOGUE_TMA_PIPELINED list")
    ap.add_argument("--tma-store-stages", dest="tma_store_stages", default=None,
                    help="override TMA_STORE_STAGES list")
    ap.add_argument("--single-tmem", dest="single_tmem", default=None,
                    help="override SINGLE_TMEM_ACCUM list")
    ap.add_argument("--single-tmem-policy", dest="single_tmem_policy",
                    choices=["all", "bn512-only"], default=None,
                    help="production pruning policy; default production uses bn512-only")
    args = ap.parse_args()
    if args.warmup_ms is not None and args.warmup_ms <= 0:
        ap.error("--warmup-ms must be positive")
    if args.rep_ms is not None and args.rep_ms <= 0:
        ap.error("--rep-ms must be positive")
    if args.cublas_samples is not None and args.cublas_samples <= 0:
        ap.error("--cublas-samples must be positive")
    if args.cublas_warmup_samples is not None and args.cublas_warmup_samples < 0:
        ap.error("--cublas-warmup-samples must be non-negative")

    M, N, K = parse_shape(args.shape)

    # Mirror the UI's scope -> (tier dirs, BN options).  The two warp-spec arms
    # share one dir (TWO_CTA distinguishes them); the sweep expands each dir to
    # all its arms, so each dir is passed once.
    ws_dirs  = list(dict.fromkeys(t["dir"] for k, t in mc.TIER_MAP.items() if t and k[0]))
    all_dirs = list(dict.fromkeys(t["dir"] for t in mc.TIER_MAP.values() if t))
    production = args.scope == "production"
    tier_dirs = ws_dirs if production else all_dirs
    filters: dict[str, list[int]] = {}
    if production:
        filters["bn"] = [256, 512]
        filters["ns"] = [x for x in mc.NS_OPTS if x >= 3]
        filters["two_cta"] = [1]
        filters["tma_store_stages"] = [1, 2]
        filters["single_tmem_policy"] = "bn512-only"
    else:
        filters["single_tmem_policy"] = "all"
    with_filter_override(filters, "bn", args.bn)
    with_filter_override(filters, "ns", args.ns)
    with_filter_override(filters, "gsm", args.gsm)
    with_filter_override(filters, "nw", args.nw)
    with_filter_override(filters, "persistent", args.persistent)
    with_filter_override(filters, "two_cta", args.two_cta)
    with_filter_override(filters, "overlap", args.overlap)
    with_filter_override(filters, "split_epilogue", args.split_epilogue)
    with_filter_override(filters, "l1_no_alloc", args.l1_no_alloc)
    with_filter_override(filters, "tma_pipelined", args.tma_pipelined)
    with_filter_override(filters, "tma_store_stages", args.tma_store_stages)
    with_filter_override(filters, "single_tmem", args.single_tmem)
    if args.single_tmem_policy is not None:
        filters["single_tmem_policy"] = args.single_tmem_policy

    print(f"# autotune {M}x{N}x{K}  scope={args.scope}  "
          f"(timing on the current GPU; this can take a while)", flush=True)
    if args.warmup_ms is not None or args.rep_ms is not None:
        warmup = args.warmup_ms if args.warmup_ms is not None else "driver-default"
        rep = args.rep_ms if args.rep_ms is not None else "driver-default"
        print(f"# do_bench override: warmup={warmup}ms rep={rep}ms", flush=True)
    if args.cublas_samples is not None or args.cublas_warmup_samples is not None:
        warmup_samples = (args.cublas_warmup_samples if args.cublas_warmup_samples is not None
                          else "driver-default")
        samples = args.cublas_samples if args.cublas_samples is not None else "driver-default"
        print(f"# cuBLAS sample override: warmup={warmup_samples} measured={samples}", flush=True)
    if filters:
        print(f"# filters={filters}", flush=True)

    job = autotune_start(tier_dirs, M, N, K, filters=filters,
                         bench_warmup_ms=args.warmup_ms,
                         bench_rep_ms=args.rep_ms,
                         cublas_samples=args.cublas_samples,
                         cublas_warmup_samples=args.cublas_warmup_samples)
    started = time.monotonic()
    last_print = 0.0
    last_done = -1
    last_total = None
    last_progress = None
    live_state = {"lines": 0}
    redraw = args.live_redraw == "always" or (
        args.live_redraw == "auto" and sys.stdout.isatty()
    )
    try:
        while True:
            done, total, finished = autotune_poll(job)
            progress = autotune_progress(job)
            now = time.monotonic()
            timed_out = args.timeout and (now - started) > args.timeout
            progress_changed = done != last_done or total != last_total or progress != last_progress
            should_print = finished or (progress_changed and (now - last_print) >= args.live_interval)
            if should_print:
                part = autotune_partial(job)
                title = "# live autotune leaderboard" if part.get("ok") else "# live autotune startup"
                block = render_leaderboard(part, M, N, K, top=args.top, title=title,
                                           done=done, total=total, progress=progress)
                emit_live_block(block, live_state, redraw=redraw)
                last_print = now
                last_done = done
                last_total = total
                last_progress = progress
            if finished:
                break
            if timed_out:
                try:
                    job["proc"].terminate()
                except Exception:
                    pass
                print(f"\nFAILED: autotune sweep timed out after {args.timeout}s")
                return 1
            time.sleep(max(1.0, min(args.live_interval, 5.0)))
    except KeyboardInterrupt:
        try:
            job["proc"].terminate()
        except Exception:
            pass
        print("\nInterrupted: autotune subprocess terminated.")
        return 130

    res = autotune_collect(job)

    if not res.get("ok"):
        if redraw and live_state.get("lines"):
            sys.stdout.write(f"\x1b[{live_state['lines']}F\x1b[J")
        print(f"\nFAILED: {res.get('error')}")
        if res.get("stderr"):
            print("--- driver stderr (tail) ---")
            print(res["stderr"])
        return 1

    final_block = render_leaderboard(res, M, N, K, top=args.top,
                                     title="# final autotune leaderboard")
    emit_live_block(final_block, live_state, redraw=redraw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
