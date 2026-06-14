"""On-the-fly B200 benchmarking for the webui.

Runs on the jump/login node (no GPU).  Renders the kernel, then submits the
compile+run to a B200 via `srun` and parses the worker's JSON.  Enabled only
when srun is available AND MMCOMPOSER_LIVE=1, so the public/Cloud deploy keeps
using the prebaked matrix.

Env overrides:
  MMCOMPOSER_LIVE=1                 enable live mode
  MMCOMPOSER_PY=<python>            interpreter for the worker (default: this one)
  MMCOMPOSER_SRUN_ARGS="..."        srun flags (default: B200 dedicated, 10 min)
"""
import os
import sys
import json
import shlex
import subprocess
import hashlib
import pathlib

import mvp_core as mc

HERE = pathlib.Path(__file__).resolve().parent
SCRATCH = HERE / "tests" / "_scratch" / "live"
WORKER = HERE / "_live_bench_worker.py"
DRIVER = HERE / "tests" / "gpu_codegen_driver.py"

DEFAULT_SRUN_ARGS = "--partition=dedicated --gres=gpu:nvidia_b200:1 --time=00:10:00"
# Autotune sweeps many combos with the driver do_bench window
# (~1s/combo), so a big grid (>1k combos) needs a multi-hour allocation.  The
# job releases when the sweep finishes — this is just the ceiling.
DEFAULT_AUTOTUNE_SRUN_ARGS = "--partition=dedicated --gres=gpu:nvidia_b200:1 --time=02:00:00"
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


def live_available() -> bool:
    """True if we're on a node that can launch B200 jobs and live mode is on."""
    if os.environ.get("MMCOMPOSER_LIVE", "0") != "1":
        return False
    from shutil import which
    return which("srun") is not None


def _sig(tier, knobs, M, N, K) -> str:
    raw = f"{tier['dir']}|{knobs}|{M}x{N}x{K}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def run_live_bench(tier, knobs: dict, M: int, N: int, K: int, timeout: int = 900) -> dict:
    """Render → srun(compile+run+cuBLAS) → parsed result dict.

    knobs: {bm,bn,bk,ns,gsm,nw,persistent,ld_width,overlap,split_epilogue,l1_no_alloc,tma_pipelined,tma_store_stages,single_tmem}.
    Returns a dict with
    ok/tflops/cublas_tflops/vs_cublas/rel_err/us (+ error/stderr on failure).
    """
    knobs = dict(knobs)
    knobs["tma_store_stages"] = mc.normalize_tma_store_stages(
        knobs.get("tma_pipelined", 0), knobs.get("tma_store_stages", 2))
    SCRATCH.mkdir(parents=True, exist_ok=True)
    d = SCRATCH / _sig(tier, knobs, M, N, K)
    d.mkdir(exist_ok=True)
    kernel_path = d / "kernel.cu"
    out_path = d / "result.json"
    if out_path.exists():
        out_path.unlink()

    kernel_path.write_text(mc.render_kernel(
        tier, knobs["bm"], knobs["bn"], knobs["bk"], knobs["ns"], knobs["gsm"],
        knobs["nw"], ld_width=knobs.get("ld_width", 8), overlap=knobs.get("overlap", 0),
        split_epilogue=knobs.get("split_epilogue", 0),
        l1_no_alloc=knobs.get("l1_no_alloc", 0),
        tma_pipelined=knobs.get("tma_pipelined", 0),
        tma_store_stages=knobs.get("tma_store_stages", 2),
        single_tmem=knobs.get("single_tmem", 0)))

    py = os.environ.get("MMCOMPOSER_PY", sys.executable)
    srun_args = shlex.split(os.environ.get("MMCOMPOSER_SRUN_ARGS", DEFAULT_SRUN_ARGS))
    cmd = ["srun", *srun_args, py, str(WORKER),
           "--kernel", str(kernel_path), "--symbol", tier["symbol"], "--out", str(out_path),
           "--cluster", str(int(tier["cluster"])), "--persistent", str(int(knobs.get("persistent", 0))),
           "--bm", str(knobs["bm"]), "--bn", str(knobs["bn"]), "--bk", str(knobs["bk"]),
           "--ns", str(knobs["ns"]), "--nw", str(knobs["nw"]),
           "--overlap", str(knobs.get("overlap", 0)),
           "--split_epilogue", str(knobs.get("split_epilogue", 0)),
           "--l1_no_alloc", str(knobs.get("l1_no_alloc", 0)),
           "--tma_pipelined", str(knobs.get("tma_pipelined", 0)),
           "--tma_store_stages", str(knobs.get("tma_store_stages", 2)),
           "--single_tmem", str(knobs.get("single_tmem", 0)),
           "-M", str(M), "-N", str(N), "-K", str(K)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"srun timed out after {timeout}s (queue/allocation?)"}

    if out_path.exists():
        try:
            res = json.loads(out_path.read_text())
            if not res.get("ok") and not res.get("error"):
                res["error"] = "kernel ran but output was incorrect (rel_err over tolerance)"
            return res
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"could not parse worker output: {e}",
                    "stderr": proc.stderr[-800:]}
    return {"ok": False, "error": "worker produced no result (compile/launch failed?)",
            "stderr": (proc.stderr or proc.stdout)[-800:]}


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
                  use_srun: bool = True, bench_warmup_ms: int | None = None,
                  bench_rep_ms: int | None = None,
                  cublas_samples: int | None = None,
                  cublas_warmup_samples: int | None = None):
    py = os.environ.get("MMCOMPOSER_PY", sys.executable)
    driver_cmd = [py, str(DRIVER), "--perf-shapes", f"{M}x{N}x{K}",
                  "--tiers", ",".join(tier_dirs), "--invalid-sample", "0",
                  "--mode", mode, "--compat-out", str(out_matrix)]
    if bench_warmup_ms is not None:
        driver_cmd += ["--bench-warmup-ms", str(bench_warmup_ms)]
    if bench_rep_ms is not None:
        driver_cmd += ["--bench-rep-ms", str(bench_rep_ms)]
    if cublas_warmup_samples is not None:
        driver_cmd += ["--cublas-warmup-samples", str(cublas_warmup_samples)]
    if cublas_samples is not None:
        driver_cmd += ["--cublas-samples", str(cublas_samples)]
    if use_srun:
        srun_args = shlex.split(os.environ.get("MMCOMPOSER_AUTOTUNE_SRUN_ARGS", DEFAULT_AUTOTUNE_SRUN_ARGS))
        cmd = ["srun", *srun_args, *driver_cmd]
    else:
        cmd = driver_cmd
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


def run_autotune(tier_dirs, M: int, N: int, K: int, filters=None, timeout: int = 3000,
                 bn_opts=None, use_srun: bool = True,
                 bench_warmup_ms: int | None = None,
                 bench_rep_ms: int | None = None,
                 cublas_samples: int | None = None,
                 cublas_warmup_samples: int | None = None) -> dict:
    """Blocking timing sweep: srun the driver in perf mode, then rank results.

    Writes a TEMP matrix (committed one untouched).  ``bn_opts`` is accepted
    only for older callers; new code should pass the general ``filters`` dict.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    filters = _normal_filters(filters, bn_opts)
    out_matrix = _out_matrix(tier_dirs, M, N, K, filters)
    jsonl = pathlib.Path(str(out_matrix)[:-5] + ".jsonl")
    n_valid = pathlib.Path(str(jsonl) + ".nvalid")
    cublas = pathlib.Path(str(jsonl) + ".cublas")
    progress = pathlib.Path(str(jsonl) + ".progress")
    for f in (out_matrix, jsonl, n_valid, cublas, progress):
        try:
            f.unlink()
        except FileNotFoundError:
            pass
    try:
        cmd = _autotune_cmd(tier_dirs, M, N, K, out_matrix, filters,
                            mode="perf", use_srun=use_srun,
                            bench_warmup_ms=bench_warmup_ms,
                            bench_rep_ms=bench_rep_ms,
                            cublas_samples=cublas_samples,
                            cublas_warmup_samples=cublas_warmup_samples) + ["--jsonl", str(jsonl)]
        proc = subprocess.run(cmd,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"autotune sweep timed out after {timeout}s"}
    res = _rank_matrix(out_matrix, M, N, K)
    if not res.get("ok") and not res.get("stderr"):
        res["stderr"] = (proc.stderr or proc.stdout)[-1200:]
    return res


def autotune_start(tier_dirs, M, N, K, filters=None, bn_opts=None,
                   use_srun: bool = True, bench_warmup_ms: int | None = None,
                   bench_rep_ms: int | None = None,
                   cublas_samples: int | None = None,
                   cublas_warmup_samples: int | None = None) -> dict:
    """Launch the sweep in the BACKGROUND (non-blocking) for a UI progress bar.
    Uses a per-run jsonl (+ .nvalid sibling) so concurrent sweeps don't clobber
    each other's progress.  Returns a job dict for autotune_poll/collect."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    filters = _normal_filters(filters, bn_opts)
    out_matrix = _out_matrix(tier_dirs, M, N, K, filters)
    jsonl = pathlib.Path(str(out_matrix)[:-5] + ".jsonl")   # autotune_<tag>.jsonl
    n_valid = pathlib.Path(str(jsonl) + ".nvalid")
    cublas = pathlib.Path(str(jsonl) + ".cublas")
    progress = pathlib.Path(str(jsonl) + ".progress")
    for f in (out_matrix, jsonl, n_valid, cublas, progress): # clean slate for fresh progress
        try:
            f.unlink()
        except FileNotFoundError:
            pass
    cmd = _autotune_cmd(tier_dirs, M, N, K, out_matrix, filters,
                        mode="perf", use_srun=use_srun,
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
    """Rank the results that have streamed into the jsonl SO FAR, for a live
    leaderboard during the sweep.  Reads the per-run jsonl (one line per combo)
    + the .cublas sidecar (written before the sweep) for vs_cublas.  Same result
    shape as _rank_matrix; ok=False (quietly) until the first correct combo lands."""
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
                    continue          # half-written trailing line — skip
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
