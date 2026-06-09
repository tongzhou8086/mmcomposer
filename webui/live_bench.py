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
# Autotune sweeps many combos, so it needs a longer allocation window.
DEFAULT_AUTOTUNE_SRUN_ARGS = "--partition=dedicated --gres=gpu:nvidia_b200:1 --time=01:00:00"


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

    knobs: {bm,bn,bk,ns,gsm,nw,tma_store,persistent}.  Returns a dict with
    ok/tflops/cublas_tflops/vs_cublas/rel_err/us (+ error/stderr on failure).
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    d = SCRATCH / _sig(tier, knobs, M, N, K)
    d.mkdir(exist_ok=True)
    kernel_path = d / "kernel.cu"
    out_path = d / "result.json"
    if out_path.exists():
        out_path.unlink()

    kernel_path.write_text(mc.render_kernel(
        tier, knobs["bm"], knobs["bn"], knobs["bk"], knobs["ns"], knobs["gsm"],
        knobs["nw"], tma_store=knobs.get("tma_store", 0),
        ld_width=knobs.get("ld_width", 8), overlap=knobs.get("overlap", 0)))

    py = os.environ.get("MMCOMPOSER_PY", sys.executable)
    srun_args = shlex.split(os.environ.get("MMCOMPOSER_SRUN_ARGS", DEFAULT_SRUN_ARGS))
    cmd = ["srun", *srun_args, py, str(WORKER),
           "--kernel", str(kernel_path), "--symbol", tier["symbol"], "--out", str(out_path),
           "--cluster", str(int(tier["cluster"])), "--persistent", str(int(knobs.get("persistent", 0))),
           "--bm", str(knobs["bm"]), "--bn", str(knobs["bn"]), "--bk", str(knobs["bk"]),
           "--ns", str(knobs["ns"]), "--nw", str(knobs["nw"]),
           "--tma_store", str(knobs.get("tma_store", 0)),
           "--overlap", str(knobs.get("overlap", 0)),
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


def _autotune_cmd(tier_dirs, M, N, K, out_matrix, bn_opts):
    py = os.environ.get("MMCOMPOSER_PY", sys.executable)
    srun_args = shlex.split(os.environ.get("MMCOMPOSER_AUTOTUNE_SRUN_ARGS", DEFAULT_AUTOTUNE_SRUN_ARGS))
    cmd = ["srun", *srun_args, py, str(DRIVER), "--perf-shapes", f"{M}x{N}x{K}",
           "--tiers", ",".join(tier_dirs), "--invalid-sample", "0", "--compat-out", str(out_matrix)]
    if bn_opts:
        cmd += ["--bn", ",".join(str(b) for b in bn_opts)]
    return cmd


def _out_matrix(tier_dirs, M, N, K, bn_opts):
    bn_csv = ",".join(str(b) for b in bn_opts) if bn_opts else ""
    tag = hashlib.sha1((",".join(tier_dirs) + f"|{M}x{N}x{K}|bn{bn_csv}").encode()).hexdigest()[:16]
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
                                                 "nw", "tma_store", "persistent")},
                            "ld_width": e.get("ld_width", 8), "overlap": e.get("overlap", 0),
                            "tflops": p["tflops"], "vs_cublas": p.get("vs_cublas"),
                            "rel_err": p.get("rel_err")})
    results.sort(key=lambda r: r["tflops"], reverse=True)
    return {"ok": bool(results), "cublas_tflops": cub, "results": results,
            "n_combos": len(results),
            "error": None if results else "no correct combos measured for this shape"}


def run_autotune(tier_dirs, M: int, N: int, K: int, bn_opts=None, timeout: int = 3000) -> dict:
    """Blocking sweep (for CLI/tests): srun the driver over every valid combo,
    then return ranked results.  Writes a TEMP matrix (committed one untouched)."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out_matrix = _out_matrix(tier_dirs, M, N, K, bn_opts)
    if out_matrix.exists():
        out_matrix.unlink()
    try:
        proc = subprocess.run(_autotune_cmd(tier_dirs, M, N, K, out_matrix, bn_opts),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"autotune sweep timed out after {timeout}s"}
    res = _rank_matrix(out_matrix, M, N, K)
    if not res.get("ok") and not res.get("stderr"):
        res["stderr"] = (proc.stderr or proc.stdout)[-1200:]
    return res


def autotune_start(tier_dirs, M, N, K, bn_opts=None) -> dict:
    """Launch the sweep in the BACKGROUND (non-blocking) for a UI progress bar.
    Uses a per-run jsonl (+ .nvalid sibling) so concurrent sweeps don't clobber
    each other's progress.  Returns a job dict for autotune_poll/collect."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out_matrix = _out_matrix(tier_dirs, M, N, K, bn_opts)
    jsonl = pathlib.Path(str(out_matrix)[:-5] + ".jsonl")   # autotune_<tag>.jsonl
    n_valid = pathlib.Path(str(jsonl) + ".nvalid")
    for f in (out_matrix, jsonl, n_valid):                   # clean slate for fresh progress
        try:
            f.unlink()
        except FileNotFoundError:
            pass
    cmd = _autotune_cmd(tier_dirs, M, N, K, out_matrix, bn_opts) + ["--jsonl", str(jsonl)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    return {"proc": proc, "out_matrix": str(out_matrix),
            "jsonl": str(jsonl), "n_valid": str(n_valid), "M": M, "N": N, "K": K}


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


def autotune_collect(job) -> dict:
    """Parse the finished sweep's matrix into ranked results."""
    res = _rank_matrix(job["out_matrix"], job["M"], job["N"], job["K"])
    if not res.get("ok") and not res.get("stderr"):
        try:
            res["stderr"] = (job["proc"].stderr.read() or "")[-1200:] if job["proc"].stderr else ""
        except Exception:
            pass
    return res
