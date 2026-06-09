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
        ld_width=knobs.get("ld_width", 8)))

    py = os.environ.get("MMCOMPOSER_PY", sys.executable)
    srun_args = shlex.split(os.environ.get("MMCOMPOSER_SRUN_ARGS", DEFAULT_SRUN_ARGS))
    cmd = ["srun", *srun_args, py, str(WORKER),
           "--kernel", str(kernel_path), "--symbol", tier["symbol"], "--out", str(out_path),
           "--cluster", str(int(tier["cluster"])), "--persistent", str(int(knobs.get("persistent", 0))),
           "--bm", str(knobs["bm"]), "--bn", str(knobs["bn"]), "--bk", str(knobs["bk"]),
           "--ns", str(knobs["ns"]), "--nw", str(knobs["nw"]),
           "--tma_store", str(knobs.get("tma_store", 0)),
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


def run_autotune(tier_dirs, M: int, N: int, K: int, bn_opts=None, timeout: int = 3000) -> dict:
    """Live sweep: srun the gpu_codegen_driver over every valid combo for the
    given tiers at one (M,N,K), then return the ranked results.

    Reuses the offline driver (parallel compile + fault-isolated run + cuBLAS),
    but writes to a TEMP matrix so the committed one is never touched.  Returns
    {ok, cublas_tflops, results:[{tier,bm..,tflops,vs_cublas,rel_err} sorted],
    n_combos, error}.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    bn_csv = ",".join(str(b) for b in bn_opts) if bn_opts else ""
    tag = hashlib.sha1((",".join(tier_dirs) + f"|{M}x{N}x{K}|bn{bn_csv}").encode()).hexdigest()[:16]
    out_matrix = SCRATCH / f"autotune_{tag}.json"
    if out_matrix.exists():
        out_matrix.unlink()

    py = os.environ.get("MMCOMPOSER_PY", sys.executable)
    srun_args = shlex.split(os.environ.get("MMCOMPOSER_AUTOTUNE_SRUN_ARGS", DEFAULT_AUTOTUNE_SRUN_ARGS))
    cmd = ["srun", *srun_args, py, str(DRIVER),
           "--perf-shapes", f"{M}x{N}x{K}",
           "--tiers", ",".join(tier_dirs),
           "--invalid-sample", "0",          # autotune wants valid combos only
           "--compat-out", str(out_matrix)]
    if bn_csv:
        cmd += ["--bn", bn_csv]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"autotune sweep timed out after {timeout}s"}

    if not out_matrix.exists():
        return {"ok": False, "error": "sweep produced no matrix (driver failed?)",
                "stderr": (proc.stderr or proc.stdout)[-1200:]}
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
                            "tflops": p["tflops"], "vs_cublas": p.get("vs_cublas"),
                            "rel_err": p.get("rel_err")})
    results.sort(key=lambda r: r["tflops"], reverse=True)
    return {"ok": bool(results), "cublas_tflops": cub, "results": results,
            "n_combos": len(results),
            "error": None if results else "no correct combos measured for this shape"}
