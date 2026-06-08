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

DEFAULT_SRUN_ARGS = "--partition=dedicated --gres=gpu:nvidia_b200:1 --time=00:10:00"


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
        knobs["nw"], tma_store=knobs.get("tma_store", 0)))

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
