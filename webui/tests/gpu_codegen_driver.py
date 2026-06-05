"""B200 codegen driver — render every valid knob combo, compile, run, check.

This is the on-GPU half of the mmcomposer MVP's standard test.  It runs
on a B200 node (one process) and, for each validator-*passing* config:

  1. renders the tier's kernel.cu with the knob values substituted,
  2. compiles it with nvcc,
  3. launches it at one small shape, and
  4. checks correctness against a torch reference.

The premise being tested: **"the validator says valid" must imply
"compiles + runs + numerically correct."**  A sample of validator-
*failing* combos is also run to confirm they genuinely fail (so the
validator isn't rejecting things that actually work).

Usage (from repo root, on a GPU node):
    srun ... python webui/tests/gpu_codegen_driver.py [--shape 2048] \
        [--tiers tier1_baseline,tier2_multistage_ws,tier3_cluster_swizzle] \
        [--invalid-sample 12] [--json out.json]

Results are printed as a table and (optionally) written as JSON.
"""

from __future__ import annotations

import argparse
import ctypes
import itertools
import json
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

WEBUI = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEBUI))
sys.path.insert(0, str(WEBUI / "kernels"))

import mvp_core as mc
import _runtime as rt

import numpy as np
import torch
from cuda.bindings import driver


SCRATCH = WEBUI / "tests" / "_scratch" / "gpu_driver"


def all_combos(tier_dirs):
    """Yield (tier_key, tier, knobs-dict) over the full dropdown grid."""
    dir_to_key = {t["dir"]: k for k, t in mc.TIER_MAP.items() if t}
    for tdir in tier_dirs:
        key = dir_to_key[tdir]
        tier = mc.TIER_MAP[key]
        for bm, bn, bk, ns, gsm, nw in itertools.product(
            mc.BM_OPTS, mc.BN_OPTS, mc.BK_OPTS, mc.NS_OPTS, mc.GSM_OPTS, mc.NW_OPTS
        ):
            yield tier, dict(bm=bm, bn=bn, bk=bk, ns=ns, gsm=gsm, nw=nw)


def launch_spec(tier, k, M, N, K):
    """Compute (grid, block, shared_bytes) for a config at shape (M,N,K)."""
    cta_group = 2 if tier["cluster"] else 1
    bn_local  = k["bn"] // cta_group
    a_slot = k["bm"] * k["bk"] * 2
    b_slot = bn_local * k["bk"] * 2
    slot   = a_slot + b_slot
    epi    = k["bm"] * (k["bn"] + 8) * 2
    shared = max(k["ns"] * slot, epi) + 1024
    block  = (k["nw"] * 32, 1, 1)
    if tier["cluster"]:
        grid_m_clusters = M // (cta_group * k["bm"])
        grid_n          = N // k["bn"]
        grid = (grid_m_clusters * grid_n * cta_group, 1, 1)
    else:
        grid = ((M // k["bm"]) * (N // k["bn"]), 1, 1)
    return grid, block, shared


def tag_for(tier, k):
    return (f"{tier['dir']}_bm{k['bm']}_bn{k['bn']}_bk{k['bk']}"
            f"_ns{k['ns']}_gsm{k['gsm']}_nw{k['nw']}")


def render_to_dir(tier, k):
    """Write the substituted kernel.cu; return its path."""
    d = SCRATCH / tag_for(tier, k)
    d.mkdir(parents=True, exist_ok=True)
    src = mc.render_kernel(tier, k["bm"], k["bn"], k["bk"], k["ns"], k["gsm"], k["nw"])
    p = d / "kernel.cu"
    p.write_text(src)
    return p


def _compile_worker(job):
    """Top-level so it pickles for the process pool.  nvcc only — no CUDA."""
    src_path, arch = job
    cubin = src_path[:-3] + f"_{arch}.cubin"
    nvcc = os.environ.get("NVCC", "nvcc")
    cmd = [nvcc, f"-arch={arch}", "-O3", "--std=c++17", "--cubin", src_path, "-o", cubin]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return src_path, r.returncode, (r.stderr[-600:] if r.returncode else "")


def launch_from_cubin(tier, k, device, arch, M, N, K, A=None, B=None, C_ref=None):
    """Load the (already compiled) cubin, launch, check correctness.

    A/B/C_ref may be passed in precomputed (they depend only on the
    shape + seed, not the kernel) so the worker computes the reference
    matmul once and reuses it across every combo."""
    src_path = str(SCRATCH / tag_for(tier, k) / "kernel.cu")
    cubin_path = src_path[:-3] + f"_{arch}.cubin"
    res = {"tier": tier["dir"], **k, "launched": False, "rel_err": None,
           "correct": False, "error": None}
    mod = None
    try:
        with open(cubin_path, "rb") as f:
            cubin = f.read()
        mod = rt.cu(driver.cuModuleLoadData(cubin))
        kernel = rt.cu(driver.cuModuleGetFunction(mod, tier["symbol"].encode()))
        grid, block, shared = launch_spec(tier, k, M, N, K)
        rt.cu(driver.cuFuncSetAttribute(
            kernel, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared))

        if A is None:
            torch.manual_seed(0)
            A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
            C_ref = (A.float() @ B.float()).to(torch.bfloat16)
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        A_tmap = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=A.data_ptr(),
            global_dim=[K, M], global_strides=[K * 2], box_dim=[k["bk"], k["bm"]],
            element_strides=[1, 1], swizzle=rt.TMA_SWIZZLE_128B)
        B_tmap = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=B.data_ptr(),
            global_dim=[N, K], global_strides=[N * 2], box_dim=[64, k["bk"]],
            element_strides=[1, 1], swizzle=rt.TMA_SWIZZLE_128B)
        args = [(ctypes.c_byte * 128).from_buffer_copy(A_tmap.tobytes()),
                (ctypes.c_byte * 128).from_buffer_copy(B_tmap.tobytes()),
                ctypes.c_void_p(C.data_ptr()),
                ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)]
        rt.launch(kernel, grid=grid, block=block, shared=shared, args=args)
        res["launched"] = True
        rel = (C.float() - C_ref.float()).abs().max().item() / C_ref.float().abs().max().item()
        res["rel_err"] = rel
        res["correct"] = rel < 5e-2
    except Exception as e:  # noqa: BLE001 — record any failure, keep going
        res["error"] = f"{type(e).__name__}: {e}"
        try:
            driver.cuCtxSynchronize()
        except Exception:
            pass
    finally:
        if mod is not None:
            try:
                rt.cu(driver.cuModuleUnload(mod))
            except Exception:
                pass
    return res


def build_to_run(tier_dirs, invalid_sample):
    """Deterministic ordered list of (tier, knobs, label).  Both the
    orchestrator and the isolated worker call this so indices line up."""
    valid, invalid = [], []
    for tier, k in all_combos(tier_dirs):
        warnings = mc.validate_config(k["bm"], k["bn"], k["bk"], k["ns"], k["gsm"], k["nw"],
                                      cluster=tier["cluster"])
        (valid if not warnings else invalid).append((tier, k))
    stepi = max(1, len(invalid) // max(1, invalid_sample))
    inv_sample = invalid[::stepi][:invalid_sample]
    to_run = [(t, k, "valid") for (t, k) in valid] + [(t, k, "invalid") for (t, k) in inv_sample]
    return to_run, len(valid), len(invalid), len(inv_sample)


def worker_loop(to_run, start, device, arch, M, N, K, jsonl_path):
    """Launch combos [start:] in one CUDA context, appending one JSON
    line per combo.  On any CUDA fault the context is poisoned, so we
    record the offending combo and exit non-zero; the orchestrator
    respawns a fresh worker at the next index."""
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    C_ref = (A.float() @ B.float()).to(torch.bfloat16)
    f = open(jsonl_path, "a")
    for idx in range(start, len(to_run)):
        tier, k, label = to_run[idx]
        r = launch_from_cubin(tier, k, device, arch, M, N, K, A=A, B=B, C_ref=C_ref)
        r["validator"] = label
        r["idx"] = idx
        f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())
        if r["error"] and ("ILLEGAL" in r["error"] or "LAUNCH" in r["error"] or "launch" in r["error"]):
            f.close()
            sys.exit(3)   # context likely poisoned — orchestrator resumes past idx
    f.close()
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=int, default=2048)
    ap.add_argument("--tiers", default="tier1_baseline,tier2_multistage_ws,tier3_cluster_swizzle")
    ap.add_argument("--invalid-sample", type=int, default=12)
    ap.add_argument("--json", default=None)
    ap.add_argument("--compat-out", default=None,
                    help="path for the committed compatibility matrix (default webui/kernels/compat_matrix.json)")
    ap.add_argument("--launch-worker", type=int, default=None,
                    help="internal: run the isolated launch worker from this index")
    ap.add_argument("--jsonl", default=None, help="internal: worker append path")
    args = ap.parse_args()

    M = N = K = args.shape
    tier_dirs = args.tiers.split(",")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    to_run, n_valid, n_invalid, n_sample = build_to_run(tier_dirs, args.invalid_sample)

    device, _ = rt.init_cuda()
    arch = rt.compute_arch(device)

    # ── Worker mode: just launch from `start`, stream results, exit ──
    if args.launch_worker is not None:
        worker_loop(to_run, args.launch_worker, device, arch, M, N, K, args.jsonl)
        return  # unreachable (worker_loop exits)

    print(f"# shape {M}^3 | arch={arch} | tiers={tier_dirs}")
    print(f"# {n_valid} valid combos to run, {n_invalid} invalid ({n_sample} sampled)")

    # ── Phase 1: render + parallel nvcc compile (CPU-bound) ──────────
    for (t, k, _) in to_run:
        render_to_dir(t, k)
    jobs = [(str(SCRATCH / tag_for(t, k) / "kernel.cu"), arch) for (t, k, _) in to_run]
    workers = min(32, (os.cpu_count() or 8))
    print(f"# compiling {len(jobs)} kernels with {workers} workers ...", flush=True)
    n_comp_ok = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for src_path, rc, stderr in ex.map(_compile_worker, jobs):
            n_comp_ok += (rc == 0)
    print(f"# compiled OK: {n_comp_ok}/{len(jobs)}", flush=True)

    # ── Phase 2: supervised isolated launches ───────────────────────
    # A fresh worker process runs until it hits a CUDA fault (sticky →
    # poisons the context), records it, and exits; we resume past it.
    jsonl = args.jsonl or str(SCRATCH / "results.jsonl")
    if os.path.exists(jsonl):
        os.remove(jsonl)
    next_idx = 0
    n_spawns = 0
    while next_idx < len(to_run):
        n_spawns += 1
        cmd = [sys.executable, os.path.abspath(__file__),
               "--launch-worker", str(next_idx), "--jsonl", jsonl,
               "--shape", str(M), "--tiers", args.tiers,
               "--invalid-sample", str(args.invalid_sample)]
        subprocess.run(cmd)
        done = set()
        if os.path.exists(jsonl):
            with open(jsonl) as fh:
                for line in fh:
                    try:
                        done.add(json.loads(line)["idx"])
                    except Exception:
                        pass
        if done:
            highest = max(done)
            resume = highest + 1
            # If the worker died without even recording `next_idx`, mark
            # it crashed so we make forward progress.
            if next_idx not in done and resume <= next_idx:
                resume = next_idx + 1
            next_idx = max(resume, next_idx + (0 if next_idx in done else 1))
        else:
            next_idx += 1  # worker crashed before recording anything

    # ── Collect + summarize ─────────────────────────────────────────
    results = {}
    with open(jsonl) as fh:
        for line in fh:
            try:
                r = json.loads(line)
                results[r["idx"]] = r
            except Exception:
                pass
    # Any index never recorded (worker crashed mid-init) → mark bad.
    for idx, (tier, k, label) in enumerate(to_run):
        if idx not in results:
            results[idx] = {"tier": tier["dir"], **k, "validator": label,
                            "launched": False, "rel_err": None, "correct": False,
                            "error": "worker crashed before recording"}
    ordered = [results[i] for i in range(len(to_run))]

    bad = [r for r in ordered if r["validator"] == "valid" and not r["correct"]]
    surprises = [r for r in ordered if r["validator"] == "invalid" and r["correct"]]
    for r in bad:
        print(f"BAD  {r['tier']:24} bn{r['bn']:>3} ns{r['ns']} gsm{r['gsm']:>2} nw{r['nw']:>2}  "
              f"rel={r['rel_err']}  {r['error'] or ''}")
    for r in surprises:
        print(f"INVALID-but-WORKS  {r['tier']} bn{r['bn']} ns{r['ns']} gsm{r['gsm']} nw{r['nw']}  rel={r['rel_err']}")

    print("\n=== SUMMARY ===")
    print(f"valid combos run:        {n_valid}   (worker spawns: {n_spawns})")
    print(f"  compiled OK:           {n_comp_ok}/{len(jobs)} (incl. invalid sample)")
    print(f"  validator-valid BAD:   {len(bad)}   (must be 0)")
    print(f"invalid combos sampled:  {n_sample}")
    print(f"  invalid-but-correct:   {len(surprises)}   (investigate if > 0)")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(ordered, indent=2))
        print(f"wrote {args.json}")

    # ── Committed compatibility matrix (the app filters against this) ─
    compat_path = args.compat_out or str(WEBUI / "kernels" / "compat_matrix.json")
    entries = [{k: r[k] for k in ("tier", "bm", "bn", "bk", "ns", "gsm", "nw")}
               | {"correct": bool(r["correct"]),
                  "rel_err": (round(r["rel_err"], 5) if r["rel_err"] is not None else None)}
               for r in ordered if r["validator"] == "valid"]
    matrix = {
        "generated_by": "webui/tests/gpu_codegen_driver.py",
        "arch": arch,
        "validated_shape": [M, N, K],
        "tolerance_rel_err": 5e-2,
        "n_entries": len(entries),
        "n_correct": sum(e["correct"] for e in entries),
        "entries": entries,
    }
    pathlib.Path(compat_path).write_text(json.dumps(matrix, indent=2))
    print(f"wrote compat matrix: {compat_path} ({matrix['n_correct']}/{matrix['n_entries']} correct)")

    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
