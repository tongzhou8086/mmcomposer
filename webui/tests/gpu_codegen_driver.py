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


def all_combos(tier_dirs, bn_opts=None):
    """Yield (tier_key, tier, knobs-dict) over the dropdown grid.

    bn_opts restricts BN for the production scope (default: all of mc.BN_OPTS).
    TCGEN05_LD_WIDTH is a first-class swept dimension like NW/NS/etc.
    """
    bn_list = bn_opts if bn_opts else mc.BN_OPTS
    ld_list = mc.TCGEN05_LD_WIDTH_OPTS
    dir_to_key = {t["dir"]: k for k, t in mc.TIER_MAP.items() if t}
    for tdir in tier_dirs:
        key = dir_to_key[tdir]
        tier = mc.TIER_MAP[key]
        # PERSISTENT is a launch knob (same cubin) — only the persistent-
        # capable tiers get the grid=#SMs variant; others stay at [0].
        pers_opts = mc.PERSISTENT_OPTS if tier.get("persistent_ok") else [0]
        # EPILOGUE_OVERLAP only applies on the persistent-capable path; most
        # overlap=1 combos are filtered by the validator (persistent/NW/SMEM).
        ov_opts = mc.EPILOGUE_OVERLAP_OPTS if tier.get("persistent_ok") else [0]
        for bm, bn, bk, ns, gsm, nw, ts, pers, ldw, ov in itertools.product(
            mc.BM_OPTS, bn_list, mc.BK_OPTS, mc.NS_OPTS, mc.GSM_OPTS, mc.NW_OPTS,
            mc.TMA_STORE_OPTS, pers_opts, ld_list, ov_opts
        ):
            yield tier, dict(bm=bm, bn=bn, bk=bk, ns=ns, gsm=gsm, nw=nw,
                             tma_store=ts, persistent=pers, ld_width=ldw, overlap=ov)


def parse_perf_shapes(spec):
    """Parse --perf-shapes: 'S' -> square (S,S,S); 'MxNxK' -> rectangular."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "x" in tok.lower():
            M, N, K = (int(v) for v in tok.lower().split("x"))
        else:
            M = N = K = int(tok)
        out.append((M, N, K))
    return out


def shape_compatible(tier, k, M, N, K):
    """Can this combo's tile geometry tile (M, N, K) exactly?  The cluster
    tier needs M divisible by CTA_GROUP*BM (2 row-blocks per cluster); all
    tiers need M%BM, N%BN, K%BK == 0.  Incompatible (tier, shape) pairs are
    skipped at sweep time, not recorded as failures."""
    bm, bn, bk = k["bm"], k["bn"], k["bk"]
    if M % bm or N % bn or K % bk:
        return False
    if tier["cluster"] and (M % (2 * bm)):
        return False
    return True


def launch_spec(tier, k, M, N, K, num_sms=None):
    """Compute (grid, block, shared_bytes) for a config at shape (M,N,K)."""
    cta_group = 2 if tier["cluster"] else 1
    bn_local  = k["bn"] // cta_group
    a_slot = k["bm"] * k["bk"] * 2
    b_slot = bn_local * k["bk"] * 2
    slot   = a_slot + b_slot
    epi    = k["bm"] * (k["bn"] if k["tma_store"] else k["bn"] + 8) * 2
    # Overlap runs ring + epilogue staging concurrently -> disjoint (ring+epi).
    shared = ((k["ns"] * slot + epi) if k.get("overlap", 0) else max(k["ns"] * slot, epi)) + 1024
    # Overlap: 2 stream warps (TMA+MMA) in warpgroup 0 + nw epilogue warps from
    # warp 4 (warps 2,3 idle for the warpgroup boundary) -> (nw+4) warps.
    block  = (((k["nw"] + 4) * 32 if k.get("overlap", 0) else k["nw"] * 32), 1, 1)
    if k.get("persistent") and num_sms:
        # Persistent grid: one CTA per SM; the kernel's tile loop walks the rest.
        # For the cluster tier the grid must stay a multiple of CTA_GROUP.
        grid = (num_sms - num_sms % cta_group, 1, 1)
    elif tier["cluster"]:
        grid_m_clusters = M // (cta_group * k["bm"])
        grid_n          = N // k["bn"]
        grid = (grid_m_clusters * grid_n * cta_group, 1, 1)
    else:
        grid = ((M // k["bm"]) * (N // k["bn"]), 1, 1)
    return grid, block, shared


def tag_for(tier, k):
    # ld_width changes the cubin (epilogue constexpr), so it's in the tag;
    # persistent does NOT (launch-only, same cubin) so it stays out.
    return (f"{tier['dir']}_bm{k['bm']}_bn{k['bn']}_bk{k['bk']}"
            f"_ns{k['ns']}_gsm{k['gsm']}_nw{k['nw']}_ts{k['tma_store']}"
            f"_ld{k.get('ld_width', 8)}_ov{k.get('overlap', 0)}")


def render_to_dir(tier, k):
    """Write the substituted kernel.cu; return its path."""
    d = SCRATCH / tag_for(tier, k)
    d.mkdir(parents=True, exist_ok=True)
    src = mc.render_kernel(tier, k["bm"], k["bn"], k["bk"], k["ns"], k["gsm"], k["nw"],
                           tma_store=k["tma_store"], ld_width=k.get("ld_width", 8),
                           overlap=k.get("overlap", 0))
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


def launch_from_cubin(tier, k, arch, shapes, do_bench=True, num_sms=None):
    """Load the (already compiled) cubin; per shape check correctness and
    (optionally) benchmark with do_bench.

    ``shapes`` is a list of dicts {M, N, K, A, B, C, C_ref} with tensors
    precomputed once by the worker and reused across every combo.  Returns
    a result with overall ``correct`` and a per-shape ``perf`` map of
    {rel_err, correct, us, tflops}."""
    src_path = str(SCRATCH / tag_for(tier, k) / "kernel.cu")
    cubin_path = src_path[:-3] + f"_{arch}.cubin"
    res = {"tier": tier["dir"], **k, "launched": False, "correct": False,
           "error": None, "perf": {}}
    mod = None
    try:
        with open(cubin_path, "rb") as f:
            cubin = f.read()
        mod = rt.cu(driver.cuModuleLoadData(cubin))
        kernel = rt.cu(driver.cuModuleGetFunction(mod, tier["symbol"].encode()))
        overall = True
        for sh in shapes:
            M, N, K = sh["M"], sh["N"], sh["K"]
            if not shape_compatible(tier, k, M, N, K):
                continue   # tile geometry can't tile this shape (e.g. cluster, M%256!=0)
            grid, block, shared = launch_spec(tier, k, M, N, K, num_sms)
            rt.cu(driver.cuFuncSetAttribute(
                kernel, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared))
            A_tmap = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=sh["A"].data_ptr(),
                global_dim=[K, M], global_strides=[K * 2], box_dim=[k["bk"], k["bm"]],
                element_strides=[1, 1], swizzle=rt.TMA_SWIZZLE_128B)
            B_tmap = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=sh["B"].data_ptr(),
                global_dim=[N, K], global_strides=[N * 2], box_dim=[64, k["bk"]],
                element_strides=[1, 1], swizzle=rt.TMA_SWIZZLE_128B)
            # Store-side descriptor (used only when TMA_STORE=1; always passed
            # since the kernel signature carries C_tmap).  SWIZZLE_NONE matches
            # the dense SMEM staging.
            C_tmap = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=sh["C"].data_ptr(),
                global_dim=[N, M], global_strides=[N * 2], box_dim=[k["bn"], k["bm"]],
                element_strides=[1, 1], swizzle=rt.TMA_SWIZZLE_NONE)
            args = [(ctypes.c_byte * 128).from_buffer_copy(A_tmap.tobytes()),
                    (ctypes.c_byte * 128).from_buffer_copy(B_tmap.tobytes()),
                    (ctypes.c_byte * 128).from_buffer_copy(C_tmap.tobytes()),
                    ctypes.c_void_p(sh["C"].data_ptr()),
                    ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)]
            sh["C"].zero_()
            rt.launch(kernel, grid=grid, block=block, shared=shared, args=args)
            res["launched"] = True
            rel = (sh["C"].float() - sh["C_ref"].float()).abs().max().item() \
                / sh["C_ref"].float().abs().max().item()
            correct = rel < 5e-2
            overall &= correct
            entry = {"rel_err": rel, "correct": correct, "us": None, "tflops": None}
            if do_bench and correct:
                us = rt.time_kernel_us(lambda: rt.launch(
                    kernel, grid=grid, block=block, shared=shared, args=args, sync=False))
                entry["us"] = us
                entry["tflops"] = (2.0 * M * N * K) / (us * 1e-6) / 1e12
            res["perf"][mc.shape_key(M, N, K)] = entry
        res["correct"] = overall
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


def build_to_run(tier_dirs, invalid_sample, bn_opts=None):
    """Deterministic ordered list of (tier, knobs, label).  Both the
    orchestrator and the isolated worker call this so indices line up
    (so bn_opts must be passed identically in both)."""
    valid, invalid = [], []
    for tier, k in all_combos(tier_dirs, bn_opts):
        warnings = mc.validate_config(k["bm"], k["bn"], k["bk"], k["ns"], k["gsm"], k["nw"],
                                      cluster=tier["cluster"], tma_store=k["tma_store"],
                                      persistent=k.get("persistent", 0),
                                      persistent_ok=tier.get("persistent_ok", False),
                                      ld_width=k.get("ld_width", 8),
                                      overlap=k.get("overlap", 0))
        (valid if not warnings else invalid).append((tier, k))
    stepi = max(1, len(invalid) // max(1, invalid_sample))
    inv_sample = invalid[::stepi][:invalid_sample]
    to_run = [(t, k, "valid") for (t, k) in valid] + [(t, k, "invalid") for (t, k) in inv_sample]
    return to_run, len(valid), len(invalid), len(inv_sample)


def make_shapes(shape_list):
    """Precompute A, B, C, C_ref tensors once per shape (seed-fixed)."""
    shapes = []
    for (M, N, K) in shape_list:
        torch.manual_seed(0)
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        shapes.append({"M": M, "N": N, "K": K, "A": A, "B": B,
                       "C": torch.zeros(M, N, dtype=torch.bfloat16, device="cuda"),
                       "C_ref": (A.float() @ B.float()).to(torch.bfloat16)})
    return shapes


def worker_loop(to_run, start, arch, shape_list, jsonl_path, num_sms=None):
    """Launch combos [start:] in one CUDA context, appending one JSON
    line per combo (with per-shape correctness + perf).  On any CUDA
    fault the context is poisoned, so we record the offending combo and
    exit non-zero; the orchestrator respawns a fresh worker past idx."""
    shapes = make_shapes(shape_list)
    f = open(jsonl_path, "a")
    for idx in range(start, len(to_run)):
        tier, k, label = to_run[idx]
        r = launch_from_cubin(tier, k, arch, shapes, do_bench=(label == "valid"), num_sms=num_sms)
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
    ap.add_argument("--perf-shapes", default="4096,8192",
                    help="comma-separated square shapes (M=N=K) to check + benchmark")
    ap.add_argument("--tiers", default="tier1_baseline,tier2_multistage_ws,tier3_cluster_swizzle")
    ap.add_argument("--invalid-sample", type=int, default=12)
    ap.add_argument("--bn", default=None,
                    help="comma-separated BN values to sweep (default: all BN_OPTS). "
                         "e.g. 128,256 for the production sweep (skip BN=64).")
    ap.add_argument("--json", default=None)
    ap.add_argument("--compat-out", default=None,
                    help="path for the committed compatibility matrix (default webui/kernels/compat_matrix.json)")
    ap.add_argument("--launch-worker", type=int, default=None,
                    help="internal: run the isolated launch worker from this index")
    ap.add_argument("--jsonl", default=None, help="internal: worker append path")
    args = ap.parse_args()

    shape_list = parse_perf_shapes(args.perf_shapes)
    tier_dirs = args.tiers.split(",")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    bn_opts = [int(x) for x in args.bn.split(",")] if args.bn else None
    to_run, n_valid, n_invalid, n_sample = build_to_run(tier_dirs, args.invalid_sample, bn_opts)
    # Publish the valid-combo count next to the results jsonl so a UI can show
    # a progress bar (done = lines in jsonl, total = this).  Per-run path (not a
    # shared file) so concurrent sweeps don't clobber each other.  Orchestrator only.
    if args.launch_worker is None:
        try:
            nvp = (args.jsonl + ".nvalid") if args.jsonl else str(SCRATCH / "n_valid.txt")
            with open(nvp, "w") as _f:
                _f.write(str(n_valid))
        except Exception:
            pass

    device, _ = rt.init_cuda()
    arch = rt.compute_arch(device)
    num_sms = rt.cu(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device))

    # ── Worker mode: just launch from `start`, stream results, exit ──
    if args.launch_worker is not None:
        worker_loop(to_run, args.launch_worker, arch, shape_list, args.jsonl, num_sms)
        return  # unreachable (worker_loop exits)

    print(f"# perf shapes {[s[0] for s in shape_list]} | arch={arch} | tiers={tier_dirs}")
    print(f"# {n_valid} valid combos to run, {n_invalid} invalid ({n_sample} sampled)")

    # ── cuBLAS reference TFLOPS per shape (one do_bench each) ────────
    cublas_tflops = {}
    for (M, N, K) in shape_list:
        torch.manual_seed(0)
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        us = rt.time_kernel_us(lambda: torch.mm(A, B))   # A:(M,K) @ B:(K,N)
        key = mc.shape_key(M, N, K)
        cublas_tflops[key] = (2.0 * M * N * K) / (us * 1e-6) / 1e12
        print(f"# cuBLAS {key}: {cublas_tflops[key]:.0f} TFLOPS", flush=True)
        del A, B
    torch.cuda.empty_cache()

    # ── Phase 1: render + parallel nvcc compile (CPU-bound) ──────────
    for (t, k, _) in to_run:
        render_to_dir(t, k)
    # PERSISTENT isn't in tag_for (same cubin both ways), so dedup the
    # compile jobs — persistent on/off variants share one kernel.cu.
    jobs = sorted({(str(SCRATCH / tag_for(t, k) / "kernel.cu"), arch) for (t, k, _) in to_run})
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
               "--perf-shapes", args.perf_shapes, "--tiers", args.tiers,
               "--invalid-sample", str(args.invalid_sample)]
        if args.bn:                       # must match orchestrator so to_run indices align
            cmd += ["--bn", args.bn]
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
                            "launched": False, "correct": False, "perf": {},
                            "error": "worker crashed before recording"}
    ordered = [results[i] for i in range(len(to_run))]

    bad = [r for r in ordered if r["validator"] == "valid" and not r["correct"]]
    surprises = [r for r in ordered if r["validator"] == "invalid" and r["correct"]]
    for r in bad:
        print(f"BAD  {r['tier']:24} bn{r['bn']:>3} ns{r['ns']} gsm{r['gsm']:>2} nw{r['nw']:>2} "
              f"ts{r.get('tma_store', 0)}  {r.get('error') or 'incorrect'}")
    for r in surprises:
        print(f"INVALID-but-WORKS  {r['tier']} bn{r['bn']} ns{r['ns']} gsm{r['gsm']} nw{r['nw']} ts{r.get('tma_store',0)}")

    print("\n=== SUMMARY ===")
    print(f"valid combos run:        {n_valid}   (worker spawns: {n_spawns})")
    print(f"  compiled OK:           {n_comp_ok}/{len(jobs)} (incl. invalid sample)")
    print(f"  validator-valid BAD:   {len(bad)}   (must be 0)")
    print(f"invalid combos sampled:  {n_sample}")
    print(f"  invalid-but-correct:   {len(surprises)}   (investigate if > 0)")
    # Best (max-TFLOPS) valid combo per tier, at each swept shape.  A tier that
    # can't tile a shape (e.g. cluster, M%256!=0) simply has no entry there.
    for (M, N, K) in shape_list:
        key = mc.shape_key(M, N, K)
        print(f"cuBLAS {key}: {cublas_tflops[key]:.0f} TFLOPS")
        for tdir in tier_dirs:
            cand = [r for r in ordered if r["tier"] == tdir and r["validator"] == "valid"
                    and r.get("perf", {}).get(key, {}).get("tflops")]
            if cand:
                best = max(cand, key=lambda r: r["perf"][key]["tflops"])
                tf = best["perf"][key]["tflops"]
                ratio = tf / cublas_tflops[key]
                print(f"  best {tdir:24}: {tf:.0f} TFLOPS ({ratio:.0%} cuBLAS)  "
                      f"bn{best['bn']} ns{best['ns']} gsm{best['gsm']} nw{best['nw']} "
                      f"ts{best.get('tma_store',0)} pers{best.get('persistent',0)}")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(ordered, indent=2))
        print(f"wrote {args.json}")

    # ── Committed compatibility matrix (the app filters + perf source) ─
    compat_path = args.compat_out or str(WEBUI / "kernels" / "compat_matrix.json")
    entries = []
    for r in ordered:
        if r["validator"] != "valid":
            continue
        perf = {}
        for s, p in (r.get("perf") or {}).items():
            tf = p.get("tflops")
            perf[s] = {
                "rel_err": round(p["rel_err"], 5) if p.get("rel_err") is not None else None,
                "tflops": round(tf, 1) if tf is not None else None,
                "vs_cublas": round(tf / cublas_tflops[s], 4) if (tf and cublas_tflops.get(s)) else None,
            }
        entries.append({k: r[k] for k in ("tier", "bm", "bn", "bk", "ns", "gsm", "nw",
                                           "tma_store", "persistent")}
                       | {"ld_width": r.get("ld_width", 8), "overlap": r.get("overlap", 0),
                          "correct": bool(r["correct"]), "perf": perf})
    matrix = {
        "generated_by": "webui/tests/gpu_codegen_driver.py",
        "arch": arch,
        "perf_shapes": [list(s) for s in shape_list],
        "cublas_tflops": {s: round(v, 1) for s, v in cublas_tflops.items()},
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
