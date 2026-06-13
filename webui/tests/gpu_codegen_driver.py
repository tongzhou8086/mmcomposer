"""B200 codegen integration driver — render combos, compile, run, check.

This is the on-GPU half of the mmcomposer MVP's correctness test.  It runs
on a B200 node and, for each validator-*passing* config:

  1. renders the tier's kernel.cu with the knob values substituted,
  2. compiles it with nvcc,
  3. launches it at one small shape, and
  4. checks correctness against a torch reference.

With ``--mode perf`` it additionally times valid combos and writes a perf
matrix.  The public timing entry point is ``webui/autotune.py``, which calls
this driver in perf mode with a pruned search policy.

The premise being tested: **"the validator says valid" must imply
"compiles + runs + numerically correct."**  A sample of validator-
*failing* combos is also run to confirm they genuinely fail (so the
validator isn't rejecting things that actually work).

Usage (from repo root, on a GPU node):
    srun ... python webui/tests/gpu_codegen_driver.py [--perf-shapes 2048] \
        [--tiers tier1_baseline,tier3_cluster_swizzle] \
        [--invalid-sample 12] [--json out.json]

Results are printed as a table and (optionally) written as JSON/matrix files.
"""

from __future__ import annotations

import argparse
import ctypes
import itertools
import json
import statistics
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

WEBUI = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEBUI))
sys.path.insert(0, str(WEBUI / "kernels"))

import mvp_core as mc
from codegen import branch_free_issues

import numpy as np
import torch


rt = None
driver = None


def load_cuda_runtime():
    """Import CUDA pieces lazily so --help and combo enumeration work off-GPU."""
    global rt, driver
    if rt is None or driver is None:
        import _runtime as _rt
        from cuda.bindings import driver as _driver
        rt = _rt
        driver = _driver


SCRATCH = WEBUI / "tests" / "_scratch" / "gpu_driver"

# Benchmark window for the sweep — same window for the kernels AND the cuBLAS
# reference, so the ratio is apples-to-apples.  do_bench's 20ms-warmup default
# measures partly clock-boosted, so a fresh/idle B200 reports inflated TFLOPS
# (8192^3 cuBLAS: 1558 at 20/200 vs ~1360 settled).  Measured a window sweep on
# a warm B200: WARMUP is the stabilizer (it rides through the boost spike into
# sustained clocks), not rep — 500/100 was the noisiest (8.4% spread) while
# 300/200 was the tightest (3.0%, matching 1000/1000's 3.2%) at ~0.5s/combo.
# Below ~300ms warmup the first call lands on the boost curve (100/200 -> 5%,
# 50/200 -> 10%), which would reorder a leaderboard whose top combos differ by
# ~1%.  So 300/200 is the sweet spot: a ~3% noise floor (thermal drift, window-
# independent) at half the wall-clock of 500/500.  Override via env.
BENCH_WARMUP_MS = int(os.environ.get("MMCOMPOSER_BENCH_WARMUP_MS", "300"))
BENCH_REP_MS    = int(os.environ.get("MMCOMPOSER_BENCH_REP_MS", "200"))
CBLAS_WARMUP_SAMPLES = int(os.environ.get("MMCOMPOSER_CUBLAS_WARMUP_SAMPLES", "1"))
CBLAS_MEASURE_SAMPLES = int(os.environ.get("MMCOMPOSER_CUBLAS_MEASURE_SAMPLES", "3"))


def parse_int_csv(spec):
    """Parse a comma-separated integer list, or return None for no filter."""
    if spec is None:
        return None
    return [int(x) for x in spec.split(",") if x.strip()]


def measure_cublas_tflops(A, B, M, N, K):
    """Robust cuBLAS reference for perf sweeps.

    A single fresh cuBLAS do_bench can catch a transient boost-clock outlier
    (observed 4096^3: first 300/200 sample 1542 TFLOPS, steady median ~1356).
    Discard a configurable number of warmup samples, then use the median of a
    few measured samples.  This is cheap per shape and keeps the denominator
    apple-to-apple with kernels timed later in the same warmed allocation.
    """
    for _ in range(CBLAS_WARMUP_SAMPLES):
        rt.time_kernel_us(lambda: torch.mm(A, B),
                          warmup_ms=BENCH_WARMUP_MS, rep_ms=BENCH_REP_MS)
    vals = []
    for _ in range(CBLAS_MEASURE_SAMPLES):
        us = rt.time_kernel_us(lambda: torch.mm(A, B),
                               warmup_ms=BENCH_WARMUP_MS, rep_ms=BENCH_REP_MS)
        vals.append((2.0 * M * N * K) / (us * 1e-6) / 1e12)
    return statistics.median(vals), vals


def _opts(filters, name, default):
    return filters.get(name) or default


def all_combos(tier_dirs, filters=None):
    """Yield (tier, knobs-dict) over the dropdown grid.

    `filters` restricts selected knob dimensions for pruned perf sweeps.  The
    correctness integration mode can pass no filters to exercise all valid
    combos without timing them.
    """
    filters = filters or {}
    bn_list = _opts(filters, "bn", mc.BN_OPTS)
    ns_list = _opts(filters, "ns", mc.NS_OPTS)
    gsm_list = _opts(filters, "gsm", mc.GSM_OPTS)
    nw_list = _opts(filters, "nw", mc.NW_OPTS)
    ld_list = _opts(filters, "ld_width", mc.TCGEN05_LD_WIDTH_OPTS)
    l1_list = _opts(filters, "l1_no_alloc", mc.EPILOGUE_L1_NO_ALLOC_OPTS)
    tma_filter = filters.get("tma_pipelined")
    single_filter = filters.get("single_tmem")
    single_tmem_policy = filters.get("single_tmem_policy")
    # A skeleton dir may back >1 tier: the warp-spec single-CTA and 2-CTA
    # cluster tiers share one dir, distinguished by the TWO_CTA knob.  Sweep
    # every (ms_ws, two_cta) arm registered for each requested dir.
    keys_for_dir = {}
    for _key, t in mc.TIER_MAP.items():
        if t:
            keys_for_dir.setdefault(t["dir"], []).append(_key)
    tier_keys = [k for tdir in tier_dirs for k in keys_for_dir[tdir]]
    for key in tier_keys:
        tier = mc.TIER_MAP[key]
        # PERSISTENT is a launch knob (same cubin) — only the persistent-
        # capable tiers get the grid=#SMs variant; others stay at [0].
        pers_default = mc.PERSISTENT_OPTS if tier.get("persistent_ok") else [0]
        pers_opts = _opts(filters, "persistent", pers_default)
        # EPILOGUE_OVERLAP only applies on the persistent-capable path; most
        # overlap=1 combos are filtered by the validator (persistent/NW/SMEM).
        ov_default = mc.EPILOGUE_OVERLAP_OPTS if tier.get("persistent_ok") else [0]
        ov_opts = _opts(filters, "overlap", ov_default)
        # EPILOGUE_SPLIT is a Tier 3 cluster epilogue staging variant.  Keep
        # non-cluster sweeps focused on code they can actually generate.
        sp_default = mc.EPILOGUE_SPLIT_OPTS if tier.get("cluster") else [0]
        sp_opts = _opts(filters, "split_epilogue", sp_default)
        tma_default = mc.EPILOGUE_TMA_PIPELINED_OPTS if tier.get("persistent_ok") else [0]
        tma_opts = tma_filter if tma_filter is not None else tma_default
        single_default = mc.SINGLE_TMEM_ACCUM_OPTS if tier.get("persistent_ok") else [0]
        single_opts = single_filter if single_filter is not None else single_default
        for bm, bn, bk, ns, gsm, nw, pers, ldw, ov, sp, l1, tma, single_tmem in itertools.product(
            mc.BM_OPTS, bn_list, mc.BK_OPTS, ns_list, gsm_list, nw_list,
            pers_opts, ld_list, ov_opts, sp_opts,
            l1_list, tma_opts, single_opts
        ):
            if single_tmem_policy == "bn512-only":
                if bn == 512 and single_tmem != 1:
                    continue
                if bn != 512 and single_tmem != 0:
                    continue
            yield tier, dict(bm=bm, bn=bn, bk=bk, ns=ns, gsm=gsm, nw=nw,
                             persistent=pers, ld_width=ldw, overlap=ov,
                             split_epilogue=sp, l1_no_alloc=l1,
                             tma_pipelined=tma, single_tmem=single_tmem)


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
    if k.get("overlap", 0) and k.get("tma_pipelined", 0):
        epi = k["bm"] * 64 * 2 * 2
    elif k.get("overlap", 0) and tier["cluster"] and k.get("split_epilogue", 0):
        epi = k["bm"] * (k["bn"] // 2 + 8) * 2
    else:
        epi = k["bm"] * (k["bn"] + 8) * 2
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
    # two_cta (cluster) changes the cubin and shares the dir with the single-CTA
    # arm, so it must be in the tag or the two arms clobber each other's cubin.
    return (f"{tier['dir']}_tc{int(tier['cluster'])}_bm{k['bm']}_bn{k['bn']}_bk{k['bk']}"
            f"_ns{k['ns']}_gsm{k['gsm']}_nw{k['nw']}"
            f"_ld{k.get('ld_width', 8)}_ov{k.get('overlap', 0)}"
            f"_sp{k.get('split_epilogue', 0)}_l1{k.get('l1_no_alloc', 0)}"
            f"_tp{k.get('tma_pipelined', 0)}_st{k.get('single_tmem', 0)}")


def render_to_dir(tier, k):
    """Write the substituted kernel.cu; return its path."""
    d = SCRATCH / tag_for(tier, k)
    d.mkdir(parents=True, exist_ok=True)
    src = mc.render_kernel(tier, k["bm"], k["bn"], k["bk"], k["ns"], k["gsm"], k["nw"],
                           ld_width=k.get("ld_width", 8),
                           overlap=k.get("overlap", 0),
                           split_epilogue=k.get("split_epilogue", 0),
                           l1_no_alloc=k.get("l1_no_alloc", 0),
                           tma_pipelined=k.get("tma_pipelined", 0),
                           single_tmem=k.get("single_tmem", 0))
    # Codegen must emit a fully branch-free kernel; a residual #if / knob
    # if-constexpr means a forgotten conversion — fail clearly here rather than
    # as an opaque nvcc error during compile.
    issues = branch_free_issues(src)
    if issues:
        raise RuntimeError(f"non-branch-free kernel for {tag_for(tier, k)}: {issues[:3]}")
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
    load_cuda_runtime()
    src_path = str(SCRATCH / tag_for(tier, k) / "kernel.cu")
    cubin_path = src_path[:-3] + f"_{arch}.cubin"
    res = {"tier": tier["dir"], "two_cta": int(tier["cluster"]), **k,
           "launched": False, "correct": False, "error": None, "perf": {}}
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
            if k.get("tma_pipelined", 0):
                C_tmap = rt.encode_tensor_map(dtype=rt.TMA_BFLOAT16, rank=2, gptr=sh["C"].data_ptr(),
                    global_dim=[N, M], global_strides=[N * 2], box_dim=[64, k["bm"]],
                    element_strides=[1, 1], swizzle=rt.TMA_SWIZZLE_128B)
            else:
                # Staged int4 stores do not consume C_tmap; keep a dummy
                # descriptor so all generated kernels share one ABI.
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
                    kernel, grid=grid, block=block, shared=shared, args=args, sync=False),
                    warmup_ms=BENCH_WARMUP_MS, rep_ms=BENCH_REP_MS)
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


def build_to_run(tier_dirs, invalid_sample, filters=None):
    """Deterministic ordered list of (tier, knobs, label).  Both the
    orchestrator and the isolated worker call this so indices line up
    (so filters must be passed identically in both)."""
    valid, invalid = [], []
    for tier, k in all_combos(tier_dirs, filters):
        warnings = mc.validate_config(k["bm"], k["bn"], k["bk"], k["ns"], k["gsm"], k["nw"],
                                      cluster=tier["cluster"],
                                      persistent=k.get("persistent", 0),
                                      persistent_ok=tier.get("persistent_ok", False),
                                      ld_width=k.get("ld_width", 8),
                                      overlap=k.get("overlap", 0),
                                      split_epilogue=k.get("split_epilogue", 0),
                                      l1_no_alloc=k.get("l1_no_alloc", 0),
                                      tma_pipelined=k.get("tma_pipelined", 0),
                                      single_tmem=k.get("single_tmem", 0))
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


def worker_loop(to_run, start, arch, shape_list, jsonl_path, *, bench_valid: bool, num_sms=None):
    """Launch combos [start:] in one CUDA context, appending one JSON
    line per combo (with per-shape correctness + perf).  On any CUDA
    fault the context is poisoned, so we record the offending combo and
    exit non-zero; the orchestrator respawns a fresh worker past idx."""
    shapes = make_shapes(shape_list)
    f = open(jsonl_path, "a")
    for idx in range(start, len(to_run)):
        tier, k, label = to_run[idx]
        r = launch_from_cubin(tier, k, arch, shapes,
                              do_bench=(bench_valid and label == "valid"),
                              num_sms=num_sms)
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
    ap.add_argument("--mode", choices=["correctness", "perf"], default="correctness",
                    help="correctness: compile+launch+check only; perf: also time valid combos")
    # tier3_cluster_swizzle backs BOTH warp-spec arms (single-CTA + 2-CTA) via
    # the TWO_CTA knob; the sweep expands it to both arms automatically.
    ap.add_argument("--tiers", default="tier1_baseline,tier3_cluster_swizzle")
    ap.add_argument("--invalid-sample", type=int, default=12)
    ap.add_argument("--bn", default=None,
                    help="comma-separated BN values to sweep (default: all BN_OPTS). "
                         "e.g. 128,256 for the production sweep (skip BN=64).")
    ap.add_argument("--ns", default=None, help="comma-separated NS values to sweep")
    ap.add_argument("--gsm", default=None, help="comma-separated GROUP_SIZE_M values to sweep")
    ap.add_argument("--nw", default=None, help="comma-separated NUM_WARPS values to sweep")
    ap.add_argument("--persistent", default=None, help="comma-separated PERSISTENT values to sweep")
    ap.add_argument("--overlap", default=None, help="comma-separated EPILOGUE_OVERLAP values to sweep")
    ap.add_argument("--split-epilogue", default=None, help="comma-separated EPILOGUE_SPLIT values to sweep")
    ap.add_argument("--l1-no-alloc", default=None, help="comma-separated EPILOGUE_L1_NO_ALLOC values to sweep")
    ap.add_argument("--tma-pipelined", default=None, help="comma-separated EPILOGUE_TMA_PIPELINED values to sweep")
    ap.add_argument("--single-tmem", default=None, help="comma-separated SINGLE_TMEM_ACCUM values to sweep")
    ap.add_argument("--single-tmem-policy", choices=["all", "bn512-only"], default=None,
                    help="optional production pruning: bn512-only keeps SINGLE_TMEM_ACCUM=0 for BN<512 and =1 for BN=512")
    ap.add_argument("--json", default=None)
    ap.add_argument("--compat-out", default=None,
                    help="path for a compatibility/perf matrix. If omitted, perf mode writes "
                         "webui/kernels/compat_matrix.json; correctness mode skips this file.")
    ap.add_argument("--launch-worker", type=int, default=None,
                    help="internal: run the isolated launch worker from this index")
    ap.add_argument("--jsonl", default=None, help="internal: worker append path")
    args = ap.parse_args()

    shape_list = parse_perf_shapes(args.perf_shapes)
    tier_dirs = args.tiers.split(",")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    filters = {
        "bn": parse_int_csv(args.bn),
        "ns": parse_int_csv(args.ns),
        "gsm": parse_int_csv(args.gsm),
        "nw": parse_int_csv(args.nw),
        "persistent": parse_int_csv(args.persistent),
        "overlap": parse_int_csv(args.overlap),
        "split_epilogue": parse_int_csv(args.split_epilogue),
        "l1_no_alloc": parse_int_csv(args.l1_no_alloc),
        "tma_pipelined": parse_int_csv(args.tma_pipelined),
        "single_tmem": parse_int_csv(args.single_tmem),
        "single_tmem_policy": args.single_tmem_policy,
    }
    filters = {k: v for k, v in filters.items() if v is not None}
    to_run, n_valid, n_invalid, n_sample = build_to_run(tier_dirs, args.invalid_sample, filters)
    jsonl = args.jsonl
    if args.launch_worker is None and jsonl is None:
        jsonl = str(SCRATCH / f"results_{args.mode}_{os.getpid()}.jsonl")
    # Publish the valid-combo count next to the results jsonl so a UI can show
    # a progress bar (done = lines in jsonl, total = this).  Per-run path (not a
    # shared file) so concurrent sweeps don't clobber each other.  Orchestrator only.
    if args.launch_worker is None:
        try:
            nvp = jsonl + ".nvalid"
            with open(nvp, "w") as _f:
                _f.write(str(n_valid))
        except Exception:
            pass

    load_cuda_runtime()
    device, _ = rt.init_cuda()
    arch = rt.compute_arch(device)
    num_sms = rt.cu(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device))

    # ── Worker mode: just launch from `start`, stream results, exit ──
    if args.launch_worker is not None:
        worker_loop(to_run, args.launch_worker, arch, shape_list, args.jsonl,
                    bench_valid=(args.mode == "perf"), num_sms=num_sms)
        return  # unreachable (worker_loop exits)

    print(f"# mode={args.mode} | perf shapes {[s[0] for s in shape_list]} | arch={arch} | tiers={tier_dirs}")
    if filters:
        print(f"# filters={filters}")
    print(f"# {n_valid} valid combos to run, {n_invalid} invalid ({n_sample} sampled)")

    # ── cuBLAS reference TFLOPS per shape ───────────────────────────
    cublas_tflops = {}
    if args.mode == "perf":
        for (M, N, K) in shape_list:
            torch.manual_seed(0)
            A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
            key = mc.shape_key(M, N, K)
            cublas_tflops[key], samples = measure_cublas_tflops(A, B, M, N, K)
            sample_msg = ", ".join(f"{x:.0f}" for x in samples)
            print(f"# cuBLAS {key}: {cublas_tflops[key]:.0f} TFLOPS "
                  f"(median of {len(samples)} after {CBLAS_WARMUP_SAMPLES} throwaway; "
                  f"samples [{sample_msg}])", flush=True)
            del A, B
        torch.cuda.empty_cache()
    else:
        cublas_tflops = {mc.shape_key(M, N, K): None for (M, N, K) in shape_list}
        print("# cuBLAS timing skipped in correctness mode", flush=True)
    # Publish cuBLAS refs next to the jsonl so a UI can compute vs_cublas for a
    # LIVE leaderboard before the sweep finishes (mirrors the .nvalid sidecar).
    try:
        cbp = jsonl + ".cublas"
        with open(cbp, "w") as _f:
            json.dump(cublas_tflops, _f)
    except Exception:
        pass

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
    if os.path.exists(jsonl):
        os.remove(jsonl)
    next_idx = 0
    n_spawns = 0
    while next_idx < len(to_run):
        n_spawns += 1
        cmd = [sys.executable, os.path.abspath(__file__),
               "--launch-worker", str(next_idx), "--jsonl", jsonl,
               "--perf-shapes", args.perf_shapes, "--tiers", args.tiers,
               "--invalid-sample", str(args.invalid_sample), "--mode", args.mode]
        for flag, val in (
            ("--bn", args.bn), ("--ns", args.ns), ("--gsm", args.gsm), ("--nw", args.nw),
            ("--persistent", args.persistent), ("--overlap", args.overlap),
            ("--split-epilogue", args.split_epilogue), ("--l1-no-alloc", args.l1_no_alloc),
            ("--tma-pipelined", args.tma_pipelined),
            ("--single-tmem", args.single_tmem),
            ("--single-tmem-policy", args.single_tmem_policy),
        ):
            if val is not None:
                cmd += [flag, val]
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
              f"{r.get('error') or 'incorrect'}")
    for r in surprises:
        print(f"INVALID-but-WORKS  {r['tier']} bn{r['bn']} ns{r['ns']} gsm{r['gsm']} nw{r['nw']}")

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
        if cublas_tflops.get(key):
            print(f"cuBLAS {key}: {cublas_tflops[key]:.0f} TFLOPS")
        else:
            print(f"cuBLAS {key}: skipped")
        # Group by (dir, two_cta): the two warp-spec arms share a dir but are
        # distinct kernels, so report each arm's best separately.
        seen_arms = []
        for r in ordered:
            arm = (r["tier"], r.get("two_cta", 0))
            if r["tier"] in tier_dirs and arm not in seen_arms:
                seen_arms.append(arm)
        for (tdir, tc) in seen_arms:
            cand = [r for r in ordered if r["tier"] == tdir and r.get("two_cta", 0) == tc
                    and r["validator"] == "valid"
                    and r.get("perf", {}).get(key, {}).get("tflops")]
            if cand:
                best = max(cand, key=lambda r: r["perf"][key]["tflops"])
                tf = best["perf"][key]["tflops"]
                ratio = tf / cublas_tflops[key] if cublas_tflops.get(key) else None
                label = f"{tdir}{' (2-CTA)' if tc else ' (1-CTA)'}"
                ratio_msg = f" ({ratio:.0%} cuBLAS)" if ratio is not None else ""
                print(f"  best {label:32}: {tf:.0f} TFLOPS{ratio_msg}  "
                      f"bn{best['bn']} ns{best['ns']} gsm{best['gsm']} nw{best['nw']} "
                      f"pers{best.get('persistent',0)} tma{best.get('tma_pipelined',0)} "
                      f"st{best.get('single_tmem',0)}")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(ordered, indent=2))
        print(f"wrote {args.json}")

    # ── Optional compatibility/perf matrix ───────────────────────────
    compat_path = args.compat_out
    if compat_path is None and args.mode == "perf":
        compat_path = str(WEBUI / "kernels" / "compat_matrix.json")
    if compat_path:
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
                                               "persistent")}
                           | {"two_cta": r.get("two_cta", 0),
                              "ld_width": r.get("ld_width", 8), "overlap": r.get("overlap", 0),
                              "split_epilogue": r.get("split_epilogue", 0),
                              "l1_no_alloc": r.get("l1_no_alloc", 0),
                              "tma_pipelined": r.get("tma_pipelined", 0),
                              "single_tmem": r.get("single_tmem", 0),
                              "correct": bool(r["correct"]), "perf": perf})
        matrix = {
            "generated_by": "webui/tests/gpu_codegen_driver.py",
            "mode": args.mode,
            "arch": arch,
            "perf_shapes": [list(s) for s in shape_list],
            "cublas_tflops": {s: (round(v, 1) if v is not None else None)
                              for s, v in cublas_tflops.items()},
            "tolerance_rel_err": 5e-2,
            "n_entries": len(entries),
            "n_correct": sum(e["correct"] for e in entries),
            "entries": entries,
        }
        pathlib.Path(compat_path).write_text(json.dumps(matrix, indent=2))
        print(f"wrote compat matrix: {compat_path} ({matrix['n_correct']}/{matrix['n_entries']} correct)")
    else:
        print("compat matrix skipped (correctness mode; pass --compat-out to write one)")

    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
