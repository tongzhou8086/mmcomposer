#!/usr/bin/env python3
"""Study-only benchmark for pipelined TMA-store stage count.

This script intentionally does not modify mmcomposer generator templates. It
renders the current best config, patches the generated kernel copy to use a
different TMA_STORE_STAGES value, then benchmarks the variants on one or more
shapes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEBUI = ROOT / "webui"
TESTS = WEBUI / "tests"
sys.path.insert(0, str(WEBUI))
sys.path.insert(0, str(TESTS))

import mvp_core as mc  # noqa: E402
import gpu_codegen_driver as gd  # noqa: E402
from cuda.bindings import driver  # noqa: E402


HERE = pathlib.Path(__file__).resolve().parent
SCRATCH = HERE / "_scratch"

CONFIG = {
    "bm": 128,
    "bn": 256,
    "bk": 64,
    "ns": 5,
    "gsm": 1,
    "nw": 4,
    "persistent": 1,
    "ld_width": 8,
    "overlap": 1,
    "split_epilogue": 0,
    "l1_no_alloc": 0,
    "tma_pipelined": 1,
    "single_tmem": 0,
}


def parse_shape(spec: str) -> tuple[int, int, int]:
    tok = spec.lower().replace(",", "x")
    if "x" in tok:
        m, n, k = (int(x) for x in tok.split("x"))
        return m, n, k
    s = int(tok)
    return s, s, s


def parse_shapes(specs: list[str]) -> list[tuple[int, int, int]]:
    out = []
    for spec in specs:
        for tok in spec.split(";"):
            tok = tok.strip()
            if tok:
                out.append(parse_shape(tok))
    return out


def parse_int_csv(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def patch_store_stages(src: str, stages: int) -> str:
    if stages < 1:
        raise ValueError("TMA store stages must be >= 1")
    src, n = re.subn(
        r"constexpr int TMA_STORE_STAGES = \d+;",
        f"constexpr int TMA_STORE_STAGES = {stages};",
        src,
        count=1,
    )
    if n != 1:
        raise RuntimeError("could not patch TMA_STORE_STAGES")
    old = "store_stage ^= 1;"
    if old not in src:
        raise RuntimeError("could not find store_stage rotation")
    if stages == 1:
        new = "store_stage = 0;"
    elif stages == 2:
        new = old
    elif stages & (stages - 1) == 0:
        new = "store_stage = (store_stage + 1) & (TMA_STORE_STAGES - 1);"
    else:
        new = "store_stage = (store_stage + 1) % TMA_STORE_STAGES;"
    return src.replace(old, new, 1)


def shared_bytes(k: dict, stages: int) -> int:
    cta_group = 2
    bn_local = k["bn"] // cta_group
    a_slot = k["bm"] * k["bk"] * 2
    b_slot = bn_local * k["bk"] * 2
    slot = a_slot + b_slot
    epi = k["bm"] * 64 * 2 * stages
    return k["ns"] * slot + epi + 1024


def install_driver_hooks(stages: int) -> None:
    if not hasattr(gd, "tag_for_orig"):
        gd.tag_for_orig = gd.tag_for
    if not hasattr(gd, "launch_spec_orig"):
        gd.launch_spec_orig = gd.launch_spec

    def tag_for_stage(tier, k):
        return gd.tag_for_orig(tier, k) + f"_ts{k.get('tma_store_stages', stages)}"

    def launch_spec_stage(tier, k, m, n, kval, num_sms=None):
        grid, block, _shared = gd.launch_spec_orig(tier, k, m, n, kval, num_sms)
        alloc_stages = int(k.get("tma_alloc_stages", k.get("tma_store_stages", stages)))
        return grid, block, shared_bytes(k, alloc_stages)

    gd.tag_for = tag_for_stage
    gd.launch_spec = launch_spec_stage


def render_compile(tier: dict, k: dict, arch: str) -> pathlib.Path:
    src_path = gd.render_to_dir(tier, k)
    src = src_path.read_text()
    src_path.write_text(patch_store_stages(src, int(k["tma_store_stages"])))
    _src, rc, stderr = gd._compile_worker((str(src_path), arch))
    if rc != 0:
        raise RuntimeError(f"nvcc failed for stages={k['tma_store_stages']}:\n{stderr}")
    return src_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", action="append", default=None,
                    help="shape to benchmark; may be repeated or semicolon-separated")
    ap.add_argument("--stages", default="2,4", help="comma-separated TMA store stage counts")
    ap.add_argument("--bn", type=int, default=CONFIG["bn"], help="override BN for the study config")
    ap.add_argument("--ns", type=int, default=CONFIG["ns"], help="override NS for the study config")
    ap.add_argument("--single-tmem", type=int, choices=[0, 1], default=CONFIG["single_tmem"],
                    help="override SINGLE_TMEM_ACCUM for the study config")
    ap.add_argument("--alloc-stages", type=int, default=None,
                    help="allocate SMEM as if this many store stages existed, while testing --stages")
    ap.add_argument("--warmup-ms", type=int, default=1000)
    ap.add_argument("--rep-ms", type=int, default=1000)
    ap.add_argument("--cublas-warmup-samples", type=int, default=1)
    ap.add_argument("--cublas-samples", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.warmup_ms <= 0 or args.rep_ms <= 0:
        ap.error("--warmup-ms and --rep-ms must be positive")
    if args.cublas_warmup_samples < 0 or args.cublas_samples <= 0:
        ap.error("--cublas-warmup-samples must be non-negative and --cublas-samples must be positive")
    if args.alloc_stages is not None and args.alloc_stages < 1:
        ap.error("--alloc-stages must be >= 1")

    shape_list = parse_shapes(args.shape or ["32768x4608x768"])
    stages_list = parse_int_csv(args.stages)

    gd.SCRATCH = SCRATCH
    gd.BENCH_WARMUP_MS = args.warmup_ms
    gd.BENCH_REP_MS = args.rep_ms
    gd.CBLAS_WARMUP_SAMPLES = args.cublas_warmup_samples
    gd.CBLAS_MEASURE_SAMPLES = args.cublas_samples

    tier = mc.tier_for(True, True)
    if tier is None:
        raise RuntimeError("missing tier3 cluster tier")

    gd.load_cuda_runtime()
    device, _ctx = gd.rt.init_cuda()
    arch = gd.rt.compute_arch(device)
    num_sms = gd.rt.cu(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device))
    max_smem = gd.rt.cu(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device))

    shape_msg = ", ".join(f"{m}x{n}x{k}" for m, n, k in shape_list)
    print(f"# shapes={shape_msg} arch={arch} sms={num_sms}", flush=True)
    print(f"# do_bench warmup={args.warmup_ms}ms rep={args.rep_ms}ms", flush=True)
    print(f"# cuBLAS samples: warmup={args.cublas_warmup_samples} measured={args.cublas_samples} median", flush=True)
    print(f"# max opt-in dynamic shared memory per block: {max_smem} B", flush=True)

    shapes = gd.make_shapes(shape_list)
    cublas = {}
    cublas_samples_by_shape = {}
    for sh in shapes:
        m, n, kval = sh["M"], sh["N"], sh["K"]
        key = mc.shape_key(m, n, kval)
        tf, samples = gd.measure_cublas_tflops(sh["A"], sh["B"], m, n, kval)
        cublas[key] = tf
        cublas_samples_by_shape[key] = samples
        print("# cuBLAS "
              f"{key}: {tf:.1f} TFLOPS "
              f"(samples {', '.join(f'{x:.1f}' for x in samples)})",
              flush=True)

    rows = []
    for stages in stages_list:
        k = dict(CONFIG)
        k["bn"] = args.bn
        k["ns"] = args.ns
        k["single_tmem"] = args.single_tmem
        k["tma_store_stages"] = stages
        if args.alloc_stages is not None:
            k["tma_alloc_stages"] = args.alloc_stages
        install_driver_hooks(stages)
        alloc_stages = args.alloc_stages if args.alloc_stages is not None else stages
        shared = shared_bytes(k, alloc_stages)
        print(f"\n# stages={stages} alloc_stages={alloc_stages} "
              f"shared={shared} B ({shared / 1024:.1f} KiB)", flush=True)
        if shared > max_smem:
            print(f"SKIP stages={stages}: shared memory exceeds opt-in limit", flush=True)
            rows.append({"stages": stages, "shared_bytes": shared, "skipped": True})
            continue
        src_path = render_compile(tier, k, arch)
        start = time.time()
        res = gd.launch_from_cubin(tier, k, arch, shapes, do_bench=True, num_sms=num_sms)
        elapsed = time.time() - start
        ok = bool(res.get("correct"))
        err = res.get("error")
        shape_results = {}
        for sh in shapes:
            m, n, kval = sh["M"], sh["N"], sh["K"]
            key = mc.shape_key(m, n, kval)
            perf = (res.get("perf") or {}).get(key, {})
            tf = perf.get("tflops")
            rel = perf.get("rel_err")
            us = perf.get("us")
            ratio = (tf / cublas[key]) if (tf and cublas.get(key)) else None
            shape_results[key] = {
                "tflops": tf,
                "vs_cublas": ratio,
                "us": us,
                "rel_err": rel,
                "correct": perf.get("correct"),
            }
            print(
                f"stages={stages} shape={key} correct={perf.get('correct')} "
                f"tflops={(f'{tf:.1f}' if tf else 'n/a')} "
                f"vs_cublas={(f'{ratio:.1%}' if ratio else 'n/a')} "
                f"us={(f'{us:.3f}' if us else 'n/a')} rel_err={(f'{rel:.6g}' if rel else 'n/a')} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        if err:
            print(f"error: {err}", flush=True)
        rows.append({
            "stages": stages,
            "alloc_stages": alloc_stages,
            "shared_bytes": shared,
            "source": str(src_path.relative_to(ROOT)),
            "correct": ok,
            "perf": shape_results,
            "error": err,
        })

    out = {
        "shapes": [list(s) for s in shape_list],
        "config": CONFIG,
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "cublas_tflops": cublas,
        "cublas_samples": cublas_samples_by_shape,
        "results": rows,
    }
    out_path = pathlib.Path(args.json) if args.json else SCRATCH / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
