"""End-to-end test of the *downloaded artifact* — the self-contained host.py.

The GPU codegen driver (gpu_codegen_driver.py) validates the rendered
*kernels* using its own generic launch harness — it never runs the
tier launchers.  But what a user downloads is the self-contained
``host.py`` (= runtime preamble + ``launcher.py``), which has its own
independent grid math, SMEM sizing, and entry symbol.  This test renders
that exact artifact for a representative set of configs per tier, writes
it next to a rendered ``kernel.cu``, and runs it as a fresh process —
exactly as a user would with ``python host.py``.

Asserts, per config:
  * the rendered host is self-contained (no cuda_utils import / sys.path hack),
  * ``python host.py`` exits 0, and
  * every printed result line is "OK" (correctness within tolerance), no "FAIL".

Run on a B200 node:
    python webui/tests/host_artifact_test.py
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

WEBUI = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEBUI))
sys.path.insert(0, str(WEBUI.parent))  # repo root for mmcomposer
from mmcomposer import mvp_core as mc

SCRATCH = WEBUI / "tests" / "_scratch" / "host_artifacts"

# (ms_ws, two_cta, bm, bn, bk, ns, gsm, nw) — representative per tier,
# chosen to exercise the launcher's variable-warp + swizzle + (for tier3)
# cluster-grid paths, not just defaults.
CONFIGS = [
    (False, False, 128, 256, 64, 2, 8, 4),    # tier1 default
    (False, False, 128, 128, 64, 3, 16, 8),   # tier1: NW8, BN128, GSM16
    (True,  False, 128, 256, 64, 2, 1, 4),    # tier2 default (no swizzle)
    (True,  False, 128, 128, 64, 3, 8, 8),    # tier2: NW8, swizzle
    (True,  False, 128, 256, 64, 3, 8, 4,
     {"persistent": 1, "overlap": 1, "tma_pipelined": 1}),  # tier2: pipelined TMA store
    (True,  True,  128, 256, 64, 5, 8, 4),    # tier3 default
    (True,  True,  128, 256, 64, 3, 16, 16),  # tier3: NW16, GSM16
    (True,  True,  128, 256, 64, 4, 8, 4,
     {"persistent": 1, "overlap": 1, "tma_pipelined": 1}),  # tier3: pipelined TMA store
    (True,  True,  128, 512, 64, 4, 8, 4,
     {"persistent": 1, "overlap": 1, "tma_pipelined": 1,
      "single_tmem": 1}),  # tier3: BN512 two-panel MMA + single-TMEM sync
]


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    failures = []
    for cfg in CONFIGS:
        if len(cfg) == 8:
            ms_ws, two_cta, bm, bn, bk, ns, gsm, nw = cfg
            opts = {}
        else:
            ms_ws, two_cta, bm, bn, bk, ns, gsm, nw, opts = cfg
        tier = mc.tier_for(ms_ws, two_cta)
        tag = f"{tier['dir']}_bn{bn}_ns{ns}_gsm{gsm}_nw{nw}"
        if opts.get("tma_pipelined"):
            tag += "_tma"
        if opts.get("tma_store_stages", 2) != 2:
            tag += f"_ts{opts['tma_store_stages']}"
        if opts.get("single_tmem"):
            tag += "_stmem"
        d = SCRATCH / tag
        d.mkdir(parents=True, exist_ok=True)
        kernel_src = mc.render_kernel(
            tier, bm, bn, bk, ns, gsm, nw,
            overlap=opts.get("overlap", 0),
            split_epilogue=opts.get("split_epilogue", 0),
            l1_no_alloc=opts.get("l1_no_alloc", 0),
            tma_pipelined=opts.get("tma_pipelined", 0),
            tma_store_stages=opts.get("tma_store_stages", 2),
            single_tmem=opts.get("single_tmem", 0))
        host_src   = mc.render_host(
            tier, bm, bn, bk, ns, gsm, nw,
            persistent=opts.get("persistent", 0),
            overlap=opts.get("overlap", 0),
            split_epilogue=opts.get("split_epilogue", 0),
            l1_no_alloc=opts.get("l1_no_alloc", 0),
            tma_pipelined=opts.get("tma_pipelined", 0),
            tma_store_stages=opts.get("tma_store_stages", 2),
            single_tmem=opts.get("single_tmem", 0))
        (d / "kernel.cu").write_text(kernel_src)
        (d / "host.py").write_text(host_src)

        # Self-containment assertions on the artifact itself.
        self_contained = ("from cuda_utils" not in host_src
                           and "import cuda_utils" not in host_src
                           and "sys.path.insert" not in host_src)

        proc = subprocess.run([sys.executable, "host.py"], cwd=str(d),
                              capture_output=True, text=True, timeout=600)
        out = proc.stdout + proc.stderr
        result_lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(("OK", "FAIL"))]
        all_ok = bool(result_lines) and all(ln.startswith("OK") for ln in result_lines)
        ok = self_contained and proc.returncode == 0 and all_ok

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {tag:42} rc={proc.returncode} "
              f"self_contained={self_contained} results={len(result_lines)} all_ok={all_ok}")
        if result_lines:
            for ln in result_lines:
                print(f"        {ln}")
        if not ok:
            failures.append(tag)
            print("    --- output tail ---")
            print("\n".join(out.splitlines()[-15:]))

    print(f"\n=== {len(CONFIGS) - len(failures)}/{len(CONFIGS)} host artifacts ran clean ===")
    if failures:
        print("FAILURES:", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
