"""Exhaustive GPU-free property test: generate EVERY valid knob combo and assert
the result is branch-free + structurally sane.

This is the cheap CI gate that complements the B200 sweep (gpu_codegen_driver):
it can't check that a kernel *compiles/runs*, but it proves codegen produces a
fully-specialized, branch-free kernel for the entire option space — catching
codegen regressions, residual branches, or a knob someone forgot to convert.

Property-based (not golden-per-combo): pinning exact text for thousands of combos
would bloat the repo; instead we assert invariants over every generated kernel.

Run:  python webui/codegen/tests/test_generate_all.py   (or under pytest)
"""

from __future__ import annotations

import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # webui/

import mvp_core as mc                       # noqa: E402
from codegen import generate_kernel, branch_free_issues   # noqa: E402


def all_valid_configs():
    """Every (tier, knob) combo that passes validate_config -> a codegen config."""
    for (ms_ws, two_cta), tier in mc.TIER_MAP.items():
        if tier is None:
            continue
        for bm, bn, bk, ns, gsm, nw, tma, pers, ld, ov, sp in itertools.product(
                mc.BM_OPTS, mc.BN_OPTS, mc.BK_OPTS, mc.NS_OPTS, mc.GSM_OPTS, mc.NW_OPTS,
                mc.TMA_STORE_OPTS, mc.PERSISTENT_OPTS, mc.TCGEN05_LD_WIDTH_OPTS,
                mc.EPILOGUE_OVERLAP_OPTS, mc.EPILOGUE_SPLIT_OPTS):
            errs = mc.validate_config(
                bm, bn, bk, ns, gsm, nw, cluster=tier["cluster"], tma_store=tma,
                persistent=pers, persistent_ok=tier.get("persistent_ok", False),
                ld_width=ld, overlap=ov, split_epilogue=sp)
            if errs:
                continue
            cfg = mc.knob_kwargs(bm, bn, bk, ns, gsm, nw, tma, pers,
                                 ld_width=ld, overlap=ov, split_epilogue=sp)
            cfg["skeleton"] = tier["dir"]
            cfg["TWO_CTA"] = int(tier["cluster"])
            yield cfg


def _structural_issues(src: str) -> list[str]:
    issues = []
    if src.count("{") != src.count("}"):
        issues.append(f"unbalanced braces: {src.count('{')} {{ vs {src.count('}')} }}")
    if "__global__" not in src:
        issues.append("no __global__ kernel entry")
    return issues


def test_all_valid_combos_branch_free():
    n = 0
    for cfg in all_valid_configs():
        try:
            src = generate_kernel(cfg)         # raises on unresolved #if directive
        except Exception as e:                 # noqa: BLE001
            raise AssertionError(f"generate_kernel failed for {cfg}: {e}")
        problems = branch_free_issues(src) + _structural_issues(src)
        assert not problems, f"{cfg}: {problems[:3]}"
        n += 1
    assert n > 0, "enumerated no valid combos — check the OPTS / validate_config"
    print(f"  {n} valid combos generated, all branch-free + structurally sane")


if __name__ == "__main__":
    try:
        test_all_valid_combos_branch_free()
        print("PASS")
    except AssertionError as e:
        print(f"FAIL: {e}")
        raise SystemExit(1)
