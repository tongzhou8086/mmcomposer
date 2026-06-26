#!/usr/bin/env python3
"""Tests for the autotune orchestrator (webui/autotune.py).

CPU layer: the codegen/record helpers (_render, _record_config).
GPU layer: a tiny end-to-end tune() over a 1-combo filter -- exercises
enumerate -> codegen -> compile -> runtime -> verify+benchmark -> cache as a
unit.  Skipped without CUDA.

Run:  python webui/tests/test_autotune.py   (or pytest)
"""
import pathlib
import sys
import tempfile

WEBUI = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEBUI))
sys.path.insert(0, str(WEBUI.parent))  # repo root for mmcomposer
sys.path.insert(0, str(WEBUI / "kernels"))

from mmcomposer import mvp_core as mc
from mmcomposer import combos
from mmcomposer import cache as kcache
from mmcomposer import autotune

WS_DIRS = list(dict.fromkeys(t["dir"] for k, t in mc.TIER_MAP.items() if t and k[0]))
TIGHT = {"bn": [256], "ns": [4], "gsm": [8], "nw": [8], "two_cta": [1],
         "persistent": [1], "overlap": [1], "split_epilogue": [0], "l1_no_alloc": [0],
         "tma_pipelined": [1], "tma_store_stages": [2], "single_tmem": [0]}


def _one_combo():
    return next(iter(combos.valid_combos(WS_DIRS, TIGHT)))


def test_record_config_has_identity_and_knobs():
    tier, k = _one_combo()
    cfg = autotune._record_config(tier, k)
    assert cfg["dir"] == tier["dir"]
    assert cfg["symbol"] == tier["symbol"]
    assert cfg["cluster"] == tier["cluster"]
    assert cfg["ws"] is True               # tier3 is warp-spec
    assert cfg["bn"] == k["bn"] and cfg["ns"] == k["ns"]


def test_render_writes_specialized_kernel():
    import re
    tier, k = _one_combo()
    with tempfile.TemporaryDirectory() as d:
        src = autotune._render(tier, k, pathlib.Path(d))
        text = pathlib.Path(src).read_text()
        assert src.endswith("kernel.cu")
        # constexprs specialized to the combo (alignment whitespace is variable)
        assert re.search(r"constexpr int BN\s*=\s*256;", text)
        assert re.search(r"constexpr int NS\s*=\s*4;", text)


def test_tune_tiny_end_to_end():
    import torch
    if not torch.cuda.is_available():
        print("    SKIP (no CUDA)")
        return
    M = N = K = 1024
    with tempfile.TemporaryDirectory() as d:
        kc = kcache.Cache(kcache.LocalDiskBackend(root=d))
        summary = autotune.tune(M, N, K, tier_dirs=WS_DIRS, filters=TIGHT,
                                cache_obj=kc, cublas_samples=1, cublas_warmup_samples=0)
        print(f"    valid={summary['n_valid']} compiled={summary['n_compiled']} "
              f"correct={summary['n_correct']} cuBLAS={summary['cublas_tflops']:.0f} "
              f"best={summary['best']['tflops']:.0f} TFLOPS "
              f"({summary['best']['vs_cublas']:.0%})")
        assert summary["ok"]
        assert summary["n_valid"] >= 1
        assert summary["n_correct"] >= 1
        best = summary["best"]
        assert best["config"]["bn"] == 256 and best["config"]["ns"] == 4
        assert best["tflops"] > 0
        assert 0.3 < best["vs_cublas"] < 1.5
        assert best["rel_err"] < 5e-2
        # the cache holds it under the shape key
        assert kc.best(summary["key"])["tflops"] == best["tflops"]


def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
