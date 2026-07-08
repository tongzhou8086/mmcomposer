#!/usr/bin/env python3
"""Smoke test for the mmcomposer package surface (Stage A).  Pure / no GPU.

Confirms `import mmcomposer as mmc` exposes the public API and the leaf modules,
and that they're the same verified objects from the core.

Run:  python mmcomposer/tests/test_api.py   (or pytest)
"""
import pathlib
import sys

# import the package from the repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import mmcomposer as mmc


def test_public_api_present_and_callable():
    for name in ("matmul", "get_tuned_kernel", "tune"):
        assert hasattr(mmc, name) and callable(getattr(mmc, name)), name


def test_leaf_modules_exposed():
    for name in ("combos", "compiler", "runtime", "benchmark",
                 "cache", "leaderboard", "autotune", "autotune_isolated", "hopper"):
        assert hasattr(mmc, name), name


def test_cache_shape_key_works_through_package():
    assert mmc.cache.shape_key(4096, 4096, 4096) == "4096x4096x4096_bf16_sm_100a"


def test_enumerate_works_through_package():
    ws = list(dict.fromkeys(t["dir"] for k, t in
                            mmc.mvp_core.TIER_MAP.items() if t and k[0]))
    n = sum(1 for _ in mmc.combos.valid_combos(ws, {"bn": [256], "ns": [4]}))
    assert n >= 1


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
