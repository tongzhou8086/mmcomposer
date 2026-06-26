#!/usr/bin/env python3
"""Unit tests for the cache module (webui/cache.py).  Pure / no GPU.

Run:  python webui/tests/test_cache.py   (or pytest)
"""
import pathlib
import sys
import tempfile

WEBUI = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEBUI))
sys.path.insert(0, str(WEBUI.parent))  # repo root for mmcomposer

from mmcomposer import cache as C


def _cache(d):
    return C.Cache(C.LocalDiskBackend(root=d))


def test_shape_key_format():
    assert C.shape_key(4096, 4096, 4096) == "4096x4096x4096_bf16_sm_100a"
    assert C.shape_key(2048, 1024, 768, dtype="bf16", arch="sm_100a") == \
        "2048x1024x768_bf16_sm_100a"


def test_missing_key_is_empty():
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        k = C.shape_key(1, 1, 1)
        assert c.get(k) == []
        assert c.best(k) is None
        assert c.top_n(k, 5) == []


def test_put_get_best_top_n_ranking():
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        k = C.shape_key(4096, 4096, 4096)
        c.put(k, {"config": {"bn": 256, "ns": 6}, "tflops": 1354})
        c.put(k, {"config": {"bn": 512, "ns": 4}, "tflops": 1373})
        c.put(k, {"config": {"bn": 256, "ns": 5}, "tflops": 1300})
        assert [r["tflops"] for r in c.get(k)] == [1373, 1354, 1300]   # ranked desc
        assert c.best(k)["config"] == {"bn": 512, "ns": 4}
        assert [r["tflops"] for r in c.top_n(k, 2)] == [1373, 1354]


def test_put_dedups_by_config():
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        k = C.shape_key(8192, 8192, 8192)
        c.put(k, {"config": {"bn": 512, "ns": 4}, "tflops": 1400})
        c.put(k, {"config": {"bn": 512, "ns": 4}, "tflops": 1453})  # same config, re-measured
        recs = c.get(k)
        assert len(recs) == 1
        assert recs[0]["tflops"] == 1453   # updated, not duplicated


def test_persists_across_instances():
    with tempfile.TemporaryDirectory() as d:
        k = C.shape_key(1024, 1024, 1024)
        _cache(d).put(k, {"config": {"bn": 256, "ns": 4}, "tflops": 999})
        # a fresh Cache over the same dir sees the written result
        assert _cache(d).best(k)["tflops"] == 999


def test_put_requires_config_and_tflops():
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        k = C.shape_key(2, 2, 2)
        for bad in ({"tflops": 1.0}, {"config": {}}):
            raised = False
            try:
                c.put(k, bad)
            except ValueError:
                raised = True
            assert raised


def test_keys_lists_cached_shapes():
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        c.put(C.shape_key(4096, 4096, 4096), {"config": {"bn": 256}, "tflops": 1})
        c.put(C.shape_key(8192, 8192, 8192), {"config": {"bn": 512}, "tflops": 2})
        assert set(c.keys()) == {"4096x4096x4096_bf16_sm_100a",
                                 "8192x8192x8192_bf16_sm_100a"}


def test_no_tmp_files_left_behind():
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        c.put(C.shape_key(3, 3, 3), {"config": {"bn": 256}, "tflops": 1})
        leftovers = list((pathlib.Path(d) / "results").glob("*.tmp*"))
        assert leftovers == []


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
