#!/usr/bin/env python3
"""Unit tests for the leaderboard module (webui/leaderboard.py).  Pure / no GPU.

Run:  python webui/tests/test_leaderboard.py   (or pytest)
"""
import io
import pathlib
import sys

WEBUI = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEBUI))
sys.path.insert(0, str(WEBUI.parent))  # repo root for mmcomposer

from mmcomposer import leaderboard as lb


def _records():
    return [
        {"config": {"bn": 256, "ns": 6, "gsm": 8, "nw": 8, "cluster": True, "ws": True},
         "tflops": 1354, "vs_cublas": 0.98},
        {"config": {"bn": 512, "ns": 4, "gsm": 8, "nw": 8, "cluster": True, "ws": True},
         "tflops": 1373, "vs_cublas": 0.96},
        {"config": {"bn": 256, "ns": 5, "gsm": 8, "nw": 4, "cluster": True, "ws": True},
         "tflops": 1300, "vs_cublas": 0.93},
    ]


def test_progress_bar():
    s = lb.progress_bar(5, 10)
    assert "5/10" in s and "50.0%" in s
    assert lb.progress_bar(7, None).endswith("7 measured")


def test_render_orders_by_tflops_desc():
    out = lb.render(_records(), (4096, 4096, 4096), cublas_tflops=1432, top=10)
    # the 1373 row must appear before 1354 before 1300
    i1373, i1354, i1300 = out.index("1373"), out.index("1354"), out.index("1300")
    assert i1373 < i1354 < i1300


def test_render_includes_shape_and_cublas():
    out = lb.render(_records(), (4096, 2048, 768), cublas_tflops=1432)
    assert "4096x2048x768" in out
    assert "1432 TFLOPS" in out


def test_render_top_limits_rows():
    out = lb.render(_records(), (4096, 4096, 4096), top=2)
    # top=2 -> ranks 1 and 2 only; the 3rd record (1300) is excluded
    assert "1373" in out and "1354" in out and "1300" not in out
    # and the rendered "Top N" header reflects the cap
    assert "Top 2 of 3" in out


def test_render_computes_vs_cublas_when_absent():
    recs = [{"config": {"bn": 512, "ns": 4}, "tflops": 1373}]  # no vs_cublas
    out = lb.render(recs, (4096, 4096, 4096), cublas_tflops=1432)
    assert "96%" in out   # 1373/1432 = 0.958 -> 96%


def test_render_reads_flat_records_too():
    # knobs flat on the record (no nested "config")
    recs = [{"bn": 256, "ns": 6, "tflops": 1000, "cluster": False, "ws": False}]
    out = lb.render(recs, (1024, 1024, 1024))
    assert "1000" in out and "off" in out   # ws=off renders


def test_live_display_no_tty_writes_plain():
    buf = io.StringIO()
    d = lb.LiveDisplay(redraw="never", stream=buf)
    d.update("line A\nline B\n")
    d.update("line C\n")
    text = buf.getvalue()
    assert "line A" in text and "line C" in text
    assert "\x1b[" not in text   # no ANSI redraw codes when redraw is off


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
