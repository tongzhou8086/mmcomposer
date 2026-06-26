#!/usr/bin/env python3
"""Unit tests for the compile module (webui/compiler.py).

Needs nvcc on PATH but NO GPU (nvcc targets the arch).  Isolated from codegen:
uses trivial hand-written kernels so it tests compile mechanics (cubin output,
mtime cache, atomic write, parallel, error reporting), not our GEMM source.

Run:  python webui/tests/test_compiler.py   (or pytest)
"""
import os
import pathlib
import sys
import tempfile

WEBUI = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEBUI))
sys.path.insert(0, str(WEBUI.parent))  # repo root for mmcomposer

from mmcomposer import compiler

TRIVIAL = "extern \"C\" __global__ void noop() {}\n"
BROKEN = "this is not valid CUDA;\n"


def _write(d, name, body):
    p = pathlib.Path(d) / name
    p.write_text(body)
    return str(p)


def test_compile_one_produces_cubin():
    with tempfile.TemporaryDirectory() as d:
        src = _write(d, "k.cu", TRIVIAL)
        cubin = compiler.compile_one(src)
        assert cubin == compiler.cubin_path_for(src)
        assert os.path.exists(cubin) and os.path.getsize(cubin) > 0


def test_up_to_date_cubin_is_not_rebuilt():
    with tempfile.TemporaryDirectory() as d:
        src = _write(d, "k.cu", TRIVIAL)
        cubin = compiler.compile_one(src)
        mtime1 = os.path.getmtime(cubin)
        cubin2 = compiler.compile_one(src)            # up to date -> skip
        assert cubin2 == cubin
        assert os.path.getmtime(cubin) == mtime1      # untouched


def test_force_rebuilds():
    with tempfile.TemporaryDirectory() as d:
        src = _write(d, "k.cu", TRIVIAL)
        compiler.compile_one(src)
        cubin = compiler.compile_one(src, force=True)  # must not raise; rebuilds
        assert os.path.exists(cubin) and os.path.getsize(cubin) > 0


def test_compile_many_parallel_all_ok():
    with tempfile.TemporaryDirectory() as d:
        srcs = [_write(d, f"k{i}.cu", TRIVIAL) for i in range(3)]
        res = compiler.compile_many(srcs)
        assert set(res) == set(srcs)
        for s in srcs:
            assert res[s].ok and res[s].cubin and os.path.exists(res[s].cubin)


def test_compile_one_raises_on_bad_source():
    with tempfile.TemporaryDirectory() as d:
        src = _write(d, "bad.cu", BROKEN)
        raised = False
        try:
            compiler.compile_one(src)
        except compiler.CompileError as e:
            raised = True
            assert "nvcc failed" in str(e)
        assert raised
        assert not os.path.exists(compiler.cubin_path_for(src))  # no half-written cubin


def test_compile_many_reports_failures_without_raising():
    with tempfile.TemporaryDirectory() as d:
        ok_src = _write(d, "ok.cu", TRIVIAL)
        bad_src = _write(d, "bad.cu", BROKEN)
        res = compiler.compile_many([ok_src, bad_src])   # must not raise
        assert res[ok_src].ok and res[ok_src].cubin
        assert not res[bad_src].ok and res[bad_src].error and res[bad_src].cubin is None


def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
