#!/usr/bin/env python3
"""Tests for the mmc public API (webui/mmc.py).

Isolates the cache to a temp dir (env set BEFORE importing mmc).

CPU layer: input validation (_shape_dtype) -- dtype/shape/contiguity/multiples.
GPU layer: pre-tune a tiny filter, then exercise get_tuned_kernel / matmul vs
torch, the in-process kernel cache, and the tune_if_missing=False error path.
Skipped without CUDA.

Run:  python webui/tests/test_mmc.py   (or pytest)
"""
import os
import pathlib
import sys
import tempfile

os.environ["MMCOMPOSER_CACHE_DIR"] = tempfile.mkdtemp(prefix="mmc_test_")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))  # repo root

import mmcomposer.mmc as mmc


def test_validation_rejects_bad_inputs():
    import torch
    good_a = torch.zeros(256, 64, dtype=torch.bfloat16)
    good_b = torch.zeros(64, 256, dtype=torch.bfloat16)
    assert mmc._shape_dtype(good_a, good_b) == (256, 256, 64)

    def _raises(exc, a, b):
        try:
            mmc._shape_dtype(a, b)
            return False
        except exc:
            return True

    # wrong dtype
    assert _raises(TypeError, torch.zeros(256, 64), torch.zeros(64, 256))
    # inner dims disagree
    assert _raises(ValueError, torch.zeros(256, 32, dtype=torch.bfloat16),
                   torch.zeros(64, 256, dtype=torch.bfloat16))
    # M is arbitrary now (ragged M -> ceil-div grid + TMA out-of-bounds clip)
    assert mmc._shape_dtype(torch.zeros(130, 64, dtype=torch.bfloat16),
                            torch.zeros(64, 256, dtype=torch.bfloat16)) == (130, 256, 64)
    # N not a multiple of 8 (TMA 16-byte stride alignment)
    assert _raises(ValueError, torch.zeros(256, 64, dtype=torch.bfloat16),
                   torch.zeros(64, 260, dtype=torch.bfloat16))
    # K not a multiple of 64
    assert _raises(ValueError, torch.zeros(256, 65, dtype=torch.bfloat16),
                   torch.zeros(65, 256, dtype=torch.bfloat16))
    # non-contiguous
    nc = torch.zeros(256, 128, dtype=torch.bfloat16)[:, ::2]   # K=64 but strided
    assert _raises(ValueError, nc, torch.zeros(64, 256, dtype=torch.bfloat16))


def test_get_tuned_kernel_raises_when_cold_and_no_tune():
    import torch
    if not torch.cuda.is_available():
        print("    SKIP (no CUDA)")
        return
    a = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    raised = False
    try:
        mmc.get_tuned_kernel(a, b, tune_if_missing=False)   # never tuned this shape
    except RuntimeError:
        raised = True
    assert raised


def test_hopper_matmul_accepts_ragged_m():
    import torch
    if not torch.cuda.is_available():
        print("    SKIP (no CUDA)")
        return
    if torch.cuda.get_device_capability()[0] != 9:
        print("    SKIP (not Hopper)")
        return
    M, N, K = 513, 512, 512
    torch.manual_seed(1)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    ref = a.float() @ b.float()
    c = mmc.matmul(a, b, sync=True)
    assert tuple(c.shape) == (M, N)
    rel = ((c.float() - ref).norm() / ref.norm()).item()
    print(f"    Hopper ragged-M matmul rel err = {rel:.3e}")
    assert rel < 5e-2


def test_matmul_end_to_end_after_pretune():
    import torch
    if not torch.cuda.is_available():
        print("    SKIP (no CUDA)")
        return
    M = N = K = 512
    if torch.cuda.get_device_capability()[0] == 9:
        torch.manual_seed(0)
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        ref = a.float() @ b.float()
        c = mmc.matmul(a, b, sync=True)
        rel = ((c.float() - ref).norm() / ref.norm()).item()
        print(f"    Hopper fixed matmul rel err = {rel:.3e}")
        assert rel < 5e-2
        return

    from mmcomposer import autotune
    # pre-populate the (shared, temp) cache with a tiny sweep so matmul hits it
    ws = list(dict.fromkeys(t["dir"] for k, t in __import__("mmcomposer").mvp_core.TIER_MAP.items()
                            if t and k[0]))
    tight = {"bn": [256], "ns": [4], "gsm": [8], "nw": [8], "two_cta": [1],
             "persistent": [1], "overlap": [1], "split_epilogue": [0],
             "l1_no_alloc": [0], "tma_pipelined": [1], "tma_store_stages": [2],
             "single_tmem": [0]}
    s = autotune.tune(M, N, K, tier_dirs=ws, filters=tight,
                      cublas_samples=1, cublas_warmup_samples=0)
    assert s["ok"], "pre-tune failed"

    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    ref = a.float() @ b.float()

    gemm = mmc.get_tuned_kernel(a, b)        # cache hit -> build+bind
    c = gemm(a, b)
    rel = ((c.float() - ref).norm() / ref.norm()).item()
    print(f"    get_tuned_kernel rel err = {rel:.3e}")
    assert rel < 5e-2

    c2 = mmc.matmul(a, b)                     # public one-shot
    assert ((c2.float() - ref).norm() / ref.norm()).item() < 5e-2

    # second get_tuned_kernel returns the SAME cached callable (no rebuild)
    assert mmc.get_tuned_kernel(a, b) is gemm


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
