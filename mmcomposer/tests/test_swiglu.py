#!/usr/bin/env python3
"""Tests for the fixed dual-B fused GEMM+SwiGLU kernel.

CPU layer: shape/dtype validation (no GPU).
GPU layer: compile the cubin, run, and check C (packed wide GEMM) and
D (left * silu(gate)) against a torch reference.  Skipped without CUDA.

Run:  python -m pytest mmcomposer/tests/test_swiglu.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))  # repo root

import pytest
import torch

from mmcomposer import swiglu
import mmcomposer.mmc as mmc


# ---- CPU: validation ------------------------------------------------------
def test_validate_returns_packed_shape():
    a = torch.zeros(256, 128, dtype=torch.bfloat16)
    bl = torch.zeros(128, 256, dtype=torch.bfloat16)
    bg = torch.zeros(128, 256, dtype=torch.bfloat16)
    assert swiglu.validate(a, bl, bg) == (256, 512, 128)   # N = 2 * 256


def test_validate_accepts_packed_b_column_views():
    a = torch.zeros(256, 128, dtype=torch.bfloat16)
    b = torch.zeros(128, 512, dtype=torch.bfloat16)
    bl = b[:, :256]
    bg = b[:, 256:]
    assert not bl.is_contiguous()
    assert bl.stride() == (512, 1)
    assert swiglu.validate(a, bl, bg) == (256, 512, 128)


def test_validate_rejects_bad_inputs():
    a = torch.zeros(256, 128, dtype=torch.bfloat16)
    bl = torch.zeros(128, 256, dtype=torch.bfloat16)
    bg = torch.zeros(128, 256, dtype=torch.bfloat16)
    with pytest.raises(TypeError):                               # dtype
        swiglu.validate(a.float(), bl, bg)
    with pytest.raises(ValueError):                             # K mismatch
        swiglu.validate(a, torch.zeros(64, 256, dtype=torch.bfloat16), bg)
    with pytest.raises(ValueError):                            # left/gate N/2 differ
        swiglu.validate(a, bl, torch.zeros(128, 128, dtype=torch.bfloat16))
    # M is arbitrary now (ragged M -> ceil-div grid + TMA out-of-bounds clip)
    assert swiglu.validate(torch.zeros(130, 128, dtype=torch.bfloat16), bl, bg) == (130, 512, 128)
    with pytest.raises(ValueError):                          # N=2*192 not mult of 256
        swiglu.validate(a, torch.zeros(128, 192, dtype=torch.bfloat16),
                        torch.zeros(128, 192, dtype=torch.bfloat16))
    with pytest.raises(ValueError):                         # non-unit column stride
        swiglu.validate(a, torch.zeros(128, 512, dtype=torch.bfloat16)[:, ::2], bg)


def test_hopper_swiglu_prefers_packaged_cubin(monkeypatch):
    import mmcomposer.hopper_swiglu as hopper_swiglu

    monkeypatch.setattr(hopper_swiglu, "_CUBIN", None)
    monkeypatch.setattr(hopper_swiglu, "_cuda_driver_version", lambda: 13000)

    def fail_compile(reason):
        raise AssertionError(f"nvcc fallback should not run: {reason}")

    monkeypatch.setattr(hopper_swiglu, "_compile_cubin_fallback", fail_compile)
    cubin = hopper_swiglu._ensure_cubin()
    assert cubin == str(hopper_swiglu._PACKAGED_CUBIN)
    assert hopper_swiglu._PACKAGED_CUBIN.exists()


def _small_cuda_swiglu_inputs(seed=0):
    M, H, K = 257, 256, 256          # packed N = 512; ragged M exercises API shape flow
    torch.manual_seed(seed)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(K, 2 * H, dtype=torch.bfloat16, device="cuda")
    return a, b[:, :H], b[:, H:]


def test_general_api_hopper_swiglu_matches_reference():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("not Hopper")
    a, b_left, b_gate = _small_cuda_swiglu_inputs(seed=10)
    N = 2 * b_left.shape[1]

    c_ref, d_ref = _reference(a, b_left, b_gate, N)

    d = mmc.matmul_swiglu_dual_b(a, b_left, b_gate, sync=True)
    d_rel = ((d.float() - d_ref.float()).norm() / d_ref.float().norm()).item()
    print(f"    Hopper no-preact SwiGLU D rel_err={d_rel:.3e}")
    assert tuple(d.shape) == tuple(d_ref.shape)
    assert d_rel < 5e-2

    d2 = mmc.matmul_swiglu_dual_b(a, b_left, b_gate, out=d, sync=True)
    assert d2.data_ptr() == d.data_ptr()

    d_gm17 = mmc.matmul_swiglu_dual_b(a, b_left, b_gate, gm=17, sync=True)
    d_rel = ((d_gm17.float() - d_ref.float()).norm() / d_ref.float().norm()).item()
    print(f"    Hopper runtime-gm17 SwiGLU D rel_err={d_rel:.3e}")
    assert d_rel < 5e-2

    c, d3 = mmc.matmul_swiglu_dual_b(a, b_left, b_gate, store_preact=True, sync=True)
    c_rel = ((c.float() - c_ref.float()).norm() / c_ref.float().norm()).item()
    d_rel = ((d3.float() - d_ref.float()).norm() / d_ref.float().norm()).item()
    print(f"    Hopper store-preact SwiGLU C rel_err={c_rel:.3e} D rel_err={d_rel:.3e}")
    assert tuple(c.shape) == tuple(c_ref.shape)
    assert tuple(d3.shape) == tuple(d_ref.shape)
    assert c_rel < 5e-2
    assert d_rel < 5e-2

    c2, d4 = mmc.matmul_swiglu_dual_b(a, b_left, b_gate, store_preact=True,
                                      preact=c, out=d3, sync=True)
    assert c2.data_ptr() == c.data_ptr() and d4.data_ptr() == d3.data_ptr()

    with pytest.raises(ValueError, match="inference"):
        mmc.matmul_swiglu_dual_b(a, b_left, b_gate, store_preact=True, gm=17, sync=True)


def test_general_api_rejects_preact_without_store_preact():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    a, b_left, b_gate = _small_cuda_swiglu_inputs(seed=11)
    preact = torch.empty(a.shape[0], 2 * b_left.shape[1], dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="store_preact=True"):
        mmc.matmul_swiglu_dual_b(a, b_left, b_gate, preact=preact)


def _reference(a, b_left, b_gate, N):
    left = torch.mm(a, b_left)
    gate = torch.mm(a, b_gate)
    # C is the standard combined projection [ left | gate ] = a @ [b_left | b_gate]
    # (== x @ W1.t()), i.e. the preactivation a training backward pass expects.
    c_ref = torch.cat([left, gate], dim=1)
    d_ref = (left.float() * (gate.float() * torch.sigmoid(gate.float()))).to(torch.bfloat16)
    return c_ref, d_ref


# ---- GPU: correctness -----------------------------------------------------
def test_swiglu_matches_reference():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("fixed Swiglu kernel is Blackwell sm_100a only")
    M, H, K = 512, 1024, 256          # N = 2H = 2048
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b_left = torch.randn(K, H, dtype=torch.bfloat16, device="cuda")
    b_gate = torch.randn(K, H, dtype=torch.bfloat16, device="cuda")
    N = 2 * H

    c, d = mmc.matmul_swiglu_dual_b(a, b_left, b_gate, store_preact=True)
    c_ref, d_ref = _reference(a, b_left, b_gate, N)

    c_rel = ((c.float() - c_ref.float()).norm() / c_ref.float().norm()).item()
    d_rel = ((d.float() - d_ref.float()).norm() / d_ref.float().norm()).item()
    print(f"    C rel_err={c_rel:.3e}  D rel_err={d_rel:.3e}")
    assert c_rel < 5e-2, f"C rel err too high: {c_rel}"
    assert d_rel < 5e-2, f"D rel err too high: {d_rel}"

    # reuse buffers + same callable path
    c2, d2 = mmc.matmul_swiglu_dual_b(a, b_left, b_gate, store_preact=True,
                                        preact=c, out=d)
    assert c2.data_ptr() == c.data_ptr() and d2.data_ptr() == d.data_ptr()


def test_swiglu_matches_reference_for_packed_b_views():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("fixed Swiglu kernel is Blackwell sm_100a only")
    M, H, K = 512, 1024, 256          # N = 2H = 2048
    torch.manual_seed(1)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(K, 2 * H, dtype=torch.bfloat16, device="cuda")
    b_left = b[:, :H]
    b_gate = b[:, H:]
    assert not b_left.is_contiguous()
    assert b_left.stride() == (2 * H, 1)
    N = 2 * H

    c, d = mmc.matmul_swiglu_dual_b(a, b_left, b_gate, store_preact=True)
    c_ref, d_ref = _reference(a, b_left, b_gate, N)

    c_rel = ((c.float() - c_ref.float()).norm() / c_ref.float().norm()).item()
    d_rel = ((d.float() - d_ref.float()).norm() / d_ref.float().norm()).item()
    print(f"    packed-view C rel_err={c_rel:.3e}  D rel_err={d_rel:.3e}")
    assert c_rel < 5e-2, f"C rel err too high: {c_rel}"
    assert d_rel < 5e-2, f"D rel err too high: {d_rel}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
