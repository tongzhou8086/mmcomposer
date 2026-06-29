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
    with pytest.raises(ValueError):                           # M not mult of 256
        swiglu.validate(torch.zeros(128, 128, dtype=torch.bfloat16), bl, bg)
    with pytest.raises(ValueError):                          # N=2*192 not mult of 256
        swiglu.validate(a, torch.zeros(128, 192, dtype=torch.bfloat16),
                        torch.zeros(128, 192, dtype=torch.bfloat16))


def _reference(a, b_left, b_gate, N):
    BN, half = swiglu.BN, swiglu.BN // 2
    M = a.shape[0]
    left = torch.mm(a, b_left)
    gate = torch.mm(a, b_gate)
    c_ref = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    cv = c_ref.view(M, N // BN, BN)
    cv[:, :, :half] = left.view(M, N // BN, half)
    cv[:, :, half:] = gate.view(M, N // BN, half)
    d_ref = (left.float() * (gate.float() * torch.sigmoid(gate.float()))).to(torch.bfloat16)
    return c_ref, d_ref


# ---- GPU: correctness -----------------------------------------------------
def test_swiglu_matches_reference():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    M, H, K = 512, 1024, 256          # N = 2H = 2048
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b_left = torch.randn(K, H, dtype=torch.bfloat16, device="cuda")
    b_gate = torch.randn(K, H, dtype=torch.bfloat16, device="cuda")
    N = 2 * H

    c, d = mmc.matmul_swiglu_dual_b_ns6_s2(a, b_left, b_gate)
    c_ref, d_ref = _reference(a, b_left, b_gate, N)

    c_rel = ((c.float() - c_ref.float()).norm() / c_ref.float().norm()).item()
    d_rel = ((d.float() - d_ref.float()).norm() / d_ref.float().norm()).item()
    print(f"    C rel_err={c_rel:.3e}  D rel_err={d_rel:.3e}")
    assert c_rel < 5e-2, f"C rel err too high: {c_rel}"
    assert d_rel < 5e-2, f"D rel err too high: {d_rel}"

    # reuse buffers + same callable path
    c2, d2 = mmc.matmul_swiglu_dual_b_ns6_s2(a, b_left, b_gate, c=c, d=d)
    assert c2.data_ptr() == c.data_ptr() and d2.data_ptr() == d.data_ptr()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
