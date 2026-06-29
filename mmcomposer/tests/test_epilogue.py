#!/usr/bin/env python3
"""Tests for the epilogue DSL (pure -- no GPU): tracing + lowering to CUDA."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))  # repo root

import pytest

from mmcomposer import epilogue as epi
from mmcomposer.epilogue import sigmoid, relu, exp, tanh, sqrt, log, maximum, minimum


def test_identity():
    assert epi.to_cuda(lambda x: x) == "x"


def test_arithmetic_and_constants():
    assert epi.to_cuda(lambda x: x * 2.0) == "(x * 2.0f)"
    assert epi.to_cuda(lambda x: 2 * x) == "(2.0f * x)"
    assert epi.to_cuda(lambda x: x + 1) == "(x + 1.0f)"
    assert epi.to_cuda(lambda x: 1 - x) == "(1.0f - x)"
    assert epi.to_cuda(lambda x: -x) == "(-x)"
    assert epi.to_cuda(lambda x: 1.0 / x) == "(1.0f / x)"


def test_primitives_map_to_intrinsics():
    assert epi.to_cuda(lambda x: exp(x)) == "__expf(x)"
    assert epi.to_cuda(lambda x: tanh(x)) == "tanhf(x)"
    assert epi.to_cuda(lambda x: sqrt(x)) == "sqrtf(x)"
    assert epi.to_cuda(lambda x: abs(x)) == "fabsf(x)"
    assert epi.to_cuda(lambda x: maximum(x, 0.0)) == "fmaxf(x, 0.0f)"
    assert epi.to_cuda(lambda x: minimum(x, 6.0)) == "fminf(x, 6.0f)"


def test_composites_expand_to_primitives():
    # sigmoid(x) = 1/(1+exp(-x))
    assert epi.to_cuda(sigmoid) == "(1.0f / (1.0f + __expf((-x))))"
    # relu(x) = maximum(x, 0)
    assert epi.to_cuda(relu) == "fmaxf(x, 0.0f)"


def test_silu_and_def_form():
    silu_lambda = lambda x: x * sigmoid(x)            # noqa: E731
    expected = "(x * (1.0f / (1.0f + __expf((-x)))))"
    assert epi.to_cuda(silu_lambda) == expected

    def silu(x):                                       # def works too
        return x * sigmoid(x)
    assert epi.to_cuda(silu) == expected


def test_pow_small_int_expands():
    assert epi.to_cuda(lambda x: x ** 2) == "(x * x)"
    assert epi.to_cuda(lambda x: x ** 3) == "(x * x * x)"
    assert epi.to_cuda(lambda x: x ** 0) == "1.0f"


def test_relu6_compose():
    assert epi.to_cuda(lambda x: minimum(maximum(x, 0.0), 6.0)) == "fminf(fmaxf(x, 0.0f), 6.0f)"


def test_digest_stable_and_distinct():
    assert epi.digest(lambda x: x * sigmoid(x)) == epi.digest(lambda x: x * sigmoid(x))
    assert epi.digest(relu) != epi.digest(sigmoid)


def test_rejects_control_flow_and_bad_returns():
    with pytest.raises(TypeError):
        epi.to_cuda(lambda x: x if x else 0)          # bool() on Expr -> blocked
    with pytest.raises(TypeError):
        epi.to_cuda(lambda x: (x, x))                 # must return one value
    with pytest.raises(TypeError):
        epi.to_cuda(lambda x: x ** x)                 # non-constant exponent


# ---- GPU integration: every builtin/op fused, vs a torch reference ---------
# Each case is (name, edl_fn, ref_fn): the EDL epilogue and the equivalent torch
# expression on the fp32 GEMM result.  Inputs are scaled so a@b ~ N(0,1), keeping
# every op in a sane domain (sqrt/log are wrapped to stay non-negative).
_EDL_CASES = [
    ("identity",  lambda x: x,                         lambda t: t),
    ("neg",       lambda x: -x,                        lambda t: -t),
    ("add_const", lambda x: x + 0.5,                   lambda t: t + 0.5),
    ("mul_const", lambda x: 2.0 * x,                   lambda t: 2.0 * t),
    ("div_const", lambda x: x / 2.0,                   lambda t: t / 2.0),
    ("pow2",      lambda x: x ** 2,                     lambda t: t ** 2),
    ("pow3",      lambda x: x ** 3,                     lambda t: t ** 3),
    ("abs",       lambda x: abs(x),                     lambda t: t.abs()),
    ("exp",       lambda x: exp(x),                     lambda t: t.exp()),
    ("tanh",      lambda x: tanh(x),                    lambda t: t.tanh()),
    ("sqrt",      lambda x: sqrt(abs(x)),               lambda t: t.abs().sqrt()),
    ("log",       lambda x: log(abs(x) + 1.0),          lambda t: (t.abs() + 1.0).log()),
    ("maximum",   lambda x: maximum(x, 0.0),            lambda t: t.clamp_min(0.0)),
    ("minimum",   lambda x: minimum(x, 0.5),            lambda t: t.clamp_max(0.5)),
    ("sigmoid",   sigmoid,                              lambda t: t.sigmoid()),
    ("relu",      relu,                                 lambda t: t.clamp_min(0.0)),
    ("relu6",     lambda x: minimum(maximum(x, 0.0), 6.0), lambda t: t.clamp(0.0, 6.0)),
    ("silu",      lambda x: x * sigmoid(x),             lambda t: t * t.sigmoid()),
    ("gelu_tanh", lambda x: 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x ** 3))),
                  lambda t: 0.5 * t * (1.0 + (0.7978845608 * (t + 0.044715 * t ** 3)).tanh())),
    ("leaky",     lambda x: maximum(x, 0.01 * x),
                  lambda t: t.clamp_min(0.0) + 0.01 * t.clamp_max(0.0)),
]


@pytest.fixture(scope="module")
def _gpu_ctx():
    """Tight pretune in an isolated cache; return (mmc, a, b, base) with a@b~N(0,1)."""
    import os
    import tempfile
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from mmcomposer import autotune, mvp_core as mc
    import mmcomposer.mmc as mmc

    old = os.environ.get("MMCOMPOSER_CACHE_DIR")
    os.environ["MMCOMPOSER_CACHE_DIR"] = tempfile.mkdtemp(prefix="mmc_epi_test_")
    M = N = K = 512
    ws = list(dict.fromkeys(t["dir"] for k, t in mc.TIER_MAP.items() if t and k[0]))
    tight = {"bn": [256], "ns": [4], "gsm": [8], "nw": [8], "two_cta": [1],
             "persistent": [1], "overlap": [1], "split_epilogue": [0],
             "l1_no_alloc": [0], "tma_pipelined": [1], "tma_store_stages": [2],
             "single_tmem": [0]}
    s = autotune.tune(M, N, K, tier_dirs=ws, filters=tight,
                      cublas_samples=1, cublas_warmup_samples=0)
    assert s["ok"], "pre-tune failed"
    torch.manual_seed(0)
    sc = K ** -0.25                                    # a@b ~ N(0,1)
    a = (torch.randn(M, K, device="cuda") * sc).to(torch.bfloat16)
    b = (torch.randn(K, N, device="cuda") * sc).to(torch.bfloat16)
    base = a.float() @ b.float()
    try:
        yield mmc, a, b, base
    finally:
        if old is None:
            os.environ.pop("MMCOMPOSER_CACHE_DIR", None)
        else:
            os.environ["MMCOMPOSER_CACHE_DIR"] = old


def test_identity_epilogue_is_bit_exact(_gpu_ctx):
    """The identity epilogue must produce exactly the same bits as a plain matmul."""
    mmc, a, b, base = _gpu_ctx
    ci = mmc.matmul(a, b, epilogue=lambda x: x)
    cp = mmc.matmul(a, b)
    assert (ci.float() - cp.float()).abs().max().item() == 0.0


@pytest.mark.parametrize("name,edl,ref", _EDL_CASES, ids=[c[0] for c in _EDL_CASES])
def test_builtin_fused_matches_torch(_gpu_ctx, name, edl, ref):
    """Each builtin/op, fused as an epilogue, matches the torch reference in fp32."""
    mmc, a, b, base = _gpu_ctx
    c = mmc.matmul(a, b, epilogue=edl)
    want = ref(base)
    rel = ((c.float() - want).norm() / (want.norm() + 1e-12)).item()
    assert rel < 5e-2, f"{name}: rel_err {rel:.2e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
