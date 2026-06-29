#!/usr/bin/env python3
"""Tests for the epilogue DSL (pure -- no GPU): tracing + lowering to CUDA."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))  # repo root

import pytest

from mmcomposer import epilogue as epi
from mmcomposer.epilogue import sigmoid, relu, exp, tanh, sqrt, maximum, minimum


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
