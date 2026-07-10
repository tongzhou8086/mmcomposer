#!/usr/bin/env python3
"""Tests for the torch custom-op registrations (mmcomposer/torch_ops.py).

CPU layer: the ops are registered under the ``mmc`` namespace (needs torch>=2.4).
GPU layer: eager equivalence vs a torch reference, ``torch.library.opcheck``
(schema/fake/aliasing/autograd), no-graph-break under ``torch.compile(...,
fullgraph=True)``, and autograd vs a pure-torch reference.  Skipped without CUDA.

Works on both arches: on Hopper the fixed WS / SwiGLU kernels run directly; on
Blackwell the plain GEMM is pre-tuned with a tight filter first (SwiGLU is a
fixed kernel there too).

Run:  python mmcomposer/tests/test_torch_ops.py   (or pytest)
"""
import os
import pathlib
import sys
import tempfile

os.environ.setdefault("MMCOMPOSER_CACHE_DIR", tempfile.mkdtemp(prefix="mmc_ops_test_"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))  # repo root

import mmcomposer.mmc as mmc          # importing mmc registers the ops
from mmcomposer import torch_ops


# ---- helpers --------------------------------------------------------------
def _skip(msg):
    print(f"    SKIP ({msg})")
    return True


def _no_cuda():
    import torch
    return not torch.cuda.is_available()


def _is_blackwell():
    import torch
    return torch.cuda.get_device_capability()[0] == 10


def _rel(out, ref):
    return ((out.float() - ref.float()).norm() / ref.float().norm()).item()


def _ensure_matmul_ready(M, N, K):
    """On Blackwell, pre-tune (M,N,K) with a tight filter so the op doesn't kick
    off a full sweep; on Hopper the fixed kernel needs no tuning (no-op)."""
    if not _is_blackwell():
        return
    from mmcomposer import autotune, mvp_core
    ws = list(dict.fromkeys(t["dir"] for k, t in mvp_core.TIER_MAP.items() if t and k[0]))
    tight = {"bn": [256], "ns": [4], "gsm": [8], "nw": [8], "two_cta": [1],
             "persistent": [1], "overlap": [1], "split_epilogue": [0],
             "l1_no_alloc": [0], "tma_pipelined": [1], "tma_store_stages": [2],
             "single_tmem": [0]}
    s = autotune.tune(M, N, K, tier_dirs=ws, filters=tight,
                      cublas_samples=1, cublas_warmup_samples=0)
    assert s["ok"], "pre-tune failed"


# ---- CPU layer ------------------------------------------------------------
def test_ops_registered():
    import torch
    if not torch_ops.ENABLED:
        return _skip(f"torch {torch.__version__} has no torch.library.custom_op")
    ns = torch.ops.mmc
    for name in ("matmul", "matmul_out", "swiglu_dual_b", "swiglu_dual_b_out",
                 "swiglu_dual_b_preact", "swiglu_dual_b_preact_out"):
        assert hasattr(ns, name), f"mmc::{name} not registered"


# ---- plain GEMM -----------------------------------------------------------
def test_matmul_op_matches_reference():
    import torch
    if _no_cuda():
        return _skip("no CUDA")
    if not torch_ops.ENABLED:
        return _skip("custom_op unavailable")
    M = N = K = 512
    _ensure_matmul_ready(M, N, K)
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    ref = a.float() @ b.float()
    c = torch.ops.mmc.matmul(a, b)
    torch.cuda.synchronize()
    assert tuple(c.shape) == (M, N) and c.dtype == torch.bfloat16
    rel = _rel(c, ref)
    print(f"    matmul op rel err = {rel:.3e}")
    assert rel < 5e-2
    # out-variant writes into the provided buffer and matches
    buf = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    torch.ops.mmc.matmul_out(a, b, buf)
    torch.cuda.synchronize()
    assert _rel(buf, ref) < 5e-2


def test_matmul_opcheck():
    import torch
    if _no_cuda():
        return _skip("no CUDA")
    if not torch_ops.ENABLED:
        return _skip("custom_op unavailable")
    M = N = K = 512
    _ensure_matmul_ready(M, N, K)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    torch.library.opcheck(torch.ops.mmc.matmul.default, (a, b))
    print("    opcheck(mmc::matmul) passed")


def test_matmul_compiles_fullgraph():
    import torch
    if _no_cuda():
        return _skip("no CUDA")
    if not torch_ops.ENABLED:
        return _skip("custom_op unavailable")
    M = N = K = 512
    _ensure_matmul_ready(M, N, K)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    ref = a.float() @ b.float()
    torch._dynamo.reset()
    fn = torch.compile(lambda a, b: mmc.matmul(a, b), fullgraph=True)
    c = fn(a, b)                     # fullgraph=True raises on any graph break
    torch.cuda.synchronize()
    rel = _rel(c, ref)
    print(f"    compiled matmul rel err = {rel:.3e}")
    assert rel < 5e-2


def test_matmul_autograd():
    import torch
    if _no_cuda():
        return _skip("no CUDA")
    if not torch_ops.ENABLED:
        return _skip("custom_op unavailable")
    M = N = K = 512
    _ensure_matmul_ready(M, N, K)
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    g = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")   # upstream grad

    c = torch.ops.mmc.matmul(a, b)
    c.backward(g)
    # reference (same bf16 inputs; torch.matmul backward)
    ar = a.detach().clone().requires_grad_(True)
    br = b.detach().clone().requires_grad_(True)
    (ar @ br).backward(g)
    ra, rb = _rel(a.grad, ar.grad), _rel(b.grad, br.grad)
    print(f"    matmul grad rel err: a={ra:.3e} b={rb:.3e}")
    assert ra < 5e-2 and rb < 5e-2


# ---- fused SwiGLU dual-B --------------------------------------------------
def _swiglu_inputs(M=512, K=512, H=512, grad=False):
    import torch
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=grad)
    bl = torch.randn(K, H, dtype=torch.bfloat16, device="cuda", requires_grad=grad)
    bg = torch.randn(K, H, dtype=torch.bfloat16, device="cuda", requires_grad=grad)
    return a, bl, bg


def test_swiglu_preact_op_matches_reference():
    import torch
    import torch.nn.functional as F
    if _no_cuda():
        return _skip("no CUDA")
    if not torch_ops.ENABLED:
        return _skip("custom_op unavailable")
    a, bl, bg = _swiglu_inputs()
    c, d = torch.ops.mmc.swiglu_dual_b_preact(a, bl, bg)
    torch.cuda.synchronize()
    c_ref = torch.cat([a.float() @ bl.float(), a.float() @ bg.float()], dim=1)
    d_ref = (a.float() @ bl.float()) * F.silu(a.float() @ bg.float())
    rc, rd = _rel(c, c_ref), _rel(d, d_ref)
    print(f"    swiglu preact op rel err: C={rc:.3e} D={rd:.3e}")
    assert rc < 5e-2 and rd < 5e-2


def test_swiglu_compiles_fullgraph():
    import torch
    if _no_cuda():
        return _skip("no CUDA")
    if not torch_ops.ENABLED:
        return _skip("custom_op unavailable")
    a, bl, bg = _swiglu_inputs()
    torch._dynamo.reset()
    fn = torch.compile(
        lambda a, bl, bg: mmc.matmul_swiglu_dual_b(a, bl, bg, store_preact=True)[1],
        fullgraph=True)
    d = fn(a, bl, bg)                # raises on graph break
    torch.cuda.synchronize()
    import torch.nn.functional as F
    d_ref = (a.float() @ bl.float()) * F.silu(a.float() @ bg.float())
    rel = _rel(d, d_ref)
    print(f"    compiled swiglu rel err = {rel:.3e}")
    assert rel < 5e-2


def test_swiglu_autograd():
    import torch
    import torch.nn.functional as F
    if _no_cuda():
        return _skip("no CUDA")
    if not torch_ops.ENABLED:
        return _skip("custom_op unavailable")
    a, bl, bg = _swiglu_inputs(grad=True)
    _c, d = torch.ops.mmc.swiglu_dual_b_preact(a, bl, bg)
    gd = torch.randn_like(d)
    d.backward(gd)
    # reference: differentiate (a@bl)*silu(a@bg) through torch
    ar = a.detach().clone().requires_grad_(True)
    blr = bl.detach().clone().requires_grad_(True)
    bgr = bg.detach().clone().requires_grad_(True)
    d_ref = (ar @ blr) * F.silu(ar @ bgr)
    d_ref.backward(gd)
    ra, rl, rg = _rel(a.grad, ar.grad), _rel(bl.grad, blr.grad), _rel(bg.grad, bgr.grad)
    print(f"    swiglu grad rel err: a={ra:.3e} b_left={rl:.3e} b_gate={rg:.3e}")
    assert ra < 6e-2 and rl < 6e-2 and rg < 6e-2


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
