"""torch.library custom-op registrations for the MMComposer kernels.

The kernels launch cubins through the cuda-python **driver API** (see
``runtime.py`` / ``kernels/_runtime.py``), which is an opaque C call to
``torch.compile``/Dynamo -- entering a compiled region graph-breaks or errors.
Wrapping the launches in ``torch.library.custom_op`` gives Dynamo a single
traceable op node with a known output shape/dtype (``register_fake``), a correct
backward (``register_autograd``), and correct in-place semantics for the ``out=``
variants (``mutates_args``).  The ops then compose with Inductor, CUDA graphs,
and autograd.

Op inventory (namespace ``mmc``):

    mmc::matmul(a, b) -> Tensor                          functional, differentiable
    mmc::matmul_out(a, b, out!) -> ()                    eager out= reuse
    mmc::swiglu_dual_b(a, b_left, b_gate) -> Tensor      D only (Hopper inference)
    mmc::swiglu_dual_b_preact(a, b_l, b_g) -> (C, D)     functional, differentiable
    mmc::swiglu_dual_b_out(..., out!) -> ()              eager out= reuse (D)
    mmc::swiglu_dual_b_preact_out(..., preact!, out!) -> ()   eager reuse (C, D)

Forward bodies dispatch through the thin ``mmc._launch_*`` helpers (device
dispatch + kernel-callable caching + async launch on torch's current stream).
Backward runs the GEMMs through ``torch.matmul`` (cuBLAS / Inductor): the mmc
kernels can't serve them (their contraction dims -- e.g. M for grad_B -- aren't
%64, and operands would be transposed).  The SwiGLU elementwise grad is computed
in fp32 then cast back to bf16.

Requires torch >= 2.4 (the ``torch.library.custom_op`` API).  On older torch (or
if torch is unavailable), ``ENABLED`` is False and nothing is registered; the
``mmc.py`` wrappers then fall back to the direct launch (eager works, compile
does not).
"""
from __future__ import annotations

import typing

try:
    import torch
    _HAVE_CUSTOM_OP = hasattr(torch.library, "custom_op")
except Exception:                       # torch missing / broken -> stay disabled
    torch = None                        # type: ignore
    _HAVE_CUSTOM_OP = False

ENABLED = _HAVE_CUSTOM_OP


if ENABLED:
    # ---- plain GEMM: C = A @ B --------------------------------------------
    @torch.library.custom_op("mmc::matmul", mutates_args=())
    def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        from . import mmc
        return mmc._launch_matmul(a, b, None)

    @matmul.register_fake
    def _(a, b):
        return a.new_empty((a.shape[0], b.shape[1]))

    def _matmul_setup_context(ctx, inputs, output):
        a, b = inputs
        ctx.save_for_backward(a, b)

    def _matmul_backward(ctx, grad):
        a, b = ctx.saved_tensors
        grad_a = grad @ b.t() if ctx.needs_input_grad[0] else None
        grad_b = a.t() @ grad if ctx.needs_input_grad[1] else None
        return grad_a, grad_b

    matmul.register_autograd(_matmul_backward, setup_context=_matmul_setup_context)

    @torch.library.custom_op("mmc::matmul_out", mutates_args={"out"})
    def matmul_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        from . import mmc
        mmc._launch_matmul(a, b, out)

    @matmul_out.register_fake
    def _(a, b, out):
        return None

    # ---- SwiGLU dual-B (D only): inference form ---------------------------
    @torch.library.custom_op("mmc::swiglu_dual_b", mutates_args=())
    def swiglu_dual_b(a: torch.Tensor, b_left: torch.Tensor,
                      b_gate: torch.Tensor) -> torch.Tensor:
        from . import mmc
        return mmc._launch_swiglu_d(a, b_left, b_gate, None)

    @swiglu_dual_b.register_fake
    def _(a, b_left, b_gate):
        return a.new_empty((a.shape[0], b_left.shape[1]))       # D = [M, H]

    @torch.library.custom_op("mmc::swiglu_dual_b_out", mutates_args={"out"})
    def swiglu_dual_b_out(a: torch.Tensor, b_left: torch.Tensor,
                          b_gate: torch.Tensor, out: torch.Tensor) -> None:
        from . import mmc
        mmc._launch_swiglu_d(a, b_left, b_gate, out)

    @swiglu_dual_b_out.register_fake
    def _(a, b_left, b_gate, out):
        return None

    # ---- SwiGLU dual-B (C preact + D): training form ----------------------
    @torch.library.custom_op("mmc::swiglu_dual_b_preact", mutates_args=())
    def swiglu_dual_b_preact(
            a: torch.Tensor, b_left: torch.Tensor, b_gate: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        from . import mmc
        return mmc._launch_swiglu_cd(a, b_left, b_gate, None, None)

    @swiglu_dual_b_preact.register_fake
    def _(a, b_left, b_gate):
        H = b_left.shape[1]
        return (a.new_empty((a.shape[0], 2 * H)),               # C = [M, 2H]
                a.new_empty((a.shape[0], H)))                   # D = [M, H]

    def _swiglu_setup_context(ctx, inputs, output):
        a, b_left, b_gate = inputs
        c, _d = output
        ctx.save_for_backward(a, b_left, b_gate, c)
        ctx.H = b_left.shape[1]

    def _swiglu_backward(ctx, grad_c, grad_d):
        # C = [ L | G ] with L = A@B_left, G = A@B_gate;  D = L * silu(G).
        a, b_left, b_gate, c = ctx.saved_tensors
        H = ctx.H
        L = c[:, :H].float()
        G = c[:, H:].float()
        sig = torch.sigmoid(G)
        silu_g = G * sig                                   # silu(G)
        silu_gp = sig * (1.0 + G * (1.0 - sig))            # d/dG silu(G)
        gd = grad_d.float()
        grad_L = gd * silu_g
        grad_G = gd * L * silu_gp
        if grad_c is not None:                             # preact used downstream too
            grad_L = grad_L + grad_c[:, :H].float()
            grad_G = grad_G + grad_c[:, H:].float()
        grad_L = grad_L.to(torch.bfloat16)
        grad_G = grad_G.to(torch.bfloat16)
        grad_a = grad_b_left = grad_b_gate = None
        if ctx.needs_input_grad[0]:
            grad_a = grad_L @ b_left.t() + grad_G @ b_gate.t()
        if ctx.needs_input_grad[1]:
            grad_b_left = a.t() @ grad_L
        if ctx.needs_input_grad[2]:
            grad_b_gate = a.t() @ grad_G
        return grad_a, grad_b_left, grad_b_gate

    swiglu_dual_b_preact.register_autograd(
        _swiglu_backward, setup_context=_swiglu_setup_context)

    @torch.library.custom_op("mmc::swiglu_dual_b_preact_out",
                             mutates_args={"preact", "out"})
    def swiglu_dual_b_preact_out(
            a: torch.Tensor, b_left: torch.Tensor, b_gate: torch.Tensor,
            preact: torch.Tensor, out: torch.Tensor) -> None:
        from . import mmc
        mmc._launch_swiglu_cd(a, b_left, b_gate, preact, out)

    @swiglu_dual_b_preact_out.register_fake
    def _(a, b_left, b_gate, preact, out):
        return None
