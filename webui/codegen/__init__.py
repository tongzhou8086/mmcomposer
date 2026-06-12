"""mmcomposer codegen — turn a config (dict of selected option values) into a
CUDA kernel specialized to exactly that combo.

Public contract:
    generate_kernel(config) -> str   # the .cu, with every knob branch resolved away
    generate_host(config)   -> str   # a self-contained host.py for that config

`config` is the JSON-style dict the user's choices map to: the integer knobs
(BM, BN, …, EPILOGUE_OVERLAP, EPILOGUE_SPLIT, EPILOGUE_TMA_PIPELINED) plus a `skeleton` (tier dir) and,
for host generation, a `label`.  How fragments are spliced and `#if` branches
are resolved is internal (see directives.py)."""

from __future__ import annotations

from .audit import branch_free_issues
from .directives import DirectiveError, resolve as resolve_directives
from .fragments import FRAGMENTS, splice
from .generate import generate_host, generate_kernel
from .substitute import substitute_kernel_constexprs, substitute_launcher_constants

__all__ = [
    "generate_kernel", "generate_host",
    "branch_free_issues",
    "resolve_directives", "DirectiveError",
    "splice", "FRAGMENTS",
    "substitute_kernel_constexprs", "substitute_launcher_constants",
]
