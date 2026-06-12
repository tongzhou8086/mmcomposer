"""Audit a generated kernel for *unresolved* knob conditionals.

After generation a kernel must contain no preprocessor conditional directives and
no `if constexpr` on a knob that has a #if-style branch (a forgotten conversion).
This deliberately does NOT flag a plain `if constexpr (BN >= 256)` — BN is a
substituted literal, so the C++ compiler resolves it; only the knobs we route
through the #if resolver are checked."""

from __future__ import annotations

import re

# Knobs whose branches are resolved at generation (so they must NOT survive as
# `if constexpr`).  LDW is the derived alias of TCGEN05_LD_WIDTH used in the
# epilogue branches.
_BRANCH_KNOBS = ("EPILOGUE_OVERLAP", "EPILOGUE_SPLIT",
                 "TCGEN05_LD_WIDTH", "LDW", "TWO_CTA", "EPILOGUE_L1_NO_ALLOC",
                 "EPILOGUE_TMA_PIPELINED")
_DIRECTIVE = re.compile(r"^\s*#\s*(if|elif|else|endif)\b")


def branch_free_issues(src: str) -> list[str]:
    """Return a list of human-readable problems; empty means fully branch-free."""
    issues = []
    for i, line in enumerate(src.splitlines(), 1):
        s = line.strip()
        if _DIRECTIVE.match(line):
            issues.append(f"line {i}: unresolved preprocessor directive: {s}")
            continue
        if s.startswith("//"):
            continue
        if "if constexpr (" in s and any(k in s for k in _BRANCH_KNOBS):
            issues.append(f"line {i}: unresolved knob branch: {s}")
    return issues
