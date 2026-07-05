"""Constant substitution into kernel / launcher source.

Rewrites the *value* of a knob's definition line (kernel ``constexpr int NAME =
<int>;`` or launcher ``NAME = <int>``) — definition-line only, so named uses of
the constant elsewhere stay intact.  Moved verbatim from ``mvp_core``."""

from __future__ import annotations

import re


def substitute_kernel_constexprs(src: str, **values) -> str:
    """Rewrite top-of-file ``constexpr int NAME = <int>;`` lines per kwarg.

    Matches ``NAME`` only when immediately followed by optional whitespace
    then ``=`` then digits, so neighbours like ``BN_PAD`` are never touched.
    """
    for name, val in values.items():
        src = re.sub(
            rf"(constexpr\s+int\s+{re.escape(name)}\s*=\s*)\d+",
            lambda m, v=val: f"{m.group(1)}{v}",
            src,
        )
    return src


def substitute_launcher_constants(src: str, **values) -> str:
    """Rewrite the Python knob constants in a tier launcher fragment."""
    if all(k in values for k in ("BM", "BN", "BK")):
        src = re.sub(
            r"BM,\s*BN,\s*BK\s*=\s*\d+,\s*\d+,\s*\d+",
            f"BM, BN, BK = {values['BM']}, {values['BN']}, {values['BK']}",
            src,
        )
    for name in ("NS", "GROUP_SIZE_M", "NUM_WARPS", "PERSISTENT",
                 "TCGEN05_LD_WIDTH", "EPILOGUE_OVERLAP", "EPILOGUE_SPLIT",
                 "EPILOGUE_L1_NO_ALLOC", "EPILOGUE_TMA_PIPELINED",
                 "TMA_STORE_STAGES", "SINGLE_TMEM_ACCUM", "SEGMENTED_PANELS",
                 "TWO_CTA"):
        if name in values:
            src = re.sub(
                rf"^({name}\s*=\s*)\d+",
                lambda m, v=values[name]: f"{m.group(1)}{v}",
                src,
                flags=re.MULTILINE,
            )
    return src
