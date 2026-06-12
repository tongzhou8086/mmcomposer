"""Config schema for the codegen contract.

A `config` is a plain dict of selected option values (the "JSON" the user picks):
the integer knobs that become ``constexpr int`` in the kernel, plus a
``skeleton`` (which tier implementation to render) and, for host generation, a
``label``.  Domain validation of *which combos are valid* lives in
``mvp_core.validate_config`` (UI concern); this is just the presence/shape check
the generator itself needs."""

from __future__ import annotations

from .fragments import KERNELS_DIR

# Integer knobs that the kernel references (as constexpr defs and/or #if
# conditions).  PERSISTENT is launcher-only but harmless to carry in config.
REQUIRED_KNOBS = (
    "BM", "BN", "BK", "NS", "GROUP_SIZE_M", "NUM_WARPS",
    "TCGEN05_LD_WIDTH", "EPILOGUE_OVERLAP", "EPILOGUE_SPLIT",
    "TWO_CTA", "EPILOGUE_L1_NO_ALLOC", "EPILOGUE_TMA_PIPELINED",
)


def int_knobs(config: dict) -> dict:
    """The integer-valued entries of `config` — what the resolver evaluates and
    the substituter rewrites (drops `skeleton`/`label` and any non-int)."""
    return {k: v for k, v in config.items()
            if isinstance(v, int) and not isinstance(v, bool)}


def validate(config: dict, *, need_label: bool = False) -> None:
    """Raise ValueError unless `config` has everything the generator needs."""
    skel = config.get("skeleton")
    if not skel:
        raise ValueError("config is missing 'skeleton' (which tier to render)")
    if not (KERNELS_DIR / skel / "kernel.cu").exists():
        raise ValueError(f"unknown skeleton {skel!r} (no {skel}/kernel.cu under kernels/)")
    missing = [k for k in REQUIRED_KNOBS if k not in config]
    if missing:
        raise ValueError(f"config is missing required knobs: {', '.join(missing)}")
    if need_label and "label" not in config:
        raise ValueError("host generation needs a 'label' in config")
