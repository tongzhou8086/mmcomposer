"""Building-block fragment splicing.

Each tier skeleton has marker lines like ``// @@EPILOGUE@@``; rendering replaces
the marker with the contents of a shared ``.frag`` file so a building block
(epilogue, MMA chain, tcgen05.ld, overlap drain) lives in exactly one place.
Moved verbatim from ``mvp_core`` (single source of truth now lives here)."""

from __future__ import annotations

import pathlib

KERNELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "kernels"

FRAGMENTS = {
    "// @@EPILOGUE@@":         "_epilogue.cu.frag",
    "// @@MMA_CHAIN@@":        "_mma_chain.cu.frag",
    "// @@TCGEN05_LD@@":       "_tcgen05_ld.cu.frag",
    "// @@OVERLAP_EPILOGUE@@": "_overlap_epilogue.cu.frag",
}


def splice(src: str) -> str:
    """Replace each building-block marker line with its fragment file."""
    out = []
    for line in src.splitlines(keepends=True):
        marker = line.strip()
        if marker in FRAGMENTS:
            frag = (KERNELS_DIR / FRAGMENTS[marker]).read_text()
            out.append(frag if frag.endswith("\n") else frag + "\n")
        else:
            out.append(line)
    return "".join(out)
