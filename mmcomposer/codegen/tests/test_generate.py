"""Golden tests for generate_kernel — GPU-free regression protection.

For each config we pin the *specialized* rendered kernel against a known-good
file in golden/, and assert the converted knob's branch is fully resolved away
(no `#if`, no `if constexpr (<knob>`).  This catches any drift in the codegen
pipeline or the templates without needing a B200.

Regenerate goldens after an intentional template change (review the diff!):
    python webui/codegen/tests/test_generate.py --update

Run:
    python webui/codegen/tests/test_generate.py        # or under pytest
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))  # repo root

from mmcomposer.codegen import generate_kernel   # noqa: E402

GOLDEN = pathlib.Path(__file__).resolve().parent / "golden"

# name -> (config, knobs whose branch must be fully resolved away in the output)
CONFIGS = {
    # Single-CTA warp-spec (TWO_CTA=0) — rendered from the unified skeleton.
    "tier2_overlap": (
        dict(skeleton="tier3_cluster_swizzle", BM=128, BN=256, BK=64, NS=5,
             GROUP_SIZE_M=4, NUM_WARPS=8, TCGEN05_LD_WIDTH=8,
             EPILOGUE_OVERLAP=1, EPILOGUE_SPLIT=0, PERSISTENT=1, TWO_CTA=0,
             EPILOGUE_L1_NO_ALLOC=0, EPILOGUE_TMA_PIPELINED=0,
             TMA_STORE_STAGES=2, SINGLE_TMEM_ACCUM=0),
        ["EPILOGUE_OVERLAP"],
    ),
    "tier2_overlap_tma_pipelined": (
        dict(skeleton="tier3_cluster_swizzle", BM=128, BN=256, BK=64, NS=3,
             GROUP_SIZE_M=8, NUM_WARPS=4, TCGEN05_LD_WIDTH=8,
             EPILOGUE_OVERLAP=1, EPILOGUE_SPLIT=0, PERSISTENT=1, TWO_CTA=0,
             EPILOGUE_L1_NO_ALLOC=0, EPILOGUE_TMA_PIPELINED=1,
             TMA_STORE_STAGES=2, SINGLE_TMEM_ACCUM=0),
        ["EPILOGUE_OVERLAP", "EPILOGUE_TMA_PIPELINED", "TMA_STORE_STAGES", "TWO_CTA"],
    ),
    "tier2_sequential": (
        dict(skeleton="tier3_cluster_swizzle", BM=128, BN=256, BK=64, NS=3,
             GROUP_SIZE_M=8, NUM_WARPS=8, TCGEN05_LD_WIDTH=8,
             EPILOGUE_OVERLAP=0, EPILOGUE_SPLIT=0, PERSISTENT=0, TWO_CTA=0,
             EPILOGUE_L1_NO_ALLOC=0, EPILOGUE_TMA_PIPELINED=0,
             TMA_STORE_STAGES=2, SINGLE_TMEM_ACCUM=0),
        ["EPILOGUE_OVERLAP", "TCGEN05_LD_WIDTH"],
    ),
    "tier3_overlap": (
        dict(skeleton="tier3_cluster_swizzle", BM=128, BN=256, BK=64, NS=3,
             GROUP_SIZE_M=4, NUM_WARPS=8, TCGEN05_LD_WIDTH=8,
             EPILOGUE_OVERLAP=1, EPILOGUE_SPLIT=0, PERSISTENT=1, TWO_CTA=1,
             EPILOGUE_L1_NO_ALLOC=0, EPILOGUE_TMA_PIPELINED=0,
             TMA_STORE_STAGES=2, SINGLE_TMEM_ACCUM=0),
        ["EPILOGUE_OVERLAP"],
    ),
    "tier3_overlap_split": (
        dict(skeleton="tier3_cluster_swizzle", BM=128, BN=256, BK=64, NS=5,
             GROUP_SIZE_M=4, NUM_WARPS=8, TCGEN05_LD_WIDTH=8,
             EPILOGUE_OVERLAP=1, EPILOGUE_SPLIT=1, PERSISTENT=1, TWO_CTA=1,
             EPILOGUE_L1_NO_ALLOC=0, EPILOGUE_TMA_PIPELINED=0,
             TMA_STORE_STAGES=2, SINGLE_TMEM_ACCUM=0),
        ["EPILOGUE_OVERLAP"],
    ),
    "tier3_overlap_tma_pipelined": (
        dict(skeleton="tier3_cluster_swizzle", BM=128, BN=256, BK=64, NS=4,
             GROUP_SIZE_M=8, NUM_WARPS=4, TCGEN05_LD_WIDTH=8,
             EPILOGUE_OVERLAP=1, EPILOGUE_SPLIT=0, PERSISTENT=1, TWO_CTA=1,
             EPILOGUE_L1_NO_ALLOC=0, EPILOGUE_TMA_PIPELINED=1,
             TMA_STORE_STAGES=2, SINGLE_TMEM_ACCUM=0),
        ["EPILOGUE_OVERLAP", "EPILOGUE_TMA_PIPELINED", "TMA_STORE_STAGES"],
    ),
    "tier3_overlap_bn512_single_tmem": (
        dict(skeleton="tier3_cluster_swizzle", BM=128, BN=512, BK=64, NS=4,
             GROUP_SIZE_M=8, NUM_WARPS=4, TCGEN05_LD_WIDTH=8,
             EPILOGUE_OVERLAP=1, EPILOGUE_SPLIT=0, PERSISTENT=1, TWO_CTA=1,
             EPILOGUE_L1_NO_ALLOC=0, EPILOGUE_TMA_PIPELINED=1,
             TMA_STORE_STAGES=2, SINGLE_TMEM_ACCUM=1),
        ["EPILOGUE_OVERLAP", "EPILOGUE_TMA_PIPELINED", "TMA_STORE_STAGES",
         "SINGLE_TMEM_ACCUM"],
    ),
    "tier3_sequential": (
        dict(skeleton="tier3_cluster_swizzle", BM=128, BN=256, BK=64, NS=3,
             GROUP_SIZE_M=8, NUM_WARPS=8, TCGEN05_LD_WIDTH=8,
             EPILOGUE_OVERLAP=0, EPILOGUE_SPLIT=0, PERSISTENT=0, TWO_CTA=1,
             EPILOGUE_L1_NO_ALLOC=0, EPILOGUE_TMA_PIPELINED=0,
             TMA_STORE_STAGES=2, SINGLE_TMEM_ACCUM=0),
        ["EPILOGUE_OVERLAP", "TCGEN05_LD_WIDTH"],
    ),
    "tier3_overlap_split_l1noalloc": (   # exercises the EPILOGUE_L1_NO_ALLOC store macro
        dict(skeleton="tier3_cluster_swizzle", BM=128, BN=256, BK=64, NS=4,
             GROUP_SIZE_M=8, NUM_WARPS=4, TCGEN05_LD_WIDTH=8,
             EPILOGUE_OVERLAP=1, EPILOGUE_SPLIT=1, PERSISTENT=1, TWO_CTA=1,
             EPILOGUE_L1_NO_ALLOC=1, EPILOGUE_TMA_PIPELINED=0,
             TMA_STORE_STAGES=2, SINGLE_TMEM_ACCUM=0),
        ["EPILOGUE_OVERLAP", "EPILOGUE_L1_NO_ALLOC"],
    ),
}


def _assert_branch_free(src: str, knobs):
    # No preprocessor conditionals may survive resolution.
    bad = [ln for ln in src.splitlines()
           if ln.lstrip().startswith("#")
           and ln.lstrip()[1:].lstrip().startswith(("if", "elif", "else", "endif"))]
    assert not bad, f"residual directive(s): {bad[:3]}"
    # And no `if constexpr (<converted knob>` may remain.
    for k in knobs:
        assert f"if constexpr ({k}" not in src, f"unresolved `if constexpr ({k}` remained"


def _check(update: bool = False) -> int:
    GOLDEN.mkdir(exist_ok=True)
    failed = 0
    for name, (config, knobs) in CONFIGS.items():
        out = generate_kernel(config)
        _assert_branch_free(out, knobs)
        path = GOLDEN / f"{name}.cu"
        if update:
            path.write_text(out)
            print(f"  wrote {path.name} ({len(out.splitlines())} lines)")
            continue
        if not path.exists():
            print(f"  FAIL {name}: no golden (run with --update)"); failed += 1; continue
        if out == path.read_text():
            print(f"  PASS {name} ({len(out.splitlines())} lines, branch-free)")
        else:
            print(f"  FAIL {name}: output differs from golden {path.name}"); failed += 1
    return failed


def test_goldens():
    """pytest entry point — asserts every config matches its golden + is branch-free."""
    assert _check(update=False) == 0


if __name__ == "__main__":
    raise SystemExit(1 if _check("--update" in sys.argv[1:]) else 0)
