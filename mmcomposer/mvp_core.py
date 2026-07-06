"""mmcomposer MVP — pure core logic (no Streamlit, no GPU).

This module is the single source of truth for:

  * the user-facing knob option lists (so the UI and the test enumerate
    an identical configuration space),
  * the tier map (toggle combination -> implementation strategy + the
    MVP-owned kernel directory under ``webui/kernels/``),
  * constraint validation for a chosen knob combination,
  * substitution of knob values into the kernel + host source, and
  * assembly of a *self-contained* host script (runtime preamble +
    tier launcher) — the artifact the user downloads and can run with
    just ``python host.py`` (plus torch / cuda-python / nvcc).

``app.py`` imports everything from here; the tests import from here too,
so they exercise the same code the UI ships.  ``tutorial/`` is a
reference implementation only — the MVP renders this owned codebase.
"""

from __future__ import annotations

import functools
import json
import pathlib
import re

# Code generation (skeleton + knob config -> specialized kernel/host) lives in
# the `codegen` package; mvp_core keeps the UI option lists, validation and
# compat lookup, and delegates rendering.  substitute_* / FRAGMENTS are
# re-exported here for back-compat with existing callers.
from .codegen import generate_kernel as _generate_kernel
from .codegen import generate_host as _generate_host
from .codegen.fragments import FRAGMENTS  # noqa: F401  (re-export)
from .codegen.substitute import (  # noqa: F401  (re-export)
    substitute_kernel_constexprs, substitute_launcher_constants)


# ── Paths ─────────────────────────────────────────────────────────────

WEBUI_DIR   = pathlib.Path(__file__).resolve().parent
KERNELS_DIR = WEBUI_DIR / "kernels"
RUNTIME_PY  = KERNELS_DIR / "_runtime.py"
COMPAT_JSON = KERNELS_DIR / "compat_matrix.json"


# ── Knob option lists (single source of truth for UI + tests) ─────────
#
# BM is locked to 128: tcgen05.mma.kind::f16 has an M-atom of 128 (BM<128
# reads past the A tile) and TMEM has 128 lanes (BM>128 won't fit a
# single CTA).  Larger M is handled by the 2-CTA cluster tier, where each
# CTA still holds 128 of the rows — so BM stays 128 there too.
BM_OPTS  = [128]
# Ordinary tcgen05.mma.kind::f16 has an N-atom max of 256 columns, so BN>256
# needs an explicit panelized implementation.  The first BN=512 implementation
# is tightly guarded to the validated single-TMEM overlap pipeline.
BN_OPTS  = [64, 128, 256, 512]
# BK is locked to 64: the K-major B TMA descriptor uses SWIZZLE_128B, whose
# inner box is exactly one 128 B atom = 64 BF16 elements.  Other values
# don't compile / load garbage, so only 64 is offered.
BK_OPTS  = [64]
NS_OPTS  = [2, 3, 4, 5, 6, 7]
GSM_OPTS = [1, 2, 4, 8, 16, 32]
NW_OPTS  = [4, 8, 16]
# Epilogue TMEM->register load width: 32-bit elements per lane per tcgen05.ld
# (.32x32b.xN).  Wider = fewer loads + fewer wait_ld syncs, more registers.
# Must divide COLS_PER_WARP (= BN / (NW/(BM/32))).  Pinned to [8]: the B200
# sweeps showed wider widths (16/32/64) don't raise the autotuned ceiling, so
# they only bloat the grid.  The x16 code path is kept (validator allows 8/16,
# epilogue dispatch handles it) — re-add 16 here to sweep it again.
TCGEN05_LD_WIDTH_OPTS = [8]
# Persistent grid: 0 = one CTA per output tile (grid = num_tiles); 1 = one
# CTA per SM, each walking a strided run of tiles (grid = #SMs).  A host/
# launch knob — same cubin both ways — wired on Tier 2 (a small reproducible
# win) and on Tier 3 only when EPILOGUE_OVERLAP supplies the persistent
# cluster tile loop.
PERSISTENT_OPTS = [0, 1]
# Epilogue/K-loop overlap (Step B): 1 = persistent pipeline that runs each
# tile's epilogue concurrently with the next tile's K-loop (TMEM double-buffer
# + disjoint epilogue SMEM).  The 2 stream warps (TMA + MMA) take warpgroup 0;
# num_warps epilogue warps run in their own warpgroup(s) from warp 4 (warps 2,3
# idle — tcgen05.ld epilogue warps must not share warpgroup 0 with the MMA warp),
# so the block is (NW+4) warps and the epilogue scales with NW.  Requires
# persistent on, int4 store, and the disjoint-SMEM budget (NS
# small).  A win on epilogue-bound low-K shapes.  Tier 2 and Tier 3.
EPILOGUE_OVERLAP_OPTS = [0, 1]
# Split the overlapped Tier 3 int4 epilogue into two half-BN staging/store
# passes.  This roughly halves the epilogue SMEM footprint, allowing a deeper
# K-loop ring such as NS=5 at BN=256.  It is exposed as a separate experiment
# knob because it trades less SMEM for an extra epilogue pass/barrier.
EPILOGUE_SPLIT_OPTS = [0, 1]
# Write the C output with `st...L1::no_allocate` so the streamed (write-once)
# result doesn't evict A/B from L1.  Shape-dependent: a measured +3-6% when the
# epilogue is exposed (low K) and null at high K — hence a sweep knob, not
# always-on.
EPILOGUE_L1_NO_ALLOC_OPTS = [0, 1]
# Alternative overlapped epilogue mode: TMEM -> registers -> compact swizzled
# SMEM buffers -> chunked TMA stores.  Unlike the dropped simple TMA-store
# path, this is pipelined and can overlap store traffic with subsequent
# epilogue chunks / tiles.  It is mutually exclusive with the staged int4
# modifiers (split writeback and L1 no-allocate).
EPILOGUE_TMA_PIPELINED_OPTS = [0, 1]
# Number of compact STORE_N=64 SMEM buffers in the pipelined TMA-store
# epilogue.  Production timing sweeps prune this to [1, 2], but the generator
# and correctness sweep keep [1, 2, 3, 4] because higher values are valid and
# useful for controlled experiments.
TMA_STORE_STAGES_OPTS = [1, 2, 3, 4]
# Reuse one TMEM accumulator buffer in the overlap pipeline.  This changes the
# tmem_empty/tmem_full synchronization between the MMA warp and epilogue drain;
# BN512 currently requires it as part of its guarded two-panel implementation,
# but the knob itself is not an epilogue-store-style constraint.
SINGLE_TMEM_ACCUM_OPTS = [0, 1]
# BN512 segmented panel schedule: process K in segments of SEG = NS k-tiles,
# running all of panel 0 then all of panel 1 per segment (A resident for the
# segment, loaded once; one recycled FIFO B ring).  Both halves of BN512's
# single-TMEM reuse delay are hidden: panel 0's drain overlaps the last
# segment's panel-1 MMAs, and the per-half TMEM release lets panel 1's drain
# overlap the NEXT tile's first panel-0 segment.  Measured vs BN512 NS4/ts2
# (exclusive B200 node): +1.6% at 8192^3, +6% at K=2048, up to +16% on
# epilogue-bound low-K shapes; parity at worst.  Requires the BN512 bundle
# (cluster+persistent+overlap+tma_pipelined) and SINGLE_TMEM_ACCUM=1 —
# enforced by validate_config, so enumeration only pairs it with single_tmem on.
SEG_PANELS_OPTS = [0, 1]
# Shared on/off labels for non-Streamlit callers.  The main Streamlit UI
# presents these binary knobs as toggles.
ONOFF_OPTS = ["Off", "On"]

GPU_OPTS   = ["B200 (sm_100a)", "H100 (sm_90a) — coming soon", "RTX 50xx (sm_120) — coming soon"]
DTYPE_OPTS = ["bfloat16", "float16 — coming soon", "fp8 e4m3 — coming soon"]


# ── Tier map: (multistage+warpspec, two_cta) -> implementation ────────
#
# CTA swizzling (GROUP_SIZE_M) and the warp count (NUM_WARPS) are
# *universal* tunables — every tier exposes the full knob set.  The
# tiers differ only in MMA / cluster strategy:
#
#   Tier 1 — multi-stage ring + synchronous MMA (no warp spec), 1 CTA.
#   Tier 2 — multi-stage + warp-specialized async MMA, 1 CTA.
#   Tier 3 — warp-spec + 2-CTA cluster MMA (cta_group::2).
TIER_MAP = {
    (False, False): {
        "label":   "Tier 1 — Baseline",
        "desc":    "Multi-stage SMEM ring + synchronous MMA (no warp specialization), "
                   "single-CTA, generalized variable-warp epilogue, CTA-swizzle tunable.",
        "dir":     "tier1_baseline",
        "symbol":  "matmul_dbuf",
        "cluster": False,
        "persistent_ok": False,   # no tile loop yet (synchronous MMA tier)
    },
    (True, False): {
        "label":   "Tier 2 — Multi-stage + warp specialization",
        "desc":    "Multi-stage ring + dedicated TMA + MMA warps (async), single-CTA, "
                   "CTA-swizzle tunable.",
        # Unified warp-spec skeleton (shared with the 2-CTA tier); the cluster
        # vs single-CTA difference is the TWO_CTA knob (= cluster here -> 0).
        "dir":     "tier3_cluster_swizzle",
        "symbol":  "matmul_cluster",
        "cluster": False,
        "persistent_ok": True,    # persistent-capable tile loop (Step A)
    },
    (True, True): {
        "label":   "Tier 3 — + 2-CTA cluster MMA",
        "desc":    "Warp-spec + 2-CTA cluster (`cta_group::2`), half-B per CTA, deeper NS, "
                   "CTA-swizzle tunable.",
        "dir":     "tier3_cluster_swizzle",
        "symbol":  "matmul_cluster",
        "cluster": True,
        # Persistent is valid only through the overlap path; validation rejects
        # persistent cluster configs with overlap off.
        "persistent_ok": True,
    },
    (False, True): None,   # invalid: 2-CTA cluster requires warp specialization
}


def tier_for(ms_ws: bool, two_cta: bool):
    """Return the tier dict for a toggle combination, or None if invalid."""
    return TIER_MAP.get((ms_ws, two_cta))


# ── Validation ────────────────────────────────────────────────────────

def normalize_tma_store_stages(tma_pipelined=0, tma_store_stages=2) -> int:
    """Collapse the TMA-store stage knob when the TMA epilogue is disabled."""
    return int(tma_store_stages) if int(tma_pipelined) else 2


def validate_config(bm, bn, bk, ns, gsm, nw, *, cluster: bool,
                    persistent=0, persistent_ok=True, shape=None, ld_width=8,
                    overlap=0, split_epilogue=0, l1_no_alloc=0,
                    tma_pipelined=0, tma_store_stages=2,
                    single_tmem=0, seg_panels=0) -> list[str]:
    """Return a list of human-readable warnings; empty list = valid.

    ``cluster`` selects the 2-CTA geometry: each CTA owns BN/2 columns of
    B but still BM rows, so the per-CTA B SMEM slot is halved.
    ``persistent`` launches grid = #SMs with a CTA-level tile loop; it is
    only valid on tiers whose kernel actually has that loop (``persistent_ok``).
    """
    out: list[str] = []
    cta_group = 2 if cluster else 1
    tma_store_stages = normalize_tma_store_stages(tma_pipelined, tma_store_stages)

    # Shape-tiling: only BK must divide K (the K-loop has no partial-tile path).
    # M and N need not divide the tile: the kernel launches ceil-div edge tiles
    # and TMA clips the out-of-bounds box (zero-fill on load, masked on store), so
    # a ragged M/N -- including M not a multiple of 2*BM under the 2-CTA cluster,
    # where the trailing rows are simply padded away by the store mask -- is fine.
    # (N's multiple-of-8 stride-alignment requirement is enforced at the API.)
    if shape is not None:
        M, N, K = shape
        if K % bk:
            out.append(f"**K = {K}** must be a multiple of BK = {bk}.")

    # Persistent is a launch-side knob; it needs a CTA tile loop to launch
    # with grid < num_tiles.  Both warp-spec paths now have one (the merged
    # skeleton's non-overlap path is persistent for single-CTA AND 2-CTA —
    # TWO_CTA and EPILOGUE_OVERLAP are independent of PERSISTENT).
    if persistent and not persistent_ok:
        out.append(
            "**Persistent grid** is only available on warp-specialized paths; "
            "other knob combinations have no CTA tile loop, "
            "so grid = #SMs would leave most output tiles uncomputed."
        )

    # Epilogue overlap (Step B): a persistent pipeline with the 2 stream warps
    # (TMA + MMA) in warpgroup 0 and num_warps epilogue warps from warp 4, with
    # an int4 store.  num_warps is the EPILOGUE warp count (validated by the grid
    # rules below); the block is (num_warps + 4) warps (warps 2,3 idle).
    if overlap:
        if not persistent_ok:
            out.append("**Epilogue overlap** is only available on warp-specialized paths.")
        if not persistent:
            out.append("**Epilogue overlap** requires **Persistent grid** on "
                       "(it's a persistent pipeline launched with grid = #SMs).")

    if tma_pipelined:
        if not persistent_ok:
            out.append("**Pipelined TMA-store epilogue** is only available on warp-specialized paths.")
        if not persistent:
            out.append("**Pipelined TMA-store epilogue** requires **Persistent grid**.")
        if not overlap:
            out.append("**Pipelined TMA-store epilogue** requires **Epilogue overlap**.")
        if split_epilogue:
            out.append("**Pipelined TMA-store epilogue** replaces **Split epilogue writeback**; "
                       "turn split off.")
        if l1_no_alloc:
            out.append("**Pipelined TMA-store epilogue** uses TMA stores, so "
                       "**L1 no-allocate C store** does not apply.")

    if tma_store_stages not in TMA_STORE_STAGES_OPTS:
        out.append(
            f"**TMA store stages = {tma_store_stages}** must be one of "
            f"{'/'.join(str(x) for x in TMA_STORE_STAGES_OPTS)}."
        )

    if split_epilogue:
        if not cluster:
            out.append("**Split epilogue writeback** currently applies only to "
                       "the **2-CTA cluster MMA** path.")
        if not overlap:
            out.append("**Split epilogue writeback** requires **Epilogue overlap**.")
        if bn % 2 != 0 or (bn // 2) % 8 != 0:
            out.append(f"**BN = {bn}** must split into two int4-aligned halves "
                       "for split epilogue writeback.")

    if ns < 2:
        out.append(
            f"**NS = {ns}** must be >= 2 (NS=2 is the minimum double-buffer overlap; "
            "NS=1 serializes TMA and MMA on the same slot)."
        )
    # SWIZZLE_128B forces the TMA inner-box dim to one 128 B atom = 64 BF16
    # elements.  BK is that inner dim on both A's and B's descriptors.
    if bk != 64:
        out.append(
            f"**BK = {bk}**: TMA with `SWIZZLE_128B` requires the inner-box dim to be "
            "exactly 128 bytes = 64 BF16 elements, so BK is locked at 64."
        )
    # BN must be a multiple of the 64-wide K-major B TMA sub-tile, and in
    # the cluster tier must split evenly across the 2 CTAs.
    if bn % 64 != 0:
        out.append(f"**BN = {bn}** must be a multiple of 64 (the K-major B TMA sub-tile width).")
    if cluster and (bn // cta_group) % 64 != 0:
        out.append(
            f"**BN = {bn}**: in the 2-CTA cluster tier each CTA owns BN/2 = {bn // 2} "
            "columns, which must still be a multiple of 64."
        )
    # Ordinary tcgen05.mma N-atom max is 256 columns per MMA.  BN512 needs
    # explicit panelized MMA; for now we only validate the studied BN512
    # implementation bundle.  These constraints belong to BN=512, not to the
    # SINGLE_TMEM_ACCUM knob by itself.
    if bn > 512:
        out.append(f"**BN = {bn}**: only the guarded BN=512 panelized mode is implemented.")
    elif bn == 512:
        if not cluster:
            out.append("**BN = 512** is currently implemented only for the "
                       "**2-CTA cluster MMA** path.")
        if not persistent:
            out.append("**BN = 512** currently requires **Persistent grid**.")
        if not overlap:
            out.append("**BN = 512** currently requires **Epilogue overlap**.")
        if not tma_pipelined:
            out.append("**BN = 512** currently requires **Pipelined TMA-store epilogue**.")
        if not single_tmem:
            out.append("**BN = 512** currently requires **Single-TMEM accumulator sync**.")
        if split_epilogue:
            out.append("**BN = 512** currently uses the pipelined TMA-store epilogue, "
                       "so **Split epilogue writeback** must be off.")
        if l1_no_alloc:
            out.append("**BN = 512** currently uses TMA stores for C, so "
                       "**L1 no-allocate C store** does not apply.")

    # Segmented panel schedule: a BN=512-only reordering of the two 256-wide
    # MMA panels (see SEG_PANELS_OPTS).  It piggybacks on the BN512 bundle and
    # the single shared accumulator; everything else about the combo is the
    # standard pipelined-TMA overlap path.
    if seg_panels:
        if bn != 512:
            out.append("**Segmented panel schedule** applies only to **BN = 512** "
                       "(it reorders the two 256-wide MMA panels).")
        if not single_tmem:
            out.append("**Segmented panel schedule** drains one shared TMEM accumulator, "
                       "so it requires **Single-TMEM accumulator sync**.")
        if not tma_pipelined:
            out.append("**Segmented panel schedule** is implemented on the "
                       "**Pipelined TMA-store epilogue** path only.")
        if not overlap:
            out.append("**Segmented panel schedule** requires **Epilogue overlap** "
                       "(its win is the panel-0 drain overlapping panel-1 compute).")
    # Phase-1 epilogue 2D warp grid: row_warp = warp_id % (BM/32),
    # col_warp = warp_id / (BM/32).  Needs BM%32==0, NW % (BM/32)==0,
    # BN divides into the column groups in multiples of 8 (tcgen05.ld atoms).
    if bm % 32 != 0:
        out.append(f"**BM = {bm}** must be a multiple of 32 (epilogue reads 32-row TMEM strips).")
    else:
        row_strips = bm // 32
        if nw % row_strips != 0:
            out.append(
                f"**num_warps = {nw}** must be a multiple of BM/32 = {row_strips} "
                "(epilogue splits warps into row strips x column groups)."
            )
        else:
            col_groups = nw // row_strips
            if tma_pipelined and (8 % col_groups != 0):
                out.append(
                    f"**Pipelined TMA-store epilogue** needs STORE_N/8 = 8 chunk loads "
                    f"to divide across {col_groups} column warp groups."
                )
            if bn % col_groups != 0 or (bn // col_groups) % 8 != 0:
                out.append(
                    f"**BN = {bn}** doesn't divide into {col_groups} column groups of a "
                    f"multiple of 8 (got {bn // col_groups} cols/group; 8-col tcgen05.ld atoms)."
                )
            else:
                cols_per_warp = bn // col_groups
                if ld_width not in (8, 16):
                    out.append(f"**Epilogue tcgen05.ld width = {ld_width}** must be one of 8/16.")
                elif cols_per_warp % ld_width != 0:
                    out.append(
                        f"**Epilogue tcgen05.ld width = {ld_width}** must divide the per-warp column "
                        f"span COLS_PER_WARP = BN/(NW/(BM/32)) = {cols_per_warp} "
                        f"(at BN={bn}, NW={nw}).  Use a smaller ld width or fewer warps."
                    )
                if split_epilogue:
                    split_cols_per_warp = (bn // 2) // col_groups
                    if (bn // 2) % col_groups != 0 or split_cols_per_warp % ld_width != 0:
                        out.append(
                            f"**Split epilogue** needs BN/2 divided across {col_groups} column groups "
                            f"to be a multiple of LD width {ld_width} (got {split_cols_per_warp})."
                        )
    # NW < 4 triggers a separate tcgen05 quirk (wrong output); options
    # start at 4 so this is informational only.
    if nw < 4:
        out.append(f"**num_warps = {nw}** must be >= 4 (a tcgen05 epilogue quirk produces wrong output below 4).")

    # BM floor (all tiers keep BM=128 per the option list, but validate anyway).
    if bm != 128:
        out.append(
            f"**BM = {bm}**: only BM=128 is supported (tcgen05.mma.kind::f16 M-atom is 128; "
            "TMEM holds 128 lanes per CTA)."
        )
    # Phase-2 flat-walk needs BM*BN divisible by THREADS*8.
    if (bm * bn) % (nw * 32 * 8) != 0:
        out.append(
            f"**BM x BN = {bm * bn}** is not a multiple of THREADS x 8 = {nw * 32 * 8} "
            "(Phase-2 epilogue flat walk would leave positions uncovered)."
        )
    if split_epilogue and (bm * (bn // 2)) % (nw * 32 * 8) != 0:
        out.append(
            f"**BM x (BN/2) = {bm * (bn // 2)}** is not a multiple of "
            f"THREADS x 8 = {nw * 32 * 8} (split epilogue flat walk would "
            "leave positions uncovered)."
        )
    # SMEM budget (B200 = 228 KB/CTA nominal).  Normally the K-loop ring and epilogue
    # staging are time-disjoint (size for max).  With epilogue overlap they run
    # concurrently, so the staging is a *separate* region (size = ring + epi).
    a_slot   = bm * bk * 2
    b_slot   = (bn // cta_group) * bk * 2
    slot     = a_slot + b_slot
    # Tier 3 split writeback stages one half-BN column panel at a time, reducing
    # epilogue SMEM enough to try deeper rings such as NS=5 at BN=256.
    if overlap and tma_pipelined:
        epi = bm * 64 * 2 * tma_store_stages   # STORE_N=64, TMA_STORE_STAGES buffers
    elif overlap and cluster and split_epilogue:
        epi = bm * (bn // 2 + 8) * 2
    else:
        epi = bm * (bn + 8) * 2   # int4 staging buffer, +8 bank-pad
    if seg_panels:
        # Segmented layout: [ A ring (NS+1) | B ring (budget-fill) | C_store ].
        # The B ring is a pure FIFO prefetch queue, so it takes whatever remains
        # of the 14-tile budget (14 x 16 KB + 1 KB = 230400 B fits the 225 KB
        # single-TMEM cap exactly); the only structural floor is 2 slots.
        seg_na = ns + 1
        seg_nb = 14 - tma_store_stages - seg_na
        seg_b_slot = (bn // 2 // cta_group) * bk * 2
        smem = seg_na * a_slot + seg_nb * seg_b_slot + epi + 1024
        if seg_nb < 2:
            out.append(
                f"**Segmented panel schedule at NS = {ns}, TMA store stages = "
                f"{tma_store_stages}** leaves the B ring only {seg_nb} slot(s) "
                "(A ring NS+1 + C_store exhaust the 14-tile SMEM budget); "
                "use a shallower NS or fewer store stages."
            )
    else:
        smem = (ns * slot + epi if overlap else max(ns * slot, epi)) + 1024
    smem_cap = (225 if (overlap and single_tmem) else 224 if overlap else 228) * 1024
    if smem > smem_cap:
        out.append(
            f"**SMEM usage {smem / 1024:.0f} KB > B200 usable cap ({smem_cap // 1024} KB)** at "
            f"(BM={bm}, BN={bn}, BK={bk}, NS={ns}, cluster={cluster}"
            f"{', overlap' if overlap else ''}): "
            f"K-loop ring = {ns} x {slot // 1024} KB = {ns * slot // 1024} KB, "
            f"epilogue staging = {epi // 1024} KB"
            f"{' (disjoint)' if overlap else ' (aliased)'}."
        )
    return out


# ── Substitution ──────────────────────────────────────────────────────
# substitute_kernel_constexprs / substitute_launcher_constants now live in the
# codegen package (imported + re-exported at the top of this module).

def knob_kwargs(bm, bn, bk, ns, gsm, nw, persistent=0, ld_width=8,
                overlap=0, split_epilogue=0, l1_no_alloc=0,
                tma_pipelined=0, tma_store_stages=2,
                single_tmem=0, seg_panels=0) -> dict:
    """Map UI knobs to the constant names used in the source files.

    PERSISTENT only appears in the launcher (it's a grid choice, not a
    kernel constexpr); substitute_kernel_constexprs simply finds no match
    in kernel.cu and leaves it untouched.
    """
    tma_store_stages = normalize_tma_store_stages(tma_pipelined, tma_store_stages)
    return {"BM": bm, "BN": bn, "BK": bk, "NS": ns, "GROUP_SIZE_M": gsm,
            "NUM_WARPS": nw, "PERSISTENT": persistent,
            "TCGEN05_LD_WIDTH": ld_width, "EPILOGUE_OVERLAP": overlap,
            "EPILOGUE_SPLIT": split_epilogue, "EPILOGUE_L1_NO_ALLOC": l1_no_alloc,
            "EPILOGUE_TMA_PIPELINED": tma_pipelined,
            "TMA_STORE_STAGES": tma_store_stages,
            "SINGLE_TMEM_ACCUM": single_tmem,
            "SEGMENTED_PANELS": seg_panels}


# (_strip_module_docstring moved to codegen.generate, used by generate_host.)


def render_kernel(tier: dict, bm, bn, bk, ns, gsm, nw, ld_width=8,
                  overlap=0, split_epilogue=0, l1_no_alloc=0,
                  tma_pipelined=0, tma_store_stages=2,
                  single_tmem=0, seg_panels=0, epilogue=None, n_extra=0) -> str:
    """Return the kernel.cu specialized to this knob combo (delegates to codegen).

    `epilogue` is an optional CUDA fp32 expression in terms of `x` (and `c0..` for
    `n_extra` extra inputs) -- the elementwise epilogue fused into the store path."""
    config = knob_kwargs(bm, bn, bk, ns, gsm, nw, ld_width=ld_width,
                         overlap=overlap, split_epilogue=split_epilogue,
                         l1_no_alloc=l1_no_alloc,
                         tma_pipelined=tma_pipelined,
                         tma_store_stages=tma_store_stages,
                         single_tmem=single_tmem, seg_panels=seg_panels)
    config["skeleton"] = tier["dir"]
    config["TWO_CTA"] = int(tier["cluster"])   # cluster tier -> the cta_group::2 #if arms
    if epilogue:
        config["EPILOGUE_FN"] = epilogue
    config["MMC_N_EXTRA"] = int(n_extra)
    return _generate_kernel(config)


def render_host(tier: dict, bm, bn, bk, ns, gsm, nw, persistent=0, ld_width=8,
                overlap=0, split_epilogue=0, l1_no_alloc=0,
                tma_pipelined=0, tma_store_stages=2,
                single_tmem=0, seg_panels=0) -> str:
    """Return a self-contained host script for this knob combo (delegates to codegen)."""
    config = knob_kwargs(bm, bn, bk, ns, gsm, nw, persistent=persistent,
                         ld_width=ld_width, overlap=overlap, split_epilogue=split_epilogue,
                         l1_no_alloc=l1_no_alloc,
                         tma_pipelined=tma_pipelined,
                         tma_store_stages=tma_store_stages,
                         single_tmem=single_tmem, seg_panels=seg_panels)
    config["skeleton"] = tier["dir"]
    config["label"] = tier["label"]
    config["TWO_CTA"] = int(tier["cluster"])
    return _generate_host(config)


def parse_shapes(text: str):
    out = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        try:
            m, n, k = (int(x.strip()) for x in line.split(","))
            out.append((m, n, k))
        except Exception:
            continue
    return out


# ── Empirical compatibility matrix (B200 ground truth) ───────────────
#
# Produced by webui/tests/gpu_codegen_driver.py: every static-valid
# combo compiled + run + correctness-checked on a real B200.  The app
# filters/annotates against this so the online experience reflects what
# actually runs, not just what the static checker predicts.

@functools.lru_cache(maxsize=1)
def load_compat() -> dict:
    """Load the committed compat matrix; {} if absent.  Cached."""
    try:
        return json.loads(COMPAT_JSON.read_text())
    except FileNotFoundError:
        return {}


@functools.lru_cache(maxsize=1)
def _compat_index() -> dict:
    """(tier_dir, two_cta, bm, bn, bk, ns, gsm, nw, persistent,
    ld_width, overlap, split_epilogue, l1_no_alloc, tma_pipelined,
    tma_store_stages, single_tmem, seg_panels) -> entry.

    two_cta is part of the key because the warp-spec single-CTA and 2-CTA
    cluster tiers share one ``tier`` dir, distinguished only by that knob."""
    idx = {}
    for e in load_compat().get("entries", []):
        idx[(e["tier"], e.get("two_cta", 0), e["bm"], e["bn"], e["bk"], e["ns"],
             e["gsm"], e["nw"], e.get("persistent", 0),
             e.get("ld_width", 8), e.get("overlap", 0), e.get("split_epilogue", 0),
             e.get("l1_no_alloc", 0), e.get("tma_pipelined", 0),
             e.get("tma_store_stages", 2),
             e.get("single_tmem", 0), e.get("seg_panels", 0))] = e
    return idx


def compat_status(tier_dir, bm, bn, bk, ns, gsm, nw, persistent=0,
                  ld_width=8, overlap=0, split_epilogue=0, two_cta=0,
                  l1_no_alloc=0, tma_pipelined=0, tma_store_stages=2,
                  single_tmem=0, seg_panels=0):
    """Return ('verified'|'failed'|'unknown', entry|None) for a combo.

    'verified'/'failed' come from the empirical B200 sweep; 'unknown'
    means the combo wasn't in the swept grid (fall back to static)."""
    tma_store_stages = normalize_tma_store_stages(tma_pipelined, tma_store_stages)
    e = _compat_index().get((tier_dir, two_cta, bm, bn, bk, ns, gsm, nw,
                             persistent, ld_width, overlap, split_epilogue,
                             l1_no_alloc, tma_pipelined, tma_store_stages,
                             single_tmem, seg_panels))
    if e is None:
        return "unknown", None
    return ("verified" if e["correct"] else "failed"), e


def shape_key(M, N, K):
    """Canonical perf-matrix key for a shape: ``'S'`` for a square SxSxS
    (back-compatible with the old int-string keys), else ``'MxNxK'``."""
    return str(M) if (M == N == K) else f"{M}x{N}x{K}"


def compat_perf(tier_dir, bm, bn, bk, ns, gsm, nw, M, N, K,
                persistent=0, ld_width=8, overlap=0, split_epilogue=0, two_cta=0,
                l1_no_alloc=0, tma_pipelined=0, tma_store_stages=2,
                single_tmem=0, seg_panels=0):
    """Return the measured {rel_err, tflops, vs_cublas} for this combo at
    shape (M, N, K), or None if not in the matrix."""
    tma_store_stages = normalize_tma_store_stages(tma_pipelined, tma_store_stages)
    e = _compat_index().get((tier_dir, two_cta, bm, bn, bk, ns, gsm, nw,
                             persistent, ld_width, overlap, split_epilogue,
                             l1_no_alloc, tma_pipelined, tma_store_stages,
                             single_tmem, seg_panels))
    if e is None:
        return None
    return (e.get("perf") or {}).get(shape_key(M, N, K))


def cublas_tflops(M, N, K):
    """Measured cuBLAS TFLOPS at shape (M, N, K), or None."""
    return load_compat().get("cublas_tflops", {}).get(shape_key(M, N, K))


def toggles_for_dir(tier_dir):
    """Reverse the tier map: tier_dir -> (ms_ws, two_cta) toggle state."""
    for (ms_ws, two_cta), t in TIER_MAP.items():
        if t and t["dir"] == tier_dir:
            return ms_ws, two_cta
    return False, False


def recommended_config(shape=None):
    """The highest-measured-TFLOPS *correct* config from the empirical matrix
    — the 'recommended' defaults.  ``shape`` is an (M, N, K) tuple (or an int
    for a square shape); None / unswept falls back to the largest swept square
    (its optimum is the stable page-load default).  Returns a dict with
    tier_dir, knobs, persistent, ms_ws, two_cta, tflops, shape;
    or None if the matrix is empty."""
    entries = [e for e in load_compat().get("entries", []) if e.get("correct")]
    shapes = perf_shapes()
    if not entries or not shapes:
        return None
    if isinstance(shape, int):
        shape = (shape, shape, shape)
    keys = {shape_key(*t) for t in shapes}
    if shape is not None and shape_key(*shape) in keys:
        ref = shape_key(*shape)
    else:
        # default / unswept: largest swept *square* shape (stable optimum).
        squares = [t for t in shapes if t[0] == t[1] == t[2]]
        ref = shape_key(*max(squares or shapes))
    best, best_tf = None, -1.0
    for e in entries:
        tf = ((e.get("perf") or {}).get(ref) or {}).get("tflops")
        if tf is not None and tf > best_tf:
            best, best_tf = e, tf
    if best is None:
        return None
    # two_cta is a recorded knob now (the two warp-spec arms share a dir), so
    # take it from the entry; the dir only tells us whether warp-spec is on.
    ms_ws, _ = toggles_for_dir(best["tier"])
    two_cta = best.get("two_cta", 0)
    knobs = {k: best[k] for k in ("tier", "bm", "bn", "bk", "ns", "gsm", "nw")}
    knobs["persistent"] = best.get("persistent", 0)
    knobs["ld_width"] = best.get("ld_width", 8)
    knobs["overlap"] = best.get("overlap", 0)
    knobs["split_epilogue"] = best.get("split_epilogue", 0)
    knobs["l1_no_alloc"] = best.get("l1_no_alloc", 0)
    knobs["tma_pipelined"] = best.get("tma_pipelined", 0)
    knobs["tma_store_stages"] = best.get("tma_store_stages", 2)
    knobs["single_tmem"] = best.get("single_tmem", 0)
    knobs["seg_panels"] = best.get("seg_panels", 0)
    return {**knobs, "ms_ws": ms_ws, "two_cta": two_cta,
            "tflops": best_tf, "shape": ref}


def perf_shapes():
    """List of (M, N, K) shapes the compat matrix recorded performance at.
    Legacy square int entries are normalized to (S, S, S)."""
    out = []
    for s in load_compat().get("perf_shapes", []):
        out.append(tuple(s) if isinstance(s, (list, tuple)) else (s, s, s))
    return out
