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
# BN caps at 256: the tcgen05.mma N-atom max is 256 columns per MMA, and
# the 2-CTA cluster splits M (not N), so BN=512 throws ILLEGAL_INSTRUCTION
# in every tier (empirically confirmed by the B200 codegen test).
BN_OPTS  = [64, 128, 256]
# BK is locked to 64: the K-major B TMA descriptor uses SWIZZLE_128B, whose
# inner box is exactly one 128 B atom = 64 BF16 elements.  Other values
# don't compile / load garbage, so only 64 is offered.
BK_OPTS  = [64]
NS_OPTS  = [2, 3, 4, 5, 6, 7]
GSM_OPTS = [1, 2, 4, 8, 16, 32]
NW_OPTS  = [4, 8, 16]
# Epilogue Phase-2 store path: 0 = all-thread int4 stores; 1 = one async
# TMA store per CTA (swizzled SMEM staging).  A universal toggle.
TMA_STORE_OPTS = [0, 1]
# On/off knobs are presented as dropdowns too, for a uniform UI (and to
# leave room for an "Auto" value once auto-tuning lands).
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
    },
    (True, False): {
        "label":   "Tier 2 — Multi-stage + warp specialization",
        "desc":    "Multi-stage ring + dedicated TMA + MMA warps (async), single-CTA, "
                   "CTA-swizzle tunable.",
        "dir":     "tier2_multistage_ws",
        "symbol":  "matmul_coalesced_epilogue",
        "cluster": False,
    },
    (True, True): {
        "label":   "Tier 3 — + 2-CTA cluster MMA",
        "desc":    "Warp-spec + 2-CTA cluster (`cta_group::2`), half-B per CTA, deeper NS, "
                   "CTA-swizzle tunable.",
        "dir":     "tier3_cluster_swizzle",
        "symbol":  "matmul_cluster",
        "cluster": True,
    },
    (False, True): None,   # invalid: 2-CTA cluster requires warp specialization
}


def tier_for(ms_ws: bool, two_cta: bool):
    """Return the tier dict for a toggle combination, or None if invalid."""
    return TIER_MAP.get((ms_ws, two_cta))


# ── Validation ────────────────────────────────────────────────────────

def validate_config(bm, bn, bk, ns, gsm, nw, *, cluster: bool, tma_store=0) -> list[str]:
    """Return a list of human-readable warnings; empty list = valid.

    ``cluster`` selects the 2-CTA geometry: each CTA owns BN/2 columns of
    B but still BM rows, so the per-CTA B SMEM slot is halved.
    ``tma_store`` selects the epilogue Phase-2 store path (0 int4 / 1 TMA);
    it changes the epilogue staging width (TMA needs a dense BM×BN buffer).
    """
    out: list[str] = []
    cta_group = 2 if cluster else 1

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
    # tcgen05.mma N-atom max is 256 columns per MMA.  The 2-CTA cluster
    # splits M (and B storage), NOT N, so the MMA N-tile is BN in every
    # tier — BN>256 throws ILLEGAL_INSTRUCTION everywhere (empirically
    # confirmed on B200 for both single-CTA and cluster tiers).
    if bn > 256:
        out.append(
            f"**BN = {bn}**: exceeds the 256-column tcgen05.mma.kind::f16 N-atom max.  "
            "The 2-CTA cluster splits M, not N, so this fails in every tier."
        )
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
            if bn % col_groups != 0 or (bn // col_groups) % 8 != 0:
                out.append(
                    f"**BN = {bn}** doesn't divide into {col_groups} column groups of a "
                    f"multiple of 8 (got {bn // col_groups} cols/group; 8-col tcgen05.ld atoms)."
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
    # SMEM budget (B200 = 228 KB/CTA).  K-loop ring (NS slots) and epilogue
    # staging share the dynamic region but are time-disjoint -> size for max.
    a_slot   = bm * bk * 2
    b_slot   = (bn // cta_group) * bk * 2
    slot     = a_slot + b_slot
    epi      = bm * (bn if tma_store else (bn + 8)) * 2   # TMA store needs a dense BM×BN buffer
    smem     = max(ns * slot, epi) + 1024
    if smem > 228 * 1024:
        out.append(
            f"**SMEM usage {smem / 1024:.0f} KB > B200 cap (228 KB)** at "
            f"(BM={bm}, BN={bn}, BK={bk}, NS={ns}, cluster={cluster}): "
            f"K-loop ring = {ns} x {slot // 1024} KB = {ns * slot // 1024} KB, "
            f"epilogue staging = {epi // 1024} KB."
        )
    return out


# ── Substitution ──────────────────────────────────────────────────────

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
    for name in ("NS", "GROUP_SIZE_M", "NUM_WARPS", "TMA_STORE"):
        if name in values:
            src = re.sub(
                rf"^({name}\s*=\s*)\d+",
                lambda m, v=values[name]: f"{m.group(1)}{v}",
                src,
                flags=re.MULTILINE,
            )
    return src


def knob_kwargs(bm, bn, bk, ns, gsm, nw, tma_store=0) -> dict:
    """Map UI knobs to the constant names used in the source files."""
    return {"BM": bm, "BN": bn, "BK": bk, "NS": ns, "GROUP_SIZE_M": gsm,
            "NUM_WARPS": nw, "TMA_STORE": tma_store}


def _strip_module_docstring(src: str) -> str:
    """Drop a leading triple-quoted module docstring, if present."""
    m = re.match(r'\s*(?:"""|\'\'\')', src)
    if not m:
        return src
    quote = src[m.end() - 3:m.end()]
    end = src.find(quote, m.end())
    if end == -1:
        return src
    return src[end + 3:].lstrip("\n")


EPI_MARKER = "// @@EPILOGUE@@"


def _splice_epilogue(src: str) -> str:
    """Replace the `// @@EPILOGUE@@` marker line with the shared epilogue
    fragment, so all tiers share one epilogue source."""
    if EPI_MARKER not in src:
        return src
    frag = (KERNELS_DIR / "_epilogue.cu.frag").read_text()
    if not frag.endswith("\n"):
        frag += "\n"
    out = []
    for line in src.splitlines(keepends=True):
        if line.strip() == EPI_MARKER:
            out.append(frag)
        else:
            out.append(line)
    return "".join(out)


def render_kernel(tier: dict, bm, bn, bk, ns, gsm, nw, tma_store=0) -> str:
    """Return the substituted, epilogue-stitched kernel.cu for a tier."""
    src = (KERNELS_DIR / tier["dir"] / "kernel.cu").read_text()
    src = _splice_epilogue(src)
    return substitute_kernel_constexprs(src, **knob_kwargs(bm, bn, bk, ns, gsm, nw, tma_store))


def render_host(tier: dict, bm, bn, bk, ns, gsm, nw, tma_store=0) -> str:
    """Return a *self-contained* host script: runtime preamble + launcher.

    The result has no ``cuda_utils`` import and no ``sys.path`` hack — it
    runs anywhere with torch + cuda-python + nvcc on PATH.
    """
    runtime  = _strip_module_docstring(RUNTIME_PY.read_text())
    launcher = (KERNELS_DIR / tier["dir"] / "launcher.py").read_text()
    launcher = substitute_launcher_constants(launcher, **knob_kwargs(bm, bn, bk, ns, gsm, nw, tma_store))
    header = (
        '"""Self-contained matmul kernel launcher generated by mmcomposer.\n'
        "\n"
        f"Tier: {tier['label']}\n"
        f"Config: BM={bm} BN={bn} BK={bk} NS={ns} GROUP_SIZE_M={gsm} NUM_WARPS={nw} TMA_STORE={tma_store}\n"
        "\n"
        "Run with:  python <this file>.py   (kernel.cu must sit alongside it)\n"
        "Requires:  torch, numpy, cuda-python (cuda.bindings), and nvcc on PATH.\n"
        '"""\n'
    )
    return f"{header}\n{runtime}\n\n# ===== tier launcher =====\n\n{launcher}"


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
    """(tier_dir, bm, bn, bk, ns, gsm, nw, tma_store) -> entry dict."""
    idx = {}
    for e in load_compat().get("entries", []):
        idx[(e["tier"], e["bm"], e["bn"], e["bk"], e["ns"], e["gsm"], e["nw"],
             e.get("tma_store", 0))] = e
    return idx


def compat_status(tier_dir, bm, bn, bk, ns, gsm, nw, tma_store=0):
    """Return ('verified'|'failed'|'unknown', entry|None) for a combo.

    'verified'/'failed' come from the empirical B200 sweep; 'unknown'
    means the combo wasn't in the swept grid (fall back to static)."""
    e = _compat_index().get((tier_dir, bm, bn, bk, ns, gsm, nw, tma_store))
    if e is None:
        return "unknown", None
    return ("verified" if e["correct"] else "failed"), e


def compat_perf(tier_dir, bm, bn, bk, ns, gsm, nw, shape_m, tma_store=0):
    """Return the measured {rel_err, tflops, vs_cublas} for this combo at a
    square shape (M=N=K=shape_m), or None if not in the matrix."""
    e = _compat_index().get((tier_dir, bm, bn, bk, ns, gsm, nw, tma_store))
    if e is None:
        return None
    return (e.get("perf") or {}).get(str(shape_m))


def cublas_tflops(shape_m):
    """Measured cuBLAS TFLOPS at a square shape, or None."""
    return load_compat().get("cublas_tflops", {}).get(str(shape_m))


def perf_shapes():
    """Square shapes (ints) the compat matrix recorded performance at."""
    return load_compat().get("perf_shapes", [])
