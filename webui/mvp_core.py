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
# BN caps at 256.  The tcgen05.mma N-atom max is 256 columns per MMA, so a
# single MMA can't do BN=512 (empirically: ILLEGAL_INSTRUCTION).  BN=512 is
# *technically* doable with two N=256 MMAs per K-step, but we considered it
# and decided against it — both ways of tiling negate large-BN's main
# benefit (amortizing A reuse across N):
#   * outer 2-pass over the K-loop re-loads A each pass → no amortization;
#   * inner N-tile doubles B SMEM → NS drops 7→4 (cluster) → loses pipelining.
# So BN=512 is unlikely to beat BN=256, for non-trivial kernel surgery.
# Revisit only if a restructuring shares A across the N passes.
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
# Persistent grid: 0 = one CTA per output tile (grid = num_tiles); 1 = one
# CTA per SM, each walking a strided run of tiles (grid = #SMs).  A host/
# launch knob — same cubin both ways — wired on Tier 2 (a small reproducible
# win).  Tried on the Tier 3 cluster but measured a wash (cluster-barrier
# overhead cancels the gain), so it stays off there.
PERSISTENT_OPTS = [0, 1]
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
        "persistent_ok": False,   # no tile loop yet (synchronous MMA tier)
    },
    (True, False): {
        "label":   "Tier 2 — Multi-stage + warp specialization",
        "desc":    "Multi-stage ring + dedicated TMA + MMA warps (async), single-CTA, "
                   "CTA-swizzle tunable.",
        "dir":     "tier2_multistage_ws",
        "symbol":  "matmul_coalesced_epilogue",
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
        # Persistent tried on the cluster tier (Step A) but measured a wash-
        # to-loss: the extra cross-CTA cluster barrier per tile cancels the
        # scheduling gain.  Kept non-persistent (the fast proven kernel).
        "persistent_ok": False,
    },
    (False, True): None,   # invalid: 2-CTA cluster requires warp specialization
}


def tier_for(ms_ws: bool, two_cta: bool):
    """Return the tier dict for a toggle combination, or None if invalid."""
    return TIER_MAP.get((ms_ws, two_cta))


# ── Validation ────────────────────────────────────────────────────────

def validate_config(bm, bn, bk, ns, gsm, nw, *, cluster: bool, tma_store=0,
                    persistent=0, persistent_ok=True, shape=None) -> list[str]:
    """Return a list of human-readable warnings; empty list = valid.

    ``cluster`` selects the 2-CTA geometry: each CTA owns BN/2 columns of
    B but still BM rows, so the per-CTA B SMEM slot is halved.
    ``tma_store`` selects the epilogue Phase-2 store path (0 int4 / 1 TMA);
    it changes the epilogue staging width (TMA needs a dense BM×BN buffer).
    ``persistent`` launches grid = #SMs with a CTA-level tile loop; it is
    only valid on tiers whose kernel actually has that loop (``persistent_ok``).
    """
    out: list[str] = []
    cta_group = 2 if cluster else 1

    # Shape-tiling: does this config's tile geometry tile (M, N, K) exactly?
    # Pure divisibility, known statically — mirrors the sweep's skip rule.
    if shape is not None:
        M, N, K = shape
        if M % bm:
            out.append(f"**M = {M}** must be a multiple of BM = {bm}.")
        if N % bn:
            out.append(f"**N = {N}** must be a multiple of BN = {bn}.")
        if K % bk:
            out.append(f"**K = {K}** must be a multiple of BK = {bk}.")
        if cluster and (M % (cta_group * bm)):
            out.append(
                f"**M = {M}**: the 2-CTA cluster tiles {cta_group}×BM = {cta_group * bm} "
                f"rows per cluster, so M must be a multiple of {cta_group * bm} "
                f"(M % {cta_group * bm} = {M % (cta_group * bm)}).  Turn off the 2-CTA cluster for this M."
            )

    # Persistent is a launch-side knob, but only the warp-spec single-CTA path
    # has the CTA tile loop needed to launch with grid < num_tiles.
    if persistent and not persistent_ok:
        out.append(
            "**Persistent grid** is only available with warp specialization on and "
            "the 2-CTA cluster off; other knob combinations have no CTA tile loop, "
            "so grid = #SMs would leave most output tiles uncomputed."
        )

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
    for name in ("NS", "GROUP_SIZE_M", "NUM_WARPS", "TMA_STORE", "PERSISTENT"):
        if name in values:
            src = re.sub(
                rf"^({name}\s*=\s*)\d+",
                lambda m, v=values[name]: f"{m.group(1)}{v}",
                src,
                flags=re.MULTILINE,
            )
    return src


def knob_kwargs(bm, bn, bk, ns, gsm, nw, tma_store=0, persistent=0) -> dict:
    """Map UI knobs to the constant names used in the source files.

    PERSISTENT only appears in the launcher (it's a grid choice, not a
    kernel constexpr); substitute_kernel_constexprs simply finds no match
    in kernel.cu and leaves it untouched.
    """
    return {"BM": bm, "BN": bn, "BK": bk, "NS": ns, "GROUP_SIZE_M": gsm,
            "NUM_WARPS": nw, "TMA_STORE": tma_store, "PERSISTENT": persistent}


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


# Shared building-block fragments stitched into each kernel at a marker.
# (The epilogue and the MMA-issue chain live in one place and are spliced
# into every tier — the start of the composable-fragments architecture.)
FRAGMENTS = {
    "// @@EPILOGUE@@":  "_epilogue.cu.frag",
    "// @@MMA_CHAIN@@": "_mma_chain.cu.frag",
}


def _splice_fragments(src: str) -> str:
    """Replace each building-block marker line with its fragment."""
    out = []
    for line in src.splitlines(keepends=True):
        marker = line.strip()
        if marker in FRAGMENTS:
            frag = (KERNELS_DIR / FRAGMENTS[marker]).read_text()
            out.append(frag if frag.endswith("\n") else frag + "\n")
        else:
            out.append(line)
    return "".join(out)


def render_kernel(tier: dict, bm, bn, bk, ns, gsm, nw, tma_store=0) -> str:
    """Return the substituted, fragment-stitched kernel.cu for a tier."""
    src = (KERNELS_DIR / tier["dir"] / "kernel.cu").read_text()
    src = _splice_fragments(src)
    return substitute_kernel_constexprs(src, **knob_kwargs(bm, bn, bk, ns, gsm, nw, tma_store))


def render_host(tier: dict, bm, bn, bk, ns, gsm, nw, tma_store=0, persistent=0) -> str:
    """Return a *self-contained* host script: runtime preamble + launcher.

    The result has no ``cuda_utils`` import and no ``sys.path`` hack — it
    runs anywhere with torch + cuda-python + nvcc on PATH.
    """
    runtime  = _strip_module_docstring(RUNTIME_PY.read_text())
    launcher = (KERNELS_DIR / tier["dir"] / "launcher.py").read_text()
    launcher = substitute_launcher_constants(
        launcher, **knob_kwargs(bm, bn, bk, ns, gsm, nw, tma_store, persistent))
    header = (
        '"""Self-contained matmul kernel launcher generated by mmcomposer.\n'
        "\n"
        f"Tier: {tier['label']}\n"
        f"Config: BM={bm} BN={bn} BK={bk} NS={ns} GROUP_SIZE_M={gsm} NUM_WARPS={nw} "
        f"TMA_STORE={tma_store} PERSISTENT={persistent}\n"
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
    """(tier_dir, bm, bn, bk, ns, gsm, nw, tma_store, persistent) -> entry."""
    idx = {}
    for e in load_compat().get("entries", []):
        idx[(e["tier"], e["bm"], e["bn"], e["bk"], e["ns"], e["gsm"], e["nw"],
             e.get("tma_store", 0), e.get("persistent", 0))] = e
    return idx


def compat_status(tier_dir, bm, bn, bk, ns, gsm, nw, tma_store=0, persistent=0):
    """Return ('verified'|'failed'|'unknown', entry|None) for a combo.

    'verified'/'failed' come from the empirical B200 sweep; 'unknown'
    means the combo wasn't in the swept grid (fall back to static)."""
    e = _compat_index().get((tier_dir, bm, bn, bk, ns, gsm, nw, tma_store, persistent))
    if e is None:
        return "unknown", None
    return ("verified" if e["correct"] else "failed"), e


def shape_key(M, N, K):
    """Canonical perf-matrix key for a shape: ``'S'`` for a square SxSxS
    (back-compatible with the old int-string keys), else ``'MxNxK'``."""
    return str(M) if (M == N == K) else f"{M}x{N}x{K}"


def compat_perf(tier_dir, bm, bn, bk, ns, gsm, nw, M, N, K, tma_store=0, persistent=0):
    """Return the measured {rel_err, tflops, vs_cublas} for this combo at
    shape (M, N, K), or None if not in the matrix."""
    e = _compat_index().get((tier_dir, bm, bn, bk, ns, gsm, nw, tma_store, persistent))
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
    tier_dir, knobs, tma_store, persistent, ms_ws, two_cta, tflops, shape;
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
    ms_ws, two_cta = toggles_for_dir(best["tier"])
    knobs = {k: best[k] for k in ("tier", "bm", "bn", "bk", "ns", "gsm", "nw", "tma_store")}
    knobs["persistent"] = best.get("persistent", 0)
    return {**knobs, "ms_ws": ms_ws, "two_cta": two_cta,
            "tflops": best_tf, "shape": ref}


def perf_shapes():
    """List of (M, N, K) shapes the compat matrix recorded performance at.
    Legacy square int entries are normalized to (S, S, S)."""
    out = []
    for s in load_compat().get("perf_shapes", []):
        out.append(tuple(s) if isinstance(s, (list, tuple)) else (s, s, s))
    return out
