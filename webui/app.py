"""mmcomposer — MVP web UI (kernel selector).

Simplified MVP: instead of generating new code, this page selects among
the existing tutorial chapters (the "ladder") based on the user's
optimization toggles, substitutes their tile parameters, and displays
the kernel + host code.  Pre-baked benchmark numbers stand in for
on-server execution (Streamlit Cloud has no GPU).

The full-vision UI with the LLM-codegen flow lives at
`pages/01_full_vision.py` for reference.

Run locally:
    pip install -r webui/requirements.txt
    streamlit run webui/app.py
"""

from __future__ import annotations

import os
import re
import pathlib

import streamlit as st


# ── Page setup ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="mmcomposer — MVP",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Repo paths ────────────────────────────────────────────────────────

REPO_ROOT     = pathlib.Path(__file__).resolve().parent.parent
TUTORIAL_CODE = REPO_ROOT / "tutorial" / "code"

# Chapter map per (warpspec, two_cta) toggle state.
# Note: the dedicated "baseline" chapter (sync MMA, double buffer, CTA
# swizzle as a knob) is not yet written; we display ch07 as a stand-in
# and flag the pending status to the user.
TIER_MAP = {
    (False, False): {
        "label":  "Tier 1 — Baseline",
        "desc":   "Single-CTA, 2-slot SMEM double buffer, sync MMA (no warp split), "
                  "coalesced epilogue.  CTA-swizzle factor still applies.",
        "chapter": "03b_double_buffer",
        "pending": False,
    },
    (True,  False): {
        "label":  "Tier 2 — Multi-stage + warp specialization",
        "desc":   "Single-CTA, NS-deep SMEM ring, dedicated TMA + MMA warps, "
                  "coalesced epilogue.",
        "chapter": "07_coalesced_epilogue",
        "pending": False,
    },
    (True,  True):  {
        "label":  "Tier 3 — + 2-CTA cluster MMA + CTA swizzle",
        "desc":   "2-CTA cluster with `cta_group::2`, half-B per CTA, NS up to 7, "
                  "chunked-walk CTA swizzle for L2 reuse.",
        "chapter": "09_cta_swizzle",
        "pending": False,
    },
    (False, True): None,   # invalid: 2-CTA without warp-spec isn't a tutorial point
}


# ── Header ────────────────────────────────────────────────────────────

st.title("mmcomposer")
st.markdown(
    "*MVP: pick optimization toggles and tile parameters; we hand you a "
    "kernel from the ladder plus the host code to launch it.*"
)
st.divider()


# ── Sidebar: minimal controls ────────────────────────────────────────

with st.sidebar:
    st.header("Kernel configuration")

    st.subheader("Target")
    gpu = st.selectbox(
        "GPU",
        ["B200 (sm_100a)", "H100 (sm_90a) — coming soon", "RTX 50xx (sm_120) — coming soon"],
        help="The first supported target is NVIDIA B200.  Other backends will "
             "land as the tutorial expands.",
    )
    dtype = st.selectbox(
        "Data type",
        ["bfloat16", "float16 — coming soon", "fp8 e4m3 — coming soon"],
        help="Input data type for A and B.  C is fp32 accumulator → output dtype.",
    )

    st.subheader("Tile shape")
    bm  = st.selectbox(
        "BM", [128], index=0,
        help="M-dimension tile size per CTA.  Fixed at 128 for the "
             "single-CTA baseline — this is a hardware floor, not a TODO: "
             "tcgen05.mma.kind::f16 has an M-atom of 128 (BM=64 reads past "
             "the A tile → illegal address), and TMEM has 128 lanes (BM=256 "
             "won't fit).  Larger M needs the 2-CTA cluster tier (toggle "
             "below), where each CTA holds 128 of the 256 rows.",
    )
    bn  = st.selectbox(
        "BN", [64, 128, 256, 512], index=2,
        help="N-dimension tile size per CTA.  Each CTA owns BN output cols.  "
             "Must be a multiple of 64 (the K-major B TMA sub-tile width).",
    )
    bk  = st.selectbox(
        "BK", [32, 64, 128], index=1,
        help="K-dimension tile size per stage — one SMEM tile streamed per "
             "K-loop iteration.  Locked at 64 in the tutorial: the K-major B "
             "descriptor uses SWIZZLE_128B, which constrains the inner box "
             "to one 128 B swizzle atom = 64 BF16 elements.",
    )
    ns  = st.selectbox(
        "NS (pipeline stages)", [2, 3, 4, 5, 6, 7], index=0,
        help="Number of SMEM ring slots — how many K-tiles can be in flight "
             "at once.  NS=2 is plain double buffering; NS>2 lets multiple "
             "TMA loads pipeline in front of the MMAs.  Capped by SMEM: "
             "NS × (A_slot + B_slot) ≤ 228 KB per CTA.",
    )
    gsm = st.selectbox(
        "CTA swizzling factor (GSM)", [1, 2, 4, 8, 16, 32], index=3,
        help="Chunked block-id rasterization for L2 reuse on B.  GSM CTAs in "
             "a row share the same B-stripe; consecutive CTAs walk M-fast "
             "inside chunks of GSM × grid_n.  GSM=1 disables swizzle "
             "(natural N-fast walk).  See ch09 for the full L2-cache rationale.",
    )
    nw  = st.selectbox(
        "num_warps", [4, 8, 16], index=0,
        help="Total warps per CTA = threads / 32.  The Phase-1 epilogue "
             "splits warps as a 2D grid: BM/32 row strips × (NW / (BM/32)) "
             "column groups, so every warp does real work (e.g. NW=8 at "
             "BM=128 → 4 row strips × 2 col halves).  NW must be a multiple "
             "of BM/32 (= 4 at BM=128).  NW<4 produces wrong output (a "
             "separate ≥4-warp tcgen05 quirk), so NW=2 is dropped.",
    )

    st.subheader("Optimizations")
    ms_ws = st.toggle(
        "Multi-staging + warp specialization", value=False,
        help="Deepen the SMEM ring (NS ≥ 3) AND split TMA + MMA into "
             "dedicated warps so the producer (TMA) and consumer (MMA) can "
             "run their own K-loops, synchronizing only at the per-slot "
             "mbarriers.  Decouples async producer and consumer.",
    )
    two_cta = st.toggle(
        "2-CTA cluster MMA", value=False,
        help="`__cluster_dims__(2, 1, 1)` + `cta_group::2` MMA.  Two CTAs "
             "cooperate inside a single tcgen05.mma — halves per-CTA B SMEM, "
             "doubles M per MMA, unlocks deeper NS.  Requires multi-stage + "
             "warp specialization (above) — the cluster MMA only fits in the "
             "warp-specialized kernel.",
    )

    st.subheader("Problem shapes")
    shapes_text = st.text_area(
        "Target shapes (one M,N,K per line)",
        value="4096,4096,4096\n8192,8192,8192",
        height=80,
        help="Shapes the generated kernel will be benchmarked at.  Each line "
             "is one (M, N, K) triple in row-major, comma-separated.  Only "
             "shapes with a cached number in the pre-baked benchmark table "
             "display a TFLOPS value — others show `—`.",
    )

    st.divider()
    generate = st.button("🛠  Generate kernel", type="primary", use_container_width=True)


# ── Gate everything on the Generate button ────────────────────────────
#
# Values from the sidebar widgets are "staged"; nothing on the right
# side changes until Generate is clicked.  We snapshot the staged
# config into st.session_state so subsequent reruns (re-clicks,
# expander toggles, tab switches) display the same code.

if generate:
    st.session_state.applied = dict(
        bm=bm, bn=bn, bk=bk, ns=ns, gsm=gsm, nw=nw,
        ms_ws=ms_ws, two_cta=two_cta,
        shapes_text=shapes_text,
    )

if "applied" not in st.session_state:
    st.info(
        "Configure parameters in the sidebar, then click **🛠  Generate kernel**.  "
        "The kernel + host code + pre-baked benchmark will appear here."
    )
    st.stop()

cfg = st.session_state.applied
bm, bn, bk          = cfg["bm"], cfg["bn"], cfg["bk"]
ns, gsm, nw         = cfg["ns"], cfg["gsm"], cfg["nw"]
ms_ws, two_cta      = cfg["ms_ws"], cfg["two_cta"]
shapes_text         = cfg["shapes_text"]


# ── Validate the toggle combination ───────────────────────────────────

if two_cta and not ms_ws:
    st.error(
        "**2-CTA cluster MMA** requires **Multi-staging + warp specialization** "
        "to be on.  (Cluster MMA only fits in the warp-specialized multi-stage kernel.)"
    )
    st.stop()

tier = TIER_MAP[(ms_ws, two_cta)]


# ── Header summary of selected tier ───────────────────────────────────

st.markdown(f"### {tier['label']}")
st.caption(tier["desc"])
if tier.get("pending"):
    st.warning(tier["pending_note"])


# ── Validation — run at Generate-time, before showing the code ──────

def validate_config(bm, bn, bk, ns, gsm, nw, chapter):
    """Return a list of human-readable warnings.  Empty list = clean."""
    out = []
    if ns < 2:
        out.append(
            f"**NS = {ns}** must be ≥ 2.  NS=2 is the minimum for the "
            "double-buffer overlap; NS=1 would serialize TMA and MMA on "
            "the same slot."
        )
    # SWIZZLE_128B forces the TMA inner-box dim to be exactly one
    # 128 B swizzle atom = 64 BF16 elements.  BK is the inner dim on
    # A's box ([BK, BM]) and on K-major B's box ([BK, ...]), so BK
    # must be 64.  Other values silently load the wrong SMEM layout.
    if bk != 64:
        out.append(
            f"**BK = {bk}**: TMA with `SWIZZLE_128B` requires the inner-box "
            "dim to be exactly 128 bytes = 64 BF16 elements.  BK is the "
            "inner dim on both A's and B's TMA descriptors in the tutorial "
            "kernels, so BK is locked at 64.  Other values either won't "
            "compile or will silently load garbage SMEM."
        )
    # BN must be a multiple of the TMA sub-tile width (64).
    if bn % 64 != 0:
        out.append(
            f"**BN = {bn}** must be a multiple of 64 (the TMA sub-tile width on K-major B)."
        )
    # Phase 1 epilogue: 2D (row_warp × col_warp) grid.  row_warp =
    # warp_id % (BM/32), col_warp = warp_id / (BM/32).  Requires:
    #   BM % 32 == 0                 (32-row TMEM strips divide evenly)
    #   NW % (BM/32) == 0            (warps split into whole col groups)
    #   BN % (NW / (BM/32)) == 0     (columns divide into col groups)
    #   (BN per col group) % 8 == 0  (8-col tcgen05.ld atoms)
    if bm % 32 != 0:
        out.append(
            f"**BM = {bm}** is not a multiple of 32 — the Phase-1 epilogue "
            "reads TMEM in 32-row strips (tcgen05.ld.32x32b), so BM must "
            "divide evenly into them."
        )
    else:
        row_strips = bm // 32
        if nw % row_strips != 0:
            out.append(
                f"**num_warps = {nw}** must be a multiple of BM/32 = {row_strips} "
                "— the Phase-1 epilogue splits warps into (BM/32) row strips × "
                "(NW / (BM/32)) column groups, so NW must divide into whole "
                "column groups."
            )
        else:
            col_groups = nw // row_strips
            if bn % col_groups != 0 or (bn // col_groups) % 8 != 0:
                out.append(
                    f"**BN = {bn}** doesn't divide into {col_groups} column "
                    f"groups of a multiple of 8 (got {bn // col_groups} cols/group). "
                    "Each Phase-1 column group is read in 8-col tcgen05.ld atoms."
                )
    # BM=128 is a hardware floor for the single-CTA kernel, not a TODO:
    # the tcgen05.mma.kind::f16 M-atom is 128 (BM<128 reads past the A
    # tile → illegal address) and TMEM is 128 lanes (BM>128 won't fit).
    if chapter == "03b_double_buffer" and bm != 128:
        out.append(
            f"**BM = {bm}**: the single-CTA baseline only supports BM=128.  "
            "tcgen05.mma.kind::f16 computes M=128 per instruction (BM<128 "
            "reads past the A tile → illegal address), and TMEM holds 128 "
            "lanes (BM>128 won't fit single-CTA).  Larger M requires the "
            "2-CTA cluster tier."
        )
    # Phase-2 flat-walk needs BM*BN divisible by THREADS*8.
    if (bm * bn) % (nw * 32 * 8) != 0:
        out.append(
            f"**BM × BN = {bm * bn}** is not a multiple of `THREADS × 8 = "
            f"{nw * 32 * 8}` — Phase-2 epilogue flat walk leaves uncovered "
            "output positions."
        )
    # SMEM budget (B200 = 228 KB/CTA).  K-loop ring (NS × slot) and
    # epilogue staging share the same dynamic SMEM region but are
    # time-disjoint, so the launcher sizes for max of the two.
    a_slot   = bm * bk * 2
    b_slot   = bn * bk * 2
    slot     = a_slot + b_slot
    epi      = bm * (bn + 8) * 2
    smem     = max(ns * slot, epi) + 1024
    if smem > 228 * 1024:
        out.append(
            f"**SMEM usage {smem/1024:.0f} KB > B200 cap (228 KB)** at "
            f"(BM={bm}, BN={bn}, BK={bk}, NS={ns}): "
            f"K-loop ring = {ns} × {slot//1024} KB = {ns*slot//1024} KB, "
            f"epilogue staging = {epi//1024} KB.  Kernel will fail to launch."
        )
    return out


config_warnings = validate_config(bm, bn, bk, ns, gsm, nw, tier["chapter"])
if config_warnings:
    st.error(
        f"⚠️  **{len(config_warnings)} configuration warning(s)** — the substituted "
        "code below likely won't compile.  Fix in the sidebar and re-generate, "
        "or accept the defaults."
    )
    for w in config_warnings:
        st.warning(w)
else:
    st.success("✓ Configuration passes all constraint checks for the selected tier.")


# ── Load chapter files ────────────────────────────────────────────────

chapter_dir = TUTORIAL_CODE / tier["chapter"]
kernel_path = chapter_dir / "kernel.cu"
main_path   = chapter_dir / "main.py"


def read_or_none(p: pathlib.Path) -> str | None:
    try:
        return p.read_text()
    except FileNotFoundError:
        return None


kernel_src = read_or_none(kernel_path)
main_src   = read_or_none(main_path)


# ── Substitute tile params (best-effort) into the displayed sources ──

def substitute_kernel_constexprs(src: str | None, **values) -> str | None:
    """Rewrite top-of-file `constexpr int NAME = ...;` lines for each kwarg.

    The kernel files use these constants as the single source of
    truth — all other tile-dependent sizing is derived from them.
    The substitution edits the displayed/downloadable source; the
    user recompiles locally with `nvcc` to get a binary at their
    chosen knobs.
    """
    if src is None:
        return None
    for name, val in values.items():
        src = re.sub(
            rf"(constexpr\s+int\s+{re.escape(name)}\s*=\s*)\d+",
            lambda m, v=val: f"{m.group(1)}{v}",
            src,
        )
    return src


def substitute_main_constants(src: str | None, **values) -> str | None:
    """Rewrite the matching `NAME = ...` constants in main.py."""
    if src is None:
        return None
    # Handle the combined `BM, BN, BK = a, b, c` line specially.
    if all(k in values for k in ("BM", "BN", "BK")):
        src = re.sub(
            r"BM,\s*BN,\s*BK\s*=\s*\d+,\s*\d+,\s*\d+",
            f"BM, BN, BK = {values['BM']}, {values['BN']}, {values['BK']}",
            src,
        )
    # Single-name constants on their own line.
    for name in ("NS", "GROUP_SIZE_M", "NUM_WARPS"):
        if name in values:
            src = re.sub(
                rf"^({name}\s*=\s*)\d+",
                lambda m, v=values[name]: f"{m.group(1)}{v}",
                src,
                flags=re.MULTILINE,
            )
    return src


KNOBS = {"BM": bm, "BN": bn, "BK": bk, "NS": ns, "GROUP_SIZE_M": gsm, "NUM_WARPS": nw}
kernel_view = substitute_kernel_constexprs(kernel_src, **KNOBS)
main_view   = substitute_main_constants(main_src, **KNOBS)


# ── Pre-baked benchmark lookup (stub for MVP) ────────────────────────
#
# A real implementation would index a JSON file by
# (tier, BM, BN, BK, GSM, NW, shape).  For the MVP we ship the b41_w8
# numbers at M=N=K=8192 as a placeholder so the layout is real.
# Indexed by (chapter, NS) → {(M, N, K): TFLOPS}.
# Only valid when the other knobs (BM, BN, BK, GSM, NUM_WARPS) are at
# the chapter default — otherwise we don't have a number and show "—".
PRE_BAKED = {
    ("03b_double_buffer", 2): {(2048, 2048, 2048): 541,
                               (4096, 4096, 4096): 758,
                               (8192, 8192, 8192): 830},
    ("03b_double_buffer", 3): {(2048, 2048, 2048): 621,
                               (4096, 4096, 4096): 819,
                               (8192, 8192, 8192): 915},
    ("03b_double_buffer", 4): {(2048, 2048, 2048): 621,
                               (4096, 4096, 4096): 825,
                               (8192, 8192, 8192): 930},
    ("07_coalesced_epilogue", 2): {(8192, 8192, 8192): 1110},
    ("09_cta_swizzle", 5):        {(8192, 8192, 8192): 1272},
}
CUBLAS_REF = {(4096, 4096, 4096): 1413, (8192, 8192, 8192): 1461, (16384, 16384, 16384): 1490}

# Each chapter has a "default-knob" set; if the user deviates from any
# of these (other than NS, which keys the pre-baked table), no cached
# number applies and the bench tab shows "—".
DEFAULT_NON_NS_KNOBS = {
    "03b_double_buffer":     {"BM": 128, "BN": 256, "BK": 64, "GROUP_SIZE_M": 8,  "NUM_WARPS": 4},
    "07_coalesced_epilogue": {"BM": 128, "BN": 256, "BK": 64, "GROUP_SIZE_M": 1,  "NUM_WARPS": 4},
    "09_cta_swizzle":        {"BM": 128, "BN": 256, "BK": 64, "GROUP_SIZE_M": 8,  "NUM_WARPS": 4},
}


def lookup_tflops(chapter, *, ns, shape, bm, bn, bk, gsm, nw):
    """Return the pre-baked TFLOPS, or None if the config isn't cached."""
    defaults = DEFAULT_NON_NS_KNOBS.get(chapter, {})
    user_non_ns = {"BM": bm, "BN": bn, "BK": bk, "GROUP_SIZE_M": gsm, "NUM_WARPS": nw}
    if user_non_ns != defaults:
        return None
    return PRE_BAKED.get((chapter, ns), {}).get(shape)


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


shapes = parse_shapes(shapes_text)


# ── Right side: tabs ─────────────────────────────────────────────────

tab_kernel, tab_host, tab_bench = st.tabs(["Kernel code", "Host code", "Benchmark (pre-baked)"])

with tab_kernel:
    if kernel_view is None:
        st.error(f"`{kernel_path}` not found.")
    else:
        st.caption(f"Source: `tutorial/code/{tier['chapter']}/kernel.cu`  ·  "
                   f"BM={bm} BN={bn} BK={bk} substituted into the top-of-file constants.")
        st.code(kernel_view, language="cpp", line_numbers=True)
        st.download_button(
            "⬇ Download kernel.cu",
            data=(kernel_view or ""),
            file_name=f"mm_b200_{tier['chapter']}_bm{bm}_bn{bn}_bk{bk}.cu",
            mime="text/x-c",
        )

with tab_host:
    if main_view is None:
        st.error(f"`{main_path}` not found.")
    else:
        st.caption(f"Source: `tutorial/code/{tier['chapter']}/main.py`  ·  "
                   "this is the Python launcher (TMA descriptor encoding + cuLaunchKernel).")
        st.code(main_view, language="python", line_numbers=True)
        st.download_button(
            "⬇ Download main.py",
            data=(main_view or ""),
            file_name=f"main_{tier['chapter']}.py",
            mime="text/x-python",
        )

with tab_bench:
    st.caption(
        "Benchmark numbers are **pre-baked** (Streamlit Cloud has no GPU).  "
        "Download the code, run `python main.py` on a B200 to reproduce locally."
    )
    rows = []
    for (m, n, k) in shapes:
        tf = lookup_tflops(tier["chapter"],
                           ns=ns, shape=(m, n, k),
                           bm=bm, bn=bn, bk=bk, gsm=gsm, nw=nw)
        cublas = CUBLAS_REF.get((m, n, k))
        tf_str = f"{tf:.0f}" if tf else "—"
        cublas_str = f"{cublas:.0f}" if cublas else "—"
        ratio = f"{tf/cublas:.0%}" if (tf and cublas) else "—"
        rows.append({
            "Shape": f"{m}³" if (m == n == k) else f"{m}×{n}×{k}",
            "TFLOPS (pre-baked)": tf_str,
            "cuBLAS TFLOPS": cublas_str,
            "vs cuBLAS": ratio,
        })
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Enter at least one valid `M,N,K` shape in the sidebar.")
    st.caption(
        f"Pre-baked numbers only apply when **non-NS knobs are at the chapter "
        f"default**.  This chapter's defaults: "
        + ", ".join(f"`{k}={v}`" for k, v in DEFAULT_NON_NS_KNOBS.get(tier['chapter'], {}).items())
        + f".  NS is keyed separately (you picked NS={ns}); any other knob "
        f"deviating from the default shows `—`."
    )


# ── Footer ────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "**mmcomposer** &nbsp;·&nbsp; "
    "[📘 Tutorial](https://mmcomposer.readthedocs.io) &nbsp;·&nbsp; "
    "[💻 Source](https://github.com/tongzhou8086/mmcomposer)  &nbsp;·&nbsp; "
    "*See `pages/01 Full Vision` (sidebar) for the original full-codegen UI design.*"
)
