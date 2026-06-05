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
    )
    dtype = st.selectbox(
        "Data type",
        ["bfloat16", "float16 — coming soon", "fp8 e4m3 — coming soon"],
    )

    st.subheader("Tile shape")
    bm = st.number_input("BM", value=128, min_value=64, max_value=256, step=64)
    bn = st.number_input("BN", value=256, min_value=64, max_value=256, step=64)
    bk = st.number_input("BK", value=64,  min_value=64, max_value=64,  step=64,
                         help="Currently locked at 64 — the tutorial kernels' inner contract.")
    gsm = st.number_input("CTA swizzling factor (GSM)", value=8,
                          min_value=1, max_value=32, step=1,
                          help="Chunk size for the L2-friendly grid walk. "
                               "GSM=1 disables swizzle (natural N-fast walk).")
    nw  = st.number_input("num_warps", value=8, min_value=4, max_value=16, step=4)

    st.subheader("Optimizations")
    ms_ws = st.toggle(
        "Multi-staging + warp specialization",
        value=False,
        help="Deepen the SMEM ring (NS≥3) AND split TMA/MMA into dedicated warps.",
    )
    two_cta = st.toggle(
        "2-CTA cluster MMA",
        value=False,
        help="Use `__cluster_dims__(2,1,1)` + `cta_group::2` MMA.  "
             "Requires multi-stage + warp specialization (above).",
    )

    st.subheader("Problem shapes")
    shapes_text = st.text_area(
        "Target shapes (one M,N,K per line)",
        value="4096,4096,4096\n8192,8192,8192",
        height=80,
    )


# ── Validate the toggle combination ───────────────────────────────────

if two_cta and not ms_ws:
    st.error(
        "**2-CTA cluster MMA** requires **Multi-staging + warp specialization** to be on. "
        "(Cluster MMA only fits in the warp-specialized multi-stage kernel.)"
    )
    st.stop()

tier = TIER_MAP[(ms_ws, two_cta)]


# ── Header summary of selected tier ───────────────────────────────────

st.markdown(f"### {tier['label']}")
st.caption(tier["desc"])
if tier.get("pending"):
    st.warning(tier["pending_note"])


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


KNOBS = {"BM": bm, "BN": bn, "BK": bk, "NS": 2, "GROUP_SIZE_M": gsm, "NUM_WARPS": nw}
kernel_view = substitute_kernel_constexprs(kernel_src, **KNOBS)
main_view   = substitute_main_constants(main_src, **KNOBS)


# ── Validation — warn (don't block) on constraint violations ────────

def validate_config(bm, bn, bk, gsm, nw, chapter):
    """Return a list of human-readable warnings.  Empty list = clean."""
    out = []
    # SWIZZLE_128B inner-box: B's inner dim is 64 BF16 = 128 B (one
    # swizzle atom).  Changing BK breaks make_desc_K_major's LBO math.
    if bk != 64:
        out.append(
            f"**BK = {bk}**: the tutorial's K-major B descriptor + SWIZZLE_128B "
            "TMA constrain BK to 64.  Other values won't compile without "
            "rewriting `make_desc_K_major` and the LBO math."
        )
    # BN must be a multiple of the TMA sub-tile width (64).
    if bn % 64 != 0:
        out.append(f"**BN = {bn}** must be a multiple of 64 (the TMA sub-tile width on K-major B).")
    # Epilogue partitioning: my_row = warp_id*32 + lane, needs BM == NUM_WARPS*32.
    if bm != nw * 32:
        out.append(
            f"**BM = {bm}** but **num_warps = {nw}**: the coalesced epilogue "
            f"assumes BM == num_warps × 32 (= {nw * 32}).  Phase 1 will write "
            "out of range."
        )
    # Phase-2 flat-walk needs BM*BN divisible by THREADS*8.
    if (bm * bn) % (nw * 32 * 8) != 0:
        out.append(
            f"**BM × BN = {bm * bn}** is not a multiple of `THREADS × 8 = "
            f"{nw * 32 * 8}` — the Phase-2 epilogue flat walk leaves "
            "uncovered output positions."
        )
    # SMEM budget (B200 = 228 KB/CTA).
    a_slot = bm * bk * 2
    b_slot = bn * bk * 2
    slot   = a_slot + b_slot
    ns     = 2   # baseline
    epi    = bm * (bn + 8) * 2
    smem   = max(ns * slot, epi) + 1024
    if smem > 228 * 1024:
        out.append(
            f"**SMEM usage {smem/1024:.0f} KB > B200 cap (228 KB)** at this "
            f"(BM={bm}, BN={bn}, BK={bk}, NS={ns}).  Kernel will fail to launch."
        )
    if gsm < 1:
        out.append(f"**GSM = {gsm}** must be ≥ 1.")
    return out


warnings = validate_config(bm, bn, bk, gsm, nw, tier["chapter"])
if warnings:
    with st.expander(f"⚠️  {len(warnings)} configuration warning(s) — click for details",
                      expanded=True):
        for w in warnings:
            st.warning(w)
        st.caption(
            "The substituted code is still displayed below so you can see what "
            "the change would look like, but it likely won't compile.  Fix the "
            "warnings in the sidebar or accept the defaults."
        )


# ── Pre-baked benchmark lookup (stub for MVP) ────────────────────────
#
# A real implementation would index a JSON file by
# (tier, BM, BN, BK, GSM, NW, shape).  For the MVP we ship the b41_w8
# numbers at M=N=K=8192 as a placeholder so the layout is real.
PRE_BAKED = {
    "03b_double_buffer":     {(2048, 2048, 2048): 540,
                              (4096, 4096, 4096): 770,
                              (8192, 8192, 8192): 832},
    "07_coalesced_epilogue": {(8192, 8192, 8192): 1110},
    "09_cta_swizzle":        {(8192, 8192, 8192): 1272},
}
CUBLAS_REF = {(4096, 4096, 4096): 1413, (8192, 8192, 8192): 1461, (16384, 16384, 16384): 1490}


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
        ch_lookup = PRE_BAKED.get(tier["chapter"], {})
        tf = ch_lookup.get((m, n, k))
        cublas = CUBLAS_REF.get((m, n, k))
        tf_str = f"{tf:.0f}" if tf else "—"
        cublas_str = f"{cublas:.0f}" if cublas else "—"
        ratio = f"{tf/cublas:.0%}" if (tf and cublas) else "—"
        rows.append({
            "Shape (M=N=K)": f"{m}^3" if (m == n == k) else f"{m}×{n}×{k}",
            "TFLOPS (pre-baked)": tf_str,
            "cuBLAS TFLOPS": cublas_str,
            "vs cuBLAS": ratio,
        })
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Enter at least one valid `M,N,K` shape in the sidebar.")
    st.caption(
        "Only shapes with cached numbers show real values; others display "
        "`—`.  The cache will grow as the MVP fills out."
    )


# ── Footer ────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "**mmcomposer** &nbsp;·&nbsp; "
    "[📘 Tutorial](https://mmcomposer.readthedocs.io) &nbsp;·&nbsp; "
    "[💻 Source](https://github.com/tongzhou8086/mmcomposer)  &nbsp;·&nbsp; "
    "*See `pages/01 Full Vision` (sidebar) for the original full-codegen UI design.*"
)
