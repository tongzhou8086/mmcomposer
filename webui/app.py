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
        "desc":   "Single-CTA, double-buffered SMEM, sync MMA, coalesced epilogue. "
                  "The CTA-swizzle knob still applies.",
        "chapter": "07_coalesced_epilogue",   # placeholder
        "pending": True,
        "pending_note":
            "**Baseline chapter (`03b_double_buffer`) is under development.**  "
            "For now we display **ch07** (warp-spec multi-stage, single CTA) as a "
            "placeholder — close in spirit but not yet the no-warp-spec, NS=2 baseline "
            "this tier will eventually point at.",
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

def substitute_tile_params(src: str | None, *, bm: int, bn: int, bk: int) -> str | None:
    """Patch the top-of-file `constexpr int BM/BN/BK = ...;` lines.

    Best-effort: only the kernel.cu has these as compile-time constants.
    We don't actually recompile here — this is for display fidelity so
    the downloaded code matches what the user requested.
    """
    if src is None:
        return None
    src = re.sub(r"constexpr\s+int\s+BM\s*=\s*\d+", f"constexpr int BM = {bm}", src)
    src = re.sub(r"constexpr\s+int\s+BN\s*=\s*\d+", f"constexpr int BN = {bn}", src)
    src = re.sub(r"constexpr\s+int\s+BK\s*=\s*\d+", f"constexpr int BK = {bk}", src)
    return src


def substitute_main_params(src: str | None, *, bm: int, bn: int, bk: int) -> str | None:
    if src is None:
        return None
    src = re.sub(r"BM,\s*BN,\s*BK\s*=\s*\d+,\s*\d+,\s*\d+",
                 f"BM, BN, BK = {bm}, {bn}, {bk}", src)
    return src


kernel_view = substitute_tile_params(kernel_src, bm=bm, bn=bn, bk=bk)
main_view   = substitute_main_params(main_src,   bm=bm, bn=bn, bk=bk)


# ── Pre-baked benchmark lookup (stub for MVP) ────────────────────────
#
# A real implementation would index a JSON file by
# (tier, BM, BN, BK, GSM, NW, shape).  For the MVP we ship the b41_w8
# numbers at M=N=K=8192 as a placeholder so the layout is real.
PRE_BAKED = {
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
