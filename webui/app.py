"""mmcomposer — MVP web UI (kernel configurator).

Pick optimization toggles + tile parameters; the app renders a kernel
from its owned codebase (``webui/kernels/``), validates the combination
against the B200's constraints, and hands back the kernel + a
*self-contained* host script you can run with ``python host.py``.

All logic lives in ``mvp_core`` (no Streamlit there) so the UI and the
test suite exercise the same code.  ``tutorial/`` is a reference
implementation only — the MVP renders its own codebase.

The full-vision LLM-codegen UI lives at ``pages/01_full_vision.py``.

Run locally:
    pip install -r webui/requirements.txt
    streamlit run webui/app.py
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# Make `mvp_core` importable whether launched via `streamlit run` (which
# adds the script dir to sys.path) or via AppTest / other harnesses.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mvp_core as mc


# ── Page setup ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="mmcomposer — MVP",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("mmcomposer")
st.markdown(
    "*MVP: pick optimization toggles and tile parameters; we render a "
    "kernel from the codebase plus a self-contained host script to launch it.*"
)
st.divider()


# ── Sidebar: controls ─────────────────────────────────────────────────

with st.sidebar:
    st.header("Kernel configuration")

    st.subheader("Target")
    gpu = st.selectbox("GPU", mc.GPU_OPTS,
                       help="First supported target is NVIDIA B200.  Others land as the codebase grows.")
    dtype = st.selectbox("Data type", mc.DTYPE_OPTS,
                         help="Input dtype for A and B.  C is fp32 accumulator → output dtype.")

    st.subheader("Tile shape")
    bm = st.selectbox("BM", mc.BM_OPTS, index=0,
                      help="M tile per CTA.  Locked at 128: tcgen05.mma.kind::f16 M-atom is 128 and "
                           "TMEM holds 128 lanes.  Larger M is served by the 2-CTA cluster tier.")
    bn = st.selectbox("BN", mc.BN_OPTS, index=2,
                      help="N tile per CTA.  Multiple of 64 (K-major B TMA sub-tile).  Caps at 256: "
                           "the tcgen05.mma N-atom max is 256, and the cluster splits M, not N.")
    bk = st.selectbox("BK", mc.BK_OPTS, index=1,
                      help="K tile per stage.  Locked at 64: the K-major B descriptor uses "
                           "SWIZZLE_128B → inner box = one 128 B atom = 64 BF16.")
    ns = st.selectbox("NS (pipeline stages)", mc.NS_OPTS, index=0,
                      help="SMEM ring slots — how many K-tiles in flight.  NS=2 is double buffering; "
                           "capped by SMEM: NS × slot ≤ 228 KB/CTA.")
    gsm = st.selectbox("CTA swizzle factor (GROUP_SIZE_M)", mc.GSM_OPTS, index=3,
                       help="Chunked block-id walk for L2 reuse on B.  GSM=1 disables swizzle.  "
                            "A universal tunable — every tier supports it.")
    nw = st.selectbox("num_warps", mc.NW_OPTS, index=0,
                      help="Warps per CTA.  The epilogue splits warps as a 2D grid: BM/32 row strips × "
                           "NW/(BM/32) column groups, so NW must be a multiple of BM/32 (= 4 at BM=128).")

    st.subheader("Optimizations")
    ms_ws = st.toggle("Multi-staging + warp specialization", value=False,
                      help="Dedicated TMA + MMA warps (async producer/consumer) on top of the "
                           "multi-stage ring.  Off = synchronous-MMA baseline (Tier 1).")
    two_cta = st.toggle("2-CTA cluster MMA", value=False,
                        help="`__cluster_dims__(2,1,1)` + `cta_group::2`: two CTAs cooperate in one "
                             "tcgen05.mma (half-B per CTA, doubles M per MMA).  Requires warp "
                             "specialization (toggle above).")

    st.subheader("Problem shapes")
    shapes_text = st.text_area(
        "Target shapes (one M,N,K per line)",
        value="4096,4096,4096\n8192,8192,8192", height=80,
        help="Shapes to benchmark.  Only shapes with a cached number show TFLOPS; others show —.")

    st.divider()
    generate = st.button("🛠  Generate kernel", type="primary", width="stretch")


# ── Gate on Generate; snapshot config into session_state ──────────────

if generate:
    st.session_state.applied = dict(bm=bm, bn=bn, bk=bk, ns=ns, gsm=gsm, nw=nw,
                                    ms_ws=ms_ws, two_cta=two_cta, shapes_text=shapes_text)

if "applied" not in st.session_state:
    st.info("Configure parameters in the sidebar, then click **🛠  Generate kernel**.")
    st.stop()

cfg = st.session_state.applied
bm, bn, bk = cfg["bm"], cfg["bn"], cfg["bk"]
ns, gsm, nw = cfg["ns"], cfg["gsm"], cfg["nw"]
ms_ws, two_cta = cfg["ms_ws"], cfg["two_cta"]
shapes_text = cfg["shapes_text"]


# ── Resolve tier ──────────────────────────────────────────────────────

tier = mc.tier_for(ms_ws, two_cta)
if tier is None:
    st.error("**2-CTA cluster MMA** requires **Multi-staging + warp specialization** to be on.  "
             "(Cluster MMA only fits in the warp-specialized kernel.)")
    st.stop()

st.markdown(f"### {tier['label']}")
st.caption(tier["desc"])


# ── Validate (static checker) ─────────────────────────────────────────

warnings = mc.validate_config(bm, bn, bk, ns, gsm, nw, cluster=tier["cluster"])
if warnings:
    st.error(f"⚠️  **{len(warnings)} configuration warning(s)** — this combination won't run.  "
             "Fix in the sidebar and re-generate.")
    for w in warnings:
        st.warning(w)
else:
    st.success("✓ Configuration passes all static constraint checks for the selected tier.")
    # Empirical ground truth from the committed B200 compatibility matrix.
    status, entry = mc.compat_status(tier["dir"], bm, bn, bk, ns, gsm, nw)
    cm = mc.load_compat()
    shape = cm.get("validated_shape")
    if status == "verified":
        st.success(f"✅ Empirically verified on B200 ({cm.get('arch','sm_100a')}) at "
                   f"{shape[0]}³: compiles, runs, correct (rel err {entry['rel_err']:.2%}).")
    elif status == "failed":
        st.error("❌ This combination is in the B200 compatibility matrix as **failing** "
                 "at runtime despite passing static checks — do not use.")
    else:
        st.info("ℹ️ Not in the B200 compatibility matrix (outside the swept grid); "
                "static checks only.")


# ── Render kernel + self-contained host ──────────────────────────────

kernel_src = mc.render_kernel(tier, bm, bn, bk, ns, gsm, nw)
host_src   = mc.render_host(tier, bm, bn, bk, ns, gsm, nw)

tab_kernel, tab_host, tab_bench = st.tabs(["Kernel code", "Host code (self-contained)", "Benchmark (pre-baked)"])

with tab_kernel:
    st.caption(f"`webui/kernels/{tier['dir']}/kernel.cu` · entry `{tier['symbol']}` · "
               f"BM={bm} BN={bn} BK={bk} NS={ns} GROUP_SIZE_M={gsm} NUM_WARPS={nw} substituted.")
    st.code(kernel_src, language="cpp", line_numbers=True)
    st.download_button("⬇ Download kernel.cu", data=kernel_src,
                       file_name=f"mm_b200_{tier['dir']}_bm{bm}_bn{bn}_bk{bk}.cu", mime="text/x-c")

with tab_host:
    st.caption("Self-contained launcher (runtime plumbing inlined): runs with "
               "`python host.py` given torch + cuda-python + nvcc.  Put `kernel.cu` alongside it.")
    st.code(host_src, language="python", line_numbers=True)
    st.download_button("⬇ Download host.py", data=host_src,
                       file_name=f"host_{tier['dir']}.py", mime="text/x-python")

with tab_bench:
    st.caption("Benchmark numbers are **pre-baked** (Streamlit Cloud has no GPU).  Download the "
               "kernel + host and run `python host.py` on a B200 to reproduce.")
    rows = []
    for (m, n, k) in mc.parse_shapes(shapes_text):
        tf = mc.lookup_tflops(tier["dir"], ns=ns, shape=(m, n, k), bm=bm, bn=bn, bk=bk, gsm=gsm, nw=nw)
        cub = mc.CUBLAS_REF.get((m, n, k))
        rows.append({
            "Shape": f"{m}³" if (m == n == k) else f"{m}×{n}×{k}",
            "TFLOPS (pre-baked)": f"{tf:.0f}" if tf else "—",
            "cuBLAS TFLOPS": f"{cub:.0f}" if cub else "—",
            "vs cuBLAS": f"{tf / cub:.0%}" if (tf and cub) else "—",
        })
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("Enter at least one valid `M,N,K` shape in the sidebar.")
    defaults = mc.DEFAULT_NON_NS_KNOBS.get(tier["dir"], {})
    st.caption("Pre-baked numbers apply only at this tier's default non-NS knobs: "
               + ", ".join(f"`{k}={v}`" for k, v in defaults.items())
               + f".  You picked NS={ns}; other deviations show —.")


# ── Footer ────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "**mmcomposer** &nbsp;·&nbsp; "
    "[📘 Tutorial](https://mmcomposer.readthedocs.io) &nbsp;·&nbsp; "
    "[💻 Source](https://github.com/tongzhou8086/mmcomposer)  &nbsp;·&nbsp; "
    "*See `pages/01 Full Vision` for the original full-codegen UI design.*"
)
