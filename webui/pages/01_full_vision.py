"""mmcomposer — pitch-demo web UI.

A Streamlit front-end that lets the user configure a matmul kernel via
hyperparameter dropdowns, see the resulting JSON-intent file, and then
trigger a (mocked) agent-loop run that shows the step-by-step
optimization progression.

The agent backend is not yet implemented — this UI presents what the
finished product will look like, using real data from the b1 → b41_w8
kernel-development journey as the mocked agent output.

Run locally:
    pip install -r webui/requirements.txt
    streamlit run webui/app.py

Deploy to Streamlit Community Cloud:
    Push to GitHub, sign in to streamlit.io with the same account,
    click "New app", point at this repo + webui/app.py.
"""

import json
import time
from textwrap import dedent

import streamlit as st


# ── Page setup ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="mmcomposer",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Header ───────────────────────────────────────────────────────────────────

st.title("mmcomposer")
st.markdown(
    "*A code generator for matrix-multiplication kernels on modern GPUs. "
    "Tell us your constraints — we compose the kernel, optimization by "
    "optimization.*"
)
st.divider()


# ── Sidebar: configuration ───────────────────────────────────────────────────

with st.sidebar:
    st.header("Kernel configuration")

    # ── Target ────────────────────────────────────────────────────────
    st.subheader("Target")
    gpu = st.selectbox(
        "GPU",
        ["B200 (sm_100a)", "H100 (sm_90a) — coming soon", "RTX 50xx (sm_120) — coming soon"],
        help="The first supported target is NVIDIA B200.  More backends will land as the tutorial expands.",
    )
    dtype = st.selectbox(
        "Data type",
        ["bfloat16", "float16 — coming soon", "fp8 e4m3 — coming soon"],
        help="Input data type for A and B.  C is fp32 accumulator → output dtype.",
    )

    # ── Operand layout ───────────────────────────────────────────────
    st.subheader("Operand layout")
    st.caption(
        "Layout of the input matrices A (M × K) and B.  A is always stored "
        "row-major, dense.  B's transpose and density are configurable."
    )
    b_transposed = st.selectbox(
        "B transposed?",
        ["No", "Yes"],
        help="`No`  → B is stored as (K, N) row-major — PyTorch's natural layout for `A @ B`.  "
             "`Yes` → B is stored as (N, K) row-major (i.e. transposed).  "
             "Currently no plan to support A transposed.",
    )
    a_format = st.selectbox(
        "A's format",
        ["Dense", "Blocksparse"],
        help="`Dense`        → A is a fully populated row-major matrix.  "
             "`Blocksparse`  → A has a regular block-sparse pattern (e.g. 2:4, 4:8 structured sparsity).  "
             "Currently no plan to support a blocksparse B.",
    )

    # ── Tile shape ───────────────────────────────────────────────────
    st.subheader("Tile shape  (BM × BN × BK)")
    st.caption(
        "Per-CTA tile dimensions.  Use `auto` to let mmcomposer pick a "
        "candidate set and autotune, or specify a single value, or a "
        "comma-separated list to autotune from."
    )
    bm = st.text_input("BM", value="128",   help="M-dimension tile size per CTA.  Common: 64, 128.")
    bn = st.text_input("BN", value="256",   help="N-dimension tile size per CTA.  Common: 64, 128, 256.")
    bk = st.text_input("BK", value="64",    help="K-dimension tile size per stage.  Common: 32, 64, 128.")

    # ── Data movement ─────────────────────────────────────────────────
    st.subheader("Data movement")
    tile_fetch = st.selectbox(
        "Tile fetch method",
        ["auto", "TMA 2D", "TMA 3D", "TMA 1D", "cp.async"],
        index=1,
        help="How A and B tiles are moved from HBM into SMEM.",
    )
    num_stages = st.text_input(
        "Pipeline stages (NS)", value="auto",
        help="Number of in-flight tile loads.  `auto` autotunes; constrained by per-CTA SMEM budget.",
    )

    # ── Tensor cores ──────────────────────────────────────────────────
    st.subheader("Tensor cores")
    mma_method = st.selectbox(
        "MMA method",
        ["auto", "tcgen05.mma", "mma.sync"],
        index=1,
        help="`tcgen05.mma` is Blackwell-native async via TMEM.  "
             "`mma.sync` is the legacy sync path; still supported on B200.",
    )

    # ── Optimizations ─────────────────────────────────────────────────
    st.subheader("Optimizations")
    st.caption("Each toggle adds one step to the agent's optimization ladder.")

    smem_multistage = st.toggle(
        "SMEM multi-stage buffering",
        value=True,
        help="Ring-buffer N tiles in SMEM to hide TMA latency behind MMA work.",
    )
    two_cta = st.toggle(
        "2-CTA cluster MMA",
        value=True,
        help="`__cluster_dims__(2,1,1)` + `cta_group::2` MMA — halves per-CTA B SMEM, "
             "enables deeper pipelining.",
    )
    warp_spec = st.toggle(
        "Warp specialization",
        value=True,
        help="Dedicated TMA + MMA warps.  Decouples async producer and consumer.",
    )
    cta_swizzle = st.toggle(
        "CTA swizzling",
        value=True,
        help="Chunked block-id rasterization for L2 reuse.  Helps at large shapes.",
    )
    epilogue_8w = st.toggle(
        "8-warp epilogue",
        value=True,
        help="Use 8 warps in the TMEM → SMEM → GMEM epilogue (vs default 4).  "
             "Bigger phase-2 GMEM throughput.",
    )

    # ── Problem shapes ────────────────────────────────────────────────
    st.subheader("Problem shapes")
    shapes_text = st.text_area(
        "Target shapes (one M,N,K per line)",
        value="4096,4096,4096\n8192,8192,8192\n16384,16384,16384",
        height=110,
        help="Shapes the generated kernel will be tuned for.",
    )

    # ── Epilogue fusion ───────────────────────────────────────────────
    st.subheader("Epilogue fusion")
    st.caption(
        "Optional elementwise function applied to the matmul result "
        "*before* writing back to global memory.  Must be a single "
        "expression in `x` (the per-element FP32 accumulator value)."
    )
    epilogue_fusion = st.text_area(
        "Function of x",
        value="",
        height=70,
        placeholder="e.g.  max(0, x)         — ReLU\n"
                    "      x / (1 + exp(-x)) — SiLU / Swish",
        help="Leave empty for no fusion.  Must be a valid expression in `x` "
             "using only the supported functions listed below.",
    )
    with st.expander("Supported functions (epilogue fusion)"):
        st.markdown(
            "- **Arithmetic:** `+`, `-`, `*`, `/`, `**`  (power)\n"
            "- **Comparison:** `max(a, b)`, `min(a, b)`\n"
            "- **Unary:** `abs(x)`, `exp(x)`, `log(x)`, `sqrt(x)`, `tanh(x)`\n"
            "- **Constants:** integer / float literals, `pi`, `e`\n\n"
            "Examples:\n"
            "- `max(0, x)`           — ReLU\n"
            "- `min(6, max(0, x))`   — ReLU6\n"
            "- `x / (1 + exp(-x))`   — SiLU / Swish\n"
            "- `1 / (1 + exp(-x))`   — sigmoid\n"
            "- `tanh(x)`             — tanh\n"
            "- `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))`\n"
            "                        — GELU (approximate)\n\n"
            "*This set will grow as the generator matures.  More complex "
            "patterns (matmul + bias-add + activation, scaled outputs, "
            "etc.) are out-of-scope for now.*"
        )


# ── Intent JSON: live preview ────────────────────────────────────────────────

def parse_dim(s: str):
    """Parse a tile-dim input as either `auto`, a single int, or a list of ints."""
    s = s.strip().lower()
    if s == "auto":
        return "auto"
    if "," in s:
        return [int(x.strip()) for x in s.split(",")]
    try:
        return int(s)
    except ValueError:
        return s


def parse_shapes(text: str):
    out = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        try:
            m, n, k = (int(x.strip()) for x in line.split(","))
            out.append({"M": m, "N": n, "K": k})
        except Exception:
            continue
    return out


intent = {
    "target": {
        "gpu":   gpu.split(" — ")[0],
        "dtype": dtype.split(" — ")[0],
    },
    "operand_layout": {
        "a_format":     a_format.lower(),       # "dense" | "blocksparse"
        "b_transposed": (b_transposed == "Yes"),
    },
    "tile":  {"BM": parse_dim(bm), "BN": parse_dim(bn), "BK": parse_dim(bk)},
    "data_movement": {
        "tile_fetch_method": tile_fetch,
        "num_stages":        parse_dim(num_stages),
    },
    "tensor_cores": {
        "mma_method": mma_method,
    },
    "optimizations": {
        "smem_multistage":     smem_multistage,
        "two_cta_cluster_mma": two_cta,
        "warp_specialization": warp_spec,
        "cta_swizzling":       cta_swizzle,
        "epilogue_8_warps":    epilogue_8w,
    },
    "shapes": parse_shapes(shapes_text),
    "epilogue_fusion": (epilogue_fusion.strip() or None),
}


# ── Two-column body: JSON intent + generation pane ──────────────────────────

col_intent, col_run = st.columns([1, 2], gap="large")

with col_intent:
    st.subheader("Intent JSON")
    st.caption("This is what gets handed to the mmcomposer agent loop.")
    st.code(json.dumps(intent, indent=2, default=str), language="json", line_numbers=False)

with col_run:
    st.subheader("Code generation")
    st.caption(
        "The agent loop adds one optimization at a time, compiling and "
        "checking correctness + performance after each step.  Click "
        "**Generate kernel** to walk through the build."
    )

    go = st.button("🛠  Generate kernel", type="primary", use_container_width=True)
    out = st.container()

# ── Mocked agent-loop run ────────────────────────────────────────────────────

# Each entry: (step_label, description, perf_tflops_at_8192).
# Real numbers from the b1 → b41_w8 development journey on B200 BF16, M=N=K=8192.
LADDER = [
    ("00 — Naive",                "Single tile, no SMEM staging, no tensor cores.",                          12),
    ("01 — Blocked tiling",       "Register-blocked accumulation over (BM, BN, BK).",                       180),
    ("02 — SMEM staging",         "Stage A and B tiles into shared memory.",                                350),
    ("03 — cp.async loads",       "Async global → SMEM with cp.async.ca.",                                  520),
    ("04 — TMA 2D",               "Hardware bulk copy with tensor-map descriptors.",                        780),
    ("05 — Multi-stage NS=4",     "Ring-buffer 4 tiles in SMEM to hide TMA latency.",                       890),
    ("06 — tcgen05.mma",          "Async tensor cores via TMEM.",                                          1050),
    ("07 — Warp specialization",  "Dedicated TMA warp + MMA warp.",                                        1110),
    ("08 — TMA 3D",               "Single-issue 3D TMA bulk per stage.",                                   1112),
    ("09 — 2-CTA cluster MMA",    "Cluster + cta_group::2 → half-B per CTA, NS up to 7.",                  1184),
    ("10 — CTA swizzle (GSM=8)",  "Chunked walk for L2 reuse on A.",                                       1272),
    ("11 — 8-warp epilogue",      "Split TMEM→SMEM phase across 8 warps; halve GMEM stores per thread.",   1328),
]

# Demo kernel preview (truncated b41_w8).
KERNEL_PREVIEW = dedent(r"""
    // Generated by mmcomposer for the configuration above.
    // Target: B200 (sm_100a), BF16, BM=128 BN=256 BK=64 NS=7
    // Optimizations: cluster + warp-spec + TMA + tcgen05 + CTA-swizzle + 8-warp epilogue

    template <int BLOCK_N, int BLOCK_K, int NUM_STAGES>
    __device__ __forceinline__ void mm_impl(
        const CUtensorMap* A_tmap,
        const CUtensorMap* B_tmap,
        __nv_bfloat16* C_ptr,
        int M, int N, int K
    ) {
        constexpr int BLOCK_M       = 128;
        constexpr int BLOCK_N_LOCAL = BLOCK_N / 2;   // 2-CTA cluster
        constexpr int MMA_K         = 16;
        constexpr int NUM_WARPS     = 8;

        // ── Triton-style chunked CTA swizzle at cluster-tile granularity ──
        constexpr int GROUP_SIZE_M = 8;
        // ... (cluster-id math omitted) ...

        // ── Warp-spec main loop ──
        if (warp_id == 0 && elect_sync()) {
            // TMA warp — issues bulks, signals via mbarriers
            for (int k = 0; k < num_k_iters; k++) { LOAD_TILE(slot, k); ... }
        } else if (cta_rank == 0 && warp_id == 1 && elect_sync()) {
            // MMA warp — only CTA 0 issues; cta_group::2 result lands in both CTAs' TMEM
            for (int k = 0; k < num_k_iters; k++) {
                mbarrier_wait(tile_ready_mbar[slot], phase);
                tcgen05_fence_after_thread_sync();
                #pragma unroll
                for (int kk = 0; kk < K_MMAS; kk++) {
                    tcgen05_mma_g2(taddr, a_desc[kk], b_desc[kk], idesc, accumulate);
                }
                tcgen05_commit_mcast_g2(mma_done_mbar[slot], cta_mask);
            }
        }

        // ── Epilogue: TMEM → SMEM → GMEM, 8 warps ──
        // (Phase 1 split across row-warps × col-warps, Phase 2 uses all 8 warps.)
        // ...
    }
""").strip()


if go:
    with out:
        st.info("Running agent loop — each step adds one optimization, compiles, checks correctness, benchmarks.")
        progress = st.progress(0.0, text="Initializing...")

        for i, (label, desc, tflops) in enumerate(LADDER):
            time.sleep(0.45)   # demo pause; replace with real agent call later
            progress.progress((i + 1) / len(LADDER), text=f"Step {i + 1}/{len(LADDER)}: {label}")

        progress.empty()
        st.success(
            f"✅ Generated.  Final kernel at **{LADDER[-1][2]} TFLOPS** "
            f"(8192³, BF16) — within ~3% of cuBLAS."
        )

        # ── Output artifacts ─────────────────────────────────────
        tab_kernel, tab_report = st.tabs(["📄 Generated kernel", "📊 Performance report"])

        with tab_kernel:
            st.caption("Preview of the generated CUDA kernel (truncated for display).")
            st.code(KERNEL_PREVIEW, language="cpp")
            st.download_button(
                label="⬇ Download full kernel",
                data="// full kernel would be written here\n",
                file_name="mm_b200_bf16.cu",
                mime="text/x-c",
            )

        with tab_report:
            st.caption("Step-by-step performance lineage (BF16, M=N=K=8192).")
            st.dataframe(
                {
                    "Step":            [f"{i + 1:02d} — {label}" for i, (label, _, _) in enumerate(LADDER)],
                    "Description":     [desc for _, desc, _ in LADDER],
                    "TFLOPS @ 8192³":  [tflops for _, _, tflops in LADDER],
                    "vs cuBLAS":       [f"{tflops/1373:.0%}" for _, _, tflops in LADDER],
                },
                use_container_width=True,
                hide_index=True,
            )


# ── Footer ──────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "**mmcomposer** &nbsp;·&nbsp; "
    "[📘 Tutorial](https://mmcomposer.readthedocs.io) &nbsp;·&nbsp; "
    "[💻 Source](https://github.com/tongzhou8086/mmcomposer)"
)
