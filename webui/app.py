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

import json
import os
import sys
import time

import streamlit as st

# Make `mvp_core` importable whether launched via `streamlit run` (which
# adds the script dir to sys.path) or via AppTest / other harnesses.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit.components.v1 as components
import mvp_core as mc
import live_bench


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

# ── Generate button (main area, top — no scrolling) ──
generate = st.button("🛠  Generate kernel", type="primary", width="stretch")
st.caption("Tip: press **Ctrl/Cmd + Enter** to (re)generate — no mouse needed.")


# ── Sidebar: controls ─────────────────────────────────────────────────

# Recommended defaults = the best-measured config at the default shape, pulled
# from the empirical matrix (data-driven, not hardcoded).  Falls back to fixed
# indices if the matrix is unavailable.
_rec = mc.recommended_config() or {}
def _idx(opts, key, fallback):
    v = _rec.get(key)
    return opts.index(v) if v in opts else fallback
def _default_on(key):
    return bool(_rec.get(key))

with st.sidebar:
    st.header("Kernel configuration")

    st.subheader("Target")
    gpu = st.selectbox("GPU", mc.GPU_OPTS,
                       help="First supported target is NVIDIA B200.  Others land as the codebase grows.")
    dtype = st.selectbox("Data type", mc.DTYPE_OPTS,
                         help="Input dtype for A and B.  C is fp32 accumulator → output dtype.")

    st.subheader("Options")
    _rec_note = (f"  Defaults = the best-measured config at {_rec['shape']}³ "
                 f"(~{_rec['tflops']:.0f} TFLOPS)." if _rec else "")
    st.caption("Composable knobs on one kernel — tile sizes *and* on/off toggles.  "
               "No knob is guaranteed to help; the measured TFLOPS tell you what actually does."
               + _rec_note)
    bm = st.selectbox("BM", mc.BM_OPTS, index=_idx(mc.BM_OPTS, "bm", 0),
                      help="M tile per CTA.  Locked at 128: tcgen05.mma.kind::f16 M-atom is 128 and "
                           "TMEM holds 128 lanes.  Larger M is served by turning on 2-CTA cluster MMA.")
    bn = st.selectbox("BN", mc.BN_OPTS, index=_idx(mc.BN_OPTS, "bn", 2),
                      help="N tile per CTA.  Multiple of 64 (K-major B TMA sub-tile).  A single "
                           "tcgen05.mma N atom caps at 256 columns; BN512 uses the guarded "
                           "two-panel path.")
    bk = st.selectbox("BK", mc.BK_OPTS, index=_idx(mc.BK_OPTS, "bk", 0),
                      help="K tile per stage.  Locked at 64: the K-major B descriptor uses "
                           "SWIZZLE_128B → inner box = one 128 B atom = 64 BF16.")
    ns = st.selectbox("NS (pipeline stages)", mc.NS_OPTS, index=_idx(mc.NS_OPTS, "ns", 0),
                      help="SMEM ring slots — how many K-tiles in flight.  NS=2 is double buffering; "
                           "capped by SMEM: NS × slot ≤ 228 KB/CTA.")
    gsm = st.selectbox("CTA swizzle factor (GROUP_SIZE_M)", mc.GSM_OPTS, index=_idx(mc.GSM_OPTS, "gsm", 3),
                       help="Chunked block-id walk for L2 reuse on B.  GSM=1 disables swizzle.  "
                            "A universal tunable — works with any option combination.")
    nw = st.selectbox("num_warps", mc.NW_OPTS, index=_idx(mc.NW_OPTS, "nw", 0),
                      help="Warps per CTA.  The epilogue splits warps as a 2D grid: BM/32 row strips × "
                           "NW/(BM/32) column groups, so NW must be a multiple of BM/32 (= 4 at BM=128).")

    ms_ws = st.toggle(
        "Multi-staging + warp specialization", value=_default_on("ms_ws"),
        help="Dedicated TMA + MMA warps (async producer/consumer) on top of the "
             "multi-stage ring.  Off = synchronous MMA (no producer/consumer split).")
    two_cta = st.toggle(
        "2-CTA cluster MMA", value=_default_on("two_cta"),
        help="`__cluster_dims__(2,1,1)` + `cta_group::2`: two CTAs cooperate in one "
             "tcgen05.mma (half-B per CTA, doubles M per MMA).  Requires "
             "multi-staging + warp specialization (above).")
    persistent = st.toggle(
        "Persistent grid", value=_default_on("persistent"),
        help="Launch one CTA per SM and loop over output tiles inside the "
             "kernel (grid = #SMs) instead of one CTA per tile.  Trims "
             "launch/tail overhead — config-dependent (a clear win on low-K / "
             "many-tile shapes).  Available on the warp-specialized single-CTA "
             "path, and on the 2-CTA cluster path when epilogue overlap is on.")
    ld_width = mc.TCGEN05_LD_WIDTH_OPTS[0]
    overlap = st.toggle(
        "Epilogue overlap (persistent)", value=_default_on("overlap"),
        help="Run each tile's epilogue concurrently with the next tile's K-loop "
             "(TMEM double-buffer).  Launches 2 stream warps (TMA + MMA) on top of "
             "num_warps epilogue warps, so num_warps scales the epilogue.  A win on "
             "epilogue-bound low-K shapes.  Requires Persistent grid on.")
    split_epilogue = st.toggle(
        "Split epilogue writeback", value=_default_on("split_epilogue"),
        help="For Tier 3 overlap, stage/store the epilogue in two half-BN column "
             "passes.  This reduces epilogue SMEM and can allow a deeper K-loop "
             "ring, but adds an extra epilogue pass/barrier.")
    tma_pipelined = st.toggle(
        "Pipelined TMA-store epilogue", value=_default_on("tma_pipelined"),
        help="Alternative overlapped epilogue mode: drain TMEM in 64-column chunks "
             "through compact swizzled SMEM buffers and issue TMA stores.  "
             "Requires Persistent grid + Epilogue overlap, and replaces split/L1 "
             "staged-store modifiers.")
    tma_store_stages = st.selectbox(
        "TMA store stages", mc.TMA_STORE_STAGES_OPTS,
        index=_idx(mc.TMA_STORE_STAGES_OPTS, "tma_store_stages", 1),
        disabled=not tma_pipelined,
        help="Number of compact STORE_N=64 SMEM buffers in the pipelined "
             "TMA-store epilogue.  Shape-dependent: store-only paths often "
             "prefer 1, while fused epilogues can prefer 2; 3/4 are exposed "
             "for controlled experiments.")
    single_tmem = st.toggle(
        "Single-TMEM accumulator sync", value=_default_on("single_tmem"),
        help="Reuse one TMEM accumulator by making the MMA warp wait until the "
             "epilogue warps have safely drained the tile from TMEM.  BN512 "
             "currently requires this as part of its guarded two-panel path; "
             "for smaller BN it is an independent buffering/sync choice.")
    l1_no_alloc = st.toggle(
        "L1 no-allocate C store", value=_default_on("l1_no_alloc"),
        help="Write the C output with `st...L1::no_allocate` so the write-once "
             "result doesn't evict A/B from L1.  A measured win when the epilogue "
             "is exposed (low K), null at high K.")

    st.subheader("Problem shape")
    shapes_text = st.text_area(
        "Target shape (one M,N,K)",
        value="8192,8192,8192", height=68,
        help="The single (M, N, K) you're tuning for.  One shape at a time: different shapes "
             "have different optimal knob configs, so a kernel is composed/verified per shape.  "
             "Extra lines are ignored (with a warning).")


# ── Gate on Generate; snapshot config into session_state ──────────────

if generate:
    applied_tma_store_stages = mc.normalize_tma_store_stages(
        tma_pipelined, tma_store_stages)
    st.session_state.applied = dict(bm=bm, bn=bn, bk=bk, ns=ns, gsm=gsm, nw=nw,
                                    ms_ws=ms_ws, two_cta=two_cta,
                                    persistent=int(persistent), ld_width=int(ld_width),
                                    overlap=int(overlap), split_epilogue=int(split_epilogue),
                                    l1_no_alloc=int(l1_no_alloc),
                                    tma_pipelined=int(tma_pipelined),
                                    tma_store_stages=applied_tma_store_stages,
                                    single_tmem=int(single_tmem), shapes_text=shapes_text)
    st.session_state.run_live = True   # fire the on-the-fly B200 bench (if live mode)

if "applied" not in st.session_state:
    st.info("Configure parameters in the sidebar, then click **🛠  Generate kernel**.")
    st.stop()

cfg = st.session_state.applied
bm, bn, bk = cfg["bm"], cfg["bn"], cfg["bk"]
ns, gsm, nw = cfg["ns"], cfg["gsm"], cfg["nw"]
ms_ws, two_cta = cfg["ms_ws"], cfg["two_cta"]
persistent = cfg.get("persistent", 0)
ld_width = cfg.get("ld_width", 8)
overlap = cfg.get("overlap", 0)
split_epilogue = cfg.get("split_epilogue", 0)
l1_no_alloc = cfg.get("l1_no_alloc", 0)
tma_pipelined = cfg.get("tma_pipelined", 0)
tma_store_stages = mc.normalize_tma_store_stages(
    tma_pipelined, cfg.get("tma_store_stages", 2))
single_tmem = cfg.get("single_tmem", 0)
shapes_text = cfg["shapes_text"]

# One shape at a time: different shapes have different optimal configs.
all_shapes = mc.parse_shapes(shapes_text)
if len(all_shapes) > 1:
    st.warning(
        f"⚠️ Only one target shape is supported at a time — you entered {len(all_shapes)}.  "
        "Different shapes have different optimal knob configs, so mmcomposer composes/verifies a "
        f"kernel per shape.  Using the first (**{all_shapes[0][0]}×{all_shapes[0][1]}×{all_shapes[0][2]}**); "
        "the rest are ignored."
    )
shapes = all_shapes[:1]


# ── Resolve the implementation for the chosen options ─────────────────
# (Internally these map to one of a few kernel shapes, but there's no
# user-facing "tier" ladder — you just compose options freely.)

tier = mc.tier_for(ms_ws, two_cta)
if tier is None:
    st.error("**2-CTA cluster MMA** requires **Multi-staging + warp specialization** to be on.  "
             "(Cluster MMA only fits in the warp-specialized kernel.)")
    st.stop()

# Load the empirical compat matrix once, up front, so it's always defined
# (the Benchmark tab reads `cm` even when the config is invalid).  It's an
# enhancement, so any load issue degrades gracefully rather than crashing.
try:
    cm = mc.load_compat()
except Exception:
    cm = {}


# ── Validate (static checker) ─────────────────────────────────────────

warnings = mc.validate_config(bm, bn, bk, ns, gsm, nw, cluster=tier["cluster"],
                              persistent=persistent,
                              persistent_ok=tier.get("persistent_ok", False),
                              shape=shapes[0] if shapes else None, ld_width=ld_width,
                              overlap=overlap, split_epilogue=split_epilogue,
                              l1_no_alloc=l1_no_alloc, tma_pipelined=tma_pipelined,
                              tma_store_stages=tma_store_stages,
                              single_tmem=single_tmem)
if warnings:
    st.error(f"⚠️  **{len(warnings)} configuration warning(s)** — this combination won't run.  "
             "Fix in the sidebar and re-generate.")
    for w in warnings:
        st.warning(w)
else:
    st.success("✅ Configuration passes all static constraint checks.")
    # Empirical ground truth from the committed B200 compatibility matrix.
    try:
        two_cta_k = int(tier["cluster"])   # recorded compat knob (shared-dir discriminator)
        status, entry = mc.compat_status(tier["dir"], bm, bn, bk, ns, gsm, nw,
                                          persistent=persistent,
                                          ld_width=ld_width, overlap=overlap,
                                          split_epilogue=split_epilogue, two_cta=two_cta_k,
                                          l1_no_alloc=l1_no_alloc, tma_pipelined=tma_pipelined,
                                          tma_store_stages=tma_store_stages,
                                          single_tmem=single_tmem)
        if status == "verified":
            # Prefer perf at the shape the user is tuning; else the largest swept square.
            em, en, ek = shapes[0]
            p = mc.compat_perf(tier["dir"], bm, bn, bk, ns, gsm, nw, em, en, ek,
                               persistent=persistent, ld_width=ld_width,
                               overlap=overlap, split_epilogue=split_epilogue, two_cta=two_cta_k,
                               l1_no_alloc=l1_no_alloc, tma_pipelined=tma_pipelined,
                               tma_store_stages=tma_store_stages,
                               single_tmem=single_tmem)
            ref = (em, en, ek)
            if not (p and p.get("tflops")):
                squares = [t for t in mc.perf_shapes() if t[0] == t[1] == t[2]]
                if squares:
                    ref = max(squares)
                    p = mc.compat_perf(tier["dir"], bm, bn, bk, ns, gsm, nw, *ref,
                                       persistent=persistent, ld_width=ld_width,
                                       overlap=overlap, split_epilogue=split_epilogue, two_cta=two_cta_k,
                                       l1_no_alloc=l1_no_alloc, tma_pipelined=tma_pipelined,
                                       tma_store_stages=tma_store_stages,
                                       single_tmem=single_tmem)
            msg = f"✅ Empirically verified on B200 ({cm.get('arch', 'sm_100a')}): compiles, runs, correct."
            if p and p.get("tflops"):
                lbl = f"{ref[0]}³" if ref[0] == ref[1] == ref[2] else f"{ref[0]}×{ref[1]}×{ref[2]}"
                msg += f"  {p['tflops']:.0f} TFLOPS at {lbl} ({p['vs_cublas']:.0%} of cuBLAS)."
            st.success(msg)
        elif status == "failed":
            st.error("❌ This combination is in the B200 compatibility matrix as **failing** "
                     "at runtime despite passing static checks — do not use.")
        else:
            st.info("ℹ️ Not in the B200 compatibility matrix (outside the swept grid); "
                    "static checks only.")
    except Exception:
        pass  # no compat annotation; static checks already shown above


# ── Render kernel + self-contained host ──────────────────────────────

kernel_src = mc.render_kernel(tier, bm, bn, bk, ns, gsm, nw,
                              ld_width=ld_width, overlap=overlap,
                              split_epilogue=split_epilogue, l1_no_alloc=l1_no_alloc,
                              tma_pipelined=tma_pipelined,
                              tma_store_stages=tma_store_stages,
                              single_tmem=single_tmem)
host_src   = mc.render_host(tier, bm, bn, bk, ns, gsm, nw,
                            persistent=persistent, ld_width=ld_width, overlap=overlap,
                            split_epilogue=split_epilogue, l1_no_alloc=l1_no_alloc,
                            tma_pipelined=tma_pipelined,
                            tma_store_stages=tma_store_stages,
                            single_tmem=single_tmem)

def ssh_copy_button(name, content, label):
    """One-click 'copy the heredoc to clipboard' for SSH use.

    Streamlit Cloud gates every URL behind a browser session, so curl/wget
    can't fetch a link — instead we copy a `cat > file <<'EOF' ... EOF` block
    to the clipboard; paste it into the SSH terminal to recreate the file.
    The clipboard API is usually blocked inside the component iframe, so we
    use document.execCommand('copy') on a hidden textarea (works in-iframe),
    with a visible confirmation so a failure is never silent."""
    body = content if content.endswith("\n") else content + "\n"
    cmd = f"cat > {name} <<'MMCOMPOSER_EOF'\n{body}MMCOMPOSER_EOF\n"
    payload = json.dumps(cmd)   # safe-escape the full source into a JS string
    components.html(f"""
      <style>
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; }}
        /* Mimic Streamlit's secondary button (the Download button's style). */
        .mmc-btn {{ font-family: "Source Sans Pro", sans-serif; font-size: 0.875rem;
                    font-weight: 400; width: 100%; min-height: 2.5rem;
                    padding: 0.25rem 0.75rem; border: 1px solid rgba(49,51,63,0.2);
                    border-radius: 0.5rem; background-color: #ffffff;
                    color: rgb(49,51,63); cursor: pointer; transition: all .1s; }}
        .mmc-btn:hover {{ border-color: #ff4b4b; color: #ff4b4b; }}
        .mmc-btn.ok {{ color: #09ab3b; border-color: #09ab3b; }}
        @media (prefers-color-scheme: dark) {{
          .mmc-btn {{ background-color: rgb(19,23,32); color: rgb(250,250,250);
                      border-color: rgba(250,250,250,0.2); }}
        }}
      </style>
      <button class="mmc-btn" id="b">📋 Copy {label} command for SSH</button>
      <script>
        const cmd = {payload};
        const btn = document.getElementById('b');
        const orig = btn.textContent;
        btn.onclick = () => {{
          const ta = document.createElement('textarea');
          ta.value = cmd; ta.style.position = 'fixed'; ta.style.opacity = '0';
          document.body.appendChild(ta); ta.focus(); ta.select();
          let ok = false; try {{ ok = document.execCommand('copy'); }} catch (e) {{}}
          document.body.removeChild(ta);
          // Feedback lives in the button label itself — no extra line / layout shift.
          btn.textContent = ok ? '✅ Copied — paste into your SSH terminal' : '⚠️ Copy blocked — use the code below';
          btn.classList.toggle('ok', ok);
          setTimeout(() => {{ btn.textContent = orig; btn.classList.remove('ok'); }}, 1800);
        }};
      </script>
    """, height=46)


tab_kernel, tab_host, tab_bench = st.tabs(["Kernel code", "Host code (self-contained)", "Benchmark (measured on B200)"])

with tab_kernel:
    dc, sc = st.columns([1, 2])
    with dc:
        st.download_button("⬇ Download kernel.cu", data=kernel_src, file_name="kernel.cu",
                           mime="text/x-c", width="stretch")
    with sc:
        ssh_copy_button("kernel.cu", kernel_src, "kernel.cu")
    st.code(kernel_src, language="cpp", line_numbers=True)

with tab_host:
    dc, sc = st.columns([1, 2])
    with dc:
        st.download_button("⬇ Download host.py", data=host_src, file_name="host.py",
                           mime="text/x-python", width="stretch")
    with sc:
        ssh_copy_button("host.py", host_src, "host.py")
    st.code(host_src, language="python", line_numbers=True)

with tab_bench:
    # ── On-the-fly B200 benchmark (jump-node / live mode) ────────────
    # Renders this exact config and submits compile+run+cuBLAS to a B200 via
    # srun — real measured numbers for the *entered* shape, not a matrix lookup.
    if live_bench.live_available() and shapes:
        m0, n0, k0 = shapes[0]
        knobs = dict(bm=bm, bn=bn, bk=bk, ns=ns, gsm=gsm, nw=nw,
                     persistent=persistent, ld_width=ld_width,
                     overlap=overlap, split_epilogue=split_epilogue,
                     l1_no_alloc=l1_no_alloc, tma_pipelined=tma_pipelined,
                     tma_store_stages=tma_store_stages,
                     single_tmem=single_tmem)
        sig = (tier["dir"], tuple(sorted(knobs.items())), m0, n0, k0)
        cache = st.session_state.setdefault("live_cache", {})
        clicked = st.button("▶  Benchmark this config on a B200 (live)", type="primary",
                            disabled=bool(warnings),
                            help="Compile + run this kernel + cuBLAS on a real B200 via srun.")
        auto = st.session_state.pop("run_live", False)
        if (clicked or auto) and not warnings:
            with st.spinner(f"Submitting {m0}×{n0}×{k0} to a B200 via srun "
                            "(queue + compile + run + cuBLAS)…"):
                cache[sig] = live_bench.run_live_bench(tier, knobs, m0, n0, k0)
        res = cache.get(sig)
        if res and res.get("ok"):
            st.success(
                f"✅ **Measured live on B200** — {res['tflops']:.0f} TFLOPS at "
                f"{m0}×{n0}×{k0} · **{res['vs_cublas']:.0%} of cuBLAS** "
                f"({res['cublas_tflops']:.0f} TFLOPS) · {res['us']:.1f} µs/call · "
                f"rel err {res['rel_err']:.2%} · grid {res.get('grid')}")
        elif res:
            st.error(f"❌ Live benchmark failed: {res.get('error')}")
            if res.get("stderr"):
                st.code(res["stderr"], language="text")
        elif warnings:
            st.info("Fix the configuration warnings above, then re-generate to benchmark live.")
        else:
            st.info("Click **Benchmark this config on a B200 (live)** to measure this exact shape.")
        st.divider()

        # ── Autotune: live sweep of valid knob combinations, ranked ──────
        # No "tier" in the UI — that's just our internal skeleton split; users
        # see knobs.  Warp-spec-on = the knob combos with warp specialization
        # enabled (the practical production set); full also sweeps the
        # warp-spec-off combos kept for education.
        # The two warp-spec arms share one dir (TWO_CTA distinguishes them); the
        # sweep expands each dir to all its arms, so pass each dir once.
        WS_DIRS  = list(dict.fromkeys(
            t["dir"] for k, t in mc.TIER_MAP.items() if t and k[0]))  # k = (ms_ws, two_cta)
        ALL_DIRS = list(dict.fromkeys(t["dir"] for t in mc.TIER_MAP.values() if t))
        st.markdown("**🔧 Autotune** — sweep valid knob combinations on a B200 for this shape and rank by TFLOPS.")
        scope = st.radio(
            "Sweep scope",
            ["Warp specialization on (production)", "Full sweep (incl. warp-spec off)"],
            horizontal=True, key="autotune_scope",
            help="Production: warp specialization essentially always helps and (assuming N is "
                 "reasonably large) BN=256/512 and NS>=3 are the practical timing subset, so it "
                 "sweeps only those warp-spec-on combos — a much smaller search. Full sweeps everything, including "
                 "the warp-spec-off and BN=64 combos kept for educational comparison.")
        production = scope.startswith("Warp")
        at_dirs = WS_DIRS if production else ALL_DIRS
        at_filters = {"bn": [256, 512],
                      "ns": [x for x in mc.NS_OPTS if x >= 3],
                      "two_cta": [1],
                      "tma_store_stages": [1, 2],
                      "single_tmem_policy": "bn512-only"} if production else {"single_tmem_policy": "all"}
        at_sig = (tuple(at_dirs), json.dumps(at_filters, sort_keys=True), m0, n0, k0)
        at_cache = st.session_state.setdefault("autotune_cache", {})

        def _knob_cols(row):
            # Warp-spec is on for any non-baseline dir; 2-CTA is the recorded
            # two_cta knob (the two warp-spec arms share a dir).
            ws, _ = mc.toggles_for_dir(row["tier"])
            cta = bool(row.get("two_cta", 0))
            return ("On" if ws else "Off", "On" if cta else "Off")

        def _render_leaderboard(at, *, live):
            """Best-line + ranked table.  live=True: 'best so far' + fixed top-10
            (re-rendered each poll); live=False: final result + a top-N slider."""
            b = at["results"][0]
            bws, bcta = _knob_cols(b)
            vs_b = f"{b['vs_cublas']:.0%}" if b.get("vs_cublas") else "—"
            cub = f"{at['cublas_tflops']:.0f}" if at.get("cublas_tflops") else "?"
            head = (f"📈 **Best so far** ({at['n_combos']} done)" if live
                    else f"🏆 **Best of {at['n_combos']} combos**")
            st.markdown(
                f"{head} at {m0}×{n0}×{k0}: **{b['tflops']:.0f} TFLOPS** ({vs_b} of cuBLAS "
                f"{cub}) — Warp-spec={bws} · 2-CTA cluster={bcta} · "
                f"BN={b['bn']} NS={b['ns']} GSM={b['gsm']} NW={b['nw']} "
                f"PERSISTENT={b['persistent']} "
                f"OVERLAP={b.get('overlap', 0)} "
                f"SPLIT={b.get('split_epilogue', 0)} L1NA={b.get('l1_no_alloc', 0)} "
                f"TMA_PIPE={b.get('tma_pipelined', 0)} "
                f"TMA_STAGES={b.get('tma_store_stages', 2)} "
                f"SINGLE_TMEM={b.get('single_tmem', 0)}")
            n_res = len(at["results"])
            if live:
                top_n = min(10, n_res)
            else:
                top_n = st.slider("Show top", min_value=3, max_value=min(50, n_res),
                                  value=min(10, n_res), key="autotune_topn") if n_res > 3 else n_res
            rows = []
            for i, r in enumerate(at["results"][:top_n]):
                ws, cta = _knob_cols(r)
                rows.append({"#": i + 1, "Warp-spec": ws, "2-CTA": cta,
                             "BN": r["bn"], "NS": r["ns"], "GSM": r["gsm"], "NW": r["nw"],
                             "PERS": r["persistent"],
                             "OV": r.get("overlap", 0),
                             "SPLIT": r.get("split_epilogue", 0), "L1NA": r.get("l1_no_alloc", 0),
                             "TMA": r.get("tma_pipelined", 0),
                             "TMS": r.get("tma_store_stages", 2),
                             "STMEM": r.get("single_tmem", 0),
                             "TFLOPS": f"{r['tflops']:.0f}",
                             "vs cuBLAS": f"{r['vs_cublas']:.0%}" if r.get("vs_cublas") else "—"})
            st.dataframe(rows, width="stretch", hide_index=True)

        # Stateful: a background sweep (job) + a polled progress bar + LIVE leaderboard.
        job = st.session_state.get("autotune_job")
        if job is not None:
            done, total, finished = live_bench.autotune_poll(job)
            if finished:
                at_cache[job["sig"]] = live_bench.autotune_collect(job)
                st.session_state.pop("autotune_job", None)
                st.rerun()
            else:
                if total:
                    pct = min(done / total, 0.99)
                    st.progress(pct, text=f"Sweeping on a B200 — {done}/{total} combos ({pct:.0%})")
                else:
                    st.progress(0.0, text="Compiling kernels on a B200 (combo count pending)…")
                if st.button("✖  Cancel sweep", key="autotune_cancel"):
                    try:
                        job["proc"].terminate()
                    except Exception:
                        pass
                    st.session_state.pop("autotune_job", None)
                    st.rerun()
                part = live_bench.autotune_partial(job)   # rank what's streamed in so far
                if part.get("ok"):
                    _render_leaderboard(part, live=True)
                time.sleep(2.0)
                st.rerun()
        elif st.button("🔧  Autotune: sweep combos on a B200", key="autotune_btn"):
            j = live_bench.autotune_start(at_dirs, m0, n0, k0, filters=at_filters)
            j["sig"] = at_sig
            st.session_state.autotune_job = j
            st.rerun()

        at = at_cache.get(at_sig)
        if at and at.get("ok"):
            _render_leaderboard(at, live=False)
            st.caption("Set the sidebar to the winning knobs (and re-generate) to download that kernel.")
        elif at:
            st.error(f"Autotune failed: {at.get('error')}")
            if at.get("stderr"):
                st.code(at["stderr"], language="text")
        elif st.session_state.get("autotune_job") is None:
            st.caption("Autotune submits one srun that compiles + benchmarks the selected valid combo set "
                       "(tens to hundreds of kernels), with a live progress bar + leaderboard. "
                       "Production is the pruned timing sweep; Full is the complete timed search.")
        st.divider()

    try:
        pshapes = mc.perf_shapes()
    except Exception:
        pshapes = []
    swept = ", ".join(f"{s[0]}³" if s[0] == s[1] == s[2] else f"{s[0]}×{s[1]}×{s[2]}"
                      for s in pshapes) if pshapes else "the swept shapes"
    st.caption(f"Numbers are **measured on a real B200** ({cm.get('arch', 'sm_100a')}) for this exact "
               f"config, recorded at {swept} via `do_bench`.  Download the kernel + host to reproduce.")
    rows = []
    for (m, n, k) in shapes:
        square = (m == n == k)
        try:
            p = mc.compat_perf(tier["dir"], bm, bn, bk, ns, gsm, nw, m, n, k,
                               persistent=persistent, ld_width=ld_width,
                               overlap=overlap, split_epilogue=split_epilogue,
                               two_cta=int(tier["cluster"]), l1_no_alloc=l1_no_alloc,
                               tma_pipelined=tma_pipelined,
                               tma_store_stages=tma_store_stages,
                               single_tmem=single_tmem)
            cub = mc.cublas_tflops(m, n, k)
        except Exception:
            p, cub = None, None
        rows.append({
            "Shape": f"{m}³" if square else f"{m}×{n}×{k}",
            "TFLOPS (B200)": f"{p['tflops']:.0f}" if (p and p.get("tflops")) else "—",
            "cuBLAS TFLOPS": f"{cub:.0f}" if cub else "—",
            "vs cuBLAS": f"{p['vs_cublas']:.0%}" if (p and p.get("vs_cublas")) else "—",
        })
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("Enter at least one valid `M,N,K` shape in the sidebar.")
    st.caption(f"Measured TFLOPS exist only for {swept} and matrix-covered knob combos; "
               "other shapes/combos show —.")


# ── Footer ────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "**mmcomposer** &nbsp;·&nbsp; "
    "[📘 Tutorial](https://mmcomposer.readthedocs.io) &nbsp;·&nbsp; "
    "[💻 Source](https://github.com/tongzhou8086/mmcomposer)  &nbsp;·&nbsp; "
    "*See `pages/01 Full Vision` for the original full-codegen UI design.*"
)


# ── Keyboard shortcut: Ctrl/Cmd+Enter clicks Generate ─────────────────
# Injected at the end (not under the button) so its zero-height iframe
# doesn't add a vertical gap to the content.  It attaches a one-time
# keydown listener on the parent doc — position on the page is irrelevant.
components.html(
    """
    <script>
    const doc = window.parent.document;
    if (!doc.__mmcomposerGenBound) {
        doc.__mmcomposerGenBound = true;
        doc.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                const btn = [...doc.querySelectorAll('button')]
                    .find(b => b.innerText.trim().includes('Generate kernel'));
                if (btn) { e.preventDefault(); btn.click(); }
            }
        });
    }
    </script>
    """,
    height=0,
)
