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
_onoff = lambda key: 1 if _rec.get(key) else 0

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
                      help="N tile per CTA.  Multiple of 64 (K-major B TMA sub-tile).  Caps at 256: "
                           "the tcgen05.mma N-atom max is 256, and the cluster splits M, not N.")
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

    ms_ws = st.selectbox(
        "Multi-staging + warp specialization", mc.ONOFF_OPTS, index=_onoff("ms_ws"),
        help="Dedicated TMA + MMA warps (async producer/consumer) on top of the "
             "multi-stage ring.  Off = synchronous MMA (no producer/consumer split).") == "On"
    two_cta = st.selectbox(
        "2-CTA cluster MMA", mc.ONOFF_OPTS, index=_onoff("two_cta"),
        help="`__cluster_dims__(2,1,1)` + `cta_group::2`: two CTAs cooperate in one "
             "tcgen05.mma (half-B per CTA, doubles M per MMA).  Requires "
             "multi-staging + warp specialization (above).") == "On"
    tma_store = st.selectbox(
        "TMA store epilogue", mc.ONOFF_OPTS, index=_onoff("tma_store"),
        help="Epilogue Phase 2: write the result to GMEM with one async TMA store "
             "per CTA (swizzled SMEM staging) instead of all-thread int4 stores.  "
             "A universal knob — often *not* a win (see the measured "
             "TFLOPS), kept as an honest mechanism comparison.") == "On"
    persistent = st.selectbox(
        "Persistent grid", mc.ONOFF_OPTS, index=_onoff("persistent"),
        help="Launch one CTA per SM and loop over output tiles inside the "
             "kernel (grid = #SMs) instead of one CTA per tile.  Trims "
             "launch/tail overhead — config-dependent (a clear win on low-K / "
             "many-tile shapes).  Available with warp specialization on and the "
             "2-CTA cluster off.") == "On"

    st.subheader("Problem shape")
    shapes_text = st.text_area(
        "Target shape (one M,N,K)",
        value="8192,8192,8192", height=68,
        help="The single (M, N, K) you're tuning for.  One shape at a time: different shapes "
             "have different optimal knob configs, so a kernel is composed/verified per shape.  "
             "Extra lines are ignored (with a warning).")


# ── Gate on Generate; snapshot config into session_state ──────────────

if generate:
    st.session_state.applied = dict(bm=bm, bn=bn, bk=bk, ns=ns, gsm=gsm, nw=nw,
                                    ms_ws=ms_ws, two_cta=two_cta, tma_store=int(tma_store),
                                    persistent=int(persistent), shapes_text=shapes_text)
    st.session_state.run_live = True   # fire the on-the-fly B200 bench (if live mode)

if "applied" not in st.session_state:
    st.info("Configure parameters in the sidebar, then click **🛠  Generate kernel**.")
    st.stop()

cfg = st.session_state.applied
bm, bn, bk = cfg["bm"], cfg["bn"], cfg["bk"]
ns, gsm, nw = cfg["ns"], cfg["gsm"], cfg["nw"]
ms_ws, two_cta = cfg["ms_ws"], cfg["two_cta"]
tma_store = cfg["tma_store"]
persistent = cfg.get("persistent", 0)
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
                              tma_store=tma_store, persistent=persistent,
                              persistent_ok=tier.get("persistent_ok", False),
                              shape=shapes[0] if shapes else None)
if warnings:
    st.error(f"⚠️  **{len(warnings)} configuration warning(s)** — this combination won't run.  "
             "Fix in the sidebar and re-generate.")
    for w in warnings:
        st.warning(w)
else:
    st.success("✅ Configuration passes all static constraint checks.")
    # Empirical ground truth from the committed B200 compatibility matrix.
    try:
        status, entry = mc.compat_status(tier["dir"], bm, bn, bk, ns, gsm, nw,
                                          tma_store=tma_store, persistent=persistent)
        if status == "verified":
            # Prefer perf at the shape the user is tuning; else the largest swept square.
            em, en, ek = shapes[0]
            p = mc.compat_perf(tier["dir"], bm, bn, bk, ns, gsm, nw, em, en, ek,
                               tma_store=tma_store, persistent=persistent)
            ref = (em, en, ek)
            if not (p and p.get("tflops")):
                squares = [t for t in mc.perf_shapes() if t[0] == t[1] == t[2]]
                if squares:
                    ref = max(squares)
                    p = mc.compat_perf(tier["dir"], bm, bn, bk, ns, gsm, nw, *ref,
                                       tma_store=tma_store, persistent=persistent)
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

kernel_src = mc.render_kernel(tier, bm, bn, bk, ns, gsm, nw, tma_store=tma_store)
host_src   = mc.render_host(tier, bm, bn, bk, ns, gsm, nw, tma_store=tma_store, persistent=persistent)

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
                     tma_store=tma_store, persistent=persistent)
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
        WS_DIRS  = [t["dir"] for k, t in mc.TIER_MAP.items() if t and k[0]]  # k = (ms_ws, two_cta)
        ALL_DIRS = [t["dir"] for t in mc.TIER_MAP.values() if t]
        st.markdown("**🔧 Autotune** — sweep valid knob combinations on a B200 for this shape and rank by TFLOPS.")
        scope = st.radio(
            "Sweep scope",
            ["Warp specialization on (production)", "Full sweep (incl. warp-spec off)"],
            horizontal=True, key="autotune_scope",
            help="In production warp specialization essentially always helps, so the production "
                 "sweep skips the warp-spec-off combos (about half the search). Full sweeps "
                 "everything, including the warp-spec-off combos kept for educational comparison.")
        at_dirs = WS_DIRS if scope.startswith("Warp") else ALL_DIRS
        at_sig = (tuple(at_dirs), m0, n0, k0)
        at_cache = st.session_state.setdefault("autotune_cache", {})

        def _knob_cols(tier_dir):
            ws, cta = mc.toggles_for_dir(tier_dir)
            return ("On" if ws else "Off", "On" if cta else "Off")

        if st.button("🔧  Autotune: sweep combos on a B200", key="autotune_btn"):
            with st.spinner(f"Sweeping valid knob combinations for {m0}×{n0}×{k0} on a B200 "
                            "(compiles + runs each + cuBLAS — this takes minutes)…"):
                at_cache[at_sig] = live_bench.run_autotune(at_dirs, m0, n0, k0)
        at = at_cache.get(at_sig)
        if at and at.get("ok"):
            b = at["results"][0]
            bws, bcta = _knob_cols(b["tier"])
            st.success(
                f"🏆 **Best of {at['n_combos']} combos** at {m0}×{n0}×{k0}: "
                f"**{b['tflops']:.0f} TFLOPS** ({b['vs_cublas']:.0%} of cuBLAS "
                f"{at['cublas_tflops']:.0f}) — Warp-spec={bws} · 2-CTA cluster={bcta} · "
                f"BN={b['bn']} NS={b['ns']} GSM={b['gsm']} NW={b['nw']} "
                f"TMA_STORE={b['tma_store']} PERSISTENT={b['persistent']}")
            n_res = len(at["results"])
            top_n = st.slider("Show top", min_value=3, max_value=min(50, n_res),
                              value=min(10, n_res), key="autotune_topn") if n_res > 3 else n_res
            rows = []
            for i, r in enumerate(at["results"][:top_n]):
                ws, cta = _knob_cols(r["tier"])
                rows.append({"#": i + 1, "Warp-spec": ws, "2-CTA": cta,
                             "BN": r["bn"], "NS": r["ns"], "GSM": r["gsm"], "NW": r["nw"],
                             "TMA": r["tma_store"], "PERS": r["persistent"],
                             "TFLOPS": f"{r['tflops']:.0f}",
                             "vs cuBLAS": f"{r['vs_cublas']:.0%}" if r.get("vs_cublas") else "—"})
            st.dataframe(rows, width="stretch", hide_index=True)
            st.caption("Set the sidebar to the winning knobs (and re-generate) to download that kernel.")
        elif at:
            st.error(f"Autotune failed: {at.get('error')}")
            if at.get("stderr"):
                st.code(at["stderr"], language="text")
        else:
            st.caption("Autotune submits one srun that compiles + benchmarks every valid combo "
                       "(tens to hundreds of kernels). Production (warp-spec on) is faster; "
                       "Full is the complete search.")
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
                               tma_store=tma_store, persistent=persistent)
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
