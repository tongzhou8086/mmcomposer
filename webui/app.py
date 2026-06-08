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
             "multi-stage ring.  Off = synchronous-MMA baseline (Tier 1).") == "On"
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
             "launch/tail overhead — a small, config-dependent win on Tier 2.  "
             "Wired on the warp-specialized single-CTA tier only.") == "On"
    no_allocate = st.selectbox(
        "Streaming output store (.L1::no_allocate)", mc.ONOFF_OPTS, index=_onoff("no_allocate"),
        help="Tag the epilogue's global store with `.L1::no_allocate` — C is "
             "write-once, so don't evict A/B from L1 for it.  Measured "
             "perf-neutral on B200 (the reuse that matters is L2), but correct "
             "and never-negative; a free composable knob.  Affects the int4 "
             "store path only (no-op under the TMA-store epilogue).") == "On"

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
                                    persistent=int(persistent), no_allocate=int(no_allocate),
                                    shapes_text=shapes_text)

if "applied" not in st.session_state:
    st.info("Configure parameters in the sidebar, then click **🛠  Generate kernel**.")
    st.stop()

cfg = st.session_state.applied
bm, bn, bk = cfg["bm"], cfg["bn"], cfg["bk"]
ns, gsm, nw = cfg["ns"], cfg["gsm"], cfg["nw"]
ms_ws, two_cta = cfg["ms_ws"], cfg["two_cta"]
tma_store = cfg["tma_store"]
persistent = cfg.get("persistent", 0)
no_allocate = cfg.get("no_allocate", 0)
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

kernel_src = mc.render_kernel(tier, bm, bn, bk, ns, gsm, nw, tma_store=tma_store, no_allocate=no_allocate)
host_src   = mc.render_host(tier, bm, bn, bk, ns, gsm, nw, tma_store=tma_store,
                            persistent=persistent, no_allocate=no_allocate)

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
