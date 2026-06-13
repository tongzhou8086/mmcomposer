# mmcomposer web UI

A Streamlit front-end for `mmcomposer` — a configurator for Blackwell (B200)
matrix-multiplication kernels.  You pick knobs (tile sizes + on/off options),
and it renders a CUDA kernel plus a **self-contained** host script to launch
it, and shows **measured** performance for that exact config.

## What it does

- **Composable kernel, exposed as knobs.** One kernel family with uniform
  options: tile sizes (`BM`, `BN`, `BK`, `NS`, `GROUP_SIZE_M`, `num_warps`)
  and on/off optimizations (warp specialization, 2-CTA cluster MMA, TMA-store
  epilogue).  The toggles select one of three implementation tiers; the knobs
  tune within it.
- **Hybrid validation.**
  - A fast *static* checker (`mvp_core.validate_config`) rejects the obviously
    invalid (SMEM over the 228 KB cap, bad divisibility, `BK≠64`, …) with an
    explanation, instantly.
  - An *empirical* compatibility matrix (`kernels/compat_matrix.json`) is the
    ground truth: every static-valid combo was compiled, run, and
    correctness-checked on a real B200, and benchmarked at 4096³/8192³ vs
    cuBLAS.  The app annotates each config with its measured TFLOPS and a
    "verified on B200" badge, and flags any combo the hardware rejects even
    though the static checker passed.
  - Validation constraints are documented in `VALIDATION_CONSTRAINTS.md`,
    including which limits are hardware constraints, implementation guardrails,
    or autotune policy.
- **Self-contained downloads.** The host script inlines its runtime, so the
  downloaded `kernel.cu` + `host.py` run with just `python host.py` (given
  torch + cuda-python + `nvcc`) — no repo, no sibling modules.

`tutorial/` is a *reference implementation* only; the MVP renders its own
owned codebase under `webui/kernels/`.

## Why composable — a worked example

The whole premise is that **there is no globally-best config**: the
optimizations don't add up linearly, and the optimal *combination* shifts with
the problem shape. A knob that's the biggest win on one shape can be dead weight
on another. So nothing is on by default — the autotune sweeps the grid and lets
**measured TFLOPS decide, per shape.**

A concrete example, same kernel family + same knobs, three shapes (M×N×K) →
three different winners:

| shape (M×N×K)       | regime                    | winning combo               | overlap Δ | persistent Δ |
|---------------------|---------------------------|-----------------------------|-----------|--------------|
| 32768×4608×**768**  | low-K, epilogue-bound     | persistent + overlap        | **+7–11%**| helps        |
| 32768×**768**×4608  | small-N, mid-K            | 2-CTA + persistent + overlap| **+2.5%** | **+2.0%**    |
| **8192³**           | big square, compute-bound | 2-CTA, *plain* (no overlap) | **~0%**   | wash / loss  |

The same knob (epilogue/K-loop overlap) swings from the single biggest lever to
a no-op purely on shape — its gain tracks the epilogue's fraction of per-tile
time, which decays smoothly with K (**+7–11% at K=768 → +2.5% at K=4608 → ~0 at
K=8192**), so it's *not* "low-K only." The per-knob attribution above comes from
median Δ over hundreds of matched-tile-knob A/B pairs in one autotune sweep;
trust the winning *cluster*, not the exact #1, since there's a ~3% thermal-drift
noise floor (confirm sub-1.5% wins with ≥3 fresh runs, and compare ratios — % of
cuBLAS — not absolute TFLOPS across separate GPU allocations).

## Layout

```
webui/
  app.py                 # Streamlit UI (imports mvp_core)
  mvp_core.py            # pure logic: options, tier map, validate, render, compat lookup
  kernels/               # MVP-owned rendered codebase
    _runtime.py          # vendored cuda-python plumbing (inlined into host downloads)
    _epilogue.cu.frag    # shared epilogue, stitched into every tier at // @@EPILOGUE@@
    compat_matrix.json   # committed B200 ground-truth (correctness + perf)
    tier1_baseline/        { kernel.cu, launcher.py }   # no warp-spec (synchronous MMA)
    tier3_cluster_swizzle/ { kernel.cu, launcher.py }   # unified warp-spec skeleton;
                                                        # TWO_CTA knob = single-CTA / 2-CTA cluster
  tests/
    gpu_codegen_driver.py # B200 integration test: render→compile→run every valid combo
    host_artifact_test.py # runs the actual downloaded host.py end-to-end per tier
```

## Run locally

```bash
pip install -r webui/requirements.txt
streamlit run webui/app.py
```

The app opens at `http://localhost:8501` and hot-reloads on save.  Tip: inside
the app, **Ctrl/Cmd + Enter** (re)generates without the mouse.

## Correctness integration test (needs a B200)

```bash
srun --partition=dedicated --gres=gpu:nvidia_b200:1 \
    python webui/tests/gpu_codegen_driver.py --mode correctness --perf-shapes 2048
```

This renders every static-valid combo, compiles in parallel, launches each in
a fault-isolated worker (a faulting kernel can't poison the rest), checks
correctness against torch, and does not time kernels.

## Timing autotune (needs a B200)

```bash
python webui/autotune.py 32768x4608x768 --top 20
```

This uses the same render/compile/run backend in perf mode, but applies the
production timing policy by default (`BN=256`, `NS>=3`) instead of timing every
valid combination.  Use `--scope full` only for the expensive timed all-combo
search.

## Deploy (Streamlit Community Cloud)

Point a new app at this repo + `webui/app.py`.  Pushes to the tracked branch
auto-redeploy.  Note: editing an imported module (e.g. `mvp_core`) may require
a full app **Reboot** so Streamlit reimports it rather than reusing the cached
module.
