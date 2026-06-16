# Project Ideas

## MMCustomizer — canonical skeleton + fusion hooks

A possible sibling project to MMComposer.

**The distinction:**
- **MMComposer** (today): compose any GEMM kernel by *freely combining knobs*
  (warp-spec on/off, 2-CTA, persistent, epilogue modes, tile sizes, …). The
  "canonical" high-performance design is just one point in that knob space and
  isn't highlighted.
- **MMCustomizer** (idea): fix the **canonical skeleton** (the general design
  that performs well across most shapes — the one the presentation builds up)
  and expose **fusion hooks** instead of knobs. The goal isn't to vary the
  matmul; it's to *extend* it with custom epilogues/prologues.

**Why the skeleton makes this natural:** the canonical design already has named
phase boundaries (the containers in the dataflow), and those boundaries *are*
the fusion hook points. There are **three hook types**:

1. **Prologue hook** — transform input tiles as they land (in SMEM / before the
   MMA). Example: weight dequantization (W4A16).
2. **Input-reduction hook** — reduce an input operand along K into a per-row (or
   per-column) statistic that is made available later in the epilogue. Example:
   RMSNorm's sum-of-squares over the hidden dim.
3. **Epilogue hook** — operate on the result tile while it is in registers
   (after the TMEM drain, before the SMEM stage). Examples: SwiGLU combine,
   bias + activation, residual add, per-row scaling.

UI: reuse the MMComposer web UI, but default to the canonical skeleton and add
options to specify custom fusions. **Epilogue fusion is the main theme.**

**Motivating examples (real, modern-ML):**
- *Epilogue:*
  - **SwiGLU / GeGLU** gated activation — LLM MLP up-projection. The existing
    `study-swiglu-extra-store/fused_matmul_swiglu_out_*` kernel is the flagship
    example.
  - **Bias + activation** (GELU/SiLU/ReLU) — every linear layer.
  - **Residual / skip add** — `D = A·B + residual` from GMEM; every transformer
    block.
  - **FP8/INT8 dequant-scale + requantize** — scale the FP32 accumulator by
    per-tensor/-row/-block scales, cast output to FP8/INT8. High impact on B200;
    strong second example after SwiGLU.
  - **Back-to-back / GEMM→GEMM** (advanced) — the epilogue feeds another MMA,
    e.g. attention `S=Q·Kᵀ → softmax → O=S·V`. Only works when the intermediate
    tile stays on-chip (small intermediate N); usually a dedicated fused kernel.
    This is the "accumulator × another matrix from GMEM" idea — it's real.
- *Prologue:*
  - **Weight dequantization (W4A16 / W8A16)** — INT4/INT8 weights with per-group
    scales (+ zero-points) dequantized on-chip before a BF16 MMA. The basis of
    Marlin / Machete / AWQ / GPTQ inference kernels. Clean prologue hook.
- *Input reduction + epilogue (RMSNorm before a GEMM):* the pre-norm pattern
  `RMSNorm(x)·W` fuses cleanly because the **norm axis = the GEMM contraction
  axis K**.
  - The per-row scale factors out of the matmul:
    `Y[m,:] = (1/rms[m]) · (X[m,:] · W′)`, applied as an **epilogue** per-row
    multiply.
  - The gain `g` folds into the weight offline: `W′ = diag(g)·W`.
  - The sum-of-squares `ssq[m] = Σ_k x[m,k]²` is accumulated during the K-sweep
    (the **input-reduction hook**); the epilogue scales each row by
    `rsqrt(ssq[m]/H + ε)`.
  - This is the motivating example for the input-reduction hook (more general
    than pointwise epilogue fusion).
  - Caveats: RMSNorm *after* a GEMM reduces over N (split across CTAs) → needs a
    cross-CTA reduction, not cleanly fusable. And a norm output often feeds
    several GEMMs (Q/K/V, gate/up), so fusing into one recomputes it for the
    rest — best when single-consumer.

**Boundary note:** block-scaled FP8/FP4 (MXFP8/NVFP4) is *not* a fusion hook —
on Blackwell the scales are a native operand of the block-scaled `tcgen05.mma`
(`kind::mxf8f6f4`, `kind::mxf4nvf4`, scales in TMEM), plus an extra scale input
stream. That belongs on MMComposer's "pick the compute path" axis, not
MMCustomizer's "inject fusion code" axis.

**Idea: a Python-like DSL for hook programs.** Let users write a small program
for each hook — a **prologue** program, an **input-reduction** program, and an
**epilogue** program — in a Python-like DSL. MMCustomizer compiles them and
splices them into the canonical skeleton organically: the compiler maps DSL ops
onto the register / SMEM / TMEM containers and the correct phase, and supplies
all the orchestration, swizzles, pipelining, and buffer management. The user
writes only the *fusion math*. Examples:
- SwiGLU → an epilogue program combining the two projections.
- RMSNorm → an input-reduction program (`Σ x²`) + an epilogue program
  (`rsqrt(ssq/H + ε)` scale).
- W4A16 → a prologue program (`(int4 − zero) · scale`).

Status: idea only (2026-06-16). Revisit after the presentation's framing settles.
