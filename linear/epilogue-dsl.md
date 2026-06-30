# Fused elementwise epilogue DSL for `mmc.matmul`

**Status:** Phase 1 implemented and GPU-verified on B200. Phase 2 designed, not started.

## Summary

Add an optional **elementwise epilogue** to the matmul API: the caller passes a
small Python function describing an op to apply to every output element, and
mmcomposer fuses it directly into the GEMM kernel's epilogue — so the activation
runs **in-register, in fp32, before the bf16 store**, with no extra kernel launch
and no extra global-memory round trip.

```python
from mmcomposer.epilogue import sigmoid
c = mmc.matmul(a, b, epilogue=lambda x: x * sigmoid(x))   # fused SiLU
```

The function looks like Python but is a restricted **description language** (its
own semantics, formally specified). It is *traced* once into an expression DAG and
*lowered* to a CUDA fp32 expression that is spliced into the kernel; it never runs
as Python at matmul time.

## Motivation

In real workloads (e.g. FFN blocks) a matmul is almost always followed by an
elementwise activation. Done the usual eager way that's a second kernel reading
and rewriting the whole output tensor through GMEM. Fusing it into the GEMM
epilogue removes that traffic and the launch — the activation rides on values
already in registers. (Cf. the dual-B SwiGLU kernel, where the gate is ~free over
the bare GEMM.) The DSL makes this available for *arbitrary* elementwise ops
without hand-writing a kernel per activation.

## Language design

A valid epilogue is a callable (`lambda` or `def`) with the contract:
**one input, one output, straight-line math, no control flow.**

- **Semantics:** an elementwise scalar map `f: float -> float`, applied
  independently to each output element, evaluated in **fp32** (inputs are the
  fp32 accumulators; the result is rounded to bf16 on store). Pure / deterministic.
- **Two-tier builtins** (in `mmcomposer.epilogue`):
  - *primitives* lower 1:1 to a CUDA intrinsic: `exp, tanh, sqrt, log, abs,
    maximum, minimum`.
  - *composites* are defined **in the DSL itself** over primitives, so the
    lowering backend only ever knows primitives: `sigmoid(x) = 1/(1+exp(-x))`,
    `relu(x) = maximum(x, 0)`. New activations are just compositions — no backend
    change.
- **Operators:** `+ - * / **` (constant exponent; small int powers expand to
  repeated multiply), unary `-`, `abs()`.
- **Rejected at trace time:** control flow / comparisons / `and`/`or`, multiple
  args or returns, non-constant exponents, foreign functions (`math`, `numpy`,
  `torch`). Use `maximum`/`minimum` instead of branching.

Formal spec: `mmcomposer/EPILOGUE.md`.

### Example lowerings

| epilogue | lowered CUDA (fp32, in terms of `x`) |
|----------|--------------------------------------|
| `lambda x: x * sigmoid(x)` (SiLU) | `(x * __fdividef(1.0f, (1.0f + __expf((-x)))))` |
| `relu` | `fmaxf(x, 0.0f)` |
| `lambda x: minimum(maximum(x,0.),6.)` (ReLU6) | `fminf(fmaxf(x, 0.0f), 6.0f)` |
| GELU (tanh approx) | `((0.5f*x) * (1.0f + tanhf((0.79788456f*(x + (0.044715f*(x*x*x)))))))` |

## Applications

- Fused activations for FFN / MLP blocks: SiLU, GELU, ReLU/ReLU6, sigmoid, tanh,
  leaky-ReLU, hardswish-style clamps — anything elementwise.
- Fused scale/bias and dequant-style affine maps (`a*x + b`).
- Removes a full activation kernel + GMEM round trip per matmul in inference and
  forward passes.

## Design decisions

1. **An epilogue is a tuned matmul *variant*.** A fused kernel is keyed by
   `(shape, epilogue digest)` and auto-tuned **with the epilogue spliced into every
   candidate** on first use — like any other knob (BN, NS, …), it produces a custom
   kernel that gets its own sweep. The winning config is the best one *for the fused
   kernel*, which differs from the plain GEMM: at FFN 32768×4608×768 the plain GEMM
   picks `nw=4`, but fused SiLU picks `nw=8` (more epilogue warps to hide the
   activation ALU). (We first shipped a reuse-the-plain-config version; tuning the
   variant recovered the residual ~10–20% gap.) Cost: one ~100 s sweep per
   (shape, epilogue), one-time + cached; many *calls* of the same epilogue reuse it.
   Verification of a fused candidate uses `epilogue.to_torch(fn)` — the same `Expr`
   DAG lowered to torch — since the candidate outputs `f(a@b)`, not `a@b`.
2. **The DSL is a real language, not arbitrary Python.** Python *syntax*, but a
   restricted, control-flow-free expression language with formally defined
   elementwise/fp32 semantics — so it can be traced and compiled deterministically.
3. **Two-tier builtins (compose, don't special-case).** Primitives map 1:1 to
   CUDA intrinsics; high-level activations are DSL compositions of primitives. The
   lowering backend stays tiny; new activations need zero backend changes.
4. **Compute in fp32, store bf16.** The op runs on the fp32 accumulators in
   registers, exactly at the existing TMEM→register→(pack)→SMEM→GMEM boundary —
   the one place every epilogue path converts fp32→bf16. Matches a fp32 reference
   to ~1e-3 relative (bf16 output precision), not bit-exact (fast intrinsics).
5. **Universal, single-point injection.** Every fp32→bf16 conversion in the
   epilogue fragments is routed through a generated `mmc_epi(float)`; identity
   (`return x;`) inlines to a no-op. One mechanism covers all tiers and epilogue
   variants (overlap / TMA-pipelined / non-overlap).
6. **No cost to the plain matmul path.** When `epilogue=None`, `matmul` takes the
   exact original code path (one extra `None` check). The EDL adds nothing to
   regular matmuls — see "Performance" below.
7. **Approximate fast math in the epilogue.** Division lowers to `__fdividef`
   (`rcp.approx`) and `exp` to `__expf` (`ex2.approx`) — the same approximate
   intrinsics the hand-tuned SwiGLU kernel uses. bf16 output makes the ~2-ULP
   error invisible, and it's what keeps the activation nearly free (an IEEE
   reciprocal in the epilogue was ~25× more expensive and made fused SiLU 2× slower
   than torch).

## Performance

**Kernel (measured, B200, FFN 32768×4608×768):** bare GEMM 0.175 ms; fused
matmul+SiLU **0.187 ms / 1237 TFLOPS** — only **1.09×** the bare GEMM (activation
~free), and **1.7× faster** than torch doing matmul + a separate SiLU kernel
(0.313 ms). The tuned variant picks a different config than plain (`nw=8` vs `nw=4`
— more epilogue warps to hide the activation). The
win is largest on memory-bound shapes (small K, large output), where torch's
separate activation pass is a full extra GMEM round trip; on compute-bound squares
the GEMM dominates and the gap is small. (This relies on the fast-math lowering —
design decision #7; with IEEE division the fused SiLU was 0.666 ms / 2× *slower*.)

**Host:**

- **Plain `mmc.matmul(a, b)`:** unchanged. `epilogue=None` short-circuits to the
  original path; no extra host work.
- **Warm fused path:** the epilogue callable is the cache key. Trace+lower is
  **memoized by the callable object** (weak-keyed), so a *reused* epilogue object
  is not re-traced — a call is just: trace-cache hit → `(shape, digest)` kernel
  lookup → launch. No per-call codegen/compile. (Tip: define the epilogue once and
  reuse the object in hot loops; a fresh inline `lambda` each iteration re-traces,
  which is cheap but avoidable.)
- **Cold (first time for a shape+epilogue):** trace → reuse tuned config → splice
  → one `nvcc` compile (~seconds) → cache the cubin on disk, keyed by
  `(geometry, epilogue digest)`.
- **Runtime cost of the op itself:** the activation runs on values already in
  registers in fp32; no extra GMEM traffic, no extra launch. Verified: identity
  epilogue is **bit-exact** vs a plain matmul (`max|diff| = 0`).

## Implementation status

**Done (phase 1):**
- `mmcomposer/epilogue.py` — the DSL: `Expr` tracer, builtins, `to_cuda`,
  `digest`. Pure, 9 unit tests.
- `mmcomposer/EPILOGUE.md` — formal language spec.
- Codegen: `generate_kernel` injects `mmc_epi(float)` from `EPILOGUE_FN`; both
  epilogue fragments route conversions through it (identity = no-op).
- API: `mmc.matmul(a, b, epilogue=...)` and `mmc.get_epilogue_kernel(a, b, fn)`;
  **tunes the fused variant** keyed by `(shape, digest)`, caches it, async on
  torch's current stream.
- Tuned variant: `autotune` threads the epilogue through the whole sweep;
  `epilogue.to_torch` provides the verify reference; the variant picks its own best
  config (FFN: `nw=8` vs plain `nw=4`).
- GPU-verified on B200: identity == plain (bit-exact); SiLU / ReLU / GELU match
  torch to ~1.7e-3. **32 tests** (CPU lowering incl. `to_torch`; op-by-op GPU
  correctness for every builtin/operator; + the tuned-variant path).
- Fast-math lowering (`div` → `__fdividef`, `exp` → `__expf`) → fused SiLU is
  ~1.09× the bare GEMM and 1.7× faster than torch at an FFN shape.
- `examples/quickstart_epilogue.py` — runnable showcase.

**Phase 2 — multiple inputs (DSL done; kernel pending):**
- n-ary epilogue: input 0 = accumulator, inputs 1.. = extra same-shape operands,
  e.g. `lambda x, c: x*c` ((a@b)*c), `lambda x, c: x+c` (residual), `lambda x, g, r:
  x*sigmoid(g)+r`. **Done:** tracer/lowering (`arity`/`to_cuda`/`to_torch` n-ary).
  **Pending:** kernel-side direct-to-register LDG of the extra tiles, `aux=` API,
  autotune threading.

**Phase 3 — multiple stores & accumulator split (design):**
- **Multi-store:** tuple return → one output matrix per element; shape inferred per
  value. `c, d = mmc.matmul(a, b, epilogue=fn)`.
- **Split:** `a, b = x.chunk(2)` splits the accumulator into column-halves (`[M,N/2]`
  each); width inference (`x`/`f(x)` = full, chunk = 1/k); mixing widths is an error.
- Together they express the **dual-B SwiGLU kernel** from the DSL:
  `def swiglu(x): a,b = x.chunk(2); return x, a*b*sigmoid(b)` → packed C `[M,N]` + D
  `[M,N/2]`. `dual_b=True` (two separate B matrices) makes it byte-equivalent to
  `matmul_swiglu_dual_b_ns6_s2`.
- Codegen impact: epilogue iterates over chunk *groups*; one TMA store + buffer per
  returned value. (n>2 chunks are a natural extension; do k=2 first.)
- See `mmcomposer/EPILOGUE.md` §8 for the full spec.

**Other designed extensions:**
- Control flow via value-level `where(cond, a, b)` + comparison operators returning
  predicate `Expr`s (no Python `if` — it can't be traced, and per-element branches
  are predication/`select` on the GPU anyway, which `where` traces directly).

## Try it

```python
import torch, mmcomposer as mmc
from mmcomposer.epilogue import sigmoid
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
c = mmc.matmul(a, b, epilogue=lambda x: x * sigmoid(x))   # fused SiLU
```
