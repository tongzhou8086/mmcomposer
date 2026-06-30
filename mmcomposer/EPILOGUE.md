# MMComposer Epilogue Description Language (EDL) — phase 1

The EDL lets you describe an **elementwise epilogue** — an operation applied to
every output element of a matmul — as an ordinary Python callable:

```python
from mmcomposer.epilogue import sigmoid
silu = lambda x: x * sigmoid(x)
c = mmc.matmul(a, b, epilogue=silu)        # c[i,j] = silu((a @ b)[i,j])
```

It is written in Python *syntax*, but it is **not** arbitrary Python: it is a
small, pure, control-flow-free expression language with its own semantics,
defined here. The function is **traced once** (called with a symbolic value),
producing an expression DAG that is **lowered to a CUDA fp32 expression** and
spliced into the kernel epilogue. It never runs as Python at matmul time.

---

## 1. Execution semantics

- **Elementwise.** The function defines a scalar map `f: float -> float`. The
  kernel applies it independently to each output element; there is no access to
  neighbours, indices, rows, columns, or reductions.
- **Where it runs.** Inside the kernel epilogue, on each accumulator element, in
  the order: tensor-memory → fp32 register → **`f` applied here** → downcast to
  bf16 → stage to SMEM → store to GMEM. So the user op is the last thing computed
  before the output is narrowed to bf16.
- **Numeric type.** `f` is evaluated in **fp32**. Inputs are the fp32
  accumulator values; the result is rounded to bf16 on store (round-to-nearest).
  Intermediate math is fp32; transcendentals use CUDA fast intrinsics
  (e.g. `__expf`), so results match a fp32 reference to ~1e-3 relative, not bit-exact.
- **Pure & deterministic.** No state, no side effects, no randomness, no I/O. The
  same input always yields the same output; the trace is order-independent.
- **Default.** No `epilogue=` (or the identity `lambda x: x`) means "store the
  GEMM result unchanged" — identical to a plain `mmc.matmul`.

---

## 2. The contract (phase 1)

A valid epilogue is a callable (`lambda` **or** `def`) such that:

1. it takes **exactly one** argument (the symbolic element `x`);
2. it returns **exactly one** value (an expression or a numeric constant);
3. its body is **straight-line math** — no control flow.

Violations are rejected at trace time (a `TypeError`), not silently miscompiled.

---

## 3. What is allowed

### 3.1 The variable
`x` — the one input element, an `Expr`. Everything is built from it.

### 3.2 Operators
| form | meaning | notes |
|------|---------|-------|
| `a + b`, `a - b`, `a * b` | fp32 arithmetic | either operand may be a constant |
| `a / b` | fast division → `__fdividef` | ~2 ULP, far below bf16 output precision |
| `-a` | negation | |
| `a ** n` | power | `n` must be a **constant**; integer `0..8` expands to repeated multiply, else `powf` |
| `abs(a)` | absolute value | Python `abs()`, lowered to `fabsf` |

Operands are either an `Expr` or a Python numeric **constant** (`int`/`float`),
which becomes an fp32 literal. (`bool` is not a number here.)

### 3.3 Builtin functions — `mmcomposer.epilogue`

Two tiers. **Primitives** lower 1:1 to a CUDA intrinsic:

| EDL | math | CUDA |
|-----|------|------|
| `exp(x)`        | eˣ              | `__expf(x)` |
| `tanh(x)`       | tanh x          | `tanhf(x)` |
| `sqrt(x)`       | √x              | `sqrtf(x)` |
| `log(x)`        | ln x            | `__logf(x)` |
| `abs(x)`        | \|x\|           | `fabsf(x)` |
| `maximum(a, b)` | max(a, b)       | `fmaxf(a, b)` |
| `minimum(a, b)` | min(a, b)       | `fminf(a, b)` |

**Composites** are defined *in the DSL itself* over primitives (no backend
special-case), so they expand to the primitives above:

| EDL | definition |
|-----|------------|
| `sigmoid(x)` | `1 / (1 + exp(-x))` |
| `relu(x)`    | `maximum(x, 0)` |

You compose new activations directly, e.g. `silu = lambda x: x * sigmoid(x)`.

---

## 4. What is NOT allowed (phase 1)

Rejected at trace time:

- **Control flow / branching:** `if/else`, ternaries, `and`/`or`, `not`, and any
  comparison (`<`, `==`, …) on the value. Use `maximum`/`minimum` instead of
  branching. (Booleanizing `x` raises immediately.)
- **More than one argument**, or returning a **tuple / multiple values**.
- **Non-constant `**` exponents** (`x ** x`).
- **Foreign functions:** `math.exp`, `numpy`, `torch`, custom Python helpers that
  aren't built from the EDL — they won't trace. Use the `mmcomposer.epilogue`
  builtins.
- **Closures over runtime tensors / Python state** that aren't numeric constants.

---

## 5. Grammar

```
epilogue   ::= callable taking one Expr, returning one expr
expr       ::= var | const
             | expr "+" expr | expr "-" expr
             | expr "*" expr | expr "/" expr
             | "-" expr
             | expr "**" int_const
             | "abs" "(" expr ")"
             | primitive "(" expr ["," expr] ")"
             | composite "(" expr ")"
var        ::= "x"
const      ::= python int | python float        (-> fp32 literal)
primitive  ::= "exp" | "tanh" | "sqrt" | "log" | "maximum" | "minimum"
composite  ::= "sigmoid" | "relu"
```

---

## 6. Lowering examples

`to_cuda(fn)` (in `mmcomposer.epilogue`) returns the CUDA fp32 expression in
terms of `x`; `digest(fn)` returns a stable short hash used as the cache key /
cubin tag.

| epilogue | CUDA fp32 expression |
|----------|----------------------|
| `lambda x: x`                         | `x` |
| `relu`                                | `fmaxf(x, 0.0f)` |
| `sigmoid`                             | `__fdividef(1.0f, (1.0f + __expf((-x))))` |
| `lambda x: x * sigmoid(x)` (silu)     | `(x * __fdividef(1.0f, (1.0f + __expf((-x)))))` |
| `lambda x: minimum(maximum(x,0.),6.)` | `fminf(fmaxf(x, 0.0f), 6.0f)` |
| `lambda x: 0.5*x*(1.+tanh(0.79788456*(x+0.044715*x**3)))` | `((0.5f * x) * (1.0f + tanhf((0.79788456f * (x + (0.044715f * (x * x * x)))))))` |

The lowered expression is wrapped as `__device__ float mmc_epi(float x){ return <expr>; }`
and applied to every element before the bf16 store.

---

## 7. Caching & tuning

An epilogue is a tuned **variant** of the matmul, keyed by
`(M, N, K, dtype, arch, epilogue_digest)`. On first use of a new (shape, epilogue)
the GEMM is auto-tuned **with the epilogue spliced into every candidate**, so the
winning config is the best one *for the fused kernel* (it can differ from the plain
GEMM — e.g. more epilogue warps to hide the activation). The winner + its cubin are
cached on disk; later calls (and future sessions) reuse them. Verification during
the sweep compares against `to_torch(fn)(a @ b)` (the same op evaluated in torch),
since a fused candidate outputs `f(a @ b)`.

Cost: one autotune per (shape, epilogue), one-time per machine; the plain matmul
and every distinct epilogue are independent cache entries.

---

## 8. Roadmap — phases 2 & 3

Phase 1 (above) is the **single-input → single-output** elementwise map. Two
planned generalizations keep the same "Python-syntax DSL, traced, lowered,
spliced, tuned-as-a-variant" model; only the contract widens.

### Phase 2 — multiple inputs *(DSL done; kernel pending)*

The epilogue takes **n positional args**: input 0 is the matmul accumulator
(`x`), inputs 1.. are **extra inputs** of the **same shape `[M,N]`**, combined
per element at the same position.

```python
lambda x, c: x * c                 # (a @ b) * c        -- elementwise gate / residual mul
lambda x, c: x + c                 # (a @ b) + c        -- residual add
lambda x, g, r: x * sigmoid(g) + r # two extra inputs
```

- `#extra inputs = arity - 1`; the matmul call supplies that many same-shape
  tensors (proposed `aux=[c, ...]`). Each is bf16, row-major, exactly `[M,N]`.
- Lowers to `mmc_epi(float x, float c0, float c1, ...)`; the digest (hence the
  tuned variant) includes the arity.
- **Implemented:** the tracer/lowering (`arity`, `to_cuda`, `to_torch` all n-ary).
  **Pending:** kernel-side load of the extra tiles + `aux=` API + autotune
  threading. The chosen load strategy is a **direct `ld.global` into registers**
  (no SMEM; mirrors each thread's accumulator coords; the read is a one-shot
  epilogue read, so the across-lane stride is cheap relative to the GEMM).

### Phase 3 — multiple stores & accumulator split *(design)*

Two composable extensions that together let the DSL **generate** the hand-tuned
dual-B SwiGLU kernel:

**(a) Multiple stores.** Return a **tuple** → one output matrix per element. Each
output's shape is **inferred** from how its value is built (see width inference).
The API returns a tuple to match: `c, d = mmc.matmul(a, b, epilogue=fn)`.

```python
lambda x: (x, 1.0 + x)             # two [M,N] outputs
```

**(b) Accumulator split.** `x.chunk(k)` splits the accumulator into `k` equal
**column chunks** along N, each of **width 1/k** (`[M, N/k]`). Self-documenting,
generalizes to any `k` (implement `k=2` first — covers SwiGLU/GLU).

**Width inference & rules.** Every value carries a *width fraction*: `x` and any
`f(x)` are width 1 (`[M,N]`); a chunk or an expression over chunks is width `1/k`
(`[M, N/k]`). A returned value's output shape is `[M, N·width]`. **Mixing widths
in one expression is rejected at trace time** (e.g. `a + x` combines `[M,N/2]`
with `[M,N]`).

```python
def left_half(x):                  # one [M, N/2] output
    a, b = x.chunk(2)
    return a

def store_two_halves_and_left(x):  # three [M, N/2] outputs: left, right, left
    a, b = x.chunk(2)
    return a, b, a

def swiglu(x):                     # the dual-B SwiGLU kernel, from the DSL:
    a, b = x.chunk(2)              #   packed C [M,N]  +  D [M,N/2]
    return x, a * b * sigmoid(b)   #   D = left * silu(gate)
```

- **`dual_b=True`** (matmul arg): instead of one packed `b` of width N, take **two
  separate B matrices** and load the second chunk's tile from the second one —
  making `swiglu` above byte-equivalent to `matmul_swiglu_dual_b_ns6_s2`.
- **Codegen impact (the real work):** split makes the epilogue iterate over
  **chunk groups** (so `a`, `b` are same-position elements of the two halves)
  rather than one column at a time; multi-store needs **one output buffer + TMA
  descriptor per returned value**, each sized by width inference.

### Composition

All three layers stack uniformly — extra inputs (P2) + split + multi-store (P3):

```python
def f(x, c):                       # gated activation with an external operand
    a, b = x.chunk(2)
    return a * sigmoid(b) * c
```

`n=k` chunks beyond 2 are a natural design extension; implement 2 first.
