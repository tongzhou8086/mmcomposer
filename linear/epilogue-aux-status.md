# Status: auxiliary-input epilogue fusion (phase 2, multi-input)

**TL;DR:** the multi-input epilogue (`(a@b) op c`) is **implemented and numerically
correct** (163/198 production configs verified), but **not production-ready**: 35/198
configs hit a `CUDA_ERROR_LAUNCH_FAILED` that poisons the context and kills the
in-process autotune sweep, so there's **no runnable `mmc.matmul(..., aux=)` quickstart
yet**. Perf was measured via the lower-level build path on a known-good config, and is
**~parity with torch** (multi-input fusion is memory-bound on the extra input).

## The feature

A multi-arg epilogue combines the matmul result with **extra same-shape `[M,N]`
input tensors**, per element, fused into the GEMM's store path:

```python
gate = lambda x, c: x * c                # out = (a @ b) * c
out  = mmc.matmul(a, b, epilogue=gate, aux=[c])
# also: lambda x, c: x + c   (residual add),   lambda x, c: x*sigmoid(x)*c   (gated SiLU)
```

Input 0 (`x`) is the matmul accumulator; inputs 1.. (`c0, c1, …`) are the `aux=[…]`
tensors (bf16, row-major, exactly `[M,N]`). `#aux = arity - 1`. See `EPILOGUE.md` §8.

## How it's implemented (where the extra load goes)

The extra input is read **directly into registers** inside the epilogue, at exactly
the `(row, col)` each thread's accumulator covers — no SMEM staging, no whole-tile
buffer. Per `STORE_N=64` chunk, in `_overlap_epilogue.cu.frag` (the TMA-pipelined /
production epilogue path):

1. **Issue the extra-input load first** — an `int4 ld.global` (8 bf16) per accumulator
   group, into a small `mmc_c0v[LOADS_PER_WARP][8]` register array, computed at
   `mmc_c0[(EPI_OUT_ROW + my_row)*N + (EPI_OUT_COL_BASE + chunk*STORE_N + local_n*8)]`.
   It's issued **before** the `tcgen05.ld` (TMEM→reg) so the GMEM latency overlaps the
   TMEM load + `wait_ld`.
2. **TMEM load** the accumulator chunk (`t[..][8]`), `wait_ld`.
3. **Combine**: `mmc_epi(t[n][k], mmc_c0v[n][k])` → pack to bf16 → stage → TMA-store.

Per-chunk streaming means the register cost is ~`LOADS_PER_WARP*8` floats (reused per
chunk), **not** the 64 KB a whole-tile SMEM stage would cost.

Plumbing:
- **codegen**: `MMC_N_EXTRA` knob (`#if`-guarded); `mmc_epi(float x, float c0, …)`.
- **kernel (tier3)**: guarded `const __nv_bfloat16* mmc_c0` param (impl + wrapper).
- **runtime**: `kernel` callable / `_prepare` take `aux=`, append the pointers after M,N,K.
- **API**: `mmc.matmul(a, b, epilogue=fn, aux=[c])` (arity + `[M,N]`/bf16/contig validated).
- **DSL**: `epilogue.to_torch` (the verify reference), n-ary `to_cuda`/`arity`/`n_inputs`.

Scope: **TMA-store/production route only** (overlap + persistent + TMA-pipelined). Other
store strategies (int4 store) would need a different splice — out of scope for now.

## Status by piece

| piece | status |
|---|---|
| DSL (n-ary trace/lower/to_torch) | done; CPU-tested |
| codegen + kernel + runtime + API plumbing | done |
| single-extra-input correctness | **verified** — `(a@b)*c`, `+c`, `silu(x)*c` match torch ~1.6e-3 (37 tests) |
| works on how many configs | **163 / 198** production configs (verified correct) |
| `mmc.matmul(..., aux=)` tuned-variant sweep | **blocked** by the crash below |
| runnable `aux` quickstart | **none yet** (would crash); benchmarked via `_build_epilogue` |

## The bug

**35 / 198** production configs `CUDA_ERROR_LAUNCH_FAILED` when the `c0` read is added
(the single-extra-input path). Findings:
- **Not an out-of-bounds in the `c0` read** — the address is provably in-bounds and
  16-byte aligned for every config (base 512-aligned; `gidx` always a multiple of 8).
- **No clean knob discriminator** — crashers span `tss1`/`tss2`, `ns` 4/5/6, all `nw`/`gsm`.
  So it's a **secondary kernel interaction** (scheduling/occupancy/a latent race that the
  extra load + registers expose), not a logic error a static checker can catch.
- **Non-deterministic** — the same config sometimes fails pre-launch (`INVALID_VALUE`,
  clean) and sometimes `LAUNCH_FAILED` (poisons the context).
- **Plain matmul is unaffected** — a clean plain production tune at the same shape runs
  **198/198, 0 skips** (best 1324 TFLOPS). So this is specific to the `c0`-read path, *not*
  a pre-existing plain-matmul bug. (An earlier hypothesis that it was pre-existing was a
  measurement artifact of single-process probes where one poison cascades.)
- **Why it blocks the sweep**: a `LAUNCH_FAILED` poisons the CUDA context for the whole
  process, so the in-process tuned-variant sweep — and any post-tune build/launch — dies.

Autotune hardening already landed (commit `815354c`): drop combos whose dynamic SMEM
exceeds the device opt-in cap; wrap the whole per-combo body in try/except (so a poison
truncates the sweep instead of crashing it). These help but don't fully fix it — a
poisoned context persists process-wide.

## Performance (measured)

Lower-level build path (`mmc._build_epilogue`) on a known-good config
(`bn256 ns4 nw8 gsm8 tss2`), FFN `32768×4608×768`, bf16, rel_err 1.66e-3:

| | time | TFLOPS |
|---|---|---|
| plain `a@b` | 0.224 ms | 1035 |
| **fused `(a@b)*c`** | **0.325 ms** | **715** |
| torch (`mm` then `*c`) | 0.304 ms | 763 |

`fused/torch = 1.07` (slightly slower), `fused/plain = 1.45`. **Multi-input fusion is
~parity, not a speed win** — unlike SiLU (pure compute, hidden behind the store, ~1.7×
faster), `(a@b)*c` must **read the full `[M,N]` extra input from GMEM** (~302 MB), which is
unavoidable memory traffic landing in the already-store-bound epilogue. So the value of
multi-input fusion is **one kernel / no intermediate tensor**, not throughput. (Single
config, not the tuned optimum — but the trend is structural.)

## Planned fixes (decision pending)

1. **Process-isolate the sweep** (recommended; "skip unlaunchable configs"): run the tune
   in a subprocess so a crash kills the child, not the parent; launchable configs stream to
   the disk cache; the parent reads the winner and builds it in a clean context. Robust to
   *any* crasher and hardens plain tuning too. The bigger change.
2. **Reuse-geometry for multi-input**: use the shape's plain winner + splice the epilogue,
   skipping the variant sweep. Avoids the crash; perf is ~parity anyway. Caveat: needs the
   plain winner to be launchable-with-epilogue.
3. **Root-cause** the 35-config crash with `compute-sanitizer` (memcheck/racecheck).

## Commits

`48c6561` (DSL n-ary) · `7e0f5b6` (single-extra-input kernel path) · `96fbf25` (docs) ·
`815354c` (autotune SMEM filter + full-body guard). Single-input epilogue (SiLU etc.) is
fully working and shipped (`examples/quickstart_epilogue.py`); this doc covers the
*multi-input* extension only.
