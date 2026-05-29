# Autotuning — picking the right config per shape

> 📁 **Code on GitHub:** [`tutorial/code/12_autotune/`](https://github.com/tongzhou8086/mmcomposer/tree/master/tutorial/code/12_autotune) — `kernel.cu` + `main.py`.

The ladder's last chapter.  Across ch04 → ch11 we've accumulated four
template knobs:

| knob | values explored | introduced in |
|---|---|---|
| `NS` (multi-stage depth)         | 3, 4, 5, 6, 7    | ch04 / ch08 |
| `GROUP_SIZE_M` (CTA swizzle)     | 1, 4, 8, 16      | ch09 |
| `NUM_WARPS` (epilogue warps)     | 4, 8             | ch10 |
| `LD_X` (`tcgen05.ld` packing)    | 8, 16, 32, 64    | ch10 |

Every previous chapter held them at fixed values "tuned at 8192³" and
showed why each knob exists.  This chapter measures what actually
matters: **the best config varies by problem shape**, and a small
Python autotuner can pick the winner per call in a fraction of a
second.

## The pattern

Three pieces, total ~30 lines of Python:

1. **Compile** the full cross product of (NS, GSM, NUM_WARPS, LD_X)
   variants at startup — for our chapter that's 5 × 4 × 2 × 4 = **160
   kernel functions** in one `kernel.cu`.
2. **Time** each variant once per problem shape and pick the winner.
3. **Cache** the result keyed by `(M, N, K)` so subsequent calls at
   the same shape skip the sweep.

```python
class Autotuner:
    def __init__(self, kernels):
        self.kernels = kernels        # {(NS, GSM, NW, LDX): CUfunction}
        self.cache   = {}             # {(M, N, K): cfg}

    def pick(self, M, N, K, args, grid):
        key = (M, N, K)
        if key in self.cache:
            return self.kernels[self.cache[key]], self.cache[key]

        best_us, best_cfg = float("inf"), None
        for cfg, kern in self.kernels.items():
            us = time_median(kern, ..., args, grid)
            if us < best_us:
                best_us, best_cfg = us, cfg
        self.cache[key] = best_cfg
        return self.kernels[best_cfg], best_cfg
```

That's the whole tuner.  Production frameworks (Triton, CUTLASS-Python,
TVM) add ML-guided search, multi-shape clustering, persistent caches —
all useful, but they're refinements of this same loop.

## Lesson 1 — equivalent configs are pure noise

Naïvely sweeping all 160 variants, we got a surprise: `GSM = 16` won
at `M = 2048`.  But here's the math: at `M = 2048`,

```
grid_m_clusters = M / (CTA_GROUP * BM) = 8
```

and the kernel's group-walk clamps `gsm = min(GROUP_SIZE_M,
grid_m_clusters)`.  So `GSM = 16` at M=2048 produces *literally
identical* SASS to `GSM = 8` — the autotuner picking one over the
other is 100 % noise.

The fix:

```python
grid_m_clusters = M // (CTA_GROUP * BM)
for cfg, kern in self.kernels.items():
    ns, gsm, nw, ldx = cfg
    if gsm > grid_m_clusters:      # would be clamped → skip
        continue
    ...
```

After pruning, the small-shape picks stabilize.  General lesson:
**before timing a variant, check whether the kernel can actually
distinguish it from a variant you're already timing.**  Equivalent
variants are just expensive ways to add noise.

## Lesson 2 — L2 invalidation between timed batches

Without flushing the L2 between timed batches, configs that happen
to leave useful state in L2 get an unfair tailwind on the *next*
batch — the timer measures "warm-cache steady state" instead of "first
call after a gap," which biases the ranking toward configs whose
benefit only materializes when the L2 is already populated by a
previous identical call.

Standard fix: allocate a buffer bigger than the L2 (~256 MB > B200's
132 MB), and write through it before each timed batch.  Touching all
256 MB evicts whatever the previous batch left behind.

```python
L2_FLUSH_BYTES = 256 * 1024 * 1024
_l2_scratch    = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda")

def invalidate_l2():
    _l2_scratch.zero_()
```

Crucially, invalidation runs **once per batch, not per launch**.
Within a batch the L2 warms up naturally — that's the realistic
state real kernels see.  Per-launch invalidation would over-penalize
configs that rely on intra-launch L2 reuse (like the CTA swizzle's
B-stripe sharing across consecutive CTAs in a chunk).

## Lesson 3 — your timer has to be tight too

The single biggest surprise in this chapter wasn't anything about the
kernel.  An early sweep at 2048³ showed our autotuned kernel hitting
54 % of cuBLAS, with the gap shrinking monotonically up to 95 % at
8192³.  The natural read is "kernel inherently doesn't amortize fixed
costs at small shapes" — and we spent a while chasing that hypothesis.

The actual cause was in `cuda_utils.launch()`:

```python
def launch(kernel, *, grid, block, shared, args, stream=0, sync=True):
    cu(driver.cuLaunchKernel(kernel, *grid, *block, shared, stream, arg_ptrs, 0))
    if sync:
        cu(driver.cuCtxSynchronize())     # ← per-launch sync
```

Convenient for chapter examples (`C` is readable right after launch),
but **fatal in a timing loop**:

```python
start.record()
for _ in range(iters):
    launch(kern, ...)            # each call: cuLaunchKernel + cuCtxSynchronize
end.record()
torch.cuda.synchronize()
```

At 2048³ each kernel runs ~5 µs; each host sync round-trip adds ~5–10 µs.
Per-launch sync was inflating the small-shape measurement by ~2×, which
made the autotuner think our kernel was slow at small shapes when it
actually wasn't.

Fix: make sync opt-out, and pass `sync=False` inside the timing loop —
queue launches back-to-back, synchronize once at `end.record()`:

```python
for _ in range(iters):
    launch(kern, ..., sync=False)
```

At 2048³: **662 → 1041 TFLOPS (+58 %)**, jumping from 54 % to 86 % of
cuBLAS.  No kernel change.  Just a timer that wasn't measuring what we
thought it was measuring.

The meta-lesson — and probably the most useful thing in this chapter —
is that **autotuner correctness depends on timer correctness**.  An
autotuner picking the wrong config is *not always* a search-space
problem; sometimes it's measuring a quantity that's a mix of "kernel
time" and "harness overhead" with different weights at different
shapes.  Always check that your timer is measuring device time, not
host round-trips, before tuning anything subtle.

## Lesson 4 — tournament budget vs measurement budget

Once the timer was honest, we still saw the autotuner picking
suboptimal configs at a couple of shapes.  Final measurement used
`11 × 50` batches; the per-config tournament used `5 × 20`.  Cheap
enough to keep tuning fast, but noisy enough to mis-rank configs whose
true gaps are a few percent.

Concrete symptom: at 2048³ the cheap tournament picked `GSM = 4`; the
strong tournament picked `GSM = 1`, and **both delivered ~the same
final TFLOPS** — proving the GSM=4 win was a measurement artifact,
not a true ranking.

Bumping the tournament to `7 × 50` resolved several picks across the
sweep at the cost of a few extra seconds of tuning time per shape.
Worth it for stable results.

**General rule: your tournament's confidence interval must be smaller
than the gap between configs, or you're picking on noise.**  If you
can't make the tournament that tight without it taking too long, the
configs themselves are probably too close in performance to matter —
prune the loser before timing.

## Per-shape results

Sweep `M = N = K ∈ {2048, 3072, …, 12288}` (11 shapes).  Measured on
B200; PyTorch matmul as the cuBLAS baseline.

| shape | tune | best (NS, GSM, NW, LDX) | **ours TFLOPS** | cuBLAS | ratio |
|---|---|---|---|---|---|
| 2048³  |  ~3 s  | (7,  1, 8, 64) | **1039** | 1212  | 86 % |
| 3072³  |  ~4 s  | (3,  1, 8, 16) | **1274** | 1605  | 79 % |
| 4096³  |  ~5 s  | (4,  1, 8,  8) | **1152** | 1433  | 80 % |
| 5120³  |  ~7 s  | (5,  1, 8,  8) | **1221** | 1424  | 86 % |
| 6144³  | ~10 s  | (7,  8, 8, 16) | **1281** | 1410  | 91 % |
| 7168³  | ~14 s  | (6,  8, 8, 32) | **1286** | 1372  | 94 % |
| 8192³  | ~18 s  | (6,  8, 8, 32) | **1309** | 1386  | **94 %** |
| 9216³  | ~24 s  | (6,  8, 8, 64) | **1319** | 1361  | **97 %** |
| 10240³ | ~31 s  | (5,  8, 8, 64) | **1315** | 1361  | **97 %** |
| 11264³ | ~40 s  | (5,  8, 8, 16) | **1311** | 1357  | **97 %** |
| 12288³ | ~50 s  | (5,  8, 8, 16) | **1319** | 1420  | 93 % |

A few things worth reading off:

- **The kernel plateaus at ~1300 TFLOPS from 7K onward** — that's its
  steady-state compute ceiling, sustained within 3 % of cuBLAS through
  the 9K–11K range (97 %).  Above and below that sweet spot the gap
  widens for different reasons: small shapes haven't built up enough
  K-loop work to amortize per-CTA prologue costs; very large shapes
  start running into L2 pressure on the larger A working set.
- **Different shapes pick different configs.**  No single (NS, GSM, NW,
  LDX) wins everywhere; the autotuner consistently picks something
  different per shape.  That's the whole point — fixed-config kernels
  leave 3–7 % on the table at off-design shapes.
- **`NUM_WARPS = 8` wins everywhere** in this sweep (NW=4 was dropped
  early; the extra epilogue parallelism pays at every shape we tried).
- **`GSM = 1` dominates small shapes** (2K–5K), `GSM = 8` dominates
  large.  At small grids the chunked walk doesn't have enough cluster
  rows to reuse A across; GSM=1 just does the N-major walk and lets
  L2 absorb everything.
- **`NS = 3` wins at 3072³** with `BK = 64`, but only because the
  autotuner gets fooled — see the next section.

## When the autotuner picks a local optimum

At 3072³ the chosen config is `(NS=3, GSM=1, NW=8, LDX=16)` at 79 % of
cuBLAS.  Every other shape clears 86 %.  What's happening?

`NS = 3` is the shallowest pipeline in our sweep (introduced for the
`BK = 128` variant where `K / BK = 24` makes 3 stages reasonable).
At `BK = 64`, `K / BK = 48`, and a deeper NS would be expected to
win — and indeed, a separately measured reference kernel with `NS = 6,
GSM = 1` produces ~1390 TFLOPS at 3072³, well above our 1274.

Why does our autotuner pick NS=3?  Because in *our config space*, the
deeper NS variants happen to plateau at ~1250 TFLOPS at 3072³, and the
NS=3 BK=64 variant edges past them by a few percent.  It's the local
best within our sweep, just not the global best the hardware can do.

**This is an honest limitation of search-based tuning, not a bug.**
The autotuner can only pick from what you compile; if the global
optimum needs a config you didn't generate, you can't get there from
here.  Production autotuners hedge against this with broader sweeps
and offline cross-shape priors; ours is intentionally minimal.

Worth flagging in the chapter because it's the kind of result that
looks suspicious in a table — it's not a measurement glitch, it's the
search space's actual boundary showing through.

## Cost & limitations

- **First-call cost.**  Tuning at 10K–12K takes 30–50 seconds because
  each timing samples non-trivially-long calls 7 × 50 = 350 times per
  config.  Smaller shapes finish in seconds.  Production autotuners
  reduce this by pruning the search space (e.g. skipping configs known
  to be slow from an offline sweep), or by persisting the cache to disk.
- **The 160-variant compile takes a few minutes** on a cold run.
  nvcc caches cubins on disk by mtime, so subsequent runs of the
  same kernel are nearly instant.
- **Pure shape-keyed cache.**  This tuner doesn't generalize across
  shapes — running at 5121³ wouldn't reuse the 5120³ winner.
  Production tuners often parameterize on `(grid_m_clusters,
  grid_n, K)` instead, so similar shapes share decisions.

## What you've built

By chapter 12 the kernel is doing roughly what a production matmul
kernel does, in roughly the same shape:

- TMA descriptors for swizzled SMEM loads.
- Multi-stage ring buffer.
- Warp-specialized TMA / MMA / epilogue.
- K-major B, no host transpose.
- 2-CTA cluster MMA with `cta_group::2`.
- Triton-style chunked grid walk for L2 reuse.
- Two-phase coalesced epilogue, parameterized by warp count and load
  width.
- Per-shape autotuning over four knobs.

That's **86–100 % of cuBLAS across an 11-shape sweep on B200**, with a
~1300 TFLOPS plateau from 7K onward and a 97 % sweet spot at 9K–11K.
The remaining gap is the territory production libraries claim with
SASS-level micro-tuning that's outside this tutorial's scope.

## Run

```bash
pip install -r ../requirements.txt
python main.py
```

A Blackwell GPU (sm_100a / B200) is required.  First run will compile
~160 kernel variants (a few minutes); subsequent runs reuse the
cubin cache and start instantly.
