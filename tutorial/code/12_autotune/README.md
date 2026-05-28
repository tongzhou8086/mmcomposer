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

## A pruning lesson — equivalent configs are noise

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

## L2 invalidation between timed batches

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

The same `invalidate_l2()` is called before each PyTorch / cuBLAS
batch for an apples-to-apples baseline.

In practice this dropped one of our cells (8192³) from a previously
measured 98 % → 91 %.  The earlier number wasn't a lie — it was a
real measurement of a different scenario.  The L2-flushed result is
the one to trust for "what happens when this kernel is called once
after a long gap"; the warm-cache result is the one to trust for
"called back-to-back in a tight loop."  Both are valid; the autotuner
should commit to one and apply it uniformly, which is what flushing
does.

## A pruning lesson — equivalent configs are noise

Autotuning is finicky when the gaps between variants are a few
percent.  Arithmetic mean across N launches mixes one slow outlier
into the answer and can flip the ranking.  We use **median over
several timed batches** instead:

```python
def time_median(kern, ..., n_batches=5, iters=5):
    times = []
    for _ in range(n_batches):
        start = torch.cuda.Event(...); end = torch.cuda.Event(...)
        start.record()
        for _ in range(iters):
            launch(kern, ...)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / iters * 1e3)
    times.sort()
    return times[len(times) // 2]
```

Each "batch" is a small group of launches we average into one sample;
we collect several samples and pick the median.  Cheap to run, more
stable than averaging everything together.  This kind of detail is
boring until you don't do it — then you spend half a day chasing
"why does the autotuner change its mind between runs."

## Per-shape results

Sweep `M = N = K ∈ {2048, 3072, …, 10240}` (9 shapes).  Measured on
B200; PyTorch matmul as the cuBLAS baseline.

| shape | tune | best (NS, GSM, NW, LDX) | **ours TFLOPS** | cuBLAS | ratio |
|---|---|---|---|---|---|
| 2048³  |  0.4s | (5,  1, 8, 32)         | **646**  | 1216  | 53 % |
| 3072³  |  0.7s | (4,  4, 8, 16)         | **1131** | 1669  | 68 % |
| 4096³  |  2.1s | (7, 16, 8, 16)         | **1164** | 1535  | 76 % |
| 5120³  |  4.0s | (5,  1, 4, 16)         | **1182** | 1475  | 80 % |
| 6144³  |  6.5s | (5,  1, 4,  8)         | **1223** | 1387  | 88 % |
| 7168³  | 10.1s | (6,  8, 8,  8)         | **1303** | 1362  | 96 % |
| 8192³  | 14.9s | (5,  8, 4, 16)         | **1312** | 1442  | 91 % |
| 9216³  | 21.3s | (6,  8, 8, 64)         | **1313** | 1373  | 96 % |
| 10240³ | 28.9s | (7,  8, 4,  8)         | **1323** | 1368  | **97 %** |

A few things worth reading off:

- **Different shapes pick different configs.**  No single (NS, GSM, NW,
  LDX) wins everywhere; the autotuner consistently picks something
  different per shape.  That's the whole point — fixed-config kernels
  leave 5–15 % on the table at off-design shapes.
- **`NUM_WARPS = 8` wins at small/mid shapes; 4 wins at very large.**
  Smaller shapes are more epilogue-sensitive (the K-loop is shorter),
  so the extra epilogue parallelism pays.  Very large shapes are
  K-loop-bound, where extra warps cost registers without buying
  throughput.
- **`GSM = 8` dominates the middle range** (5K – 10K), where B's L2
  working set straddles capacity.  Below that range, L2 absorbs B
  naturally so `GSM = 1` or `16` is fine; above it, the chunk size
  is constrained by the M-dim aspect ratio.
- **Approach to cuBLAS** climbs monotonically from 52 % at 2048³ to
  ~96 % at 10K — the kernel was tuned around 8K and small shapes are
  inherently harder (less compute to amortize fixed costs).

## Cost & limitations

- **First-call cost is non-trivial.**  Tuning at 10240³ takes ~8 s
  because each timing samples non-trivially-long calls.  Smaller
  shapes finish in <1 s.  Production autotuners reduce this by
  pruning the search space (e.g. skipping configs known to be slow
  from an offline sweep), or by persisting the cache to disk.
- **We only swept 2K → 10K.**  Larger shapes (12K–16K) would require
  more HBM and more time to time properly.  Easy extension —
  one-line change to `SHAPES = range(2048, 16384 + 1, 1024)`.
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

That's ~96 % of cuBLAS on a B200 — the remaining ~4 % is the gap
between this tutorial's clean focus and the SASS-level micro-tuning
that lives in production libraries.

## Run

```bash
pip install -r ../requirements.txt
python main.py
```

A Blackwell GPU (sm_100a / B200) is required.  First run will compile
~160 kernel variants (a few minutes); subsequent runs reuse the
cubin cache and start instantly.
