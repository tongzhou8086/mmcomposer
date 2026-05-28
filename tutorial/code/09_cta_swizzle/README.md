# CTA swizzling — chunked grid walk for L2 reuse

> 📁 **Code on GitHub:** [`tutorial/code/09_cta_swizzle/`](https://github.com/tongzhou8086/mmcomposer/tree/master/tutorial/code/09_cta_swizzle) — `kernel.cu` + `main.py`.

Ch05 introduced the grid mapping in the simplest possible form:

```cpp
const int bid_m = cluster_id / grid_n;     // outer
const int bid_n = cluster_id % grid_n;     // inner — sweeps fast
```

CTAs sweep across all of N for the first M-row, then advance to the
next M-row, and so on.  Every chapter since has kept that walk.  It's
correct, but it leaves a lot of L2 cache on the table.

## The L2 reuse problem with the naïve walk

For a given CTA at `(bid_m, bid_n)`:

- It reads A's **M-stripe `bid_m`**.  Every CTA in the same M-row
  shares this stripe.
- It reads B's **N-stripe `bid_n`**.  Every CTA in the same N-column
  shares this stripe.

With the default walk, *adjacent* CTAs (consecutive `cluster_id`) have
the same `bid_m` but different `bid_n` — so they share an A-stripe but
not a B-stripe.  As the launch sweeps across one M-row:

- A's M-stripe `bid_m` stays hot in L2 (`grid_n` consecutive CTAs all
  want it).  ✓
- B's N-stripes get **streamed once**, no reuse — each CTA touches a
  fresh slice.  ✗

When the launch finishes one M-row and moves to the next, A's stripe
changes and B starts streaming again from `n = 0`.  Across the whole
launch, **B is read M-stripes times** instead of once.  That's a lot
of redundant HBM traffic at scale.

## The Triton-style chunked walk

Re-pack the grid order so consecutive CTAs share a **B-stripe**
instead of an A-stripe.  Inside each "group" of `GROUP_SIZE_M × grid_n`
cluster IDs, walk M fast and N slow:

```
   chunk 0:  (m=0,n=0) (m=1,n=0) … (m=GSM-1,n=0)        ← GSM CTAs share B's n=0 stripe
             (m=0,n=1) (m=1,n=1) … (m=GSM-1,n=1)        ← GSM CTAs share B's n=1 stripe
             …
             (m=0,n=grid_n-1) … (m=GSM-1,n=grid_n-1)
   chunk 1:  (m=GSM,n=0) … (m=2·GSM-1,n=0)
             …
```

Reading off the reuse story:

- **B reuse**: `GSM` consecutive CTAs share the same B-stripe.  B is
  hot in L2 for `GSM × ~(1 MMA-stage time)` instead of getting evicted
  immediately.
- **A working set**: `GSM` M-stripes are *all* in flight at once.
  They cycle as you advance `n` through the group — visited
  `grid_n` times within one group before moving on.

`GROUP_SIZE_M` is the tunable knob.  Larger `GSM` → more B reuse but
larger A working set in L2.  Smaller `GSM` → less B reuse, but A
fits more easily.  `GSM = 1` recovers exactly chapter 08's walk.

## Code change — a few lines in the per-CTA setup

The entire change vs ch08 is the `(cluster_m, cluster_n)` derivation
at the top of the kernel.  Everything else — the warp-specialized
TMA/MMA loops, the multi-stage ring, the cluster MMA, the epilogue —
is **bit-identical** to ch08.

```cpp
template <int NS, int GROUP_SIZE_M>
__device__ __forceinline__ void matmul_swizzle_impl(...) {
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    const int cluster_id        = blockIdx.x / CTA_GROUP;
    const int grid_n            = N / BN;
    const int grid_m_clusters   = M / (CTA_GROUP * BM);

    // Triton-style chunked walk: M-fast inside groups of size
    // GROUP_SIZE_M × grid_n.  `gsm` shrinks for the (possibly ragged)
    // last group so we stay inside the M-grid.
    const int num_cluster_in_group = GROUP_SIZE_M * grid_n;
    const int group_id             = cluster_id / num_cluster_in_group;
    const int first_cluster_m      = group_id * GROUP_SIZE_M;
    const int gsm                  = min(grid_m_clusters - first_cluster_m,
                                         GROUP_SIZE_M);
    const int cluster_m            = first_cluster_m + (cluster_id % gsm);
    const int cluster_n            = (cluster_id % num_cluster_in_group) / gsm;

    const int off_m_cluster = cluster_m * (CTA_GROUP * BM);
    const int off_n         = cluster_n * BN;
    const int off_m_local   = off_m_cluster + cta_rank * BM;
    const int off_n_local   = off_n + cta_rank * BN_LOCAL;
    // ... rest of the kernel is ch08's, verbatim
}
```

That's it.  No new PTX instructions, no new layouts, no new
mbarriers — just a different formula for "where in the output grid am
I?"

## Composing with the 2-CTA cluster — the unit of grouping

A point worth being precise about: the swizzle groups *clusters*, not
individual CTAs.  In ch08 each cluster already owns `CTA_GROUP × BM =
2 × 128 = 256` M-rows of output.  When we say `GROUP_SIZE_M = 8`, we
mean 8 *cluster-rows* per chunk → **`8 × 256 = 2048` M-rows of
output per chunk** (= 16 BM-tiles in M).

| `GSM` | M-rows per chunk | A working set per chunk @ K = 8192 |
|---|---|---|
| 1  |  256 (one cluster row)  |  4 MB |
| 4  | 1024                    | 16 MB |
| **8**  | **2048 = 16 BM-tiles**  | **32 MB** |
| 16 | 4096                    | 64 MB |

For our 8192³ headline run, GSM=8 chunks the M-dimension into
groups of 2048 rows (16 BM-tiles, or 8 cluster-tiles), giving a 32 MB
A working set per chunk — comfortably inside L2 — while still leaving
~100 MB of L2 for B to stay hot across the `grid_n` N-sweeps inside
the chunk.  That's the sweet spot for this geometry; `GSM = 16` would
push A's working set to 64 MB and start crowding B out.

So when you read `GROUP_SIZE_M = N` in the kernel, the practical
meaning is **"how many BM-tiles in M do we serialize before
advancing N"** is `2 · N`, not `N` — the factor of 2 comes for free
from the cluster.

## Sanity check — does `GSM = 1` recover ch08?

Yes.  With `GROUP_SIZE_M = 1`:

```
   num_cluster_in_group = 1 * grid_n = grid_n
   group_id             = cluster_id / grid_n
   gsm                  = 1   (assuming non-ragged)
   cluster_m            = group_id + cluster_id % 1 = group_id = cluster_id / grid_n
   cluster_n            = (cluster_id % grid_n) / 1 = cluster_id % grid_n
```

Exactly the same formulas as ch08's `bid_m = cluster_id / grid_n`,
`bid_n = cluster_id % grid_n`.  So `GSM = 1` is the natural
"no-swizzle" baseline.

## Performance — `GSM` sweep at one shape, fixed `NS`

To isolate the swizzle effect we hold everything else constant:

- **Problem**: `M = N = K = 8192`.  At this size,
  `B = K × N × 2 = 128 MB`, which is roughly **B200's L2 capacity
  (132 MB)** — so the L2 can't naturally hold B across the launch.
  *That* is when re-ordering for B-reuse pays off.  (Smaller shapes
  are addressed below.)
- **`NS = 5`**: a healthy stage depth from ch08's sweep at this shape
  — close enough to optimal that the GSM contribution is what we're
  measuring.

Measured on B200:

| `GSM` | walk pattern | TFLOPS |
|---|---|---|
| **1** (= ch08's walk) | N-fast within each M-row              | **1178** |
| 4                     | M-fast within chunks of 4 cluster-rows | 1219 (+3.5 %) |
| **8**                 | M-fast within chunks of 8 cluster-rows | **1225** (+4 %) |
| 16                    | M-fast within chunks of 16 cluster-rows | 1199 (+1.8 %) |

`GSM = 8` is the sweet spot at 8192³: enough M-stripes in flight per
chunk to give B genuine L2 reuse, not so many that A's working set
spills the L2.  `GSM = 16` is past the peak — the extra A working set
costs more than the extra B reuse buys.

Against cuBLAS at the same shape: ch09 best = 1225 / cuBLAS ≈ 1264 =
**~97 %**.  Very close to library parity at this scale.

### When the swizzle *doesn't* help — smaller shapes

A surprise the first time you measure it: at `M = N = K = 4096`,
the same kernel makes things slightly *worse* with `GSM > 1`.  The
reason is the L2 working-set math:

| shape | B total (`K·N·2`) | fits in 132 MB L2? |
|---|---|---|
| 2048³ | 8 MB    | trivially |
| 4096³ | 32 MB   | comfortably |
| **8192³** | **128 MB**  | **right at capacity** |

At 2K and 4K, **B is already small enough that the L2 holds it
across the launch** regardless of walk order — the swizzle has
nothing to rescue, and you pay a tiny cost for the extra address math
and the less-natural traversal.  Only when the L2 budget starts to
bind does re-ordering pay back.

This is the headline lesson of the chapter:

> **CTA swizzling is an *L2-pressure-dependent* optimization.**  It's
> a free knob when L2 is the bottleneck and a small tax otherwise.
> Picking the right `GSM` per shape is exactly the kind of thing an
> autotuner does — and is what the next chapter is about.


So this is the last *concept-level* optimization in the ladder.  The
next chapter pulls the tuning knobs together into an actual autotuner.

## Run

```bash
pip install -r ../requirements.txt
python main.py
```

A Blackwell GPU (sm_100a / B200) is required.
