# Streaming K — the outer K-loop

Chapter 02's kernel fit the whole problem in a single SMEM tile.  That
only works while `K` is small enough that A and B fit in SMEM at once
— which in practice means `K ≤ 64` for our `BM = 128, BN = 256` setup.
Real matmuls have `K` in the thousands.  This chapter introduces the
one structural change that unlocks arbitrary `K`: the **outer K-loop**.

## The idea

The same SMEM tile is **reused** across K-iterations.  Each iteration:

1. TMA streams the *next* `(BM × BK)` slab of A and `(BN × BK)` slab of
   B into SMEM, *overwriting* the previous one.
2. The MMA reads from SMEM (4 back-to-back `tcgen05.mma` calls for
   `BK = 64`) and **accumulates into the same TMEM accumulator** that
   the previous K-iteration's MMAs left behind.

TMEM holds the running `(BM × BN)` FP32 sum across all K-iterations;
SMEM is just the per-stage scratch.  After `K / BK` iterations, TMEM
holds the final `C` tile and the epilogue writes it out — same epilogue
as chapter 02.

```
   K_global = K / BK iterations:
   ─────────────────────────────────────────────────────────────
     iter 0    TMA → SMEM  →  MMA (P = false; overwrites TMEM)
     iter 1    TMA → SMEM  →  MMA (P = true;  accumulates)
     iter 2    TMA → SMEM  →  MMA (P = true;  accumulates)
       ...
   iter K/BK-1 TMA → SMEM  →  MMA (P = true;  accumulates)
   ─────────────────────────────────────────────────────────────
     epilogue: tcgen05.ld TMEM → registers → GMEM
```

Three things change vs chapter 02; everything else is identical.

## Change 1 — phase alternates across iterations

The mbarrier's parity bit flips on every completion.  Chapter 02 had
exactly *one* completion per mbar (single shot), so we always passed
`phase = 0` to `try_wait.parity`.  In a K-loop, the mbarrier completes
*every iteration*, so its parity goes `0 → 1 → 0 → 1 → …`.  The
software side has to mirror this:

```cpp
const uint32_t phase = k_iter & 1;
mbarrier_wait_phase(tile_ready_mb, phase);
```

That's the *only* new thing about the mbarrier — same instructions,
same handshake, just the operand alternates.  Pass `phase = 0` on even
iterations, `phase = 1` on odd ones.  We covered the *why* of phase
parity in chapter 00 and the [Part 1 mbarrier chapter](../part1_gpu_arch/mbarrier);
this is the first chapter where you actually have to track it.

The same applies to `mma_done_mb` — also one completion per iteration,
also alternating phase.

## Change 2 — the accumulate predicate across iterations

In chapter 02, the K-loop was just the 4 inner MMA calls (`kk = 0..3`)
covering `K = 64` in one stage.  `P` was `false` only for `kk = 0`, and
`true` for `kk = 1, 2, 3`.

In chapter 03 the same logic extends *one level up*.  Now `P = false`
applies to **the very first MMA of the very first iteration** — `(k_iter
== 0, kk == 0)` — and **every other MMA, in every other iteration,
accumulates**:

```cpp
const bool first_ever = (k_iter == 0 && kk == 0);
tcgen05_mma(taddr, a_desc, b_desc, idesc, /*enable_d=*/ !first_ever);
```

The TMEM accumulator is initialized by that one `P = false` MMA at the
very start and then read-modify-written by every subsequent MMA — both
within an iteration (the 4 inner `kk` calls) and across iterations.
That's how the final TMEM sum captures `Σ_k A[:, k] · B[k, :]` across
the whole K dimension.

## Change 3 — TMA coordinate advances along K

The descriptor for A is built on the global `(M, K)` tensor.  In
chapter 02 we always loaded the box starting at `(coord_x=0,
coord_y=0)` because the whole problem fit in one tile.  In chapter 03
each iteration loads the box starting at `(coord_x = k_iter * BK,
coord_y = 0)` — same M-rows (we still own only the first BM = M rows
in this single-CTA kernel), advancing K-cols by `BK` per iter:

```cpp
tma_2d_load(A_BASE, &A_tmap, /*x=*/ k_iter * BK, /*y=*/ 0, tile_ready_mb);
tma_2d_load(B_BASE, &B_tmap, /*x=*/ k_iter * BK, /*y=*/ 0, tile_ready_mb);
```

`B` is presented to TMA as `B^T` (transposed on the host), so its
descriptor is built on `(N, K)` and the same `x = k_iter * BK` slides
the K-window — see chapter 02's host launcher for why.

## The full loop, structurally

Everything wrapped together — this is the entire main body of the
chapter-03 kernel:

```cpp
const int num_k_iters = K / BK;

for (int k_iter = 0; k_iter < num_k_iters; k_iter++) {
    const uint32_t phase = k_iter & 1;

    // 1) TMA the next K-tile (one thread issues).
    if (warp_id == 0 && elect_sync()) {
        tma_2d_load(A_BASE, &A_tmap, k_iter * BK, 0, tile_ready_mb);
        tma_2d_load(B_BASE, &B_tmap, k_iter * BK, 0, tile_ready_mb);
        mbarrier_arrive_expect_tx(tile_ready_mb, TILE_BYTES);
    }

    // 2) Wait for SMEM, publish to tcgen05 proxy.
    mbarrier_wait_phase(tile_ready_mb, phase);
    tcgen05_fence_after_thread_sync();

    // 3) 4 MMAs covering BK = 64 (K = 16 each).  The very first one
    //    (k_iter == 0 && kk == 0) overwrites TMEM; everything else
    //    accumulates.
    if (warp_id == 1 && elect_sync()) {
        #pragma unroll
        for (int kk = 0; kk < K_MMAS; kk++) {
            const uint64_t a_desc = make_desc(A_BASE + kk * 32);
            const uint64_t b_desc = make_desc(B_BASE + kk * 32);
            const bool first_ever = (k_iter == 0 && kk == 0);
            tcgen05_mma(taddr, a_desc, b_desc, idesc, !first_ever);
        }
        tcgen05_commit(mma_done_mb);
    }

    // 4) Wait for the MMAs to drain before TMA can overwrite SMEM.
    mbarrier_wait_phase(mma_done_mb, phase);
}
```

The epilogue (chapter 02's `tcgen05.ld` + GMEM store) follows
unchanged.

## Why this is *slow* — and what's next

Look at the loop carefully: step 4 (wait for MMAs to drain) **blocks
the next iteration's TMA** from starting.  And step 2 (wait for SMEM)
blocks the MMAs.  So TMA and MMA strictly alternate — only one of them
is active at any moment.

```
 time →
   TMA[0] ──► (idle)──► TMA[1] ──► (idle) ──► TMA[2] ──► (idle) ...
            MMA[0]──────────────  MMA[1] ──────────────  MMA[2] ...
```

The hardware is capable of overlapping them — TMA and the tensor cores
are entirely separate units — but our single-stage SMEM layout forces
the serialization: there's nowhere for TMA to write iter `k+1` while
MMA is still reading iter `k`.

**Chapter 04** fixes this by adding a second SMEM slot ("double
buffering"): TMA writes one slot while MMA reads the other, and the
two run concurrently.  That's the most impactful single optimization
in the whole ladder — it roughly doubles throughput.

## What this chapter doesn't do

To stay focused on the K-loop concept:

- **Single CTA.**  Grid is still `(1, 1, 1)`, so the kernel only
  computes one `(BM × BN)` tile of output.  Going to a full M×N output
  means launching a grid of CTAs and deriving each one's `(bid_m,
  bid_n)` from `blockIdx` — covered in the chapter on grid mapping.
- **Single SMEM stage.**  As just discussed — multi-stage buffering is
  chapter 04.
- **No warp specialization.**  The TMA-issuing warp and the
  MMA-issuing warp are different *warps* but they're both stalled
  waiting on the same mbar at any given moment.  True warp
  specialization (TMA warp and MMA warp running concurrently, never
  blocking each other) needs the pipelining infrastructure from
  chapter 04 to be useful.

## Take-away

The outer K-loop is the smallest possible structural change that lifts
the kernel from "one-tile demo" to "arbitrary K matmul."  Three pieces
of machinery were enough: alternating mbar phases, restricting `P =
false` to the very first MMA of the very first iteration, and
advancing the TMA coordinate by `BK` per iter.  The kernel is now
*correct* for any K that's a multiple of `BK`; it's just leaving
performance on the table by serializing TMA and MMA, which is the
next chapter's problem.

## Run

```bash
pip install -r ../requirements.txt
python main.py
```

A Blackwell GPU (sm_100a / B200) is required.
