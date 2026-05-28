# Hoisting descriptor builds above the mbarrier wait

> 📁 **Code on GitHub:** [`tutorial/code/11_hoist_desc/`](https://github.com/tongzhou8086/mmcomposer/tree/master/tutorial/code/11_hoist_desc) — `kernel.cu` + `main.py`.

The MMA warp in ch04–ch09 follows this structure inside the K-loop:

```cpp
mbarrier_wait_phase(tile_ready[slot], ...);   // spin until TMA done
tcgen05_fence_after_thread_sync();
for (int kk = 0; kk < K_MMAS; kk++) {
    // build descriptors using `slot`-derived addresses ...
    tcgen05_mma_g2(taddr, a_desc, b_desc, idesc, ...);
}
tcgen05_commit_mcast_g2(...);
```

Look carefully: **the descriptor math only depends on `slot`**, which
is known at the top of the K-iter, *before* the wait.  It doesn't
depend on whether the TMA bytes have actually landed — just on which
SMEM slot the MMA is going to read.  Yet we compute it *after* the
wait, serializing it with both the wait's spin and the MMA issue.

This chapter hoists those descriptor builds **above** the wait:

```cpp
// Build all K_MMAS descriptors first, into local arrays.
uint64_t a_desc[K_MMAS], b_desc[K_MMAS];
#pragma unroll
for (int kk = 0; kk < K_MMAS; kk++) {
    a_desc[kk] = make_desc(A_base(slot) + kk * MMA_K * BF16_BYTES);
    b_desc[kk] = make_desc_K_major(B_base(slot) + kk * MMA_K * SWIZZLE_ROW_BYTES,
                                    BK * SWIZZLE_ROW_BYTES);
}

// Now wait — the descriptor math has either already run or runs
// concurrently with the spin.
mbarrier_wait_phase(tile_ready[slot], ...);
tcgen05_fence_after_thread_sync();

// Issue all MMAs back-to-back from the pre-built arrays.
#pragma unroll
for (int kk = 0; kk < K_MMAS; kk++) {
    tcgen05_mma_g2(taddr, a_desc[kk], b_desc[kk], idesc, ...);
}
tcgen05_commit_mcast_g2(...);
```

Two effects come together:

1. **Latency hiding through the wait.**  The wait is implemented as a
   spin-loop on `mbarrier.try_wait.parity`.  Each iteration of the
   spin is a few register ops; the warp scheduler can co-issue the
   descriptor-build arithmetic in the gaps, hiding most of it.
2. **Tighter MMA issue burst.**  With descriptors pre-built, the
   second loop fires K_MMAS=4 `tcgen05.mma` instructions
   back-to-back with no descriptor-arithmetic stalls between them.
   `tcgen05.mma` is itself async-issue, so the four issues happen in
   tight succession and the tensor-core pipeline sees a clean burst.

This is the canonical "rearrange the source so the compiler can do
the right thing" micro-opt — `nvcc` won't move the descriptor
arithmetic across a memory-fence-shaped barrier on its own, but the
data dependence makes it safe and we can do it by hand.

The change is purely a code reordering of the MMA warp's K-loop body.
Nothing else moves — same kernel structure, same SMEM layout, same
descriptors, same correctness invariants.  The TMA warp is untouched.

## What `nvcc` can do automatically (and what it can't)

Modern compilers will:

- Inline the helpers (`make_desc`, lambdas).
- Schedule independent instructions for ILP.
- Fold constants and unroll `#pragma unroll` loops.

What `nvcc` won't do, because it would change the memory-ordering
semantics:

- **Move code across `mbarrier.try_wait.parity`.**  The compiler sees
  the wait as an opaque memory-ordering instruction (the `"memory"`
  clobber in the inline-asm tells it so).  Even though the
  descriptor builds don't *actually* depend on anything the wait
  guards, the compiler conservatively keeps everything in source
  order around the barrier.

So we move the code by hand — which is safe because we know the
real dependency (`slot` is set at the top of the iter, not by the
TMA).

## Performance

Same shape as ch09 (`M = N = K = 8192`, `NS = 5`), head-to-head against
ch09 at every `GSM`.  Measured on B200:

| `GSM` | ch09 TFLOPS | ch11 TFLOPS | speedup |
|---|---|---|---|
| 1  | 1238 | 1195 | 0.97× |
| 4  | 1209 | 1249 | 1.03× |
| 8  | 1287 | 1288 | 1.00× |
| 16 | 1267 | 1264 | 1.00× |

**Honest read — within run-to-run noise.**  Mixed sub-3 % deltas, no
consistent sign.  Two plausible reasons:

1. The K-loop at 8192³ is already so well-pipelined that the descriptor
   builds were *already* mostly hidden — the warp scheduler may have
   been co-issuing them with the wait spin even without our help.
2. Modern GPU schedulers tolerate short arithmetic chains well; the
   win from rearranging is bounded by how much latency was actually
   visible to begin with.

The *principle* is still correct, and on kernels with longer waits or
heavier pre-wait work the win is measurable.  On *this* kernel at
*this* shape we don't have enough margin to show it cleanly — we've
only swept a single shape and a single `NS`, so this is more "no
measurable signal yet" than "definitely doesn't help."  We keep the
chapter for the pattern; future revisions may remove or replace it if
broader measurements support a different conclusion.

## Why this is a *pattern*, not just a one-off

This is one instance of a general rule: **whenever a piece of work
in your kernel doesn't depend on a particular synchronization, look
at whether it can run *before* the sync instead of after.**
Production kernels (CUTLASS, gau-nernst, the b42 lineage) do this in
several places:

- Pre-built mbar addresses cached above wait loops.
- Predicate computations for the next iter computed during the
  current iter's wait.
- Loop-induction-variable updates pulled into the wait shadow.

`tcgen05.mma`-based kernels in particular benefit because the wait
is non-trivial (the spin reads a cluster-scoped mbar) and the
descriptor math is non-trivial enough to be worth hiding.

## Run

```bash
pip install -r ../requirements.txt
python main.py
```

A Blackwell GPU (sm_100a / B200) is required.
