# Multi-stage buffering — overlapping TMA and MMA

> 📁 **Code on GitHub:** [`tutorial/code/04_multi_stage/`](https://github.com/tongzhou8086/mmcomposer/tree/master/tutorial/code/04_multi_stage) — `kernel.cu` + `main.py`.

Chapter 03 worked, but each K-iteration is dead time for the other
unit: the MMA sits idle while TMA fetches the next tile, then TMA sits
idle while the MMA chews through it.  The hardware *can* run them in
parallel — they're entirely separate engines — but the kernel held them
in lock-step because there's only one SMEM tile to share.

This chapter introduces **multi-stage buffering**: a small ring of
SMEM tiles (`NUM_STAGES = 2` here) so TMA can write the *next* slot
while MMA reads the *current* one.  The two units overlap.  This is
the single highest-impact optimization in the ladder — roughly **2×
throughput** in our measurements below.

```
   chapter 03 (1 slot):     │TMA│       │TMA│       │TMA│
                                    │MMA│      │MMA│      │MMA│
                                                            (only one active at a time)

   chapter 04 (2 slots):    │TMA0│TMA1│TMA0│TMA1│TMA0│
                                  │MMA0│MMA1│MMA0│MMA1│MMA0│
                                  (TMA and MMA pipelined, both busy)
```

The whole change is structural — same `tcgen05.mma`, same descriptors,
same epilogue.  What changes is the dependency graph between TMA and
MMA, and the kernel layout that expresses it.

## Three changes

### 1. SMEM and mbarriers become arrays of `NUM_STAGES`

One slot per pipeline stage, one mbar pair per slot.  A single CTA
still owns one output tile, so the TMEM allocation is unchanged:

```cpp
constexpr int NS = 2;

extern __shared__ __align__(1024) char smem[];
__shared__ uint64_t tile_ready[NS], mma_done[NS];
```

The slot for iteration `k_iter` is `k_iter % NS` — classic ring
buffer.

### 2. TMA and MMA run as independent warps with their own loops

In chapter 03 every iteration was `TMA → wait → fence → MMA → wait`,
and *all threads* waited on each mbar — the all-threads-wait line is
what forced TMA and MMA to alternate.  Multi-stage drops the
shared-loop structure entirely: warp 0 has its own TMA loop, warp 1
has its own MMA loop, they sync only at the mbarrier pairs.

```cpp
if (warp_id == 0 && elect_sync()) {
    // TMA warp — its own K-loop
    for (int k = ...; ...; ...) {
        wait mma_done[slot]   // is the slot free?
        LOAD into slot
    }
} else if (warp_id == 1 && elect_sync()) {
    // MMA warp — its own K-loop
    for (int k = ...; ...; ...) {
        wait tile_ready[slot]   // is the slot full?
        fence
        4 × tcgen05.mma into TMEM
        tcgen05.commit → mma_done[slot]
    }
}
```

The producer waits for "slot is free" before overwriting; the consumer
waits for "slot is full" before reading.  Classic single-producer /
single-consumer ring buffer.

### 3. Prologue + steady-state + epilogue drain

The TMA warp's first iteration shouldn't have to wait for an `mma_done`
that hasn't been signalled yet.  Standard trick: **pre-arrive
`mma_done[NS-1]` once at init**, then run a tiny prologue that
front-loads the first `NS - 1` tiles, then enter the steady state where
the per-iter wait actually has something to wait on.

```cpp
// pre-arrive once: arms mma_done[NS-1] so iter 0's wait returns
// immediately even though no MMA has actually fired yet
asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
    :: "r"(__cvta_generic_to_shared(&mma_done[NS - 1])) : "memory");
```

The TMA warp then looks like:

```cpp
uint32_t mma_done_phase[NS] = {};

// Prologue: front-load NS-1 tiles unconditionally
for (int s = 0; s < NS - 1; s++)
    LOAD(s, /*k_iter=*/ s);

// Steady-state: TMA the tile that's NS-1 ahead of the MMA
for (int k = 0; k < num_k_iters - (NS - 1); k++) {
    const int slot = (k + NS - 1) % NS;
    mbarrier_wait_phase(mma_done[slot], mma_done_phase[slot]);
    LOAD(slot, /*k_iter=*/ k + NS - 1);
    mma_done_phase[slot] ^= 1;
}
```

The MMA warp has a single straight K-loop, no prologue:

```cpp
uint32_t tile_ready_phase[NS] = {};

for (int k = 0; k < num_k_iters; k++) {
    const int slot = k % NS;
    mbarrier_wait_phase(tile_ready[slot], tile_ready_phase[slot]);
    tcgen05_fence_after_thread_sync();

    // 4 MMAs covering BK = 64; very first MMA of the kernel
    // (k == 0 && kk == 0) overwrites TMEM, everything else accumulates.
    for (int kk = 0; kk < K_MMAS; kk++) { ... }

    tcgen05_commit(mma_done[slot]);
    tile_ready_phase[slot] ^= 1;
}
```

Note the **per-slot** phase tracking.  Each mbar in the ring has its
own parity counter — they flip independently of each other because the
TMA warp and MMA warp are running their own loops.  Slot 0's mbar
fires on iters 0, 2, 4, …; slot 1's on 1, 3, 5, ….  Each lane keeps a
2-entry array (or 3, or however many stages) and XORs only the entry
for the slot it just synchronized with.

The epilogue stays mostly the same as ch02/ch03 — `tcgen05.ld` from
TMEM, pack to BF16, store to GMEM — but it now has to wait until **all
MMAs are done**, not just the last one in this warp's loop, before
reading TMEM.  Easiest way: a single mbar `all_mmas_done` armed by an
extra `tcgen05.commit` issued after the MMA warp's loop finishes.

## The accumulate predicate, unchanged

Still `P = false` only when `(k_iter == 0 && kk == 0)`.  The MMA warp
sees a flat K-loop, so it's literally the same check as ch03 — nothing
about pipelining changes this.

## Per-slot phase tracking — why?

The mbar's parity bit is hardware state; it flips on every completion.
With one slot (ch03), all completions go to the same mbar and we
tracked one global phase that flipped each iter.  With NS slots, slot
`s` completes on iters `s, s + NS, s + 2·NS, …` — separately from the
other slots — so each slot's mbar has its own parity sequence and
needs its own software counter.

```
   iter:   0   1   2   3   4   5   6   7
   slot:   0   1   0   1   0   1   0   1     (= k % NS)

   slot-0 mbar parity:  → 1 →   → 0 →   → 1 →   → 0   (flips on iters 0, 2, 4, 6)
   slot-1 mbar parity:    → 1 →   → 0 →   → 1 →   → 0 (flips on iters 1, 3, 5, 7)
```

A `uint32_t mma_done_phase[NS]` tracks each independently; XOR after
each successful wait.  The kernel below uses one such array on the
TMA side and another on the MMA side.

## Measured speedup over chapter 03

Same problem (`M=128, N=256, K=4096`), same single CTA, same epilogue.
Mean kernel time over 1000 launches on B200:

| Kernel | Time / call | TFLOPS |
|---|---|---|
| ch03 (single-stage)     | reported by `main.py` | reported |
| ch04 (NS=2, two stages) | reported by `main.py` | reported |

The kernel below times itself and prints both numbers — run it to see
your B200's measured speedup.  On our reference run (`K = 4096`, single
CTA), `main.py` measured **67.7 µs/call → 43.5 µs/call = 1.56× speedup**
just from flipping `NS = 1 → 2`.  Going to `NS = 3` or 4 gives
diminishing returns at this problem size because MMA latency dominates
once TMA is fully hidden.

(Absolute TFLOPS are low here — ~4 → ~6 — because a single CTA can't
saturate a B200.  What we're measuring is the ratio, which captures the
TMA/MMA overlap.  Multi-CTA + full-grid kernels in later chapters get
the absolute numbers up.)

## What's still constrained

- **Single CTA.**  Still only one `(BM × BN) = (128 × 256)` output
  tile per launch.  Multi-CTA / grid mapping is the next chapter.
- **Same uncoalesced direct writeback.**  SMEM-staged coalescing is
  the chapter after that.
- **No 2-CTA cluster MMA.**  This kernel is `cta_group::1` only.

## Take-away

Multi-stage buffering is structurally simple — a ring of SMEM slots
plus per-slot mbar pairs — but it changes the kernel's dependency
graph fundamentally: TMA and MMA stop blocking each other and become
two independent flows synchronized at the buffer boundaries.  The
prologue / steady-state / pre-arrive trick is the standard pattern,
generalizes to any `NUM_STAGES ≥ 2`, and is the same shape every
production matmul kernel uses.

## Run

```bash
pip install -r ../requirements.txt
python main.py
```

A Blackwell GPU (sm_100a / B200) is required.
