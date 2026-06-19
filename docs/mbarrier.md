# mbarrier semantics (as used in the MMComposer kernels)

How the producer/consumer "ready" and "free" signals in the warp-specialized
kernels are actually implemented. The signals in the design diagrams (dashed
arrows) are all one primitive: a hardware **mbarrier** in shared memory.

## The mbarrier object

A 64-bit shared-memory word with this mutable state:

| field               | meaning                                            |
|---------------------|----------------------------------------------------|
| `expected_arrivals` | constant, set by `mbarrier.init` (e.g. 1, or 2)    |
| `pending_arrivals`  | counts down within the current phase               |
| `tx_count`          | pending async-transfer **bytes** for this phase    |
| `phase`             | parity bit (0/1)                                   |

**Completion rule (hardware):** when `pending_arrivals == 0` **and**
`tx_count == 0`, the hardware atomically **flips `phase`**, reloads
`pending_arrivals = expected_arrivals`, and resets `tx_count = 0`. No thread
polls or flips it.

## The operations

- `mbarrier.init [mb], N` — set `expected_arrivals = N`, `pending_arrivals = N`.
- `mbarrier.arrive [mb]` — `pending_arrivals -= 1` (immediate). Plain "I'm here /
  I'm done." Used for resource-free signals and for pre-arming at init.
- `mbarrier.arrive.expect_tx [mb], bytes` — **one atomic op doing two things**:
  1. `tx_count += bytes`  (expect-tx: declare how many bytes to wait for)
  2. `pending_arrivals -= 1`  (arrive: the producer checks in)
  Then the completion check runs. Returns immediately; never blocks.
- `…mbarrier::complete_tx::bytes` (carried by the **TMA copy** instruction) —
  the copy *engine* does `tx_count -= delivered` as data lands. No thread issues
  this per-transfer; it's a side effect of the async DMA finishing.
- `tcgen05.commit.…mbarrier::arrive` — an **async-deferred arrive**: the thread
  issues `commit`, but the arrival (`pending_arrivals -= 1`) is delivered by the
  **tensor core when the MMA completes**, not when the instruction is issued.
- `mbarrier.try_wait.parity [mb], phase` — **non-blocking** test: returns a
  predicate "has the barrier reached this parity?" The blocking `wait` is a spin
  loop around it (`mbarrier_wait_phase`); it can park the warp so it isn't a hot
  spin. The waiter tracks its own phase bit and flips it (`^= 1`) each round.

## Hardware vs. software

- **arrival count** — moved by *instructions*: immediate via `mbarrier.arrive`,
  or async-deferred via `tcgen05.commit` (tensor core arrives on MMA completion).
- **byte count** — *armed* by software (`expect_tx`), *credited* by hardware (the
  TMA copy engine's `complete_tx`, with no per-transfer instruction).
- **phase flip** — always hardware, when both counters hit zero.

## Why an arrival count *and* a byte count

The byte count alone answers "did the data physically land?" The arrival count
answers "have all the **producers** feeding this barrier checked in?" — it counts
producers. With one producer it's nearly a formality; with N producers it
synchronizes them.

Concretely, in **2-CTA MMA** the `cta_group::2` MMA reads *both* CTAs' SMEM, so
the compute-full barrier is init'd with `expected_arrivals = CTA_GROUP = 2`. Each
rank's TMA lane does `arrive.expect_tx` arming its own half; the phase flips only
once **both ranks have checked in and all bytes (both halves) have landed** — so
the joint MMA never reads a half-loaded tile.

General form: producer1 `arrive.expect_tx(10K)` + producer2 `arrive.expect_tx(10K)`
on a count-2 barrier ⇒ flip when both arrived **and** 20K total has landed.

## State trace — TMA data-ready (single-CTA, count = 1)

```
init:                       pending=1, tx=0,    phase=0
issue TMA copies (reference mb; will complete_tx later)
arrive.expect_tx(SLOT):     pending=0, tx=SLOT  → check: tx≠0, no flip
engine delivers bytes, each complete_tx:  tx -= n
tx hits 0:                  pending=0, tx=0     → FLIP phase=1, pending reloads to 1
consumer wait_phase: try_wait.parity sees phase 1 → unblocks, reads SMEM
```

The producer does **not** wait after `arrive.expect_tx` — it advances to the next
slot and races ahead over the NS-deep ring. The copy engine drives `tx_count` to
zero and flips the barrier; the consumer wakes on the flip. No thread polls the
transfer; producer and consumer never lock-step. That decoupling *is* the
load↔compute pipeline.

## Map to the kernel's signals

| signal               | barrier (init count)            | how it's raised                                  |
|----------------------|----------------------------------|--------------------------------------------------|
| SMEM data-ready      | `smem_compute_full_mbar` (2)     | `mbarrier_arrive_expect_tx(SLOT_BYTES)` + TMA `complete_tx` |
| SMEM slot-free       | `smem_compute_free_mbar` (1, pre-armed) | `tcgen05_commit_mcast_g2(free_mbar)` — rides on commit, fires after the MMA's SMEM reads |
| TMEM data-ready      | `tmem_full` (1)                  | `tcgen05_commit_mcast_g2(tmem_full)` — arrives on MMA completion |
| TMEM buffer-free     | `tmem_free_mbar` (2, pre-armed)  | `EPI_TMEM_FREE_ARRIVE` — plain `mbarrier.arrive` |

### The real distinction (not data-ready vs. resource-free)

It is tempting to say "data-ready = hardware signal, resource-free = plain
arrive," but that is **wrong** — the SMEM slot-free is also hardware-signaled.
The actual axis is **whether the work the signal gates is still in flight when
you raise it**:

- **Still async in-flight** → defer the signal to the hardware completion
  mechanism (`expect_tx`/`complete_tx`, or `tcgen05.commit`), so it fires at true
  completion. (SMEM data-ready, TMEM data-ready, **and SMEM slot-free** — the
  MMA's SMEM reads are still in flight when the MMA warp moves on.)
- **Already finished synchronously by the issuing thread** → a plain
  `mbarrier.arrive` suffices. Only the TMEM buffer-free qualifies: the epilogue
  calls `tcgen05_wait_ld()` (synchronously waiting for the TMEM→register loads)
  *before* it arrives, so the reads are provably done. The MMA can't do that for
  its SMEM slot-free without stalling the pipeline, so it rides on `commit`.

So three of the four signals ride on hardware completion; only the TMEM-free is a
plain arrive, and only because its work was already awaited.

vs. `__syncthreads()`: that is a fused, symmetric, all-threads barrier (arrive +
wait together, count = blockDim). An mbarrier here is split-phase and
point-to-point — the **arriver and the waiter are different warps**, the count is
small, and arrivals can come from hardware units (TMA, tensor cores). That
asymmetry is what wires distinct warps and engines into one pipeline.
