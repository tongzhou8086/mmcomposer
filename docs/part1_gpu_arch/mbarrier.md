# mbarrier — async synchronization primitive

> **Status:** stub — content TBD.

`mbarrier` is the SMEM-resident synchronization object that ties async
producers (TMA, tcgen05 commit) to their consumers (MMA warp, epilogue
warps).  It replaces ad-hoc spin-polling and works across CTAs in a
thread block cluster.

## Why mbarrier exists

* **Async ops need a completion signal.**  TMA and tcgen05.commit both
  fire-and-forget; the issuing thread doesn't stall.  We need a way for
  *another* thread (or warp) to know when those ops have retired.
* **Counter-based instead of sequence-based.**  Mbarrier tracks two
  counters (arrival and tx-count); completion = both zero.  This
  lets producers and consumers coordinate without a global ordering.
* **Hardware-resident.**  Lives in SMEM as a 64-bit value (per
  mbarrier).  Polling is via a single PTX instruction, not a memory
  fence + load loop.

## The lifecycle

1. **Init.**  `mbarrier.init.shared::cta.b64 [addr], count` sets
   `{arrival_count = count, tx_count = 0, phase = 0}`.
2. **Producer side.**  Async op fires; either explicitly via
   `mbarrier.arrive.expect_tx [addr], bytes` (which sets the expected
   tx-count and pre-decrements the arrival counter), or implicitly via
   the `mbarrier::complete_tx::bytes` modifier on TMA bulks (which
   auto-decrements tx-count by the bytes transferred).
3. **Wait.**  `mbarrier.try_wait.parity.shared::cta.b64 P, [addr],
   phase` blocks the consumer until the current phase has completed.
   Phase parity flips automatically each completion.
4. **Reuse.**  After completion, the counters auto-reset to
   `{count, 0}` and the consumer waits on the *next* phase.

## The two counters

* **arrival_count** — decremented by `mbarrier.arrive[.expect_tx]`.
  Reaches zero when N arrivals have happened.
* **tx_count** — decremented by TMA bulks via the
  `mbarrier::complete_tx::bytes` modifier.  Reaches zero when the
  expected bytes have arrived.

Both must reach zero before the phase completes.

## Phase parity (the tricky part)

The mbarrier maintains a 1-bit phase parity that flips on each
completion.  `try_wait.parity P, [mbar], phaseParity` succeeds when
the mbarrier's *current* parity bit is **opposite** to the
`phaseParity` operand.

This means each waiter keeps a software mirror of the parity bit it
*expects to see*, flips it after each successful wait, and passes the
flipped value to the next `try_wait`:

```cuda
uint32_t phase = 0;
for (;;) {
    mbarrier_wait(mb, phase);   // succeeds when mbar parity != phase
    phase ^= 1;
    // ...
}
```

A common trick: **pre-arrive** on an mbarrier just after init so the
first wait returns immediately.  This avoids a branch at iter 0:

```cuda
mbarrier_init(&mb, 1);
asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(addr));
asm volatile("fence.mbarrier_init.release.cluster;");
// ... now wait(phase=0) returns immediately
```

## The instruction family

```
mbarrier.init.shared::cta.b64 [addr], count;
mbarrier.arrive.shared::cta.b64 _, [addr];
mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [addr], bytes;
mbarrier.try_wait.parity.shared::cta.b64 P, [addr], phaseParity;
```

* `.shared::cta` vs `.shared::cluster` — destination address space.
* `.release.cta` — release-semantics at CTA scope.  For cross-CTA mbar
  use `.release.cluster` if the consumer is on a peer CTA.
* `.acquire.cta` — implicit on `try_wait.parity` when paired with a
  `.release` arrive.

## Scope and cluster mbarriers

Within a single CTA, mbarriers are just SMEM addresses.  In a 2-CTA
cluster, **peer-CTA mbarriers are addressed via a special bit pattern**:
clearing bit 24 of the SMEM address routes the address to CTA 0's
mbarrier from CTA 1.  In our codebase:

```cuda
const uint32_t mb_local = (uint32_t)__cvta_generic_to_shared(&mb);
const uint32_t mb_cta0  = mb_local & 0xFEFFFFFF;   // route to CTA 0
```

This is how cross-CTA tx-count bookkeeping works: both CTAs' TMA bulks
target the same CTA 0 mbarrier even though they're issued from different
SMs.

## Common pitfalls

* **Arrival count mismatch.**  Init with count=N but only N-1 arrivals
  ever happen → permanent hang.
* **Forgetting `expect_tx`.**  If the TMA bulks have a tx-count modifier
  but no `arrive.expect_tx` was issued, tx-count never reaches zero.
* **Wrong scope qualifier.**  Cross-CTA mbarrier touched with
  `.release.cta` instead of `.release.cluster` → memory ordering
  inconsistent, races possible.
* **Stale phase.**  If your software phase mirror drifts (e.g. you flip
  it twice for one completion), every subsequent wait blocks forever.
* **Init not yet visible.**  `mbarrier.init` is followed by
  `fence.mbarrier_init.release.cluster` before any async ops touch
  the mbar — otherwise the producer might race ahead of the init.
