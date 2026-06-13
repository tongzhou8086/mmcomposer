# Generator Validation Constraints

This document records why the generator rejects or narrows some knob
combinations.  Keep it in sync with `mvp_core.validate_config`, the launchers,
the autotune filters, and the GPU integration tests.

The distinction matters:

- **Hardware or instruction constraint**: the combination is not valid for the
  target primitive.
- **Implementation guardrail**: the combination may be valid in principle, but
  this generator intentionally does not implement it yet.
- **Autotune policy**: the combination is valid, but the timing sweep may prune
  it to keep runs practical.

When adding a new guardrail, use wording like "currently" or "initially" in UI
messages so it is clear that the limitation is not necessarily fundamental.

## Hardware And Instruction Constraints

- Target architecture is NVIDIA B200, `sm_100a`.
- `BK = 64` is required by the current 128-byte TMA swizzle layout.  The inner
  TMA box dimension is one 128-byte BF16 atom.
- `BM = 128` is the only implemented M tile.  It matches the tcgen05 F16 MMA
  shape and the current TMEM/epilogue row mapping.
- The ordinary MMA path has a tcgen05 F16 N atom limit of 256 columns.  A
  logical `BN > 256` must be implemented as multiple 256-column MMA panels; it
  cannot use the ordinary single-idesc path.
- `BN` must be a multiple of 64 for the K-major B TMA sub-tile.  In the 2-CTA
  cluster path, each CTA owns `BN/2` B columns, so `BN/2` must also be a
  multiple of 64.
- Shapes must tile exactly: `M % BM == 0`, `N % BN == 0`, and `K % BK == 0`.
  In the 2-CTA cluster path, `M` must also be a multiple of `2 * BM`.

## Current Implementation Guardrails

- Persistent launch is available only on warp-specialized paths with a CTA tile
  loop.  Non-warp-specialized paths do not have the loop needed for `grid =
  #SMs`.
- Epilogue overlap requires persistent launch.  The current overlap skeleton
  uses stream warps plus epilogue worker warps inside a persistent tile loop.
- Pipelined TMA-store epilogue requires persistent launch and epilogue overlap.
  It is an alternative epilogue mode, so it currently rejects split epilogue and
  L1 no-allocate C stores.
- Split epilogue writeback currently applies only to the 2-CTA cluster overlap
  path.  This is an implementation guardrail, not a known hardware limit.
- `num_warps` must map cleanly onto `BM/32` row strips and column groups.  The
  epilogue reads 32-row TMEM strips, so `num_warps % (BM/32) == 0` is required.
- `num_warps >= 4` is kept as a tcgen05 safety guard.  Smaller values produced
  incorrect output in earlier experiments and are not exposed by the option
  list.
- In pipelined TMA-store mode, `STORE_N = 64`, so `STORE_N/8 = 8` tcgen05 load
  atoms must divide across the epilogue column warp groups.
- The UI hides the epilogue tcgen05 load-width knob and currently uses
  `TCGEN05_LD_WIDTH = 8`.  Some non-TMA epilogue paths still validate 8 or 16,
  but the pipelined TMA-store branch currently emits x8 loads.
- Shared memory validation uses the current B200 usable limits: about 228 KiB
  for non-overlap paths and 224 KiB for overlap paths.  Non-overlap paths size
  for `max(K-loop ring, epilogue staging)`, while overlap paths size for
  `K-loop ring + epilogue staging` because those regions are live together.

## BN512 Panelized Implementation Bundle

The BN512 study shows a useful performance win for some large square shapes,
but the first generator integration is intentionally tight.  BN512 is the
thing that requires a special implementation: it exceeds the 256-column
`tcgen05.mma.kind::f16` N atom, so the compute path is panelized into two
256-column MMA panels.  `SINGLE_TMEM_ACCUM` is a separate synchronization and
buffer-reuse knob inside the currently validated BN512 bundle; it is not, by
itself, an epilogue-style constraint.

Initial guardrails:

- `BN = 512` is valid only for the studied implementation bundle:
  - warp-specialized 2-CTA cluster path
  - `BM = 128`, `BK = 64`
  - persistent launch on
  - epilogue overlap on
  - pipelined TMA-store epilogue on
  - single-TMEM accumulator synchronization on
  - split epilogue off
  - L1 no-allocate off
- The logical `BN=512` tile is implemented as two 256-column MMA panels:
  - panel 0 writes TMEM columns `0..255`
  - panel 1 writes TMEM columns `256..511`
- TMEM allocation is one logical buffer of `BN` columns, not the usual
  `2 * BN` ping-pong allocation.
- The MMA warp must wait on `tmem_empty[0]` before starting the next output
  tile.  In the sync-fixed design, `tmem_empty[0]` is released only after all
  epilogue worker warps complete the final chunk's `tcgen05.ld`.
- Outstanding TMA stores do not block TMEM reuse.  After the final TMEM load,
  the old tile's data lives in registers and the two SMEM TMA-store buffers.
- Shared memory for the studied config is:

```text
BN_LOCAL = 512 / 2 = 256
A slot   = 128 * 64 * 2 = 16 KiB
B slot   = 256 * 64 * 2 = 32 KiB
slot     = 48 KiB
epilogue = 2 * 128 * 64 * 2 = 32 KiB

NS=4: 4 * 48 KiB + 32 KiB + 1 KiB = 230400 B  fits
NS=5: 5 * 48 KiB + 32 KiB + 1 KiB = 279552 B  does not fit
```

Potential later expansions:

- single-CTA support
- other BN values implemented as 256-column panels
- single-TMEM accumulator synchronization on more non-BN512 paths
- different `STORE_N` or tcgen05 load widths
- less restrictive interactions with split epilogue or non-TMA stores

These should be added only after a study proves correctness and gives useful
performance wins.

## Autotune Policy

Autotune may intentionally prune valid configurations.  For example, production
timing sweeps currently prefer practical subsets such as `BN=256` and `NS>=3`
instead of timing every valid educational combination.

Policy filters should not be presented as correctness requirements.  If a
combination is valid but pruned, document it as a timing-scope decision in the
autotune script or UI text.
