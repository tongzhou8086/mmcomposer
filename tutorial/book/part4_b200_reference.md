# B200 Reference

> **Status:** stub — content TBD.

## PTX cheat sheet

A focused, GEMM-relevant subset of the PTX ISA on sm_100a.

- `cp.async.bulk.tensor.{1d,2d,3d}` (with and without `.cta_group::N` and
  `.multicast::cluster` modifiers)
- `mbarrier.{init, arrive, arrive.expect_tx, try_wait.parity}`
- `tcgen05.{alloc, dealloc, mma, commit, ld, wait::ld}`
- `barrier.cluster.{arrive, wait}.aligned`
- `cvt.rn.bf16x2.f32` (epilogue casts)

TODO: fill out with concrete operand syntax + scope modifiers.

## Common errors

- Missing `.cta_group::2` on TMA bulks in cluster mode → silent
  hang on cross-CTA mbarriers.
- Forgetting `tcgen05.fence::after_thread_sync` between MMA and the
  consumer load → garbage TMEM reads.
- Mbarrier init count mismatch with arrival count → either deadlock or
  premature wake.

## Shared-memory layouts

- 128B swizzle pattern (`CU_TENSOR_MAP_SWIZZLE_128B`) and what it implies
  for the SMEM-to-MMA-descriptor offsets.
- The K-major vs MN-major B layouts and their MMA descriptor differences
  (LBO field, idesc bit 16).
- The 8-bf16 row padding trick for avoiding 32-way bank conflicts in the
  C staging SMEM (`C_sh[BM][BN + 8]`).

## TODO

- TMEM addressing details for `cta_group::2` (cluster-wide logical
  indexing → per-CTA physical storage).
- SMEM descriptor encoding for tcgen05.
- A table of `__launch_bounds__` settings that worked vs didn't.
