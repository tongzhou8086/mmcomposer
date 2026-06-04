# TMA store epilogue

> 📁 **Code on GitHub:** [`tutorial/code/13_tma_store_epilogue/`](https://github.com/tongzhou8086/mmcomposer/tree/master/tutorial/code/13_tma_store_epilogue) — `kernel.cu` + `main.py`.

One small structural change on top of ch12: the epilogue's Phase 2
(SMEM → GMEM coalesced writeback) goes from a **128-thread int4-store
loop** to a **single async TMA store** issued by one thread.

The TMA load engine has been with us since chapter 00.  We've used it
for every A/B load through the K-loop.  But for output we kept using
plain `st.global` int4 stores — same throughput per byte, but each
warp spends 16 stores worth of LSU bandwidth doing them.  The TMA
store mechanism is the mirror image of the TMA load: same
tensor-map descriptor, same `(x, y)` coordinates, one PTX
instruction, asynchronous.

## The diff

This chapter changes three things on top of ch12 (everything else —
TMA loads, K-loop, MMA, Phase 1 of the epilogue — is identical):

1. **A C-side tensor map** built host-side and passed as an extra
   `__grid_constant__` kernel argument.
2. **The Phase 2 store loop** is replaced with a single
   `cp.async.bulk.tensor.2d.global.shared::cta.bulk_group` PTX
   instruction issued by warp 0 lane 0, plus a `commit_group` and a
   `wait_group.read 0` at the end of the kernel.
3. **A `fence.proxy.async.shared::cta`** between Phase 1's SMEM
   writes and the TMA store's SMEM read.  This is the one new
   subtlety the chapter introduces — see below.

## The PTX

The store instruction mirrors the load:

```
load:   cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier... [smem], [tmap, {x, y}], [mbar];
store:  cp.async.bulk.tensor.2d.global.shared::cta.bulk_group       [tmap, {x, y}], [smem];
```

The first operand is now the destination tensor map (GMEM) and the
second is the SMEM source.  No mbarrier — the bulk-store engine uses
its own counter, accessed via `commit_group` / `wait_group`:

```cpp
if (warp_id == 0 && elect_sync()) {
    tma_2d_store(C_tmap,
                 (uint32_t)__cvta_generic_to_shared(&C_sh[0][0]),
                 /*x=*/ off_n,
                 /*y=*/ off_m_cluster + cta_rank * BM);
    tma_commit_group();
}
tma_wait_group<0>();           // drain ALL outstanding bulk stores
```

`commit_group` enqueues a barrier after the in-flight stores;
`wait_group.read 0` blocks until *all* outstanding bulk-store groups
drain.  Pre-warp counter — warps that didn't issue a store no-op
through the wait.

## The cross-proxy fence — the one subtlety

```cpp
asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
```

This sits between Phase 1 (regular `st.shared` int4 stores written by
the LSU's *generic proxy*) and the TMA store (the bulk-copy engine
reads SMEM via the *async proxy*).  Without it, the async-proxy read
can observe stale or partially-written SMEM — manifests as
nondeterministic data errors in `C`.

We've encountered other proxy fences along the way
(`tcgen05.fence::after_thread_sync`, `fence.mbarrier_init.release.cluster`)
but `fence.proxy.async.shared::cta` is new.  The rule of thumb:
**when one proxy writes SMEM and another proxy reads it, fence
between them.**

## The cost: no row padding in C_sh

Chapter 07 added `BN_PAD = BN + 8` row padding to `C_sh` to defang
32-way bank conflicts on Phase 1's int4 stores (lanes 0/8/16/24 hit
the same banks at stride 256 bf16 = exactly 128 bank-words).  That
padding mitigated to 4-way.

**TMA can't tolerate the padding.**  The store engine reads `C_sh` as
a tightly-packed `BM × BN` box; an extra 8 cols per row in SMEM would
become 8 cols of garbage written to GMEM.  So `C_sh` in ch13 is
`bf16[BM][BN]` with no padding — and Phase 1's int4 stores climb
back up to 32-way bank conflict.

This is the real trade.  Async store vs. unconflicted Phase 1.

## Gotcha — dealloc must come AFTER the TMA store

Discovered while debugging ch13: the order

```cpp
__syncthreads();
if (warp_id == 0 && elect_sync()) tcgen05_dealloc_g2(taddr, BN);   // ← BEFORE
asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
if (warp_id == 0 && elect_sync()) {
    tma_2d_store(...);
    tma_commit_group();
}
tma_wait_group<0>();
```

**deadlocks**.  `cp.async.bulk.wait_group.read 0` never completes.

Reversing the order — TMA store + commit + wait FIRST, then dealloc —
runs correctly.  The cause appears to be that `tcgen05.dealloc.cta_group::2.sync.aligned`
leaves the cluster's bulk-engine state in a way that prevents a
subsequent bulk-copy store from completing.  We bisected with a
hello-world TMA store (store then dealloc → works; dealloc then store
→ hangs) so the dependency is real, not a tensormap or address bug.

So in ch13 we **drain the TMA store first, then dealloc**:

```cpp
__syncthreads();
asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
if (warp_id == 0 && elect_sync()) {
    tma_2d_store(...);
    tma_commit_group();
    tma_wait_group<0>();
}
if (warp_id == 0 && elect_sync()) tcgen05_dealloc_g2(taddr, BN);   // ← AFTER
```

This is the only ordering constraint in the chapter where the order
matters for correctness rather than just for perf.

## Results

`M = N = K ∈ {2048, …, 12288}` (11 shapes), B200, `triton.testing.do_bench`.
Compared head-to-head against ch12's autotuned best at the same shape:

| shape  | ch12 best (NS,GSM) | ch12 TF | **ch13 TF** | Δ TF | ratio |
|---|---|---|---|---|---|
| 2048³  | (5, 1)  |  802 |   727 |  −75 | 91 % |
| 3072³  | (6, 1)  | 1260 |  1067 | −193 | 85 % |
| 4096³  | (7, 1)  | 1231 |  1073 | −158 | 87 % |
| 5120³  | (6, 8)  | 1266 |  1115 | −151 | 88 % |
| 6144³  | (6, 8)  | 1320 |  1189 | −131 | 90 % |
| 7168³  | (6, 8)  | 1329 |  1193 | −136 | 90 % |
| 8192³  | (6, 8)  | 1322 |  1224 |  −98 | 93 % |
| 9216³  | (6, 8)  | 1339 |  1213 | −125 | 91 % |
| 10240³ | (7, 16) | 1346 |  1230 | −116 | 91 % |
| 11264³ | (7, 8)  | 1339 |  1229 | −111 | 92 % |
| 12288³ | (7, 8)  | 1337 |  1244 |  −93 | 93 % |

All shapes correct (max relative error ≤ 0.5 % vs PyTorch matmul),
but ch13 is **consistently slower** than ch12 — by 7-15 % across the
sweep.  The chapter does NOT improve perf on its own.

## Why ch13 is slower than ch12 at this shape

Two compounding losses, no compensating overlap win:

1. **Phase 1 bank conflicts.** Ch12 used `BN_PAD = BN + 8 = 264` row
   padding (introduced in ch07) to defang Phase 1's int4 SMEM stores
   from 32-way bank conflicts down to 4-way.  TMA can't tolerate the
   padding (the engine would write 8 cols of garbage to GMEM per row),
   so ch13 uses `bf16[BM][BN]` with no padding — and Phase 1's bank
   conflicts go back up to 32-way.  Roughly an 8× slowdown on the
   Phase-1 SMEM stores themselves; Phase 1 is ~5 % of total time, so
   the end-to-end cost is ~3-4 %.
2. **Async drain stalls the kernel.** `wait_group.read 0` at the end
   of the kernel blocks the kernel from exiting until the bulk-copy
   engine has drained.  There's no other work to overlap that drain
   with (in this non-persistent kernel, each CTA processes exactly
   one tile and exits).  So `wait_group` effectively serialises the
   GMEM writes that the int4 store loop in ch12 just fired in parallel.
   Another ~4-10 %, growing as a fraction at small shapes where the
   epilogue is a bigger slice of total runtime.

## When TMA store would be worth it

The TMA store mechanism is structurally *correct* — it's just on the
wrong side of a perf trade at this point in the ladder.  Two future
changes would flip the calculation:

- **K-loop / epilogue overlap.**  Tile T+1's K-loop and TMA loads run
  concurrently with tile T's TMA-store drain.  Now `wait_group` no
  longer stalls anything — the device is busy with T+1's compute.
  Combined with a persistent grid (so the kernel actually has a
  "tile T+1"), TMA store becomes a clear win.  This was confirmed
  empirically in a sibling project (`fused_swiglu_kernel`).
- **Wider epilogue.**  At shapes where the epilogue is a meaningful
  fraction of total runtime (small K, or large BN), the LSU
  saturation in ch12's int4 store loop becomes the bottleneck, and
  the TMA engine's single-instruction tile transfer becomes the
  faster path even without overlap.

Neither applies at our `M = N = K` headline shapes with current
ladder structure, so ch13 lands as a teaching point ("here's how the
TMA store works") without a perf headline.  The chapter is honest
about that.

## Run

```bash
pip install -r ../requirements.txt
python main.py
```

A Blackwell GPU (sm_100a / B200) is required.  Compiles a single
kernel variant (~5 s) plus ch12's 20-variant sweep (~30 s) for the
head-to-head.
