# 2-CTA cluster MMA — pairing CTAs, doubling M, unlocking deeper NS

> 📁 **Code on GitHub:** [`tutorial/code/08_cluster_mma/`](https://github.com/tongzhou8086/mmcomposer/tree/master/tutorial/code/08_cluster_mma) — `kernel.cu` + `main.py`.

Up to ch07, every CTA computed its own `BM × BN = 128 × 256` output
tile independently.  This chapter introduces **CTA clusters** — small
groups of CTAs that cooperate inside a single `tcgen05.mma`.  We use
the smallest cluster (size 2) along the M dimension, which gives the
chapter three concrete wins:

1. **M per MMA doubles** to `2·BM = 256`.  One instruction covers
   twice the output, halving per-tile setup overhead.
2. **B is split across the cluster.**  Each CTA owns only `BN/2` cols
   of B in SMEM — per-CTA B SMEM cost drops from 32 KB to 16 KB.
3. **The freed SMEM unlocks deeper pipelines.**  The single-CTA
   kernels of ch04–ch07 ran out of room at `NS = 4` per-CTA; the
   2-CTA cluster fits **up to `NS = 7`**.

The first two are direct consequences of `cta_group::2` cooperation.
The third is indirect but compounding: more available SMEM ⇒ playable
range of the pre-existing `NS` knob extends.  Each delivers something
on its own, and they stack.

## Per-CTA SMEM budget (the headline number)

B200's per-CTA dynamic-SMEM cap (with `cuFuncSetAttribute` opt-in) is
**228 KB**.  After ~1 KB for `__align__(1024)` padding and the small
static `__shared__` (mbar arrays, `tmem_addr_holder`), the usable
budget is ~225 KB.  Per-stage cost:

| | per-slot per-CTA | `NS_max` |
|---|---|---|
| ch04–ch07 single-CTA | `A + B = 16 + 32 = 48 KB` | **4** (`NS = 5` needs 241 KB) |
| ch08 2-CTA cluster   | `A + B/2 = 16 + 16 = 32 KB` | **7** (`NS = 7` uses 225 KB; `NS = 8` needs 257 KB) |

`NS_max` almost doubles, and that's the lever this chapter pulls.

## The five new pieces

This chapter adds more new concepts in one go than any prior — but
they're tightly coupled.  Together they take you from "lots of
independent CTAs" to "groups of CTAs cooperating on one MMA tile."

### 1. Cluster launch — `__cluster_dims__(2, 1, 1)`

The kernel is decorated with `__cluster_dims__`, which fixes that
every `2 × 1 × 1` block of CTAs in the grid runs together as a
*cluster*: co-resident on neighbouring SMs, able to address each
other's SMEM via cluster-scoped instructions:

```cpp
extern "C" __global__
__cluster_dims__(2, 1, 1)
__launch_bounds__(THREADS, 1)
void matmul_cluster(...) { ... }
```

Grid math changes accordingly: the host launches
`(M / (2·BM)) × (N / BN)` *clusters* worth of CTAs, flattened to a
1-D grid.  Each cluster covers a `2·BM × BN = 256 × 256` output tile.

### 2. `cta_rank` — who am I in the cluster

A per-CTA hardware register, accessed via the `%cluster_ctarank`
special:

```cpp
int cta_rank;
asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
// cta_rank ∈ {0, 1} for a 2-CTA cluster
```

Drives every per-CTA offset.  CTA 0 owns the lower `BM` rows of the
cluster's output (and lower `BN/2` cols of B in SMEM); CTA 1 owns
the upper halves.

### 3. `tcgen05.alloc.cta_group::2` + `tcgen05.mma.cta_group::2`

The TMEM allocator and MMA both get `cta_group::2` variants:

* **`tcgen05.alloc.cta_group::2`** reserves TMEM cluster-wide.  Each
  CTA gets the same `taddr` back and addresses *cluster-logical* row
  indices `[0, 2·BM)` — its own physical TMEM holds rows
  `[cta_rank·BM, (cta_rank+1)·BM)`.
* **`tcgen05.mma.cta_group::2`** issues one MMA that contracts a
  `2·BM × BN` output from `2·BM × MMA_K` × `MMA_K × BN` operands.
  **Only `cta_rank == 0` issues**; the result lands in *both* CTAs'
  TMEM (the cluster-wide logical rows).

`idesc` carries the doubled M-dim: `make_idesc_bf16_cluster(2·BM, BN)`
sets `m_dim = (2·BM)/16 = 16` instead of `BM/16 = 8`.

#### Asymmetric work — only CTA 0 issues the MMA

| phase | CTA 0 | CTA 1 |
|---|---|---|
| TMA warp | loads its half of A + B | loads its half of A + B |
| MMA warp | issues `tcgen05.mma.cta_group::2` | **idle, waiting on `mma_done`** |
| epilogue | reads its TMEM half + GMEM writes | reads its TMEM half + GMEM writes |

CTA 1 isn't idle overall — TMA and epilogue still run there, and the
cluster MMA reads its A/B from CTA 1's SMEM via fan-out.  But the MMA
*instruction itself* has a single issuer per cluster (two would race
the pipeline), and that issuer is always `cta_rank == 0`.

### 4. Cross-cluster mbarrier accounting

Two new wrinkles in the handshake compared to ch07:

**(a)** `tile_ready` mbar's arrival count is `2`, not `1` — both
CTAs' TMA warps must arrive before the MMA proceeds.  The peer-CTA
addressing routes both arrivals to one shared mbar:

```cpp
// In each CTA's TMA warp:
const uint32_t mb_local = (uint32_t)__cvta_generic_to_shared(&tile_ready[s]);
const uint32_t mb_cta0  = mb_local & 0xFEFFFFFFu;   // clear CTA-rank bit
                                                    //  → always addresses CTA 0's mbar
...
mbarrier_arrive_expect_tx(mb_cta0, SLOT_BYTES_PER_CTA);
```

The mask `0xFEFFFFFFu` clears the bit that encodes which CTA in the
cluster a shared address refers to, so the resulting address is
always *CTA 0's* shared region.  Both CTAs' `expect_tx` thereby
arrive at CTA 0's `tile_ready[s]` mbar.

**(b)** The MMA's completion needs to fire mbarriers on **both**
CTAs.  That's the **multicast commit**:

```ptx
tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64
    [mbar_addr], cta_mask;
```

`cta_mask = (1 << CTA_GROUP) - 1 = 0b11` says "fire on every CTA in
the cluster."  One `commit` arms both CTAs' `mma_done` mbars from
CTA 0's MMA issuer; CTA 1 sits idle during the MMA but its mbar still
flips when the work completes.

### 5. Cluster barrier replaces `__syncthreads` at init

`__syncthreads` synchronizes within one CTA.  To make peer CTAs see
each other's mbar inits, we need a cluster-wide barrier:

```cpp
asm volatile("barrier.cluster.arrive.release.aligned;");
asm volatile("barrier.cluster.wait.acquire.aligned;");
```

These bracket the post-init point so that *both* CTAs' subsequent TMA
arrivals reach mbarriers that are already initialised cluster-wide.
The acquire/release pair plays the role of the
`fence.mbarrier_init.release.cluster` we used in single-CTA kernels,
extended to the cluster scope.

## SMEM layout — `B` is half-width per CTA

The kernel's SMEM accounting changes only on the B side.  Per CTA per
stage slot:

```
   A : [BM rows][BK K-cols]                     16 KB    (unchanged)
   B : [BN/2 / 64 sub-tiles][BK rows][64 cols]  16 KB    (half of ch07)
```

Each CTA owns its **N-half** of B: CTA 0 covers cols `[0, BN/2)`,
CTA 1 covers `[BN/2, BN)`.  TMA still uses the chapter-06 K-major B
descriptor (one inner sub-tile per call, looped); each CTA loops over
`BN/2 / 64 = 2` sub-tiles per stage instead of ch07's 4.

The MMA descriptor for B continues to use `make_desc_K_major`; the
LBO field walks N-sub-tiles inside the local CTA's `BN/2`, and the
cluster MMA fans across both CTAs to assemble the full N=BN slice.

## Sizing the dynamic SMEM — a footgun the cluster exposes

This chapter introduces a sizing requirement on the launcher that
ch04–ch07 happened to satisfy by accident.  The dynamic SMEM the
kernel needs is the **max of two non-overlapping uses**, not just the
K-loop term:

1. During the K-loop: `NS × SLOT_BYTES` bytes of A + B ring buffer.
2. During the epilogue: `BM × (BN+8) × 2 = 67584 B ≈ 66 KB` of staging
   for the coalesced TMEM → SMEM → GMEM writeback (the `+8` is the
   bank-conflict pad from ch07).

The two phases never overlap in time — `all_mmas_done` cleanly
separates them — so SMEM is reused.  But the *allocation* must be
sized to cover both:

```python
shared_bytes_per_CTA = max(NS × SLOT_BYTES, EPILOGUE_STAGING_BYTES) + 1024
```

| | `NS × SLOT_BYTES` | staging | which dominates? |
|---|---|---|---|
| ch07 NS=2 (single-CTA, 48 KB/slot)  | 96 KB | 66 KB | K-loop ✓ |
| ch07 NS=3 (single-CTA, 48 KB/slot)  | 144 KB | 66 KB | K-loop ✓ |
| **ch08 NS=2 (2-CTA, 32 KB/slot)**   | **64 KB** | **66 KB** | **staging ⚠️** |
| ch08 NS=3 (2-CTA, 32 KB/slot)       | 96 KB | 66 KB | K-loop ✓ |
| ch08 NS≥3 (2-CTA)                   | ≥ 96 KB | 66 KB | K-loop ✓ |

ch07 never hit the staging-dominates case because its K-loop SMEM
was always ≥ 96 KB.  ch08 at `NS = 2` is the *first* configuration
in the tutorial where the K-loop term falls below the staging — so
naïvely setting `shared_bytes = NS * SLOT_BYTES + 1024` (the pattern
that's been correct through ch04–ch07) **under-allocates the dynamic
SMEM, and the epilogue writes past the allocation → CUDA_ERROR_ILLEGAL_ADDRESS**.

### Why we don't just shrink the staging

A few alternatives don't work:

- **Smaller pad than `+8`.**  `+8` is already the smallest 16-byte-
  aligned padding that breaks the 32-way SMEM bank conflict from
  ch07's columnar phase-1 writes.  Going to `+4` BF16 = 8 B isn't
  `int4`-aligned, breaking the phase-2 stores.
- **Chunk the epilogue in row strips.**  Each warp would need its own
  17 KB staging slice, so 4 warps × 17 KB = 68 KB — same total.
- **Don't reuse A/B SMEM at all.**  The kernel would then need
  `K-loop + staging ≈ 130 KB` regardless of NS, wasting more than the
  reuse pattern saves.

So the `max(...)` is the production-correct pattern.  CUTLASS and
`gau-nernst` both do the same.  ch08 just makes it impossible to
ignore.

### What we did about it

- **`EPILOGUE_STAGING_BYTES`** is a `constexpr` at the top of
  `kernel.cu`, with a comment block calling out the dual use of
  dynamic SMEM and what the launcher must do.
- **`shared_for(ns)`** in `main.py` computes the correct allocation
  per NS, and a comment cross-references the kernel-side constant.
- The TMEM `__syncthreads` between phase 1 and phase 2 (which already
  exists for correctness) is the natural seam between the two SMEM
  uses, so the reuse is safe.

If you write a future kernel built on this one, **inherit the
`max(K-loop, staging)` pattern verbatim** — the bug doesn't show up
in correctness tests until NS or geometry crosses the threshold, so
it's easy to miss.

## Performance — `NS` sweep at constant kernel structure

`main.py` compiles ch07 (single-CTA, fixed `NS = 2`) and ch08 with
`NS = 2..7` and runs them all at three problem sizes.  Measured on
B200, TFLOPS (higher is better):

| config | per-CTA SMEM | 2K³ | 4K³ | 8K³ |
|---|---|---|---|---|
| ch07 NS=2 (single-CTA)  | 97 KB   |  568 |  805 |  820 |
| **ch08 NS=2** (2-CTA)   | **67 KB** | **574** | **768** | **825** |
| ch08 NS=3 (2-CTA)       | 97 KB   |  651 | 1020 | 1009 |
| ch08 NS=4 (2-CTA)       | 129 KB  |  665 | 1191 | 1127 |
| ch08 NS=5 (2-CTA)       | 161 KB  |  **671** | **1209** | 1153 |
| ch08 NS=6 (2-CTA)       | 193 KB  |  657 | 1144 | **1168** |
| ch08 NS=7 (2-CTA)       | 225 KB  |  662 | 1133 | 1163 |

Two cleanly separable effects:

**Direct 2-CTA structural change** (rows 1 → 2, holding `NS = 2`):
**roughly a wash**.  M=256 per MMA + multicast B do *something*, but
at this `(BM, BN, BK)` and `NS = 2`, the gain is within noise.  By
itself, 2-CTA is not the headline.

**The freed SMEM unlocking deeper `NS`** (rows 2 → 5, 4K and 8K):
**+45–50%**.  At 4K, 768 → 1209 TFLOPS as NS climbs from 2 to 5.  At
8K, 825 → 1168 TFLOPS as NS climbs to 6.  *This* is where the chapter
earns its keep.

The best `NS` shifts with problem size (5 at 4K, 6 at 8K).  That's a
preview of the autotuning chapter — picking `NS` per-shape becomes
worthwhile once the playable range is large enough.

cuBLAS comparison (PyTorch's reference at the same shapes):

| shape | ch07 best | **ch08 best** | cuBLAS | ch08 / cuBLAS |
|---|---|---|---|---|
| 2048³ |  568 |  **671** | ~1278 | **53%** |
| 4096³ |  805 | **1209** | ~1500 | **81%** |
| 8192³ |  820 | **1168** | ~1264 | **92%** |

At 8192³ we're closing in on parity.

## The relationship between cluster and NS — complementary, not coupled

A precise framing matters here.  These two knobs are **not strictly
coupled** — you *can* sweep `NS` without 2-CTA (chs 04–07 already
support arbitrary `NS`), and you *can* do 2-CTA without sweeping `NS`
(holding it at 2 still gives you the direct structural wins).  They
just **compose well**: 2-CTA halves the per-CTA SMEM cost of a stage,
which extends the playable range of `NS` upward.

So the right reading of the chapter isn't "to get more stages you
need clusters" — it's "clusters let you push `NS` past where
single-CTA hits the SMEM ceiling."

## What's still on the table

After this chapter the cuBLAS gap closes further but isn't gone.  The
remaining items, roughly in impact order:

- **CTA-tile L2 swizzling** (Triton-style chunked walk) — improves
  A-stripe L2 reuse across the grid.  Pure launcher-side change.
- **8-warp kernel with epilogue parallelism** — split the now-coalesced
  epilogue across more warps; epilogue becomes ~2× faster.
- **`NS` autotuning** — the sweep this chapter does manually becomes
  systematic, with the right `NS` chosen per-shape.

## Run

```bash
pip install -r ../requirements.txt
python main.py
```

A Blackwell GPU (sm_100a / B200) is required.
