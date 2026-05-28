# `tcgen05.mma` — the Blackwell async tensor-core MMA

> 📁 **Code on GitHub:** [`tutorial/code/02_tcgen05_mma/`](https://github.com/tongzhou8086/mmcomposer/tree/master/tutorial/code/02_tcgen05_mma) — `kernel.cu` + `main.py`.

Chapter 01 set up swizzled SMEM tiles.  This chapter introduces the
consumer those tiles were built for: **`tcgen05.mma`**, the native MMA
instruction on Blackwell.

The headline differences from `mma.sync` + `ldmatrix`:

* **One thread issues, the tensor core does the work.**  `mma.sync`
  is a *warp-collective* instruction — all 32 lanes participate, each
  holding a fragment of an `m16n8k16` MMA in their registers.
  `tcgen05.mma` flips this entirely: **a single thread fires off one
  instruction**, and that one instruction is offloaded to the tensor
  core engine, which then chews through a `M = 128 × N ≤ 256 × K_dtype`
  tile on its own (or `M = 256` across a 2-CTA cluster).  The issuing
  thread is free the moment the instruction is launched. It's the same "issue a big async unit of work" model as
  TMA, applied to the tensor cores.
* **No register operands.**  The tensor core reads operand tiles
  *directly from SMEM* via a 64-bit matrix descriptor.  There is no
  `ldmatrix`, no warp-cooperative fragment load, no register fragment
  to manage.  The cost: the SMEM layout has to be the one the
  descriptor knows how to walk — i.e., the swizzled layout from
  chapter 01.
* **The accumulator lives in TMEM, not registers.**  The result of
  `tcgen05.mma` is written into a brand-new memory space on Blackwell
  called **tensor memory (TMEM)** — fast on-die storage adjacent to
  the tensor cores.  Registers don't see the accumulator until the
  kernel explicitly loads it back via `tcgen05.ld`.
* **Async, like TMA.**  Issue and continue.  Completion is signalled to
  an mbarrier via `tcgen05.commit`, same handshake pattern you've used
  for TMA loads.

## The four moving parts

To issue one `tcgen05.mma` you need four pieces of state, and the rest
of this chapter walks each one:

1. **A TMEM allocation** — where the accumulator lives.
2. **A matrix descriptor** for A and B — 64-bit values that tell the
   tensor core where the operand tiles sit in SMEM and how they're laid
   out (swizzle, strides).
3. **An instruction descriptor (`idesc`)** — 32-bit value encoding the
   instruction shape (`M`, `N`, transpose flags) and operand dtypes.
4. **The instruction itself** — `tcgen05.mma` with the TMEM destination,
   the two matrix descriptors, the `idesc`, and an accumulate predicate.

Plus two pieces of glue: an `after_thread_sync` fence between TMA and
MMA, and an mbarrier `commit`/wait pair after the MMA so you know when
TMEM is consumable.

```
                                                                  ┌────────────────┐
                                                                  │ TMEM (D)       │ <─ alloc once
                                                                  │ accumulator    │
                                                                  └───────┬────────┘
                                                                          ▲
                                                                          │  writes
   SMEM A tile ──► matrix desc A ──┐                                      │
                                   ├─►  tcgen05.mma [D], A, B, idesc, P  ─┘
   SMEM B tile ──► matrix desc B ──┘
                                                  ▲
                                                  └── idesc: shape + dtype
```

## Part 1 — TMEM, the accumulator memory

TMEM is the cleanest mental break from the pre-Blackwell world.  Each
SM gets a slab of on-die memory dedicated to the tensor cores, viewed
as a grid of **128 rows × 512 columns of 32-bit cells** (= 256 KB per
SM), with the hardware managing physical placement.  An accumulator tile is some
rectangular sub-region of that grid.

You don't read or write TMEM with regular SMEM/RF instructions.  You
get three dedicated PTX operations:

* `tcgen05.alloc` — reserve `n_cols` columns; writes the starting TMEM
  address into an SMEM slot for the rest of the kernel to read.
* `tcgen05.mma` — writes into TMEM as its output.
* `tcgen05.ld` — pulls 32-bit values from TMEM into registers.

A TMEM address is a 32-bit handle with two halves: bits `[31..16]` are
the **row index** (0..127) and bits `[15..0]` are the **column index**
(0..511).  So `(taddr_base + (row_offset << 16) + col_offset)` is the
TMEM cell at `(base_row + row_offset, base_col + col_offset)`.

The matmul convention:

```cpp
__shared__ uint32_t tmem_addr_holder[1];

// One warp allocates BN columns of TMEM (cta_group::1 here):
tcgen05_alloc(/*dst smem*/ ..., /*n_cols=*/ BLOCK_N);

__syncthreads();
const uint32_t taddr = tmem_addr_holder[0];   // every thread reads it
```

The allocator returns a *base* TMEM address — typically row 0,
column 0 — and the kernel addresses into the BM × BN accumulator tile
relative to it.  At the end of the kernel the warp calls
`tcgen05.dealloc(taddr, n_cols)` to release.

A few practical notes:

* TMEM allocations are warp-level operations; only one warp issues each
  `alloc`/`dealloc`.  Other warps see the result via the SMEM holder.
* `cta_group::2` allocations span two CTAs in a cluster — the allocator
  reserves the same column range on both CTAs' TMEM, and the kernel
  addresses the resulting `2 × BM × BN` accumulator using cluster-wide
  logical row indices.  (We'll see this in the 2-CTA-cluster chapter.)
* TMEM rows are *fixed at 128 rows* per allocation block; you size by
  picking `n_cols`.  `BN = 256` means 256 columns.

> **Why TMEM exists.**  In `mma.sync` the accumulator sits in
> registers, which means the warp can't do *much else* while
> accumulating large M × N tiles — registers are scarce and the warp
> threads each hold a fragment.  Moving the accumulator to a dedicated
> on-die memory frees the warp to do other work between MMAs (TMA
> stream-in, epilogue prep, …) and removes the register-fragment
> bookkeeping entirely.  It's the change that makes warp specialization
> (chapter to come) practical at this tile size.

## Part 2 — the matrix descriptor (SMEM operand)

A 64-bit value telling the tensor core "here's the operand tile."  It
encodes:

| Bits  | Field | Meaning |
|-------|-------|---------|
| `0..13`  | start address (`addr >> 4`) | operand base in SMEM (16-byte granular, 14 bits ≈ 256 KB range) |
| `16..29` | leading byte offset (LBO `>> 4`) | byte stride along the inner-tile dimension (used for K-major B; zero for MN-major) |
| `32..45` | stride byte offset (SBO `>> 4`) | byte stride between 8-row "core matrix" groups |
| `46`     | layout mode bit | distinguishes layout variants; `1` for the common case |
| `49..51` | "base offset" / matrix-A vs B alignment hints | mostly zero in our kernels |
| `61..63` | swizzle mode | `0` = none, `1` = 32B, `2` = 64B (wait — see encoding below) |

The exact `swizzle` field encoding has a few historical conventions in
circulation.  What matters in practice is that the same value is written
on TMA's descriptor side and the matrix-descriptor side, so they agree.
We'll use `2 << 61 = SWIZZLE_128B`, which is what the matmul kernels in
this tutorial use throughout.

The whole thing comes together as a single bit-packed `uint64_t`:

```cpp
__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr) {
    constexpr uint64_t SBO = 8 * 128;            // stride between 8-row groups
    uint64_t a = ((uint64_t)smem_addr >> 4) & 0x3FFF;     // bits 0-13
    uint64_t b = ((SBO)              >> 4) & 0x3FFF;     // bits 32-45
    return a
         | (b << 32)
         | (1ULL << 46)     // layout mode
         | (2ULL << 61);    // SWIZZLE_128B
}
```

* **`smem_addr`** is the 32-bit shared-state address of the operand
  tile (`__cvta_generic_to_shared(...)` of the array).  The descriptor
  holds it shifted right by 4 — the operand base must be 16-byte
  aligned (and, for `SWIZZLE_128B`, 128-byte aligned at minimum — see
  chapter 01's rule of thumb).
* **`SBO = 8 * 128`** is the byte stride between consecutive 8-row
  groups.  Recall from chapter 01 that the descriptor walks SMEM as a
  grid of **8-row × 16-byte core matrices**.  `SBO = 8 × 128` is
  "advance 8 rows × 128-byte-wide = one full swizzle row vertical."
* The **`1 << 46`** picks the "operand is MN-major" layout variant
  (rows along M for A, along N for B).  For K-major B (a variant we'll
  see in a later chapter) you set bit 16 of `idesc` and use a
  `make_desc_K_major` variant that also fills in the `LBO` field.
* **`2 << 61`** = `SWIZZLE_128B`.  This must match the swizzle the TMA
  descriptor used; otherwise the MMA's swizzle XOR won't agree with the
  TMA's, and the operand reads land in the wrong physical chunks
  → garbage.

You build a fresh descriptor for **every MMA call** — its `smem_addr`
points to the *current* K-strip within the SMEM tile.  An MMA with
`K = 16` (BF16) consumes 16 K-elements per A row = 32 bytes, so each
inner MMA-K step advances `smem_addr` by 32 bytes:

```cpp
for (int kk = 0; kk < K_MMAS; kk++) {           // K_MMAS = BK / 16
    uint64_t a_desc = make_desc(A_BASE + kk * 32);   // 32 B per K-step
    uint64_t b_desc = make_desc(B_BASE + kk * 32);
    tcgen05_mma(taddr, a_desc, b_desc, idesc, /*accumulate=*/true);
}
```

(Real kernels' loops are slightly more elaborate because the SMEM
layout typically splits the K-dimension into 64-element "sub-tiles" —
an outer index selects the sub-tile and an inner index steps inside
one — but the pattern is the same: a new descriptor per MMA, K-strip
walked by 32 bytes at a time.)

## Part 3 — the instruction descriptor (`idesc`)

Where the matrix descriptor describes *operand placement*, `idesc`
describes the **MMA itself**: shape and dtype.  32 bits, also
bit-packed:

```cpp
__device__ __forceinline__ uint32_t make_idesc_bf16(int M, int N) {
    uint32_t d = 0;
    d |= (1u << 4);                                    // c_format = F32
    d |= (1u << 7);                                    // a_format = BF16
    d |= (1u << 10);                                   // b_format = BF16
    d |= (((uint32_t)(N >> 3) & 0x3F) << 17);          // n_dim = N/8 (6 bits)
    d |= (((uint32_t)(M >> 4) & 0x1F) << 24);          // m_dim = M/16 (5 bits)
    return d;
}
```

The relevant fields:

| Bits | Field | Encoding | Example |
|------|-------|----------|---------|
| `2..5`  | C format (accumulator dtype) | `1 = F32`, others = lower-precision | F32 |
| `7..9`  | A format | `1 = BF16`, `0 = F16`, FP8 codes, … | BF16 |
| `10..12`| B format | (same enum as A) | BF16 |
| `15`    | A transpose | `1` flips A's K-major-ness | 0 |
| `16`    | B transpose | `1` flips B's K-major-ness | 0 for MN-major B, 1 for K-major B |
| `17..22`| N dim | `N / 8` (so N ≤ 504 representable) | `BN/8` |
| `24..28`| M dim | `M / 16` (so M ≤ 240 representable on one CTA) | `BM/16 = 8` |

So one `idesc` value commits the kernel to a specific *(M, N, dtype)*
shape for that MMA.  A and B's K dimension is implied by the `kind::`
on the instruction (16 for BF16, 32 for FP8 — see chapter 01).  `idesc`
is built **once** outside the K-loop and reused for every MMA call.

For 2-CTA cluster MMA, the M-dim spans both CTAs — you'd build the
`idesc` with `M = BLOCK_M × 2 = 256`, so `m_dim = 256/16 = 16`.

## Part 4 — the instruction

```
tcgen05.mma.cta_group::N.kind::<dtype> [d_tmem], a_desc, b_desc, idesc, P;
```

* **`cta_group::N`** — `::1` issues from one CTA, `::2` is the cluster
  MMA that spans two CTAs' TMEM as one logical M=256 tile.
* **`kind::<dtype>`** — selects the inner K and operand interpretation:
  `kind::f16` (16-bit BF16/FP16, K=16), `kind::tf32` (K=8),
  `kind::f8f6f4` (K=32), etc.
* **`[d_tmem]`** — TMEM destination address (32-bit), holds the
  accumulator.
* **`a_desc`, `b_desc`** — the matrix descriptors built in Part 2
  (64-bit each).
* **`idesc`** — the instruction descriptor from Part 3 (32-bit).
* **`P`** — the **accumulate predicate**.  `P = true` → `D = A·B + D`
  (read-modify-write the existing accumulator); `P = false` →
  `D = A·B` (overwrite).  You pass `false` on the very first MMA of a
  tile to initialise the accumulator, then `true` for every subsequent
  K-step.

In CUDA C++, wrapped:

```cpp
__device__ __forceinline__ void tcgen05_mma(
    uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
    uint32_t idesc, bool enable_d)
{
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "setp.ne.b32 P, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, P;\n\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
           "r"((uint32_t)enable_d) : "memory");
}
```

Only **one thread per warp** issues each MMA (the standard "`elect_sync`
+ one issuer" pattern).  Multiple warps issuing in parallel would race
on the tensor-core pipeline and produce undefined results.

## Part 5 — async semantics: fence, commit, wait

The MMA is async, so you owe it the same hand-off ceremony TMA needs:

### Before issuing MMAs — `tcgen05.fence::after_thread_sync`

TMA writes SMEM from the **async proxy** (a separate memory pipeline);
`tcgen05.mma` reads SMEM from a different async unit.  Without an
explicit fence between them, the MMA can issue *before* TMA's bytes
are visible to the tensor cores, even though the mbarrier said the
load was complete.

```cpp
mbarrier_wait(tile_ready, phase);                 // load done in TMA proxy
asm volatile("tcgen05.fence::after_thread_sync;"); // publish to MMA proxy
// now safe to issue MMAs
```

### After issuing MMAs — `tcgen05.commit`

`tcgen05.mma` doesn't itself signal completion.  You issue all the MMAs
for a stage's K-strip back-to-back, then one `tcgen05.commit` that arms
an mbarrier to fire once **every outstanding MMA in this thread**
completes:

```cpp
for (int kk = 0; kk < K_MMAS; kk++) {
    tcgen05_mma(taddr, a_desc[kk], b_desc[kk], idesc, accumulate);
}
tcgen05_commit(mma_done_mbar);   // arms mbar to fire when all kk above finish
```

PTX:

```
tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [mbar];
```

It's the analog of `mbarrier.arrive.expect_tx` for TMA: a deferred
arrival that fires when the asynchronous work actually finishes.

### Then — `mbarrier.try_wait.parity`

Same primitive as TMA: any thread that needs the MMA result waits via
`try_wait.parity` on `mma_done_mbar`.  In a pipelined matmul that wait
gates *the next stage's TMA load into the same SMEM slot* — you can't
overwrite SMEM while a still-running MMA is reading it.

The full per-stage event chain in a single-stage matmul:

```
TMA load ──► tile_ready mbar ──► fence::after_thread_sync ──►
  K_MMAS × tcgen05.mma ──► tcgen05.commit ──► mma_done mbar ──► next iter
```

## Part 6 — reading TMEM back

When the K-loop finishes, the accumulator sits in TMEM.  To write it
to global memory you need to (a) pull it into registers, then
(b) store from registers to GMEM (typically via an SMEM staging buffer
for coalescing — chapter to come).

`tcgen05.ld.sync.aligned.<shape>.x<N>.b32` reads from TMEM into a
vector of 32-bit registers.  Common variant for BF16-output kernels:

```cpp
__device__ __forceinline__ void tcgen05_ld_32x32b_x8(
    uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
          "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7])
        : "r"(taddr));
}
```

* `32x32b` — the load shape, "32 lanes × 32-bit values."
* `x8` — packing factor: 8 32-bit values per lane = 32 BF16 (after
  converting `float → bf16` in software).
* `[taddr]` — the TMEM address, encoding the row and column the warp
  reads from.

The load is async-ish too — it can issue while the tensor cores are
still working — so a `tcgen05.wait::ld.sync.aligned` follows before
the registers are used.

A common pattern: each warp covers a 32-row strip of TMEM at a time
and loops over the N-columns:

```cpp
const uint32_t taddr_row_base = taddr + (((warp_id * 32) << 16));
for (int n = 0; n < BN; n += 8) {
    float tmp[8];
    tcgen05_ld_32x32b_x8(taddr_row_base + n, tmp);  // 8 cols at a time
    tcgen05_wait_ld();
    // convert float[8] → __nv_bfloat162[4], write to SMEM staging buffer
    ...
}
```

The pattern parallels TMA on the load side: a hardware-accelerated
bulk move (`tcgen05.ld`) followed by an explicit wait.

## What we have now

By the end of this chapter you know how each piece works in isolation:

* **TMEM** — `alloc` / `dealloc` / `ld`, addressed by
  `(row << 16) | col`.
* **Matrix descriptor** — packed 64-bit, encodes `(addr, SBO, LBO,
  layout mode, swizzle mode)`; one per MMA, advances along K.
* **`idesc`** — packed 32-bit, encodes `(c_format, a_format, b_format,
  transpose flags, n_dim, m_dim)`; built once, reused for the whole
  K-loop.
* **The instruction** — `tcgen05.mma`, async, one issuer per CTA,
  accumulate predicate flips for the first K-step.
* **Async glue** — fence between TMA and MMA, `tcgen05.commit` + mbar
  after the MMA, `tcgen05.wait::ld` after TMEM reads.

What we don't have yet is a kernel that puts them together — the K-loop
that streams swizzled TMA tiles into SMEM, runs the MMAs against them,
and reads TMEM out to GMEM.  That's chapter 03: a minimal single-stage
matmul (no pipelining, no warp specialization) that you can run on a
B200 and verify against cuBLAS.

## Run

```bash
pip install -r ../requirements.txt
python main.py
```

A Blackwell GPU (sm_100a / B200) is required.
