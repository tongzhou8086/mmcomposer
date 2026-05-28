# Coalesced SMEM-staged epilogue

> 📁 **Code on GitHub:** [`tutorial/code/07_coalesced_epilogue/`](https://github.com/tongzhou8086/mmcomposer/tree/master/tutorial/code/07_coalesced_epilogue) — `kernel.cu` + `main.py`.

Every chapter from 02 onward closed with the same caveat: *"direct
TMEM → GMEM writeback, uncoalesced — comes later."*  This is later.

Recall the problem from ch02's writeback section: each thread owns one
output row, reads 8 FP32 from TMEM with `tcgen05.ld`, packs to a 16-byte
`int4`, and stores it to `C[my_row, n..n+7]`.  Within a warp, lane `L`
and lane `L+1` are storing to rows that are **`N * 2 = 512 bytes` apart**
in GMEM at any given `n`.  That's 32 scattered transactions per warp
per call instead of one coalesced cycle.  Fine when we just wanted the
kernel to work; expensive once the rest of the kernel is tight.

The fix is a **two-phase write** that decouples the "shape that
`tcgen05.ld` hands you" from the "shape GMEM wants."

```
   ch02–06 epilogue (one phase, uncoalesced):

     TMEM ──tcgen05.ld──► registers (per-thread row layout) ──int4 store──► GMEM
                                                      └─ 32 scattered transactions / warp


   ch07 epilogue (two phases, coalesced phase 2):

     phase 1:  TMEM ──tcgen05.ld──► registers ──int4 store──► SMEM (per-thread row layout)
     phase 2:  SMEM ──re-mapped read──► registers ──int4 store──► GMEM
                                                      └─ 1 coalesced transaction / warp
```

Phase 1 keeps the natural "lane L owns row L" layout the `tcgen05.ld`
instruction produces — the stores go to a padded SMEM buffer rather
than GMEM.  Phase 2 is the new piece: after a `__syncthreads`, all
threads re-read the SMEM buffer with a **flat thread-major index** so
consecutive lanes hit consecutive GMEM addresses, producing one
coalesced GMEM transaction per warp.

## The SMEM staging buffer reuses the A/B slots

By the time the epilogue runs, the multi-stage A/B SMEM ring is no
longer needed — the K-loop has finished and `all_mmas_done` has fired.
So we **reuse the same SMEM** as the staging buffer.  No extra SMEM
allocation, no extra `cuFuncSetAttribute`:

```cpp
constexpr int BN_PAD = BN + 8;                            // padded row width — see below
auto C_sh = reinterpret_cast<__nv_bfloat16(*)[BN_PAD]>(smem);
```

The staging buffer needs `BM × BN_PAD × 2 = 128 × 264 × 2 = 67584 B`
≈ 66 KB, which comfortably fits in the existing `NS × 48 KB = 96 KB`
multi-stage allocation.  The cast is just a different *view* of the
same memory.

## Phase 1 — TMEM → SMEM (the natural layout)

Identical structure to ch02–06's writeback, but the `int4` store lands
in SMEM instead of GMEM:

```cpp
const int my_row = warp_id * 32 + lane;
const uint32_t taddr_row_base = taddr + ((uint32_t)(warp_id * 32) << 16);

for (int n = 0; n < BN; n += 8) {
    float tmp[8];
    tcgen05_ld_32x32b_x8(taddr_row_base + (uint32_t)n, tmp);
    tcgen05_wait_ld();

    __nv_bfloat162 packed[4];
    for (int i = 0; i < 4; i++)
        packed[i] = __floats2bfloat162_rn(tmp[2*i], tmp[2*i+1]);

    *reinterpret_cast<int4*>(&C_sh[my_row][n]) =
        *reinterpret_cast<int4*>(packed);
}
__syncthreads();
```

This is still the columnar pattern (32 lanes writing to 32 different
rows at the same column).  Without padding, every lane in the warp
would hit the **same** SMEM bank because the row stride is exactly
`BN × 2 = 512 B = 4 × 32 banks` (the `r × 32-bank-cycles` term
vanishes mod 32, just like chapter 01's bank analysis).  32-way
conflict.

## The padding trick — `BN_PAD = BN + 8`

Padding each SMEM row by 8 BF16 makes the row stride `(BN + 8) × 2 =
528 B = 132 × 4 B-words`, so the per-row bank-shift is no longer a
multiple of 32.  Bank for lane L's column-0 store:

```
   row stride = 132 words   →   bank shift per row = 132 mod 32 = 4 banks
   32 lanes  ×  4 banks/lane  =  128 banks  mod 32  =  8 distinct banks
   → 4-way conflict instead of 32-way.
```

We don't hit zero conflicts (would need a stride coprime with 32,
which conflicts with the 16-byte alignment the `int4` stores require),
but going from 32-way to 4-way is the cheap structural win — that's
why `+8` is the standard padding in production kernels.

> **Why `+8` specifically?**  It's the smallest 16-byte-aligned
> padding (8 BF16 = 16 B, preserving `int4`-store alignment) that
> breaks the worst-case all-on-one-bank pattern.  Larger pads
> (`+16`, `+24`, …) don't help further at this geometry.

## Phase 2 — SMEM → GMEM, coalesced

After the `__syncthreads`, every row of `C_sh` is populated.  Now we
re-read it with a **flat, thread-major layout** so consecutive lanes
write consecutive 16-byte chunks to GMEM:

```cpp
constexpr int CHUNK_BF16        = 8;                          // 16 B int4 per store
constexpr int CHUNKS_PER_ROW    = BN / CHUNK_BF16;            // 32 for BN=256
constexpr int STORES_PER_THREAD = (BM * BN) / (THREADS * CHUNK_BF16);    // 32

for (int s = 0; s < STORES_PER_THREAD; s++) {
    const int flat = tid + s * THREADS;
    const int row  = flat / CHUNKS_PER_ROW;
    const int col  = (flat % CHUNKS_PER_ROW) * CHUNK_BF16;
    const int gr   = off_m + row;
    const int gc   = off_n + col;
    *reinterpret_cast<int4*>(&C_ptr[gr * N + gc]) =
        *reinterpret_cast<const int4*>(&C_sh[row][col]);
}
```

Reading off the indexing:

- Iter `s = 0`: `flat = tid ∈ {0..127}`.  Threads `0..31` (warp 0) hit
  `row = 0`, `col = 0, 8, …, 248`.  Consecutive lanes → consecutive
  16-byte chunks → **one coalesced 512-byte GMEM transaction per
  warp**.
- Iter `s = 1`: `flat = tid + 128`, all four warps advance to the next
  row band.
- 32 iters total cover `BM × BN = 32 768` BF16 = `4 096` `int4` chunks
  = `4096 / 128 = 32` stores per thread.

The SMEM read in this phase is *not* the columnar pattern from phase 1
— it's a flat read where consecutive lanes hit consecutive SMEM
addresses.  Combined with the row-padding, both phases avoid the
worst-case bank traffic.

## Performance

`main.py` compiles both this kernel and chapter 06's, runs them on
three shapes, and reports the head-to-head.  Measured on B200:

| shape | ch06 (direct) | ch07 (coalesced) | speedup |
|---|---|---|---|
| `2048³` | 534 TFLOPS | **557 TFLOPS** | **1.04×** |
| `4096³` | 791 TFLOPS | **803 TFLOPS** | **1.02×** |
| `8192³` | 811 TFLOPS | 785 TFLOPS | 0.97× |

**Honest read:** the standalone win is modest at small shapes
(`+4%` at 2K) and inverts slightly at 8K.  The reason is that the
epilogue is only a meaningful fraction of total runtime when the
K-loop is short — at `K = 8192` we run 128 K-iterations per CTA, and
the few-microsecond epilogue gets dwarfed.  The extra
`__syncthreads` + SMEM round-trip of the two-phase pattern adds a
small overhead that doesn't pay for itself when there's nothing to
amortize it against.

**Why the chapter still matters:** this is structural scaffolding for
chapter 08.  With only 4 warps, the two-phase structure is just "spend
fixed overhead to coalesce."  With 8 warps (the b41 pattern), the extra
warps split phase 1 (TMEM reads in half along N) *and* halve phase 2's
per-thread work — both phases become roughly 2× faster, the coalescing
benefit compounds with the parallelism, and the headline win lands at
8K too.  Coalescing alone doesn't move 8K; coalescing + 8 warps does,
and you need the two-phase split to *enable* the 8-warp parallelism in
the first place.

So treat ch07 as "set up the layout the next chapter exploits."  The
table here will look much friendlier after ch08.

## What's still on the table

After this chapter the gap to cuBLAS is roughly:

- **2-CTA cluster MMA** (`cta_group::2`) — two CTAs share one MMA on a
  `2·BM × BN` tile, halving per-tile setup cost.
- **CTA-tile L2 swizzle** (Triton-style chunked walk) — better L2
  reuse on A.
- **8-warp kernel + epilogue parallelism** — the natural follow-on to
  this chapter, since the two-phase structure makes it trivial to
  split epilogue work across more warps.
- **NS tuning / first taste of autotuning**.

## Take-away

The "direct writeback was uncoalesced" debt has a clean structural
fix: split the epilogue into two phases (TMEM → SMEM in the natural
layout, then SMEM → GMEM in the GMEM-coalesced layout) with a small
SMEM padding to keep both phases off the worst-case bank pattern.
SMEM is free to reuse since the A/B tiles aren't needed once the
K-loop is done.  No new instructions; just a different way to compose
the ones we already had.

## Run

```bash
pip install -r ../requirements.txt
python main.py
```

A Blackwell GPU (sm_100a / B200) is required.
