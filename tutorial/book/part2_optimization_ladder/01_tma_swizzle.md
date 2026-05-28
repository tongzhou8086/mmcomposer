# TMA with 128-byte swizzling

Chapter 00 loaded bytes from global memory into shared memory with
`SWIZZLE_NONE` — SMEM came out byte-for-byte identical to the source.
This chapter changes exactly one thing: `SWIZZLE_NONE` →
`SWIZZLE_128B`, and watches what happens to the layout.

Still no matmul.  The goal is to *see*, concretely, the byte
rearrangement that `tcgen05.mma` will depend on in the next chapter —
so that when we get there, the swizzled SMEM layout is already
familiar instead of magic.

> 📁 **Runnable code:** [`tutorial/code/01_tma_swizzle/`](https://github.com/tongzhou8086/mmcomposer/tree/master/tutorial/code/01_tma_swizzle).
> Run with `pip install -r ../requirements.txt && python main.py` on a
> Blackwell (sm_100a) GPU.  All the numbers printed in this chapter are
> its real output.

## Why swizzle at all?

When many threads read SMEM at the same time, the hardware serves them
through 32 **banks** (4 bytes wide each).  If two threads in a warp hit
two different addresses in the *same* bank, the accesses serialize —
a **bank conflict**.

A matmul consumer like `tcgen05.mma` reads operand tiles from SMEM in a
very regular strided pattern.  Stored naively (row-major, `SWIZZLE_NONE`),
that pattern makes whole groups of threads land in the same bank on
every access — worst-case conflicts, throughput cut by up to 8×.

Swizzling fixes this by permuting where each 16-byte chunk physically
lands, so the consumer's strided reads spread across all 32 banks.  The
beautiful part: **TMA applies the permutation for free on the way in.**
You set one field in the descriptor and the bytes arrive pre-arranged
for conflict-free MMA reads.  The cost is the subject of this chapter —
SMEM no longer mirrors global memory, so you can't read it back
linearly and expect the source order.

## What changes from chapter 00

Three lines, nothing else:

| | Chapter 00 | Chapter 01 |
|---|---|---|
| descriptor `swizzle` | `TMA_SWIZZLE_NONE` | `TMA_SWIZZLE_128B` |
| `box_dim` | `[COLS, 1]` (one row) | `[COLS, ROWS]` (whole tile) |
| input | `torch.randn` | structured (see below) |

We load the **whole 8×64 tile in one TMA** because the swizzle
permutation is *row-dependent* — a single row wouldn't reveal it.  And
we use a structured input instead of random noise so the permutation is
legible: each 16-byte chunk (8 consecutive BF16 values) is filled with
the constant `row*10 + chunk_index`.  Since 128B swizzle only ever
moves whole chunks — never elements within a chunk — chunk-constant
values show the reordering with nothing to distract from it.

```python
row_idx   = torch.arange(ROWS, device="cuda").view(ROWS, 1)
chunk_idx = (torch.arange(COLS, device="cuda") // 8).view(1, COLS)
g_in = (row_idx * 10 + chunk_idx).to(torch.bfloat16)   # chunk c of row r == r*10 + c
```

The descriptor call is chapter 00's, with the two field changes:

```python
tmap = encode_tensor_map(
    dtype=TMA_BFLOAT16,
    rank=2,
    gptr=g_in.data_ptr(),
    global_dim=[COLS, ROWS],            # innermost first
    global_strides=[COLS * ELEM_BYTES],
    box_dim=[COLS, ROWS],               # whole tile in one load
    element_strides=[1, 1],
    swizzle=TMA_SWIZZLE_128B,           # <-- the subject of this chapter
)
```

## The 128B swizzle rule

With BF16, a 128-byte SMEM row holds 64 elements = **eight 16-byte
chunks** of 8 BF16 each.  128B swizzle permutes those eight chunks, and
the permutation for row `r` is a single XOR:

```
physical_chunk = logical_chunk  XOR  (r mod 8)
```

That's it.  Row 0 XORs by 0 (identity — no change).  Row 1 XORs by 1
(swap neighbours).  Row 7 XORs by 7 (full reversal).  Because XOR is
its own inverse, the same formula reads both directions: physical chunk
`pc` of row `r` holds logical chunk `pc XOR (r mod 8)`.

```
        chunk:   0   1   2   3   4   5   6   7
                ┌───┬───┬───┬───┬───┬───┬───┬───┐
   row 0  XOR 0 │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │   identity
   row 1  XOR 1 │ 1 │ 0 │ 3 │ 2 │ 5 │ 4 │ 7 │ 6 │   swap pairs
   row 2  XOR 2 │ 2 │ 3 │ 0 │ 1 │ 6 │ 7 │ 4 │ 5 │
   row 3  XOR 3 │ 3 │ 2 │ 1 │ 0 │ 7 │ 6 │ 5 │ 4 │
   row 4  XOR 4 │ 4 │ 5 │ 6 │ 7 │ 0 │ 1 │ 2 │ 3 │   swap halves
   row 5  XOR 5 │ 5 │ 4 │ 7 │ 6 │ 1 │ 0 │ 3 │ 2 │
   row 6  XOR 6 │ 6 │ 7 │ 4 │ 5 │ 2 │ 3 │ 0 │ 1 │
   row 7  XOR 7 │ 7 │ 6 │ 5 │ 4 │ 3 │ 2 │ 1 │ 0 │   full reverse
                └───┴───┴───┴───┴───┴───┴───┴───┘
                  each cell = which LOGICAL chunk lands here
```

(`32B` and `64B` swizzle are the same idea over 2 and 4 chunks
respectively; `128B` is what matmul uses, so it's the one we show.)

## Seeing it on real hardware

The kernel loads the tile with `SWIZZLE_128B`, then copies SMEM out to
`g_out` in **linear physical order** — `g_out[i] = smem[i]`.  So `g_out`
is a faithful snapshot of how the bytes actually sit in shared memory.
Printing one representative value per chunk:

```
INPUT  g_in  (natural, un-swizzled chunk order):
  row 0: [ 0,  1,  2,  3,  4,  5,  6,  7]
  row 1: [10, 11, 12, 13, 14, 15, 16, 17]
  row 2: [20, 21, 22, 23, 24, 25, 26, 27]
  row 3: [30, 31, 32, 33, 34, 35, 36, 37]
  row 4: [40, 41, 42, 43, 44, 45, 46, 47]
  row 5: [50, 51, 52, 53, 54, 55, 56, 57]
  row 6: [60, 61, 62, 63, 64, 65, 66, 67]
  row 7: [70, 71, 72, 73, 74, 75, 76, 77]

OUTPUT g_out (raw SWIZZLE_128B SMEM layout):
  row 0: [ 0,  1,  2,  3,  4,  5,  6,  7]   ← unchanged (XOR 0)
  row 1: [11, 10, 13, 12, 15, 14, 17, 16]   ← neighbours swapped (XOR 1)
  row 2: [22, 23, 20, 21, 26, 27, 24, 25]   ← XOR 2
  row 3: [33, 32, 31, 30, 37, 36, 35, 34]   ← XOR 3
  row 4: [44, 45, 46, 47, 40, 41, 42, 43]   ← halves swapped (XOR 4)
  row 5: [55, 54, 57, 56, 51, 50, 53, 52]   ← XOR 5
  row 6: [66, 67, 64, 65, 62, 63, 60, 61]   ← XOR 6
  row 7: [77, 76, 75, 74, 73, 72, 71, 70]   ← fully reversed (XOR 7)
```

Read off any row and the rule holds exactly.  Row 1's first chunk
(physical chunk 0) holds the value `11` — that's logical chunk
`0 XOR 1 = 1`.  Row 4's layout is the natural order with its two halves
swapped, which is what `XOR 4` does to a 0..7 index.  This is the
hardware's actual output, verified against `pc XOR (r mod 8)` in the
script.

The takeaway for the next chapter: **after a swizzled TMA load, you
cannot index SMEM with logical `(row, col)` and expect the source
element.**  The `tcgen05.mma` matrix descriptor knows the swizzle mode
and undoes the permutation internally, so the MMA still computes the
right product — but any hand-written SMEM access has to apply the same
XOR.

## Aside — swizzle is keyed off the *absolute* SMEM offset

A subtlety that costs people hours.  The XOR amount isn't derived from
your array index; it's derived from the **absolute byte offset within
the SMEM window**.  Concretely, the chunk's XOR key is bits `[7,10)` of
that offset — i.e. `(absolute_offset / 128) mod 8`.

If your tile doesn't start at a 1024-byte-aligned window offset, the
whole pattern shifts.  In an earlier draft of this kernel the tile was
preceded by a static `__shared__ uint64_t mbar`, which — with the
tile's 128-byte alignment — pushed the tile to **window offset 128**.
That's one chunk-row of shift, and the output came out as
`pc XOR ((r+1) mod 8)`: row 0 was XOR-1, row 7 was the identity.
Everything was rotated by one row.

The fix in the runnable kernel is to place the tile **first** in
dynamic shared memory (window offset 0) and carve the mbarrier out
*after* it:

```cpp
extern __shared__ __align__(128) __nv_bfloat16 smem[];   // tile @ offset 0
uint64_t* mbar_ptr = reinterpret_cast<uint64_t*>(&smem[ROWS * COLS]);
```

Why this never bites real matmul kernels: the `tcgen05.mma` matrix
descriptor is built from the *same* SMEM address the TMA wrote to, and
it encodes the swizzle mode.  Both sides compute the XOR from the same
absolute offset, so they always agree — the permutation is
self-consistent regardless of where the tile sits.  It only bites you
here because *we* are the consumer, reading SMEM by hand and comparing
against a model that assumed offset 0.

## The kernel

Identical to chapter 00 except for the tile-first SMEM layout, the
whole-tile byte count, and the linear copy-out:

```cpp
constexpr unsigned ROWS = 8, COLS = 64;
constexpr unsigned TILE_BYTES = ROWS * COLS * 2;   // 1024

extern "C" __global__ void tma_swizzle_demo(
    const __grid_constant__ CUtensorMap tmap,
    __nv_bfloat16* __restrict__ g_out
) {
    extern __shared__ __align__(128) __nv_bfloat16 smem[];   // tile @ offset 0
    uint64_t* mbar_ptr = reinterpret_cast<uint64_t*>(&smem[ROWS * COLS]);
    const uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(mbar_ptr);
    const uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem);

    // 1) init mbarrier
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar_addr));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    // 2) one thread issues the whole-tile TMA load
    if (threadIdx.x == 0) {
        const int coord_x = 0, coord_y = 0;
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];"
            :: "r"(smem_addr), "l"(&tmap), "r"(coord_x), "r"(coord_y), "r"(mbar_addr)
            : "memory");
        asm volatile(
            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
            :: "r"(mbar_addr), "r"(TILE_BYTES) : "memory");
    }

    // 3) all threads wait
    const uint32_t phase = 0;
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE_%=;\n\t bra WAIT_%=;\n\t DONE_%=:\n\t }"
        :: "r"(mbar_addr), "r"(phase) : "memory");

    // 4) copy SMEM out in linear physical order — g_out mirrors the
    //    swizzled layout
    for (int i = threadIdx.x; i < ROWS * COLS; i += blockDim.x)
        g_out[i] = smem[i];
}
```

The mbarrier handshake (init / `arrive.expect_tx` / `try_wait.parity`)
is exactly chapter 00's — see there for the line-by-line walkthrough.
The only mechanical differences are `TILE_BYTES = 1024` instead of
`128` (we now expect the whole tile), and the copy-out loop striding
over all 512 elements.

## What to take away

* Swizzling exists to make the consumer's strided SMEM reads
  **bank-conflict-free**; `tcgen05.mma` requires it.
* TMA applies the swizzle on arrival — one descriptor field, zero extra
  instructions.
* 128B swizzle for BF16 permutes eight 16-byte chunks per row by
  `chunk XOR (row mod 8)`.  Row 0 is unchanged; row 7 is fully
  reversed.
* The XOR key is the **absolute** SMEM offset, so a tile that doesn't
  start at a 1024-byte boundary sees a rotated pattern — harmless for
  real kernels (the MMA descriptor uses the same address), but a trap
  if you read SMEM by hand.

Next chapter: the first real matmul.  We keep this exact swizzled TMA
load, bump `boxDim` to a matmul tile, and feed the SMEM straight into
`tcgen05.mma` — which consumes the swizzled layout natively.
