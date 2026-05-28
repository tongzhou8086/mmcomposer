# TMA with 128-byte swizzling

Chapter 00 loaded bytes from global memory into shared memory with
`SWIZZLE_NONE` — SMEM came out byte-for-byte identical to the source.
This chapter changes exactly one thing: `SWIZZLE_NONE` →
`SWIZZLE_128B` (Another building block of high performance matmul), and watches what happens to the layout.


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
every access — worst-case conflicts.

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
moves whole chunks — never elements within a chunk.

To give a quick preview of what SWIZZLE_128B does: it applies a
**per-row XOR** to the chunk index of each row of the tile — row 0 is
XOR-ed by 0 (untouched), row 1 by 1, row 2 by 2, …, row 7 by 7.  So
row 0 comes out identical to the input, row 7 comes out with its 8
chunks fully reversed, and the rows in between sit on a smooth gradient
between those two extremes.  We derive the rule and the full table in
[the next section](#the-128b-swizzle-rule); for now just know that
*every row gets its own permutation, keyed by its row index*.

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

One sentence to anchor everything that follows:

> **128B swizzle permutes 16-byte chunks within each 128-byte row,
> keyed by the row index.**

Three quantities, and it's worth keeping them distinct because they're
easy to blur together:

| Term | What it is | BF16 value |
|------|-----------|------------|
| **chunk** — the *unit* that moves | a fixed **16 bytes** = **4 banks** | 8 BF16 |
| **row** — the *window* it moves within | **128 bytes** = 8 chunks = 32 banks | 64 BF16 |
| **key** — the XOR amount | the **row index** `r mod 8` | — |

So a 128-byte SMEM row holds eight 16-byte chunks, and the swizzle
reshuffles those eight chunks *among themselves* — a chunk never leaves
its 128-byte row, and the 8 BF16 inside a chunk always ride together.
The permutation for row `r` is a single XOR:

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

**Read each cell as a whole 16-byte chunk, not a single element.**  This
trips everyone up at least once.  A cell is one chunk = 8 BF16 = 4
banks; it is *not* a 2-byte value.  So the 8 columns are 8 chunks =
`8 × 4 = 32` banks = one full 128-byte row.  (We return to the bank
arithmetic — and why this layout is conflict-free — in
[its own section](#why-this-eliminates-bank-conflicts) below.)

(`32B` and `64B` swizzle are the same idea over 2 and 4 chunks
respectively; `128B` is what matmul uses, so it's the one we show.)

### Reading the table by flipping bits

XOR with a value flips every bit position where that value has a `1`,
and leaves the rest unchanged.  With 3-bit chunk indices (0–7):

* `XOR 1` (`001`) → flips the **last** bit only.
* `XOR 2` (`010`) → flips the **middle** bit only.
* `XOR 4` (`100`) → flips the **top** bit only.
* `XOR 7` (`111`) → flips **all three** bits.

So you can derive any row of the table by flipping the marked bits of
each chunk index:

```
chunk:   0    1    2    3    4    5    6    7
binary: 000  001  010  011  100  101  110  111

XOR 1 → flip last bit:
        001  000  011  010  101  100  111  110
      =  1    0    3    2    5    4    7    6     (neighbour swaps)

XOR 2 → flip middle bit:
        010  011  000  001  110  111  100  101
      =  2    3    0    1    6    7    4    5

XOR 3 → flip last two bits:
        011  010  001  000  111  110  101  100
      =  3    2    1    0    7    6    5    4

XOR 4 → flip top bit:
        100  101  110  111  000  001  010  011
      =  4    5    6    7    0    1    2    3     (halves swapped)

XOR 7 → flip all three:
        111  110  101  100  011  010  001  000
      =  7    6    5    4    3    2    1    0     (full reverse)
```

A composite key like `XOR 3` (`011`) just flips the last two bits at
once (last + middle), which is why it looks like `XOR 1` and `XOR 2`
applied together.  Each row's permutation is fully determined by which
bits of `(row mod 8)` are set.

### Does the pattern depend on the dtype?

No — and this is the useful part.  128B swizzle is defined on the
**byte address**, not on elements.  Its unit is a fixed **16-byte
chunk** (= 4 banks), and a 128-byte swizzle row is always 8 such chunks.
The permutation `chunk XOR (row mod 8)` is therefore *identical* for
FP8, BF16/FP16, and FP32.  The only thing the dtype changes is how many
elements ride inside each 16-byte chunk:

| dtype        | bytes/elem | elems per 16B chunk | elems per 128B row |
|--------------|:----------:|:-------------------:|:------------------:|
| FP8 / int8   | 1          | 16                  | 128                |
| BF16 / FP16  | 2          | 8                   | 64                 |
| TF32 / FP32  | 4          | 4                   | 32                 |

So the eight-way chunk shuffle drawn above is exactly what you'd see at
any precision; switch to FP32 and each cell of the table just holds 4
elements instead of 8, and a full swizzle row spans 32 columns instead
of 64.  Practically this means the K-tile width that fills one swizzle
atom scales with the dtype: 64 K for BF16, but 128 K for FP8 and 32 K
for FP32.

(The `32B` and `64B` swizzle modes are likewise byte-defined — they
permute 2 and 4 chunks respectively. Picking a mode is about how wide a
conflict-free region the consumer needs, never about the dtype.)

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

### Full-row view — seeing the 16-byte unit

The print above shows *one representative per chunk*, which is only
valid because the input is constant within each 16-byte chunk.  Drop
that collapse and print all 64 elements of a row — grouping into chunks
with `|` — and the atomic unit becomes literal (the script does this for
rows 0, 1, 4; `N×8` here means "the value `N` repeated 8 times"):

```
  in  row 1:  10×8 | 11×8 | 12×8 | 13×8 | 14×8 | 15×8 | 16×8 | 17×8
  out row 1:  11×8 | 10×8 | 13×8 | 12×8 | 15×8 | 14×8 | 17×8 | 16×8
  in  row 4:  40×8 | 41×8 | 42×8 | 43×8 | 44×8 | 45×8 | 46×8 | 47×8
  out row 4:  44×8 | 45×8 | 46×8 | 47×8 | 40×8 | 41×8 | 42×8 | 43×8
```

Everything about the 16-byte unit is visible here:

* **Each `|`-group is one chunk** = 8 identical BF16 = 16 bytes = 4
  banks.  That run-of-8 is exactly what the collapsed print represented
  with a single number.
* **Chunks move as whole blocks.**  Row 1 (XOR 1) swaps blocks in pairs
  — the `10`-block and `11`-block trade places, neither is split.  Row 4
  (XOR 4) swaps the two halves — the four high blocks jump ahead of the
  four low ones, each block intact.
* **Within a block, order is never touched** — there's no
  `11 11 10 …` scrambling.  The permutation is strictly chunk-granular.
* **Row 0 (XOR 0)** is identical in/out — the baseline.

(The constant-within-chunk input can't reveal whether the 8 elements
*inside* a chunk keep their order — they're all equal.  Set
`g_in[r][c] = c` instead and a row prints `0 1 … 7 | 8 9 … 15 | …`; after
swizzle you'd see those ascending 8-wide blocks reordered with their
internal `0..7` runs preserved — confirming intra-chunk order is
untouched.)

## Why this eliminates bank conflicts

This is the whole point of swizzling, so it's worth doing carefully.

**Banks, in one line.**  SMEM is split into **32 banks of 4 bytes** each;
`bank = (byte_offset / 4) mod 32`.  A warp can hit all 32 banks in one
cycle, but if two threads address *different* words in the *same* bank,
the access serializes — a conflict.

### The table works at two scales

Re-reading the swizzle table with the "each cell is a 16-byte chunk"
fact, every cell, row, and column maps cleanly onto banks:

```
 one CELL    = 1 chunk  = 8 BF16 = 16 B = 4 banks
 one ROW     = 8 chunks          = 128 B = 32 banks   (a full bank cycle)
 one COLUMN  = a physical slot   = a fixed 4-bank group across all rows
```

A chunk (cell) starts at bank `4 × physical_slot`, so the 8 columns of
the table land on bank groups `0–3, 4–7, 8–11, …, 28–31` — tiling all
32 banks exactly once.

### The access that conflicts

The MMA reads operands via `ldmatrix`-style 8×8 tiles.  Crucially, **one
`ldmatrix` matrix-row is 8 BF16 = exactly one chunk.**  An 8×8 load is 8
lanes, each fetching one chunk from a *different SMEM row*, all at the
*same logical column* (say column 0 = logical chunk 0).  Eight identical
logical requests — the worst case.

**Without swizzle**, all 8 rows keep logical chunk 0 in physical slot 0
= banks 0–3.  Eight lanes pile onto 4 banks → 8-way conflict, 28 banks
idle:

```
            bank:  0  1  2  3 │ 4 ............................ 31
 lanes 0–7      →  ███████████ │          (all idle)
                   ▲ 8 lanes contend for banks 0–3  → 8 transactions
```

**With swizzle**, logical chunk 0 of row `r` sits at physical slot
`0 XOR r = r` — the diagonal of the table — so the 8 lanes fan out
across all 32 banks, 4 each:

```
            bank:  0  1  2  3 │ 4  5  6  7 │ 8  9 10 11 │ ... │28 29 30 31
 lane 0 (row 0) →  ███████████ │            │            │     │
 lane 1 (row 1) →             │ ███████████ │            │     │
 lane 2 (row 2) →             │            │ ███████████ │     │
   ...                                                          ███████████
                   └─ 8 lanes × 4 banks = 32 banks, no overlap → 1 transaction ─┘
```

The numbers close perfectly: the SMEM port delivers `32 banks × 4 B =
128 B` per cycle, and 8 lanes read `8 × 16 B = 128 B`.  A conflict-free
8-row read is *exactly* one full bank cycle — which only happens because
the 8 chunks cover all 32 banks instead of stacking on 4.

### Why XOR, and why it's free

XOR-ing a fixed logical-chunk index with `0..7` always yields a
permutation of `0..7`, so **whatever column** the MMA reads, the 8 rows
always scatter across 8 distinct 4-bank groups — no column is special.
An additive shift wouldn't guarantee that (it aliases for some
patterns).  And because XOR is its own inverse, the consumer recovers
the data by applying the *same* formula — the swizzle is simultaneously
the write layout and the read addressing, costing nothing but a few XOR
gates on the address.

This is also why both matmul operands are swizzled identically:
`ldmatrix` reads A and B the same way — each lane a contiguous 16-byte
chunk, consecutive lanes on different rows — so both hit this columnar
pattern and both need the same XOR.  (The only genuinely contiguous,
conflict-free reads in the kernel are the within-chunk 16-byte run a
single lane grabs, and the coalesced GMEM→SMEM staging copy — neither is
what the swizzle protects.)

## Who reads the swizzled tile — `ldmatrix` vs `tcgen05.mma`

The bank discussion above was framed around `ldmatrix`, the
Ampere/Ada/Hopper way of getting operands into the tensor core.  On
B200 the matmul chapters use `tcgen05.mma` instead, which reads SMEM
differently — but, satisfyingly, around the *same* 8×8 unit.

**`ldmatrix` (legacy path).**  The warp cooperatively loads operand
fragments SMEM→registers in **8×8 tiles**, then `mma.sync` consumes the
registers.  The 8×8 is a register-fragment granularity, and *you* compute
the swizzled address in software.  In the Ampere-style kernel the `XOR`
is right there in the source — both in the staging copy and the load:

```cpp
// staging into SMEM — explicit swizzle:
const int sc   = ((c/8) ^ ((r/A_SHIFT) % A_SWZ)) * 8 + (c%8);
// ldmatrix read — explicit swizzle again:
const int phys = lg ^ ((ar / A_SHIFT) % A_SWZ);
```

**`tcgen05.mma` (Blackwell path).**  There are *no* operand registers and
*no* `ldmatrix`.  A single instruction computes a whole tile —
`M = 128` (or 256 across a 2-CTA cluster) × `N ≤ 256` × `K = 16` for
16-bit — and the tensor core reads the operand slabs straight from SMEM,
addressed by a 64-bit **matrix descriptor**.  The swizzle is one field
of that descriptor, so the hardware does all the XOR addressing
internally — the kernel never computes a swizzled index.

### The 8×8 reappears as the *core matrix*

Even though the `tcgen05.mma` instruction is huge, the SMEM layout its
descriptor describes is built from **core matrices** — for 16-bit types,
**8 rows × 16 bytes = 8×8 BF16**.  The descriptor's leading/stride byte
offsets (`LBO`/`SBO`) are how the hardware steps between core matrices as
it gathers an operand (e.g. `SBO = 8 * 128` in our `make_desc`).

So the 8×8 you know from `ldmatrix` doesn't disappear — it changes role:

```
 mma.sync  (Ampere)   : 8×8 = the ldmatrix register fragment you load by hand
 wgmma     (Hopper)   : 8×8 = the SMEM core matrix the descriptor tiles
 tcgen05   (Blackwell): 8×8 = the SMEM core matrix the descriptor tiles
```

And this is the answer to "why is the swizzle period 8 rows?" — because
the **core matrix is 8 rows tall**.  The 8-row swizzle period, the 8-row
core matrix, and the 8-deep `ldmatrix` tile are the same constant for the
same reason: 8 chunks × 4 banks = the 32-bank SMEM, the granularity at
which a swizzled read is conflict-free.

> **A precision worth keeping.**  The 8×8 core matrix is the unit the
> operand *layout, addressing, and swizzle* are built around — that's
> documented and is what you program against.  Whether it's also the
> literal per-cycle SMEM read transaction is a microarchitectural detail
> Nvidia doesn't expose, so treat "8×8 core matrix" as the layout atom,
> not a claim about bus-transaction width.

| | `ldmatrix` + `mma.sync` | `tcgen05.mma` |
|---|---|---|
| operand path | SMEM → registers → MMA | SMEM → MMA (direct) |
| role of 8×8 | register-load fragment | SMEM core matrix |
| who reads SMEM | 32 warp lanes | tensor-core read ports |
| swizzle applied by | software (you write the XOR) | hardware (descriptor field) |
| per-instruction shape | `m16n8k16`-ish | `M128 × N≤256 × K16` |

Building that matrix descriptor — base address, `SBO`, swizzle field,
plus the instruction-shape `idesc` — is exactly what the next chapter
sets up.  For now the takeaway is just: the swizzled SMEM layout we built
here is consumed natively by `tcgen05.mma` via its descriptor, with the
same 8×8 / 16-byte / 32-bank structure underneath.

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
* 128B swizzle permutes **16-byte chunks** (the unit, dtype-independent
  — 8 BF16, 4 banks) **within each 128-byte row** (the window — 8
  chunks, 32 banks), keyed by the row index: `chunk XOR (row mod 8)`.
  Row 0 is unchanged; row 7 is fully reversed.
* That permutation is exactly what makes the read conflict-free: 8
  `ldmatrix` lanes asking for the same logical chunk of 8 different rows
  land on 8 distinct 4-bank groups (the table's diagonal), covering all
  32 banks in one cycle instead of stacking on 4.
* The XOR key is the **absolute** SMEM offset, so a tile that doesn't
  start at a 1024-byte boundary sees a rotated pattern — harmless for
  real kernels (the MMA descriptor uses the same address), but a trap
  if you read SMEM by hand.

Next chapter: the first real matmul.  We keep this exact swizzled TMA
load, bump `boxDim` to a matmul tile, and feed the SMEM straight into
`tcgen05.mma` — which consumes the swizzled layout natively.
