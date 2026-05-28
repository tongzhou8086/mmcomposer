# K-major B — drop the host transpose

> 📁 **Code on GitHub:** [`tutorial/code/06_k_major_b/`](https://github.com/tongzhou8086/mmcomposer/tree/master/tutorial/code/06_k_major_b) — `kernel.cu` + `main.py`.

Every kernel from chapter 02 through chapter 05 secretly did this in
its launcher:

```python
B = torch.randn(K, N, ...)           # the user's row-major B
B_t = B.t().contiguous()             # ← internal preprocessing copy
A_tmap = encode_tensor_map(..., gptr=A.data_ptr(), ...)
B_tmap = encode_tensor_map(..., gptr=B_t.data_ptr(), ...)   # TMA reads B_t, not B
```

That `B.t().contiguous()` is a full `M·K·2` bytes of extra HBM traffic
on every kernel call where B isn't already cached.  In a "weights are
reused across many calls" scenario (e.g. transformer inference) the
cost is amortized; in a one-shot matmul or any setting where B
changes, the transpose is real overhead that cuBLAS doesn't pay.

This chapter drops it.  The kernel reads from B's **native `(K, N)`
row-major GMEM** directly.  Everything else — the multi-stage ring,
the warp specialization, the grid, the epilogue — stays exactly as in
chapter 05.

## What "K-major B" means

`tcgen05.mma` understands two SMEM layouts for the B operand,
controlled by `idesc` bit 16:

| `idesc` bit 16 | tcgen05 name | B SMEM layout | Used in chs |
|---|---|---|---|
| `0` | "MN-major B" | `[BN rows][BK cols, K-innermost]` — N is the row axis, K is innermost  | 02–05 (matches the *transposed* B's natural shape) |
| `1` | "K-major B"  | `[BN/64 sub-tiles][BK rows][64 cols, N-innermost]` — K is the row axis, N is innermost  | **this chapter** (matches the *original* B's natural shape) |

The names are confusing on first read: *"K-major"* doesn't mean "K is
contiguous in memory" — it means "K is the row dimension of the SMEM
tile."  For a B matrix that's `(K, N)` row-major in GMEM (N
contiguous), K-major SMEM layout is the one TMA can produce *directly*
without the host doing any transpose.

## Three changes from chapter 05

### 1. SMEM layout: B is sub-tiled along N

SWIZZLE_128B requires each SMEM row to be exactly 128 bytes wide (= 64
BF16 elements).  When the operand's row dimension is N (as in K-major
B) and `BN > 64`, the tile has to be sub-divided along N into
**N-sub-tiles of 64 cols each**, stacked.  For our `BN = 256`:

```
   B SMEM layout (per stage slot):

       [BN/64 = 4 sub-tiles]    each sub-tile is:
       ┌───┐ ┌───┐ ┌───┐ ┌───┐    [BK rows × 64 cols]    K-major
       │ 0 │ │ 1 │ │ 2 │ │ 3 │    = BK × 128 B           one swizzle atom
       │   │ │   │ │   │ │   │                            per row
       └───┘ └───┘ └───┘ └───┘
        \─────── BN total ───────/

       sub-tile s sits at SMEM offset  s · (BK · 128) bytes
                                        = s · 8 KB        (for BK = 64)
```

Total B SMEM bytes per slot = `BN · BK · 2 = 32 KB`, same as before —
just sliced differently.

### 2. TMA: one sub-tile per call, looped over N

Chapter 05 issued one TMA per B tile.  Now we issue **`BN / 64 = 4`
TMAs per stage** for B, each loading one N-sub-tile of `(64 N-cols ×
BK K-rows)`:

```cpp
constexpr int BN_SUB = 64;
constexpr int N_SUBS = BN / BN_SUB;     // 4 for BN=256

for (int n_sub = 0; n_sub < N_SUBS; n_sub++) {
    tma_2d_load(B_base(slot) + n_sub * BK * 128,    // sub-tile s
                &B_tmap,
                /*x=*/ off_n + n_sub * BN_SUB,      // N-coord (innermost)
                /*y=*/ k_iter * BK,                 // K-coord (outer)
                ready_mb);
}
```

The TMA descriptor itself is built on B's native `(K, N)` GMEM —
`global_dim = [N, K]`, `global_strides = [N · 2]`, `box_dim = [64,
BK]`.  All four sub-tile calls share the same descriptor; they just
query it at different N-coords.

(A's TMA is unchanged from ch05 — A was already row-major-friendly.)

### 3. MMA descriptor: `make_desc_K_major` with `LBO`

The matrix descriptor needs a second variant.  K-major B uses the
**leading byte offset** field (`LBO`) to tell the tensor cores how to
walk between N-sub-tiles internally:

```cpp
__device__ __forceinline__ uint64_t make_desc_K_major(
    uint32_t smem_addr, int lbo_bytes
) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a   = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    uint64_t lbo = ((uint64_t)lbo_bytes >> 4) & 0x3FFFULL;
    uint64_t b   = ((SBO)               >> 4) & 0x3FFFULL;
    return a | (lbo << 16) | (b << 32) | (1ULL << 46) | (2ULL << 61);
}
```

The only new field versus `make_desc` is the LBO at bits 16–29.  For
our B layout, `LBO = BK · 128` (the byte distance between consecutive
N-sub-tiles).

And the MMA loop's B descriptor pointing changes accordingly — it
points at **sub-tile 0** at the current K-strip, and the hardware uses
LBO to walk to sub-tiles 1, 2, 3 on its own:

```cpp
for (int kk = 0; kk < K_MMAS; kk++) {
    const uint64_t a_desc = make_desc(A_base(slot) + kk * 32);
    const uint64_t b_desc = make_desc_K_major(
        B_base(slot) + kk * 16 * 128,    // sub-tile 0, K-row kk*16
        BK * 128);                        // LBO = inter-sub-tile stride
    tcgen05_mma(taddr, a_desc, b_desc, idesc, ...);
}
```

One MMA reads `MMA_K = 16` K-rows from each of the four sub-tiles, for
a total of `16 K × 256 N` from B — same operand shape as chapter 05.

Plus the `idesc`:

```cpp
d |= (1u << 16);     // B is K-major
```

## What the host launcher loses

Just the transpose.  The whole `B_t = B.t().contiguous()` line is
gone, B is passed straight through:

```python
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")   # pass as-is
...
B_tmap = encode_tensor_map(
    ..., gptr=B.data_ptr(),                      # ← no _t
    global_dim=[N, K], global_strides=[N * ELEM_BYTES],
    box_dim=[BN_SUB, BK], ...)                   # 64 N-cols × BK K-rows
```

## Performance

The kernel does the same total work as ch05 — same MMAs, same TMA
bytes, same epilogue — so per-call kernel time should be essentially
unchanged.  `main.py` runs three problem sizes (2048³, 4096³, 8192³)
and reports `us/call` and TFLOPS against PyTorch's cuBLAS path.

What *did* change is the **end-to-end story**: a one-shot `A @ B` no
longer pays a `K · N · 2` byte transpose before launching.  In settings
where B is cached for many calls (transformer weights), the chs 02–05
kernels are fine; in settings where B is fresh per call, ch06 is
fairer to compare against cuBLAS.

The `idesc` bit-16 flip and the sub-tile loop add a small amount of
inner-loop bookkeeping but no measurable per-call cost — the work the
SMs do is identical.

## Take-away

`tcgen05.mma` has two B-layout conventions baked into `idesc`:
**MN-major** (chs 02–05, requires the operand-row dim to be the SMEM
row → fits a transposed B naturally) and **K-major** (this chapter,
K is the SMEM row → fits a native row-major B naturally).  The choice
is just a descriptor field — once you build the matching SMEM layout
and descriptor, the rest of the kernel is identical.

For the rest of the tutorial we'd typically pick whichever matches
the operand's natural GMEM layout (here, K-major B for `(K, N)`
row-major).  The MN-major path stays available for kernels that
happen to have B already transposed in GMEM, or that want to share a
single descriptor convention with A.

## Run

```bash
pip install -r ../requirements.txt
python main.py
```

A Blackwell GPU (sm_100a / B200) is required.
