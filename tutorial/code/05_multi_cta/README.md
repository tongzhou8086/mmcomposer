# Multi-CTA grid mapping — covering the whole output

> 📁 **Code on GitHub:** [`tutorial/code/05_multi_cta/`](https://github.com/tongzhou8086/mmcomposer/tree/master/tutorial/code/05_multi_cta) — `kernel.cu` + `main.py`.

Chapter 04 ran fast for a single CTA but only computed one
`(BM × BN) = (128 × 256)` output tile per launch.  On a B200 with
~148 SMs, that left the rest of the chip idle — which is why the
absolute throughput came out at ~6 TFLOPS instead of the hundreds the
hardware can deliver.

This chapter tiles the **whole output** across many CTAs.  Each CTA
owns one `(bid_m, bid_n)` tile and runs the *unchanged* chapter-04
kernel body on it.  The grid is sized to cover `M × N`:

```
   grid = (M / BM, N / BN)  flattened into blockIdx.x
   CTAs run in parallel on different SMs
```

That's the entire idea.  No change to multi-stage buffering, warp
specialization, or the MMA inner loop — those are about overlap
*within* a tile.  This chapter is about how *many* tiles a launch
covers.

## What changes from chapter 04

Three small additions, all inside the same kernel:

### 1. Derive `(bid_m, bid_n)` from `blockIdx.x`

We launch a 1-D grid and fold it into a 2-D tile coordinate.  N-major
walk (CTAs sweep across N first, then advance in M):

```cpp
const int grid_n = N / BN;
const int bid_m  = blockIdx.x / grid_n;
const int bid_n  = blockIdx.x % grid_n;

const int off_m  = bid_m * BM;     // this CTA's M-offset into A and C
const int off_n  = bid_n * BN;     // this CTA's N-offset into B and C
```

`N`, `M` are now runtime parameters (not just `BM`, `BN`).  Walk
patterns smarter than N-major (M-major, Triton-style chunked walks
for L2 reuse) are a later optimization chapter.

### 2. TMA coordinates pick up the per-CTA offset

The K-window still slides per iteration (`k_iter * BK`); the
operand-row dimension now starts at the CTA's offset:

```cpp
tma_2d_load(A_slot, &A_tmap, /*x=*/ k_iter * BK, /*y=*/ off_m, ready_mb);
tma_2d_load(B_slot, &B_tmap, /*x=*/ k_iter * BK, /*y=*/ off_n, ready_mb);
```

Same descriptors as before — built once on the full `(M, K)` and
`(N, K)` tensors — they're just queried with different `y` coords by
different CTAs.

### 3. Output store uses the per-CTA offset

The epilogue still gives each thread one row of its own `BM × BN`
tile; now that row sits inside the global `M × N` output at offset
`(off_m, off_n)`:

```cpp
const int my_row_global = off_m + warp_id * 32 + lane;
...
*reinterpret_cast<int4*>(&C_ptr[my_row_global * N + off_n + n]) = ...;
```

That's it.  The K-loop, the warp-specialized TMA/MMA structure, the
multi-stage ring, the `tcgen05.ld` epilogue — all unchanged from
chapter 04.

## Performance — real TFLOPS, not microbenchmark numbers

Single CTA in ch04 measured ~5 TFLOPS because most of the GPU was
parked.  With a full grid the SMs actually fill up.  Measured on B200
via `triton.testing.do_bench` (`main.py` re-runs these — your numbers
will be in the same ballpark):

| shape (M, N, K) | grid | this kernel | PyTorch `A @ B` (cuBLAS) | us / cuBLAS |
|---|---|---|---|---|
| `2048³` | 16 × 8 = 128 CTAs (one wave)    | **541 TFLOPS** |  876 TFLOPS | **62%** |
| `4096³` | 32 × 16 = 512 CTAs (multi-wave) | **770 TFLOPS** | 1443 TFLOPS | **53%** |
| `8192³` | 64 × 32 = 2048 CTAs (many waves)| **826 TFLOPS** | 1454 TFLOPS | **57%** |

From ~5 TFLOPS (ch04, single CTA) to **~800 TFLOPS** — roughly **160×**
just from sizing the grid to the problem.

Our absolute throughput climbs with shape (541 → 770 → 826), but the
**us / cuBLAS** ratio doesn't move monotonically.  At `2048³` cuBLAS
itself is hobbled by the small problem (876 TFLOPS, far below its 1.4
TFLOPS plateau), so a 62% ratio there reflects cuBLAS struggling at
small shapes as much as our kernel doing well.  At `4096³` and `8192³`
cuBLAS hits its stride (~1450 TFLOPS) while we lag at ~800; that ~50%
gap is the headroom the next chapters chase — chunked grid walk for L2
reuse, 2-CTA cluster MMA, coalesced SMEM-staged epilogue.

We're not at parity yet — `cuBLAS` (what PyTorch calls under the hood)
uses CTA-tile swizzling for L2 reuse on A, 2-CTA cluster MMA, and a
coalesced SMEM-staged epilogue.  Closing the gap is the business of
the next several chapters; this one just gets the absolute numbers
off the floor and onto the same order of magnitude as the production
library.

## What this chapter *doesn't* do

- **N-major walk only.**  Adjacent CTAs in N share the same A-slabs
  but stream different B-slabs.  Adjacent CTAs in M share B-slabs but
  stream different A-slabs.  A smarter walk (M-major, or Triton-style
  chunked) trades one for the other for better L2 reuse — covered in
  the CTA-swizzle chapter.
- **No 2-CTA cluster MMA.**  Each CTA is still `cta_group::1` and
  owns one `(BM × BN)` tile.  2-CTA cluster (covered later) lets two
  CTAs cooperate on a single `(2·BM × BN)` MMA, dramatically reducing
  per-tile setup overhead.
- **No epilogue coalescing.**  Direct TMEM→GMEM, same uncoalesced
  pattern as ch02–04.  At this point in the ladder the kernel is
  bandwidth-bound on the epilogue stores more than the matmul itself.

## Take-away

Multi-CTA grid mapping is the cheapest large performance step
remaining: a few lines of indexing math turn a single-tile kernel
into a real matmul that uses the whole GPU.  Every subsequent
optimization (CTA-tile L2 swizzling, 2-CTA cluster MMA, occupancy
tuning, epilogue coalescing) is meaningful only because *this*
chapter put a real grid in front of the SMs.

## Run

```bash
pip install -r ../requirements.txt
python main.py
```

A Blackwell GPU (sm_100a / B200) is required.
