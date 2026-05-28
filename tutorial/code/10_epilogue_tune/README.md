# Tuning the epilogue — `NUM_WARPS` and `tcgen05.ld` width

> 📁 **Code on GitHub:** [`tutorial/code/10_epilogue_tune/`](https://github.com/tongzhou8086/mmcomposer/tree/master/tutorial/code/10_epilogue_tune) — `kernel.cu` + `main.py`.

Up to ch09 the kernel has used **4 warps** per CTA and the **`.x8`**
variant of `tcgen05.ld` to read TMEM in the epilogue.  Both are
defaults — neither has ever been measured against alternatives.  This
chapter introduces both as **tunable knobs**, sweeps the full cross,
and sees what wins on a representative shape.

The K-loop body is untouched.  Only the epilogue changes.

## What the two knobs do

### `NUM_WARPS` — total warps per CTA

The main loop only ever uses warp 0 (TMA) and warp 1 (MMA) — extra
warps just wait on `all_mmas_done`.  But once we enter the epilogue,
**every warp can contribute**.  With 4 warps, each lane owns a row of
the BM × BN tile and walks the full BN columns.  With 8 warps, we
split *both* M-rows and N-cols across warps in the **b41 pattern**:

```cpp
const int row_warp = warp_id & 3;       // 0..3 — picks TMEM 32-row strip
const int col_warp = warp_id >> 2;      // 0..1 — picks N-half
const int my_row   = row_warp * 32 + lane;
const int col_base = col_warp * (BN / 2);
const int col_end  = col_base + (BN / 2);
```

Phase 1 (`tcgen05.ld` → SMEM) per warp now covers `32 rows × BN/2 cols`
instead of `32 × BN` — half the work per warp.  Phase 2 (SMEM → GMEM)
likewise halves the per-thread store count because `TB_SIZE` doubles.

### `LD_X` — `tcgen05.ld` packing factor

The PTX `tcgen05.ld.32x32b.x<N>` family reads `32 lanes × N cols` of
TMEM per call.  `.x8` (what we've used through ch09) gives each lane
8 floats per call.  Wider variants:

| variant | floats per lane per call | calls per warp at full-BN | regs/lane (FP32) |
|---|---|---|---|
| `.x8`  |  8 |  32 |  8 |
| `.x16` | 16 |  16 | 16 |
| `.x32` | 32 |   8 | 32 |
| `.x64` | 64 |   4 | 64 |

Wider reads amortize the per-instruction overhead but spend more
registers per lane.  Beyond some point, register pressure starts to
hurt — finding where is the point of the sweep.

The kernel exposes a small dispatch template:

```cpp
template <int LD_X>
__device__ __forceinline__ void tcgen05_ld_packed(uint32_t taddr, float* out) {
    if constexpr (LD_X ==  8) tcgen05_ld_32x32b_x8 (taddr, out);
    else if constexpr (LD_X == 16) tcgen05_ld_32x32b_x16(taddr, out);
    else if constexpr (LD_X == 32) tcgen05_ld_32x32b_x32(taddr, out);
    else if constexpr (LD_X == 64) tcgen05_ld_32x32b_x64(taddr, out);
}
```

and the epilogue inner loop becomes:

```cpp
for (int n = col_base; n < col_end; n += LD_X) {
    float tmp[LD_X];
    tcgen05_ld_packed<LD_X>(taddr_row_base + n, tmp);
    tcgen05_wait_ld();
    // pack LD_X floats → LD_X/2 bfloat162 → LD_X/8 int4s in SMEM
}
```

## Performance sweep

`M = N = K = 8192`, `NS = 5`, `GSM = 8` (best from ch09).  Full
`NUM_WARPS × LD_X` cross:

| | `x8` | `x16` | `x32` | `x64` |
|---|---|---|---|---|
| **4 warps** | **1333** | 1308 | 1256 | 1279 |
| 8 warps     | 1305     | 1249 | 1294 | 1301 |

(TFLOPS, higher is better.  ch09's reference at the same `(NS, GSM)`
was ~1225 TFLOPS, so the whole table sits within run-to-run noise of
that baseline.)

## Interpretation — the K-loop dominates at this shape

The full sweep clusters within ~7% (1249–1333), and the **default
4-warp / `.x8` configuration is the best**.  Surprising at first — why
don't more warps or wider loads help?

Because at `K = 8192`, the K-loop runs **128 iterations per CTA**
before the epilogue runs once.  The K-loop is bound by TMA bandwidth
and tensor-core throughput, both of which are already at near-peak
utilization (we're at ~97 % of cuBLAS).  The epilogue is a small
fraction of total runtime — even a 50 % epilogue speedup would barely
register in end-to-end TFLOPS.

You can roughly read each cell as "kernel time" with the epilogue
substituted for that config.  Differences between cells are
differences in **epilogue time alone**, which is ~5-10 % of total.
50 % of 5-10 % is 2-5 %.  Right around our noise floor.

## Why the chapter is still worth its slot

Two reasons even though the headline gain at this shape is roughly
nothing:

1. **The knobs become meaningful at other shapes.**  At small K (say
   `K = 512`), the K-loop is short and the epilogue's share of total
   time is much larger — so making it 30 % faster *would* show up.
   We measured at `K = 8192` because that's where everything earlier
   in the ladder was tuned, but a different operating point would
   tilt the table.  Picking the right knob value per shape is
   exactly what the next chapter (autotuning) does.

2. **The pattern is general.**  "Split work across more warps" and
   "process larger tiles per instruction" are two of the most
   commonly tunable knobs in production kernels.  Even if they don't
   buy anything *here*, the reader has now seen the mechanics
   (templated kernel, parameterized epilogue, full sweep) and can
   apply the same shape to other parts of the kernel — or other
   kernels entirely.

## Caveat — register pressure at `.x64`

`.x64` makes each lane hold 64 FP32 = 256 bytes of registers during
the load.  At 32 lanes per warp × 8 warps × 64 = 16384 floats =
64 KB of register state per CTA just for the in-flight tcgen05.ld
data.  B200's register file is 64K × 32-bit per SM, so this fits
exactly one CTA — meaning the cluster's two CTAs each take one full
SM, no occupancy pad.  At smaller `LD_X` there's room to spare.
This is part of why `.x64` doesn't pay off here: register pressure
nibbles back what wider loads should buy.

## Run

```bash
pip install -r ../requirements.txt
python main.py
```

A Blackwell GPU (sm_100a / B200) is required.
