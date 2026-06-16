# SwiGLU Extra Store Study

Study-only experiment for the fused-SwiGLU epilogue bandwidth pattern.

The reference inspiration is:

```text
/data/home/tong/projects/fused_swiglu_kernel/fused_swiglu/triton_baseline.py
def _fused_swiglu_wide_packed_save_factors_kernel(...)
```

That Triton kernel computes a wide packed tile and writes more than the final
matmul output during the epilogue. This study keeps the MMComposer compute path
unchanged and adds one extra TMA-store output to the pipelined TMA epilogue.

Controlled variants:

- Base kernel: BN256 low-K config, `NS=5`, persistent overlap, pipelined TMA
  epilogue, `SINGLE_TMEM=0`.
- `half-only`: half-width output only, `D[M, N/2]`. The epilogue drains both
  halves of the wide tile and stores `left + right`, so both halves are live
  while final output traffic is `0.5x`.
- `base`: normal output only, `C[M, N]`, or `1.0x` final output traffic.
- `extra-half`: normal `C[M, N]` plus `D[M, N/2]`. For each `BN=256` output
  tile, `D` stores the first `BN/2=128` columns from the staged tile, packed
  into the corresponding `N/2` tile position, for `1.5x` final output traffic.
- `extra-full`: normal `C[M, N]` plus `D[M, N]`, storing every C tile chunk
  into a same-shape second output, for `2.0x` final output traffic.
- `swiglu-half`: fused SwiGLU semantics. The wide `BN=256` matmul tile is
  interpreted as `left[128] | gate[128]`; `C[M, N]` stores save-factors
  `silu(gate) | left * silu_prime(gate)`, and `D[M, N/2]` stores the final
  output `left * silu(gate)`.
- `swiglu-out`: `C[M, N]` stores the original wide matmul result
  `left | gate`, and `D[M, N/2]` stores `left * silu(gate)` using the direct
  `__expf` sigmoid form.
- `swiglu-out-fast`: same outputs as `swiglu-out`, but computes sigmoid with
  `exp2.approx(-x * log2(e))` plus `rcp.approx`, matching the CUTE kernel's
  fast epilogue style.
- `swiglu-out-fast-precompute-d`: same outputs as `swiglu-out-fast`, but
  computes and packs the final `D` chunk before waiting for the TMA-store SMEM
  buffer.
- `swiglu-out-fast-overlap-wait-d`: starts the elected TMA wait before the
  final `D` math so other epilogue warps can compute while the elected thread
  waits, then synchronizes before writing the SMEM TMA buffer.
- `swiglu-out-fast-dual-b`: same epilogue as `swiglu-out-fast`, but the wide
  `left | gate` matmul input is loaded from two separate `B_left[K, N/2]` and
  `B_gate[K, N/2]` tensor maps instead of one packed `B[K, N]` tensor.
- Generator templates are not modified; generated study copies are patched.

## Paired-Chunk SwiGLU Epilogue

The fused SwiGLU epilogue treats each `BN=256` output tile as two paired
halves:

```text
C wide tile: [ left 128 | gate 128 ]
D output:    [ left * silu(gate) 128 ]
```

The TMA-store chunk size is `STORE_N=64`, so the epilogue loops over two
paired chunks instead of four independent chunks. For each paired chunk, every
epilogue lane loads the matching `left[64]` and `gate[64]` stripe from TMEM
into registers, releases TMEM after the last pair, then emits three explicit
output chunks:

```text
1. pack/store original left chunk to C left half
2. pack/store original gate chunk to C right half
3. compute left * gate * sigmoid(gate), then pack/store to D
```

Here `pack` means converting the lane's FP32 register values into four
`__nv_bfloat162` pairs, then writing the resulting eight BF16 values to the
swizzled SMEM TMA buffer with one 128-bit store.

The left/gate data stays in registers across those three outputs. The TMA-store
stage ring is independent of the paired-chunk math: the current study advances
`store_stage` after every individual TMA output store, not after every
left/gate pair. With one stage, all three stores serialize through the same
SMEM buffer. With two stages, the stores alternate between two buffers, allowing
one earlier store to remain in flight while the epilogue prepares the next
output chunk.

## Stage-Count Conclusion

The pure matmul low-K path benefits from `TMA_STORE_STAGES=1`, largely because
it frees 16 KiB of SMEM and lets the compute ring use `NS=6`. That result does
not transfer directly to fused SwiGLU.

For `swiglu-out-fast` at `32768x4608x768`, `NS=6`:

| Variant | TMA stages | Runtime |
| --- | ---: | ---: |
| baseline source order | 1 | `220.064 us` |
| baseline source order | 2 | `191.552 us` |
| precompute final `D` before wait | 1 | `226.304 us` |
| precompute final `D` before wait | 2 | `191.616 us` |
| overlap elected wait with final `D` math | 1 | `218.048 us` |
| overlap elected wait with final `D` math | 2 | `191.584 us` |

Compared with the earlier `NS=5, TMA_STORE_STAGES=2` runs, `NS=6` gives a
small but real fused-kernel improvement when the two-store-buffer epilogue is
kept:

| Variant | NS=5, TMS=2 | NS=6, TMS=2 |
| --- | ---: | ---: |
| `swiglu-out-fast` | `195.456 us` | `191.616 us` |
| `swiglu-out-fast-dual-b` | `195.616 us` | `193.536 us` |

This corrects an earlier oversimplified explanation. The one-buffer case was
indeed disadvantaged by source structure: the original code waited before
computing and packing the final `D` chunk. But moving that math before or across
the wait did not close the gap. The main bottleneck is SMEM buffer reuse between
consecutive output stores. A paired SwiGLU chunk emits left, gate, and final
`D` stores back-to-back. With one TMA-store buffer, each next SMEM write must
wait for the previous TMA store to drain. With two buffers, the epilogue can
write the next output chunk into the alternate buffer while the previous output
store remains in flight.

Conditional rule from this study:

```text
store-only / pure matmul BN256 low-K: prefer TMA_STORE_STAGES=1
fused SwiGLU multi-output epilogue:   prefer TMA_STORE_STAGES=2
```

Run from the repo root on a B200 allocation:

```bash
srunpy study-swiglu-extra-store/bench_extra_store.py --shape 32768x4608x768 --shape 32768x9216x768 --stages 1,2
srunpy study-swiglu-extra-store/bench_extra_store.py --variants swiglu-out-fast-dual-b --stages 2 --shape 32768x4608x768
```

Default controlled-study timing is `do_bench warmup=1000ms rep=1000ms`.

The fastest shareable dual-B kernel from this study is the `NS=6`,
`TMA_STORE_STAGES=2` variant:

```text
study-swiglu-extra-store/fused_matmul_swiglu_out_fast_dual_b_ns6_s2.cu
study-swiglu-extra-store/host_fused_matmul_swiglu_out_fast_dual_b_ns6_s2.py
```

Run it directly with:

```bash
srunpy study-swiglu-extra-store/host_fused_matmul_swiglu_out_fast_dual_b_ns6_s2.py
```

The host builds the five TMA descriptors once after tensor allocation and
reuses the same launch arguments for correctness and `do_bench` timing. If
tensor pointers, shape, strides, or TMA box dimensions change, rebuild the
descriptors before launching again.

The older `fused_matmul_swiglu_dual_b.cu` and
`fused_matmul_swiglu_out_fast_dual_b_s2.cu` files are `NS=5` variants. They
are useful historical artifacts, but the measured best dual-B source is the
`ns6_s2` file above.

Other shareable fused sources are also saved as:

```text
study-swiglu-extra-store/fused_matmul_swiglu.cu
```

## STORE_N=128 Debug Artifact

The active benchmark harness is intentionally back to the known-good
`STORE_N=64` path. A copy of the temporary `STORE_N=128` experiment is kept at:

```text
study-swiglu-extra-store/bench_extra_store_store_n128_debug.py
```

That experiment compiled representative kernels, but the GPU run failed while
encoding the C/D TMA tensor maps with `TMA_SWIZZLE_128B`, before kernel launch.
The likely issue is that bf16 `STORE_N=128` makes each row 256 bytes, while the
current swizzled TMA-store layout is the 128-byte path used by `STORE_N=64`.
