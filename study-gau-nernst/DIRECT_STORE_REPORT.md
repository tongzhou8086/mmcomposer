# Direct-store epilogue investigation report

Date: 2026-06-11

Target shape: `32768 x 4608 x 768` on B200.

All performance numbers below use `triton.testing.do_bench` with
`warmup=1000` ms and `rep=1000` ms unless explicitly noted.  They are single-run
medians.  cuBLAS varies between allocations, so the main comparison is absolute
kernel TFLOPS plus the same-run cuBLAS ratio as supporting context.

## Executive summary

The original direct-store overlap bug is fixed by making the overlapped cluster
path use a cluster-wide barrier before final `tcgen05.dealloc`.  The direct
epilogue then runs correctly on multi-tile persistent workloads.

The performance gap to gau-nernst v7c is not caused by the two idle warps in our
overlap skeleton.  The main gap is the B TMA/layout path:

- Our 2D row-major-B TMA path reaches about `1237 TFLOPS` after direct-store
  fixes and `L1::no_allocate`.
- Switching to 3D TMA with gau-style transposed-B storage reaches `1278 TFLOPS`,
  effectively matching gau v7c (`1283 TFLOPS` in the same 1000/1000 timing style).
- Keeping normal row-major B storage but using a 3D TMA B path reaches
  `1266 TFLOPS`.  That recovers most of the gap while preserving the eventual
  product direction of regular row-major B.

## Experiment table

| Variant | B storage / TMA path | Correct? | TFLOPS | Delta | Notes |
|---|---|---:|---:|---:|---|
| Staged baseline (`ours_kernel.cu`) | row-major B, 2D TMA, staged split epilogue | yes | `1199` | baseline | Current stable staged config, rerun at 1000/1000. |
| Failing direct (`ours_direct_FAILING_*`) | row-major B, 2D TMA, direct x8 | no | n/a | n/a | Launch-fails at multi-tile persistent shape. |
| Fixed direct x8 (`ours_direct_FIXED_*`) | row-major B, 2D TMA, direct x8 | yes | `1007` | n/a | Barrier fix makes it correct, but x8 direct store is slow. |
| Fixed direct x16 (`ours_direct_FIXED_LD16_*`) | row-major B, 2D TMA, direct x16 | yes | `1197` | `+190` vs x8 | LD16 is required to make direct competitive with staged. |
| Direct x16 + `L1::no_allocate` (`EXP_L1NOALLOC`) | row-major B, 2D TMA | yes | `1237` | `+40` vs fixed x16 | Gau-style global-store cache modifier is a real win. |
| 6-warp layout (`EXP_6W`) | row-major B, 2D TMA | yes | `1211` | neutral | Removing the two idle warps does not explain the gap. |
| 6-warp + `L1::no_allocate` (`EXP_6W_L1NOALLOC`) | row-major B, 2D TMA | yes | `1237` | `+0` vs 8-warp + L1 | Confirms warp layout is not material here. |
| 3D TMA, transposed B (`EXP_3DTMA`) | gau-style `(N,K)` B storage, 3D TMA | yes | `1278` | `+41` vs 6W+L1 | Matches gau's memory path and essentially closes the gap. |
| 3D TMA, row-major B (`EXP_3DTMA_ROWMAJOR_B`) | normal `(K,N)` B storage, 3D TMA | yes | `1266` | `+29` vs 6W+L1 | Preserves row-major B and recovers most of the 3D-TMA benefit. |
| Staged + `L1::no_allocate` (`ours_staged_EXP_L1NOALLOC`) | row-major B, 2D TMA, staged split epilogue | yes | `1266` | `+67` vs staged baseline | L1 no-allocate helps staged coalesced stores too. |
| Staged + B 3D TMA + `L1::no_allocate` (`ours_staged_EXP_B3D_L1NOALLOC`) | normal `(K,N)` B storage, B-only 3D TMA | yes | `1258` | `-8` vs staged+L1 | B 3D alone did not add beyond the store-cache win in this run. |
| Staged + A/B 3D TMA + `L1::no_allocate` (`ours_staged_EXP_AB3D_L1NOALLOC`) | normal `(K,N)` B storage, A/B 3D TMA | yes | `1260` | `-6` vs staged+L1 | Apples-to-apples with direct row-major 3D memory path; essentially tied. |
| gau-nernst v7c (`bench_v7c_1000.py`) | gau reference | yes | `1283` | reference | Same target shape and 1000/1000 timing. |

## What helped

1. Cluster barrier before final dealloc.

   The original direct+overlap kernel failed after all tiles completed.  The
   fix is to replace the CTA-local cleanup sync before `tcgen05_dealloc` with a
   cluster-wide barrier in the 2-CTA overlap path.  This matches gau's cleanup
   ordering and makes multi-tile persistent direct store correct.

2. `TCGEN05_LD_WIDTH = 16`.

   The x8 direct-store path reached only `1007 TFLOPS`.  Switching to x16 reached
   `1197 TFLOPS`, roughly `+190 TFLOPS`, and brought direct store back to staged
   baseline performance.

3. `st.relaxed.cta.global.L1::no_allocate`.

   This raised the x16 direct path from `1197` to `1237 TFLOPS`, about
   `+40 TFLOPS`.  It also raised the staged split epilogue from `1199` to
   `1266 TFLOPS`, about `+67 TFLOPS`.  This is worth carrying into both final
   direct and staged epilogues.

4. 3D TMA for B.

   The 2D row-major-B path issues two B TMA loads per K tile per CTA.  Gau's 3D
   path issues one B TMA load for the full CTA-local B panel.  Moving to 3D TMA
   is the largest remaining performance lever after the direct-store epilogue
   itself is fixed.

## What did not help

- Removing the two idle warps did not materially improve performance.  The
  combined 6-warp + L1-no-allocate result stayed at `1237 TFLOPS`, the same as
  the 8-warp L1-no-allocate variant.
- Increasing the K-ring depth did not help in the earlier fixed direct LD16
  trials: `NS=5` measured about `1173 TFLOPS`, and `NS=6` about `1122 TFLOPS`,
  both below `NS=4`.
- A mismatched 3D B layout/MMA descriptor is incorrect.  For gau-style
  transposed-B 3D TMA, the B K-major descriptor bit must be cleared.  For normal
  row-major-B 3D TMA, the K-major B descriptor path must be retained.

## Row-major B with 3D TMA

The row-major-B 3D experiment is the most relevant product result.

It keeps `B` as normal contiguous `(K, N)` row-major storage and uses a rank-3
tensor map over the N dimension split into 64-column chunks:

```text
global_dim     = [64, K, N / 64]
global_strides = [N * sizeof(bf16), 128]
box_dim        = [64, BK, BN_LOCAL / 64]
coords         = [0, k * BK, local_n / 64]
```

The kernel keeps the K-major B MMA descriptor path (`make_desc_K_major` and the
B K-major bit in `idesc`).  This variant is correct and measured `1266 TFLOPS`,
only about `12 TFLOPS` below the transposed-B 3D experiment and about `17 TFLOPS`
below gau v7c in these runs.

Conclusion: we do not need to adopt gau's transposed-B storage for the product
path.  A row-major-B 3D TMA implementation appears to recover nearly all of the
missing performance while preserving the desired input layout.

## Staged store with the same upgrades

After the initial direct-store investigation, we also tested the original staged
split epilogue with the same two upgrades:

- `st.relaxed.cta.global.L1::no_allocate` for the final coalesced
  `C_sh -> C_ptr` int4 stores.
- 3D TMA for normal row-major `(K, N)` B storage.

The results were important:

| Variant | TFLOPS | Interpretation |
|---|---:|---|
| staged baseline | `1199` | Original split staged path at 1000/1000. |
| staged + L1 no-allocate | `1266` | Big gain from the global-store cache modifier alone. |
| staged + B-only 3D TMA + L1 no-allocate | `1258` | B 3D did not improve staged beyond L1 in this run. |
| staged + A/B 3D TMA + L1 no-allocate | `1260` | Essentially tied with direct row-major A/B 3D (`1266`). |
| direct + row-major A/B 3D + L1 no-allocate | `1266` | Best row-major direct-store experiment. |

This changes the interpretation: the earlier staged-vs-direct gap was mostly
not the SMEM staging itself.  Once the staged path uses the same L1 no-allocate
store policy, it catches up to the direct row-major 3D result.  Direct store is
still attractive because it removes epilogue SMEM and barriers, but for this
shape the cache policy and TMA/layout path explain more of the measured gap than
the staged-vs-direct distinction.

## Recommended generator changes later

Do not update the generator until these study changes are cleaned up and folded
in deliberately.  The likely final change set is:

1. Add `EPILOGUE_DIRECT` as a real tunable mode.
2. In the 2-CTA overlap path, use a cluster-wide cleanup barrier before
   `tcgen05_dealloc`.
3. Use LD16 and `st.relaxed.cta.global.L1::no_allocate` for direct stores.
4. Add a 3D TMA B-load mode for normal row-major `(K, N)` B storage, with the
   K-major B MMA descriptor retained.
5. Re-run the compatibility/perf matrix before exposing the knob in the UI.

## Study files

- `ours_direct_FAILING_kernel.cu`, `ours_direct_FAILING_host.py`: original
  multi-tile direct+overlap failure reproducer.
- `ours_direct_FIXED_kernel.cu`, `ours_direct_FIXED_host.py`: correctness fix
  for the direct path.
- `ours_direct_FIXED_LD16_kernel.cu`, `ours_direct_FIXED_LD16_host.py`: LD16
  direct-store baseline.
- `ours_direct_EXP_L1NOALLOC_kernel.cu`: L1 no-allocate store experiment.
- `ours_direct_EXP_6W_kernel.cu`, `ours_direct_EXP_6W_host.py`: gau-style
  6-warp layout experiment.
- `ours_direct_EXP_6W_L1NOALLOC_kernel.cu`,
  `ours_direct_EXP_6W_L1NOALLOC_host.py`: combined 6-warp + L1 no-allocate.
- `ours_direct_EXP_3DTMA_kernel.cu`, `ours_direct_EXP_3DTMA_host.py`: gau-style
  transposed-B 3D TMA experiment.
- `ours_direct_EXP_3DTMA_ROWMAJOR_B_kernel.cu`,
  `ours_direct_EXP_3DTMA_ROWMAJOR_B_host.py`: normal row-major-B 3D TMA
  experiment.
- `ours_staged_EXP_L1NOALLOC_kernel.cu`: staged split epilogue with
  L1-no-allocate global stores.
- `ours_staged_EXP_B3D_L1NOALLOC_kernel.cu`,
  `ours_staged_EXP_B3D_L1NOALLOC_host.py`: staged split epilogue with B-only
  row-major 3D TMA plus L1-no-allocate global stores.
- `ours_staged_EXP_AB3D_L1NOALLOC_kernel.cu`,
  `ours_staged_EXP_AB3D_L1NOALLOC_host.py`: staged split epilogue with A/B
  row-major 3D TMA plus L1-no-allocate global stores.
- `bench_v7c_1000.py`: gau v7c timing at 1000/1000.
