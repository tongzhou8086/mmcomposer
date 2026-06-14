# TMA Store Stage Count Results

This study tests only generated-kernel copies under `study-tma-store-stages/`.
No generator template is modified.

Config under test:

```text
BM=128 BN=256 BK=64 NS=5 GSM=1 NW=4
PERSISTENT=1 OVERLAP=1 SPLIT=0 L1NA=0 TMA_PIPE=1 SINGLE_TMEM=0 TWO_CTA=1
```

The production pipelined TMA epilogue currently uses two store-stage SMEM
buffers. This study patches that generated copy to use 1, 2, 3, or 4 active
TMA-store stages.

## Shared Memory

Per CTA, for `BM=128`, `BN=256`, `BK=64`, `NS=5`:

| TMA store stages | Epilogue store SMEM | Total dynamic SMEM |
| ---: | ---: | ---: |
| 1 | 16 KiB | 177 KiB |
| 2 | 32 KiB | 193 KiB |
| 3 | 48 KiB | 209 KiB |
| 4 | 64 KiB | 225 KiB |

B200 opt-in dynamic shared memory limit queried by the script:

```text
232448 B = 227 KiB
```

So four store stages fit, but with only about `2 KiB` headroom.

## Stable Timing

Default controlled-study timing is now:

```text
triton.testing.do_bench(warmup=1000ms, rep=1000ms)
cuBLAS = median of 10 measured samples after 1 throwaway sample
```

### Low-K Shapes

Artifact:
`study-tma-store-stages/_scratch/results_lowk_aspects_1000_1000.json`

| Shape | S1 TFLOPS | S2 TFLOPS | S3 TFLOPS | S4 TFLOPS | Best |
| --- | ---: | ---: | ---: | ---: | --- |
| `8192x4608x768` | 1204.7 | 1203.9 | 1203.1 | 1203.1 | S1 |
| `16384x4608x768` | 1301.7 | 1272.9 | 1269.8 | 1246.2 | S1 |
| `32768x2304x768` | 1244.5 | 1218.9 | 1216.1 | 1216.1 | S1 |
| `32768x4608x768` | 1324.8 | 1308.5 | 1279.4 | 1279.4 | S1 |
| `65536x2304x768` | 1292.4 | 1251.3 | 1237.0 | 1237.2 | S1 |
| `32768x9216x768` | 1320.9 | 1275.6 | 1254.4 | 1251.4 | S1 |

| Shape | cuBLAS TFLOPS | S1 | S2 | S3 | S4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `8192x4608x768` | 1312.0 | 91.8% | 91.8% | 91.7% | 91.7% |
| `16384x4608x768` | 1361.3 | 95.6% | 93.5% | 93.3% | 91.5% |
| `32768x2304x768` | 1330.6 | 93.5% | 91.6% | 91.4% | 91.4% |
| `32768x4608x768` | 1356.0 | 97.7% | 96.5% | 94.4% | 94.4% |
| `65536x2304x768` | 1323.9 | 97.6% | 94.5% | 93.4% | 93.5% |
| `32768x9216x768` | 1344.4 | 98.3% | 94.9% | 93.3% | 93.1% |

### Low-K Shape Trend

For this config, each persistent cluster tile covers `2 * BM = 256` rows and
`BN = 256` columns. The low-K shapes above therefore map to this cluster-tile
grid:

| Shape | Cluster tiles M x N | Total cluster tiles | S1 | S2 | S3 | S4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `8192x4608x768` | `32 x 18` | 576 | 91.8% | 91.8% | 91.7% | 91.7% |
| `16384x4608x768` | `64 x 18` | 1152 | 95.6% | 93.5% | 93.3% | 91.5% |
| `32768x2304x768` | `128 x 9` | 1152 | 93.5% | 91.6% | 91.4% | 91.4% |
| `32768x4608x768` | `128 x 18` | 2304 | 97.7% | 96.5% | 94.4% | 94.4% |
| `65536x2304x768` | `256 x 9` | 2304 | 97.6% | 94.5% | 93.4% | 93.5% |
| `32768x9216x768` | `128 x 36` | 4608 | 98.3% | 94.9% | 93.3% | 93.1% |

This suggests two separate effects:

- The one-stage variant reaches a better steady state as the persistent loop has
  more cluster tiles to process. It rises from `91.8%` of cuBLAS at 576 tiles to
  `97-98%` once there are thousands of tiles.
- Extra TMA-store stages become more clearly harmful in the same steady-state
  regime. The smallest shape is nearly flat across S1-S4, but the larger shapes
  show stage 2 below stage 1 and stage 3/4 lower still.

So the low-K autotune signal is not "we need more C-store queue depth". It is
more consistent with this kernel being epilogue/memory-schedule sensitive, while
extra outstanding C-store TMA groups contend with A/B TMA loads once the
persistent loop is otherwise running efficiently.

Shape geometry may also matter. For the two 1152-tile shapes, `64 x 18` does
better than `128 x 9`; for the two 2304-tile shapes, S1 is almost identical but
deeper stages hurt the `256 x 9` case more. That points to possible interaction
with persistent tile order / memory traffic shape, but the current data is not
enough to claim a specific scheduler effect.

### Fixed-M Low-K, Varying N

To test whether larger `N` is the main reason low-K improves, we also fixed
`M=32768`, `K=768`, and swept `N`.

Artifact:
`study-tma-store-stages/_scratch/results_lowk_vary_n_1000_1000.json`

| Shape | N cluster tiles | S1 TFLOPS | S2 TFLOPS | S3 TFLOPS | S4 TFLOPS | S1 vs cuBLAS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `32768x4608x768` | 18 | 1324.3 | 1301.9 | 1279.2 | 1279.4 | 97.6% |
| `32768x9216x768` | 36 | 1320.3 | 1268.4 | 1240.8 | 1247.6 | 99.4% |
| `32768x12288x768` | 48 | 1293.8 | 1260.8 | 1230.3 | 1230.0 | 103.1% |
| `32768x18432x768` | 72 | 1229.1 | 1193.7 | 1177.9 | 1181.0 | 94.0% |
| `32768x24576x768` | 96 | 1214.0 | 1184.1 | 1160.3 | 1162.5 | 93.2% |
| `32768x32768x768` | 128 | 1178.2 | 1141.5 | 1123.9 | 1123.2 | 90.6% |

This refines the earlier observation. Larger `N` helps up to a moderate region:
`N=9216` and `N=12288` are excellent relative to cuBLAS. But the effect is not
monotonic. Absolute TFLOPS drops once `N` becomes very large, even though the
persistent loop has more tiles.

The stage-count conclusion still holds in this slice: one active TMA-store stage
is best for every tested `N`, and deeper store staging is consistently lower.

### BN512 Variant On Fixed-M Low-K

BN512 is the practical wide-compute variant, but it is not a pure BN-only
change. The currently valid BN512 bundle is:

```text
BN=512 NS=4 PERSISTENT=1 OVERLAP=1 TMA_PIPE=1 SINGLE_TMEM=1 SPLIT=0 L1NA=0
```

It uses the pipelined TMA-store epilogue; production code uses
`TMA_STORE_STAGES=2`. In this study, S1/S2 are deliberately patched variants.
S3/S4 exceed the shared-memory limit for this BN512/NS4 config and are skipped.

Artifact:
`study-tma-store-stages/_scratch/results_lowk_vary_n_bn512_1000_1000.json`

| Shape | BN256 NS5 S1 | BN256 NS5 S2 | BN512 NS4 S1 | BN512 NS4 S2 | Best BN512 / Best BN256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `32768x4608x768` | 1324.3 | 1301.9 | 1034.5 | 1138.2 | 85.9% |
| `32768x9216x768` | 1320.3 | 1268.4 | 1050.6 | 1152.4 | 87.3% |
| `32768x12288x768` | 1293.8 | 1260.8 | 1053.8 | 1150.4 | 88.9% |
| `32768x18432x768` | 1229.1 | 1193.7 | 1053.3 | 1151.2 | 93.7% |
| `32768x24576x768` | 1214.0 | 1184.1 | 1047.6 | 1131.0 | 93.2% |
| `32768x32768x768` | 1178.2 | 1141.5 | 1014.9 | 1104.7 | 93.8% |

BN512 is not competitive for this low-K family. It narrows the gap as `N`
becomes very large, but still loses to BN256 at every tested point. This agrees
with the autotune observation that the top low-K configs did not choose BN512.

One interesting difference: BN512 prefers S2 over S1, unlike BN256. That is
plausible because BN512 has twice as many `STORE_N=64` chunks per tile and only
four compute stages, so two C-store buffers recover some store overlap. Even so,
the total BN512 tradeoff is still worse than BN256 for these low-K shapes.

### Square Shapes

Artifact:
`study-tma-store-stages/_scratch/results_square_1000_1000.json`

| Shape | S1 TFLOPS | S2 TFLOPS | S3 TFLOPS | S4 TFLOPS | Best |
| --- | ---: | ---: | ---: | ---: | --- |
| `4096` | 1303.1 | 1303.1 | 1302.7 | 1302.7 | S1/S2 |
| `5120` | 1291.3 | 1291.7 | 1291.3 | 1291.0 | S2 |
| `6144` | 1312.9 | 1298.3 | 1302.0 | 1290.6 | S1 |
| `7168` | 1272.9 | 1266.1 | 1261.7 | 1262.0 | S1 |
| `8192` | 1273.7 | 1266.3 | 1261.5 | 1261.6 | S1 |
| `10240` | 1165.4 | 1171.6 | 1170.3 | 1169.0 | S2 |
| `12288` | 1139.2 | 1139.2 | 1131.9 | 1136.0 | S1/S2 |

| Shape | cuBLAS TFLOPS | S1 | S2 | S3 | S4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `4096` | 1335.3 | 97.6% | 97.6% | 97.6% | 97.6% |
| `5120` | 1385.6 | 93.2% | 93.2% | 93.2% | 93.2% |
| `6144` | 1410.3 | 93.1% | 92.1% | 92.3% | 91.5% |
| `7168` | 1385.6 | 91.9% | 91.4% | 91.1% | 91.1% |
| `8192` | 1375.6 | 92.6% | 92.1% | 91.7% | 91.7% |
| `10240` | 1365.5 | 85.4% | 85.8% | 85.7% | 85.6% |
| `12288` | 1414.2 | 80.6% | 80.6% | 80.0% | 80.3% |

## Interpretation

The low-K autotune result did not pick `BN=512` in the top configurations.
That is a useful signal: this shape family is not asking for a wider compute
tile or more per-CTA arithmetic. The bottleneck is more exposed around the
epilogue / memory schedule.

However, "epilogue matters" does not imply "more TMA-store stages help".
Increasing active TMA-store stages only increases the number of outstanding C
store groups that the epilogue can queue from one output tile. In these runs,
that is either neutral or harmful:

- Low-K shapes consistently prefer one active TMA-store stage.
- Square shapes are mostly flat between one and two stages, with no stable win
  for three or four stages.
- Three and four stages are usually worse, despite fitting in shared memory.

The likely reason is contention rather than lack of store-buffer space. The
kernel already overlaps current-tile epilogue with next-tile MMA through TMEM
double buffering. Extra C-store TMA groups can compete with the next tile's A/B
TMA loads and memory-system resources. For `BN=256`, there are only four
`STORE_N=64` chunks per tile, so there is little steady-state store depth to
exploit.

## Mechanism Note

There are two independent buffers in the overlap path:

- TMEM accumulator buffers: with `SINGLE_TMEM_ACCUM=0`, the kernel allocates
  `2 * BN` TMEM columns and alternates `buf = ti & 1`.
- TMA-store SMEM buffers: the epilogue stages `STORE_N=64` chunks into a small
  SMEM ring before issuing C TMA stores.

For `BN=256`, the next output tile can start MMA in the alternate TMEM
accumulator while the previous tile's epilogue is still storing C. The epilogue
releases the TMEM buffer after the final chunk has been loaded out of TMEM,
before the final chunk's SMEM packing and TMA store complete.

That early TMEM release is the important cross-tile overlap. Increasing
`TMA_STORE_STAGES` does not create another independent epilogue stream; it only
allows more C TMA store groups from the same epilogue stream to be outstanding.
The measurements suggest that deeper C-store queueing is not the useful knob for
these shapes.

## Older Quick Runs

Earlier exploratory runs used `warmup=300ms`, `rep=200ms`, and several orderings
plus a constant-allocation control. They already showed the same direction:
stage 1 was generally best, stage 2 was below stage 1, and stage 3/4 were worse.

Relevant artifacts:

- `study-tma-store-stages/_scratch/results_300_200_1234.json`
- `study-tma-store-stages/_scratch/results_300_200_1234_alloc4.json`
- `study-tma-store-stages/_scratch/results_shapes_32768_4096_8192.json`
- `study-tma-store-stages/_scratch/results_square_more.json`
- `study-tma-store-stages/_scratch/results_lowk_aspects.json`
