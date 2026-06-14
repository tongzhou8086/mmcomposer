# TMA Store Stage Count Study

This study tests whether changing the pipelined TMA-store epilogue stage count
helps across representative low-K and square GEMM shapes. The original target
shape was:

```text
32768x4608x768
```

The generator/templates are not modified. `bench_store_stages.py` renders the
known best production config into this directory, then mechanically patches only
the generated study copy:

- `TMA_STORE_STAGES = 2` -> selected stage count
- `store_stage ^= 1` -> modulo/power-of-two rotation across all store buffers

Best config under test:

```text
BM=128 BN=256 BK=64 NS=5 GSM=1 NW=4
PERSISTENT=1 OVERLAP=1 SPLIT=0 L1NA=0 TMA_PIPE=1 SINGLE_TMEM=0 TWO_CTA=1
```

Run from the repo root inside a B200 allocation, for example:

```bash
srunpy study-tma-store-stages/bench_store_stages.py --shape 32768x4608x768 --stages 1,2,3,4
```

By default the script uses `triton.testing.do_bench` with `warmup_ms=1000` and
`rep_ms=1000`, which is the preferred setting for controlled performance
studies.

For a quicker ordering run:

```bash
srunpy study-tma-store-stages/bench_store_stages.py --stages 2,4 --warmup-ms 100 --rep-ms 200 --cublas-samples 10
```

Artifacts and results are written under `study-tma-store-stages/_scratch/`.
