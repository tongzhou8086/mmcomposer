# Autotuning Methodology

> **Status:** stub — content TBD.

## What this part covers

- How to use `triton.testing.do_bench` for reliable measurements.
- Why `rep=50` is often too few — the noise floor on shared GPUs.
- When autotune adds signal vs when it adds noise.
- How to harden autotune picks (`rep=200`, `warmup=20`, multiple passes).
- Strategies for autotuning across shapes (per-shape cache, key=`(M,N,K)`).

## TODO

- Cover the empirical observation from the b1 → b41_w8 journey that
  rep=50 autotune produced unstable picks at the few-% scale and rep=200
  was needed for reliable GSM/NW/NS selection.
- Cover thermal-state drift effects on shared multi-tenant GPUs.
- Cover the "tried this autotune, it didn't help" null-result pattern.
