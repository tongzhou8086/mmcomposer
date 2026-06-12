# Study: borrowing gau-nernst's direct-store epilogue

A focused investigation into adopting the lean **direct-store epilogue** from
gau-nernst's Blackwell matmul into our composable kernel, and a kernel bug that
investigation surfaced.

- Reference: <https://github.com/gau-nernst/learn-cuda/tree/main/02e_matmul_sm100>
  (write-up: <https://gau-nernst.github.io/tcgen05>)
- Target shape: **32768 × 4608 × 768** (low K — N=4608, K=768).

---

## 1. Why this study

On the low-K shape `32768×4608×768`, head-to-head with identical timing (Triton
`do_bench` 500/500, cuBLAS in the same script):

| shape | cuBLAS | ours (staged) | gau-nernst v7c (direct) |
|---|---|---|---|
| 4096³ (sanity) | 1355 | 1194 (88%) | 1350 (96%) |
| **32768×4608×768** | ~1363 | **1178 (86%)** | **1274 (92%)** |

Our best autotuned config is `2-CTA + persistent + overlap + split, BN256 NS4
NW4`. v7c is **~8% faster** at our shape (92% vs 86% of cuBLAS).

The difference is the **epilogue**:

- **Ours (staged):** TMEM → SMEM (`C_sh`) → GMEM, with a coalesced int4 store and
  one/two `bar.sync`s per tile. Optimized for high-K, where the epilogue is fully
  hidden behind the long K-loop; the coalesced store wins there.
- **gau-nernst (direct):** TMEM → registers → GMEM directly (`tcgen05.ld` → `cvt`
  → `st.global`), **no SMEM staging, no per-pass `bar.sync`**. The GMEM store is
  strided/uncoalesced, but at **low K** the K-loop is too short to hide a heavy
  epilogue, so the *lean* path stalls less and wins.

The two are **complementary** (staged wins high-K, direct wins low-K), so the goal
is to add a `EPILOGUE_DIRECT` mode to our composable kernel and let the autotune
pick per shape. Making it a first-class tunable knob is a later step; first we
just want the direct epilogue working in our kernel and a measured win.

(cuBLAS varies run-to-run, so compare the %-of-cuBLAS column, not absolute TFLOPS.)

---

## 2. The bug

We implemented `EPILOGUE_DIRECT` (the lean TMEM→reg→GMEM store) in our overlap
epilogue fragment. It is **correct in isolation** but **faults when combined with
the overlap double-buffer at ≥2 tiles per cluster**:

| config | multi-tile (≥2 tiles/cluster) |
|---|---|
| direct + **non-overlap** | ✅ works (e.g. 773 TFLOPS @ 32768×4608×768, correct) |
| direct + **overlap** | ❌ `CUDA_ERROR_LAUNCH_FAILED` |
| staged + overlap | ✅ works (1160) |
| single tile (any) | ✅ works |

So the direct store itself is fine; the fault is **specifically the overlap
double-buffer (2 TMEM accumulator buffers) interacting with the direct store**.
And gau-nernst's `v7c` does *exactly* that combination (overlap + direct +
multi-tile) and works — so it is a specific difference in **our** overlap setup,
not a fundamental incompatibility.

### Reproduce it (this directory)

`ours_direct_FAILING_kernel.cu` / `ours_direct_FAILING_host.py` are the rendered
overlap+direct kernel (`EPILOGUE_DIRECT=1, EPILOGUE_OVERLAP=1, split=0`) and a
launcher that shows the OK→FAULT transition:

```bash
srun --partition=dedicated --gres=gpu:nvidia_b200:1 --time=00:15:00 \
    python study-gau-nernst/ours_direct_FAILING_host.py
```

Observed output:
```
=== 2048x2048x768  (64 clusters, ~1 tile(s)/cluster) ===
  launched OK, rel_err=3.29e-03  (CORRECT)
=== 4096x4608x768  (288 clusters, ~4 tile(s)/cluster) ===
  *** FAILED: RuntimeError: CUDA driver error: CUDA_ERROR_LAUNCH_FAILED
```
(`ours_kernel.cu`/`ours_host.py` are the *staged* epilogue — correct — for
comparison.)

### What we know

- **`compute-sanitizer --tool memcheck`: clean** even at multi-tile (no OOB, no
  misaligned store). The fault is not a bad address.
- **`--tool synccheck`: "Barrier error / Missing init" — but the *working* staged
  kernel reports the identical warning** at the same shared address. It is a known
  false-positive for persistent-kernel mbarrier reuse → **noise**, not the cause.
- **Neutering the GMEM store (load + release, no store) → runs.** The store's
  presence is the trigger (most likely via timing, not address).
- The lean direct store interleaves `tcgen05.ld` with GMEM stores across the whole
  epilogue, so the epilogue's `tcgen05.ld` runs concurrently with the next tile's
  `tcgen05.mma` (different TMEM buffer) for a **longer window** than the staged
  store does. This concurrent-tcgen05 window is the leading suspect. (In the
  non-overlap path the epilogue runs *after* the MMA, so there is no concurrency —
  consistent with non-overlap working.)

### Ruled out (each tried as a single change, still faults)

1. A second trailing `bar.sync` in the epilogue (warp lockstep).
2. Monolithic inline-asm store (gau-nernst's exact `tcgen05.ld`+`cvt`+`st.global`),
   ruling out compiler reordering / the `int4`-through-local-memory store.
3. All-epilogue-threads arrival on the TMEM-empty mbarrier (gau-nernst's
   buffer-empty *release* scheme).
4. Single-waiter + `bar.sync` broadcast for the TMEM-full *wait* (gau-nernst's
   buffer-full wait scheme).

### Next planned step

Bisect from the **known-good v7c**: mutate v7c toward our design one change at a
time (e.g. `WIDTH`, warp layout, the `GROUP_M` swizzle, 3D→2D TMA) until it
breaks. Whichever single mutation breaks v7c *is* the bug — no guessing.

### Key terms (our names ↔ gau-nernst's)

- **Buffer-full wait** — epilogue waits until the MMA finished writing a TMEM
  accumulator buffer: `tmem_full[buf]` ↔ `mainloop_mbar`.
- **Buffer-empty release** — epilogue signals it has finished reading the buffer
  so the MMA may reuse it: `tmem_empty[buf]` ↔ `epilogue_mbar`.

---

## 3. How to benchmark

Both require a B200 (`srun`). cuBLAS is measured in the same script as the ratio
baseline (compare ratios, not absolute TFLOPS — cuBLAS varies run-to-run).

### gau-nernst's kernel (this directory)

`bench.py` JIT-builds `matmul_v7.cu` (+ `binding.cpp`, only v7a/b/c) as a torch
extension and benches v7c vs cuBLAS with Triton `do_bench` (500/500):

```bash
# from repo root, on a B200 node
srun --partition=dedicated --gres=gpu:nvidia_b200:1 --time=00:25:00 \
    python study-gau-nernst/bench.py 4096 32768x4608x768
```

Shapes are positional (`S` for square `S³`, or `MxNxK`); default is
`4096 32768x4608x768`. First run compiles the extension (~1–2 min); needs
`ninja` (`pip install ninja`).

Files: `matmul_v7.cu`, `common.h`, `profiler.h` (copied verbatim from the
reference repo), `binding.cpp` (minimal torch binding), `bench.py`.

### Our kernel — concrete best config (this directory)

A materialized copy of our current best config for this shape lives here so it can
be benched head-to-head with v7c using the *same* `do_bench` 500/500 + cuBLAS
methodology:

- `ours_kernel.cu` — rendered kernel for `2-CTA + persistent + overlap + split,
  BN256 NS4 NW4` (the staged epilogue; `EPILOGUE_DIRECT` off).
- `ours_host.py` — self-contained launcher (driver API + tensor maps) that benches
  it vs cuBLAS.

```bash
# from repo root, on a B200 node (ours_kernel.cu must sit alongside ours_host.py)
srun --partition=dedicated --gres=gpu:nvidia_b200:1 --time=00:25:00 \
    python study-gau-nernst/ours_host.py 4096 32768x4608x768
```

To regenerate after a codegen change, re-render via `webui/mvp_core` (see the
commands used to produce these files) — they are *generated artifacts*, not the
source of truth.

### Our kernel (autotune sweep)

`webui/autotune.py` sweeps every valid knob combo for the shape on a B200 and
prints the top configs. It `srun`s the driver internally, so run it from the
login node:

```bash
# from repo root
python webui/autotune.py 32768x4608x768            # production scope, top 10
python webui/autotune.py 32768x4608x768 --scope full --top 20
```

The current best for this shape (`EPILOGUE_DIRECT` off — staged epilogue) is
~1160 TFLOPS at `2-CTA + persistent + overlap + split, BN256 NS4 NW4`. The
committed `webui/kernels/compat_matrix.json` is never touched (the sweep writes a
scratch matrix).

---

## 4. Repo state (IMPORTANT)

The `EPILOGUE_DIRECT` feature is **implemented but UNCOMMITTED** — nothing is on
`master`, so the live app / committed kernels are unaffected. Working-tree status:

- The `EPILOGUE_DIRECT` knob is threaded through codegen (`spec.py`, `audit.py`,
  `mvp_core.knob_kwargs`/`render_*`, `live_bench.py`, `_live_bench_worker.py`) and
  the kernel fragments (`_overlap_epilogue.cu.frag`, `_epilogue.cu.frag`,
  `tier3_cluster_swizzle/kernel.cu`).
- **Non-overlap direct path: correct** (verified on B200, multi-tile).
- **Overlap direct path: still faults** (the open bug in §2) — *do not* enable
  `EPILOGUE_DIRECT` together with `EPILOGUE_OVERLAP` until it's fixed.
- It is **not** wired into the validator/driver/UI as a swept knob yet, and there
  are **no tests/goldens** for it. That wiring (and making it a first-class tunable
  knob) is deferred until the overlap bug is fixed.

So: treat the working tree as a work-in-progress spike. The `ours_kernel.cu` /
`ours_host.py` here use the **staged** epilogue (no `EPILOGUE_DIRECT`), so they are
correct and committable-quality regardless of the open bug.
