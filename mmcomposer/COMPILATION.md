# What happens when you call `mmc.matmul` — compilation & caching

This explains the full path from `mmc.matmul(a, b)` to a launched kernel, and
**exactly what is cached in memory vs. on disk**. The short version:

- **Nothing is preloaded at import.** All in-memory caches start empty and fill
  **lazily**, one entry per shape you actually use.
- **Disk holds the durable artifacts**: the tuned winner config (tiny JSON, per
  shape) and the compiled cubins. These survive across sessions.
- **A miss reads one file**, not the whole cache directory.

---

## The caches

### In memory (per process — empty at import, filled lazily)

| cache | where | key → value | filled when |
|-------|-------|-------------|-------------|
| `_KERNELS` | `mmc` | `shape_key` → callable | first use of a shape |
| `_EPI_KERNELS` | `mmc` | `(shape_key, epilogue_digest)` → callable | first use of a shape+epilogue |
| `_TRACE_CACHE` | `mmc` | epilogue object → `(cuda, digest)` | first trace of an epilogue object |
| `_MODULE_CACHE` | `runtime` | `(cubin_path, symbol)` → loaded CUDA module + fn | first launch from a cubin |
| launch state | inside each callable | `(M,N,K,a_ptr,b_ptr,c_ptr)` → grid/block/args/descriptors | first launch with those buffers |

### On disk (durable, per machine — `$MMCOMPOSER_CACHE_DIR`, else `~/.cache/mmcomposer/`)

| cache | path | contents |
|-------|------|----------|
| **results / config** | `results/<shape_key>.json` | ranked tuned configs for that shape (winner = rank 0). Tiny. |
| **artifact** | `build/<arch>/<tag>/kernel.cu` + `kernel_<arch>.cubin` | generated source + compiled cubin, one dir per config (`<tag>`), epilogues under `epi_<hash>/`. Big, binary. |

`shape_key = "<M>x<N>x<K>_<dtype>_<arch>"`, e.g. `4096x4096x4096_bf16_sm_100a`.

### Where the cache root lives (and cross-node reuse)

`cache_root()` picks, in order: `$MMCOMPOSER_CACHE_DIR` → `$XDG_CACHE_HOME/mmcomposer`
→ `~/.cache/mmcomposer`.

On a multi-node cluster this matters: if `XDG_CACHE_HOME` points at **node-local**
storage (e.g. `/scratch` on a local NVMe), each node has its own cache, so a tune
done on one node **re-tunes** when a job lands on another. To share one tune across
all nodes, point mmcomposer at a **shared filesystem** (e.g. `$HOME`):

```bash
export MMCOMPOSER_CACHE_DIR=$HOME/.cache/mmcomposer
```

This overrides `XDG_CACHE_HOME` for mmcomposer only (its cache is a tiny JSON +
cubins), leaving every other tool's cache on fast local scratch. Cubins are
arch-specific but valid on any same-arch node, so they're reusable too.

---

## The flow

```
mmc.matmul(a, b)
   │
   ▼
validate + build shape_key  (M, N, K, dtype, arch)
   │
   ▼
┌───────────────────────────────────────────────────────────────────┐
│ key in _KERNELS ?            (MEMORY, empty at import)              │
└───────────────────────────────────────────────────────────────────┘
   │ yes                                   │ no
   ▼                                       ▼
return cached callable          kcache.best(key)  ── reads results/<key>.json  (DISK)
   │                                       │
   │                            ┌──────────┴───────────┐
   │                            │ hit (config)         │ miss (None)
   │                            ▼                      ▼
   │                            │              AUTOTUNE  (cold, ~100 s, once per shape)
   │                            │                 1. enumerate valid combos (production scope, pruned)
   │                            │                 2. codegen kernel.cu per combo        (CPU)
   │                            │                 3. compile_many -> cubins   (PARALLEL nvcc, DISK)
   │                            │                 4. for each: load + verify + do_bench (SERIAL, GPU)
   │                            │                 5. kc.put winner -> results/<key>.json (DISK)
   │                            │                      │
   │                            └──────────┬───────────┘
   │                                       ▼
   │                            _build(config):
   │                              • render kernel.cu  (DISK, already exists from tune)
   │                              • compile_one -> cubin  (DISK; SKIPS nvcc if up-to-date)
   │                              • runtime.kernel(config, cubin):
   │                                  load module into _MODULE_CACHE  (MEMORY)
   │                                  → returns an async callable
   │                                       │
   │                            _KERNELS[key] = callable   (MEMORY)
   │                                       │
   └───────────────┬───────────────────────┘
                   ▼
        callable(a, b)  → build/reuse launch state (MEMORY) → cuLaunchKernel
                   │        (async, on torch's current stream)
                   ▼
                   c
```

---

## Three scenarios

**A. Cold — never tuned on this machine.**
`_KERNELS` miss → `results/<key>.json` miss → **autotune** (enumerate → codegen →
parallel nvcc → sequential benchmark → write winner JSON) → `_build` (cubin already
on disk from the sweep, so `compile_one` skips nvcc; load module) → cache callable →
launch. Slow once (~100 s + compiles); prints one-time progress.

**B. Warm — same process, shape already used.**
`_KERNELS` hit → return callable → launch. Pure memory; two dict lookups + launch.

**C. New session — tuned in a previous run.**
`_KERNELS` empty → `results/<key>.json` **hit** (DISK) → `_build`: render source,
`compile_one` sees the **up-to-date cubin on disk → skips nvcc**, just **loads** the
module into `_MODULE_CACHE` → cache callable → launch. No re-tune, no re-compile —
just a disk read + module load (~sub-second). After this first call, it's scenario B.

**D. With an epilogue** (`epilogue=fn`).
Same as above for the *geometry* (the shape's tune is reused), but keyed by
`(shape_key, epilogue_digest)` in `_EPI_KERNELS`. First time for a new epilogue:
trace+lower the function → reuse the tuned config → compile **one** fused cubin
(`build/<arch>/epi_<hash>/`, ~seconds, cached on disk) → load → cache. No re-tune.
See `EPILOGUE.md`.

---

## Autotune internals (the cold path, expanded)

```
autotune.tune(M, N, K)
  1. enumerate   valid combos for the scope (production = pruned: ~198 combos)   [CPU, pure]
  2. codegen     render each combo's kernel.cu into build/<arch>/<tag>/          [CPU, pure]
  3. compile     compile_many: one nvcc subprocess per .cu, in a thread pool     [CPU, PARALLEL -> DISK]
                 (atomic temp+rename; skips up-to-date; prunes compile failures)
  4. benchmark   serial on the GPU (co-running kernels corrupt timing):          [GPU, SERIAL]
                   for each compiled combo:
                     load cubin -> module/fn   (into _MODULE_CACHE)
                     verify    (one launch vs torch reference; drop if wrong)
                     do_bench  (warmup + L2 flush + median)  -> TFLOPS
                     kc.put(key, record)  -> stream into results/<key>.json      [-> DISK, live]
  5. select      winner = rank-0 record in results/<key>.json
```

Parallelism rule: **compile is parallel** (independent cubins, CPU); **timing is
serial per GPU** (a kernel must own the GPU for clean numbers). Multi-GPU shards
the benchmark step, one serial timer per GPU.

---

## Why split memory vs disk this way

- **Compilation is the expensive, reusable step** → cubins live on disk, named by
  content (config tag / epilogue hash), so they're reused across sessions and never
  recompiled unless the source changes.
- **The winner config is tiny and portable** → a small JSON per shape. (Future:
  a shared/remote results cache so one machine's tune is reusable by everyone on
  the same arch; cubins stay local since they're arch-specific.)
- **The in-memory caches are pure speed** → they avoid re-reading JSON, re-loading
  modules, and rebuilding launch state within a process. They're rebuilt cheaply
  on the next run from the durable disk artifacts.
```
import time:   all in-memory caches EMPTY
first call:    DISK (read config + load cubin)  -> MEMORY (callable, module)
later calls:   MEMORY only
```
