# MMComposer — Modular Design

One shared **core** with three **front-ends** over it:

- **CLI** — `autotune.py` (terminal timing sweep + live leaderboard)
- **Python package** — `import mmcomposer as mmc` → `mmc.matmul` / `mmc.get_tuned_kernel` / `mmc.tune`
- **Web UI** — Streamlit app for inspect / generate / download

All three call the same modules and share the same caches. No front-end owns logic
that another could need.

## Scope / constraints (v0)

- bf16 in / bf16 out, **fp32 accumulate**.
- Layouts: **A & B K-major, C row-major**.
- `M` a multiple of 128 (256 for 2-CTA); per-kernel N/K constraints.
- Arch: **B200 (`sm_100a`) only**.
- Unsupported dtype/layout/shape → a clear error (or warning), never silent.

## Modules (leaf → up)

Each is independently testable; leaves need no other mmcomposer module (and
`enumerate`/`leaderboard`/`codegen` need no GPU).

### codegen — *pure, no GPU*
- `generate_kernel(config) -> str` — the **device** `kernel.cu` (knobs as `constexpr`).
  This is the **only** artifact that must be generated and compiled.
- `generate_host(config) -> str` — a self-contained standalone `run.py` (inlines the
  `runtime` source + `nvcc` + a demo `main`). **Download / inspect artifact only**
  (web UI); **not** on the execution path. It is produced by *inlining the same
  `runtime` library source*, so the host launch logic has a single source of truth.

### enumerate — *pure, no GPU*
- `enumerate(filters) -> set[config]` — all **valid** combos for a scope/filter set.

### compile — *CPU (nvcc), disk cache*
- `compile(configs) -> {config: cubin_path}` — parallel `nvcc` over generated kernels
  → cubins on disk, keyed `(arch, config-hash)`. Dedups identical configs, skips
  up-to-date cubins (mtime/hash), **atomic write** (temp + rename), and **reports
  ok vs failed** so the orchestrator prunes compile failures before any GPU time.
- One cubin serves **all shapes** for a config (M/N/K are runtime args / grid math).

### runtime — *GPU, in-memory cache*  (the entry point)
- `kernel(config) -> callable`; `k(a, b) -> c`. **Assumes the cubin already exists.**
  Per call it is just two lookups:
  1. module/function loaded in this process? → else load the cubin once.
  2. host setup for `(config, M, N, K)` computed? → else compute grid/block/shared,
     max-SMEM attr, and TMA descriptor templates once.
  …then launch. TMA descriptor base-ptr: cache whole if buffers are stable (always
  true under `do_bench`); else cheap patch/rebuild.
- **Contract:** the entry point never compiles; the caller guarantees the cubin
  (via `compile`, typically from `autotune`).
- This callable is what `mmc.get_tuned_kernel` returns, what `autotune` feeds to
  `benchmark`, and what `mmc.matmul` invokes — one object, every use.

### benchmark — *GPU, pure*
- `benchmark(callable, ...) -> {tflops, latency}` via `do_bench`. Knows nothing about
  kernels, so it also times cuBLAS / `torch.matmul` for the ratio.
- **Correctness is separate from timing:** `verify(callable, ref)` gates whether a
  combo counts; `benchmark` stays a pure timer.

### cache — *two tiers*
- **results/config cache** — `(M, N, K, dtype, arch) -> best config(s)`. Tiny JSON.
  `put` / `get` / `top_n`. Written **live** during a sweep. local → **(future)
  remote/network**, so a tune by anyone on the same arch is reusable by everyone.
- **artifact cache** — `config -> cubin` (the `compile` disk cache). Big, binary,
  arch-specific, **local** (sync the small *config*, recompile per machine).

### leaderboard — *terminal, standalone*
- Render / redraw top-N. Polls `cache.top_n(shape)`. **Zero knowledge** of how
  results are produced.

### autotune — *orchestrator*
- `autotune(M, N, K, dtype, scope) -> best config`. Phases:
  1. `enumerate` → valid combos
  2. `codegen` (parallel, CPU)
  3. `compile` (parallel, CPU → disk; prune failures)
  4. `benchmark` (**serial per GPU**: `verify` + `do_bench`, stream each result to
     the results cache) — shard across GPUs if more than one
  5. select winner → `cache.put`
- The leaderboard polls the results cache throughout.
- Shared by the CLI and `mmc.tune`.

## Front-ends

- **CLI `autotune.py`** — parse args → `autotune()` → render leaderboard.
- **Package `mmc`**
  - `matmul(a, b)` — validate → key `(M,N,K,dtype,arch)` → results-cache **hit**:
    `kernel(winner)(a, b)`; **miss**: `autotune(shape)` then `kernel(winner)(a, b)`.
  - `get_tuned_kernel(a, b)` — same, but returns the `kernel(winner)` callable.
  - `tune(M, N, K, dtype)` — explicit offline pre-tune (no runtime stall later).
- **Web UI** — `codegen` (device + standalone host) for inspect/download + the
  prebaked-matrix perf table.

## Execution path vs download path

- **Execution** (package + autotune): `codegen`(device) → `compile` → `runtime`.
  No generated host text, no `nvcc` inside generated code.
- **Download** (web): `generate_host` → standalone `run.py` (inlines `runtime` + `nvcc`).

## Parallelism & the GPU rule

- **Compile is parallel** (CPU pool; cubins are independent).
- **Timing is serial per GPU** — `do_bench` (warmup, L2 flush, median) needs the
  kernel to *own* the GPU; co-running kernels corrupts the numbers. Real parallel
  benchmarking = **shard across GPUs**, one serial timer each.
- Single GPU: parallel compile-all, then serial time-all. (A compile→time pipeline
  that overlaps the two is a possible later optimization.)

## Overhead model (per `k(a,b)` call, warm)

- cubin: loaded from disk **once per process** (in-memory module handle).
- host setup: computed **once per `(config, shape)`** (in-memory dict).
- per call: dict lookups + (descriptor ptr patch if buffers changed) + launch.

## Future

See **`TODO.md`** (repo root) for measured per-call host costs and the
performance / future follow-ups (async `matmul`, descriptor caching, remote
cache, more dtypes).

## Dependency DAG

```
   CLI autotune.py      mmc (matmul / get_tuned_kernel / tune)      web UI
            \                /            \                            |
             ▼              ▼              ▼                           ▼
              autotune (orchestrator)   [cache + runtime]          codegen
             /    |     |      |     \        |
            ▼     ▼     ▼      ▼      ▼       ▼
      enumerate codegen compile runtime benchmark   cache        leaderboard
       (pure)   (pure)  (disk$) (callable)(do_bench)(store)       (terminal)
                            \____shared cubin cache____/
```
