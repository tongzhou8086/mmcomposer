# mmcomposer

A CUDA **matrix-multiplication kernel generator + autotuner** for NVIDIA
**Blackwell (B200, `sm_100a`)**.  Pick a point in a space of design knobs
(warp specialization, 2-CTA MMA, pipeline depth, epilogue strategy, …) and
mmcomposer splices hand-tuned fragments into a complete, compilable kernel —
then benchmarks the valid combinations on your GPU and serves the fastest one.

It reaches **~100% of cuBLAS** across a range of square and LLM-shaped GEMMs.

## Install

No PyPI yet — install straight from GitHub:

```bash
pip install git+https://github.com/tongzhou8086/mmcomposer.git
```

Requires a CUDA toolkit (`nvcc` on `PATH`) and a GPU build of `torch`; deps
`numpy`, `torch`, `cuda-python` are pulled in automatically.

## Quickstart (Python)

```python
import torch
import mmcomposer as mmc

a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")

c = mmc.matmul(a, b)        # first call for a new shape auto-tunes + caches (slow,
                            # once per machine); subsequent calls are fast.

gemm = mmc.get_tuned_kernel(a, b)   # or grab a reusable callable for the shape
c = gemm(a, b)

mmc.tune(4096, 4096, 4096)  # or pre-tune offline so nothing stalls at runtime
```

The first tune for a shape sweeps the valid configs, compiles + benchmarks them
on the local GPU, and writes the winner to `~/.cache/mmcomposer/`.  It is keyed
by `(M, N, K, dtype, arch)`, so it happens **once per machine**, not per session.

### Supported inputs (v0)

- **bf16** inputs/outputs, fp32 accumulate.
- `A` is `M×K`, `B` is `K×N`, both **row-major contiguous**; `C` is row-major.
- `M` and `N` multiples of **256**, `K` a multiple of **64**.
- Arch: **B200 (`sm_100a`)**.

Unsupported dtype/layout/shape raises a clear error.

## Command line

```bash
python -m mmcomposer.autotune 8192               # square 8192³
python -m mmcomposer.autotune 32768x4608x768     # rectangular MxNxK
mmcomposer-tune 8192 --scope full --top 20       # console-script equivalent
```

Renders → compiles → benchmarks every valid combo on the local GPU and prints a
live leaderboard by measured TFLOPS.

## Web UI

A Streamlit app to inspect/generate/download a kernel for a chosen config and
view its measured B200 performance.

**Live:** [mmcomposer.streamlit.app](https://mmcomposer.streamlit.app/)

Or run it locally:

```bash
streamlit run webui/app.py
```

## How it works

mmcomposer is a small set of composable modules — each with a clear public API,
independently unit-tested — wired by one in-process orchestrator (no
subprocess/`srun`):

```
enumerate (combos) → codegen → compile → runtime → benchmark → cache
                                                   (leaderboard renders top-N)
```

- **combos** — enumerate the valid knob combinations.
- **codegen** — a combo → device `kernel.cu` (and a standalone host script for download).
- **compiler** — `nvcc → cubin`, parallel + disk-cached.
- **runtime** — load a cubin and launch it as a plain `callable` (this *is* what
  `get_tuned_kernel` returns).
- **benchmark** — a `do_bench`-style timer over any callable; `rel_error` checks correctness.
- **cache** — ranked tuning results per shape (local disk now; pluggable for a future shared/remote tier).
- **autotune** — orchestrates the sweep; `mmc.matmul`/`get_tuned_kernel`/`tune` sit on top.

See [`mmcomposer/DESIGN.md`](mmcomposer/DESIGN.md) for the full architecture.

## Project layout

```
mmcomposer/        # the package: combos, codegen/, kernels/, compile, runtime,
                   #   benchmark, cache, leaderboard, autotune, mmc + DESIGN.md
webui/             # Streamlit UI (app.py) + GPU correctness/perf harnesses (tests/)
tutorial/          # matmul-optimization curriculum (the "how" behind the knobs)
presentation/      # the talk: "How to Design a High Performance GEMM Kernel (on Blackwell)"
docs/              # design notes
study-*/           # exploratory experiments
```

## Status & roadmap

The package, CLI, and web UI are working and B200-verified. Planned:

- a **shared/remote results cache** (tune once, reuse everywhere on the same arch);
- more **dtypes / layouts** (fp16, transposed operands);
- other architectures (Hopper).
