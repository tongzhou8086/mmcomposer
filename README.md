# mmcomposer

A code generator for matrix-multiplication kernels on modern GPUs.  Combines
a tutorial-driven LLM agent loop with a web UI for hyperparameter selection.

## Project layout

```
mmcomposer/
├── docs/                    # Jupyter Book — the comprehensive matmul tutorial
│   ├── _config.yml
│   ├── _toc.yml
│   ├── index.md
│   ├── part1_gpu_arch/      # B200 hardware overview
│   ├── part2_optimization_ladder/   # the core curriculum
│   ├── part3_autotuning.md
│   └── part4_b200_reference.md
├── mmcomposer/              # Python package (agent loop, codegen) — empty for now
├── webui/                   # Web frontend — empty for now
└── tests/
```

## Current status

Bootstrapping — the tutorial is the foundational artifact.  See `docs/index.md`.

Agent loop and web UI are scaffolded but not yet implemented; the tutorial
comes first because it is the agent's primary knowledge source.

## Supported GPUs

Initial target: **NVIDIA B200 (sm_100a)**.  Other targets (Hopper, Ada) will
be added incrementally once the tutorial structure stabilizes.
