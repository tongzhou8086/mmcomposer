# mmcomposer

A code generator for matrix-multiplication kernels on modern GPUs.  Combines
a tutorial-driven LLM agent loop with a web UI for hyperparameter selection.

## Project layout

```
mmcomposer/
├── tutorial/                # The matmul-optimization curriculum
│   ├── book/                # Jupyter Book — prose chapters
│   │   ├── _config.yml
│   │   ├── _toc.yml
│   │   ├── index.md
│   │   ├── part1_gpu_arch/  # hardware + primitive primers
│   │   ├── part2_optimization_ladder/   # the core curriculum
│   │   ├── part3_autotuning.md
│   │   └── part4_b200_reference.md
│   └── code/                # Runnable companion code, paired per chapter
│       ├── requirements.txt
│       └── 00_first_tma/    # one directory per Part 2 chapter
├── mmcomposer/              # Python package (agent loop, codegen) — empty for now
├── webui/                   # Streamlit pitch-demo UI
└── tests/
```

The tutorial is both human-facing (a standalone curriculum, published at
[mmcomposer.readthedocs.io](https://mmcomposer.readthedocs.io)) and the
primary knowledge source the agent loop consults during code generation.
Prose lives in `tutorial/book/`; runnable companion code lives in
`tutorial/code/<chapter-slug>/`.

## Current status

Bootstrapping — the tutorial is the foundational artifact.  See `tutorial/book/index.md`.

Agent loop and web UI are scaffolded but not yet implemented; the tutorial
comes first because it is the agent's primary knowledge source.

## Supported GPUs

Initial target: **NVIDIA B200 (sm_100a)**.  Other targets (Hopper, Ada) will
be added incrementally once the tutorial structure stabilizes.
