# Matrix Multiplication Optimization on B200

This book is a hands-on tutorial on optimizing BF16 matrix multiplication
on NVIDIA's B200 GPU (sm_100a, Blackwell architecture).  It is intended
to be read in two ways:

- **By humans** — as a self-contained curriculum for engineers learning to
  write high-performance GEMM kernels on Blackwell-class hardware.
- **By `mmcomposer`'s code-generation agent** — as the authoritative
  knowledge source the agent consults when synthesizing kernels per
  user-supplied hyperparameters.

## How it's organized

```{tableofcontents}
```

## Reading guide

- **Part 1** introduces the hardware constraints you must keep in mind when
  reasoning about GEMM performance on B200.  Short — just what's load-bearing.
- **Part 2** is the heart of the book: a sequence of optimization steps, each
  one a discrete kernel transformation that adds a single concept on top of
  the previous step.  Read in order.  Each chapter has the same structure:
  *why* (which bottleneck it addresses), *how* (the mechanism), *code*
  (a complete kernel pattern), *pitfalls*, and *expected perf delta*.
- **Part 3** is a B200-specific reference: PTX cheat sheet, common errors,
  shared-memory layouts.

Autotuning methodology — how to use `do_bench`, when autotune is worth
it, when it just adds noise — used to be its own part.  We folded it
into Part 2's chapter 12 instead: by that point in the ladder you've
seen enough kernels to make autotuning a concrete activity rather than
abstract advice.

## Prerequisites

The reader is expected to be comfortable with:

- CUDA C++ at the level of writing block-tiled kernels.
- PTX inline assembly (we use it heavily — most of the modern instructions
  on B200 have no C++ wrappers).
- The concept of warps, blocks, and the SM hierarchy.

We do *not* assume prior familiarity with Hopper-specific features (TMA,
WGMMA), Blackwell-specific features (tcgen05, TMEM, 2-CTA clusters), or
modern GPU memory hierarchies (HBM3e, segmented L2).  Those are taught.

## A note on roofline

Roofline analysis is a useful mental model for thinking about GPU
performance in general, and it is covered briefly in Part 1's hardware
chapter.  But on Hopper and Blackwell-class hardware, the proliferation
of asynchronous mechanisms (TMA, warp specialization, TMEM, multi-stage
buffering) means that simple roofline arguments often *fail to predict*
which optimizations will help and by how much.  We have intentionally
omitted a dedicated chapter on roofline modeling because we found, during
the kernel-development work that informed this book, that empirical
measurement — compile, run, measure — was a more reliable guide than
analytic prediction.
