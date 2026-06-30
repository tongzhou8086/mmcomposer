# TODO / performance follow-ups

Measured, deferred work for the `mmcomposer` package. See `mmcomposer/DESIGN.md`
for the architecture.

## Measured per-call host cost

B200, 4096³, host-side timing with `sync=False` (no GPU wait), median of 300:

| step | cost |
|---|---|
| `torch.empty(M,N)` output alloc | ~1.9 µs |
| `encode_tensor_map` (one TMA descriptor) | ~5.3 µs |
| `_descriptors` (A + B + C, 3 encodes) | ~16 µs |
| `_prepare` (3 descriptors + SMEM attr + arg marshalling) | ~21 µs |
| `gemm(a, b, c)` reused buffer (launch-state cache hit) | ~6.5 µs |
| `gemm(a, b)` fresh output | ~14 µs |

**Why a descriptor costs ~5 µs:** a TMA descriptor is not a raw pointer.
`encode_tensor_map` is a *driver round trip* — a call into the CUDA user-mode
driver (`cuTensorMapEncodeTiled`) that validates the dims/strides/swizzle/
alignment and encodes the opaque 128-byte `CUtensorMap`. It's host-only (no GPU),
but the binding crossing + driver validation cost ~5 µs. A plain-pointer kernel
arg is just an int handed to `cuLaunchKernel` → ~free.

**With a reused output buffer** the launch state is cached (encode happens once),
so per-call host cost is ~6.5 µs — on par with `torch`. The overhead only appears
when the output is freshly allocated each call, and is negligible at realistic
shapes (kernel time ≫ host time).

## Done

### ~~1. Async `matmul` on torch's stream~~  *(done — commit `3537623`)*
The `runtime.kernel` callable now defaults to `sync=False` and launches on
`torch.cuda.current_stream(device).cuda_stream`, matching `torch.matmul`: the
result is stream-ordered before any following torch op, and host reads
(`.item()`/`.cpu()`) sync as usual. Removes a full `cuCtxSynchronize` per call
and makes mmc capturable by CUDA graphs / `torch.compile` (a blocking sync
can't be captured). `sync=True` and an explicit `stream=` stay opt-in;
`mmc.matmul` gained `out=` / `sync=` passthrough. B200-verified: all paths
correct (rel_err 1.66e-3); 50 launches return to host in 0.55 ms vs 4.66 ms GPU.

## Follow-ups (rough priority)

### 2. Descriptor-cache split  *(modest; fresh-output path only)*
Cache `(fn, grid, block, shared)` + the **A & B** descriptors keyed by
`(config, M, N, K, a_ptr, b_ptr)` (stable across a loop) and re-encode **only C**
keyed by `c_ptr`. Turns a full rebuild (~21 µs) into ~one encode (~7 µs). Note:
the `CUtensorMap` is opaque, so the base pointer can't be byte-patched — a changed
output pointer requires a re-encode. Worth little when the output buffer is reused
(already cached) or at large shapes.

### 3. Tune the epilogue as a matmul *variant*  *(perf; epilogue feature)*
Phase-1 epilogues **reuse the shape's plain-matmul winner** and splice the op in.
But the extra epilogue ALU shifts the optimal config (register pressure, epilogue↔
K-loop overlap), so the plain winner isn't best for the fused kernel: at FFN
32768×4608×768 the bare GEMM is ~0.175 ms (1300 TFLOPS) but fused SiLU is
~0.194–0.214 ms (1100–1200 TFLOPS), a ~10–20% gap. Make the epilogue a tuned
variant: key the results/cubin cache by `(shape, epilogue digest)`, thread the
epilogue through `autotune` so every candidate is compiled+benchmarked *with* it,
and add an `epilogue.to_torch(fn)` backend (lower the same `Expr` DAG to torch)
so the tune's correctness check references the activation too. Cost: ~100 s tune
per (shape, epilogue), one-time + cached. First step: confirm an epilogue-aware
sweep actually beats the reuse-geometry number (i.e. the gap is tunable, not
irreducible activation ALU). The hand-tuned SwiGLU (gate ~free) suggests it is.

## Larger / future

- **Remote results cache**: upload-on-tune, query-at-runtime; shared across
  machines on the same GPU arch (the "online config registry").
- **More dtypes / layouts** beyond bf16 / (A,B K-major, C row-major).
- **Optional process isolation** for broad/experimental sweeps (a crashing kernel
  can corrupt an in-process CUDA context).

## Guidance for now
`matmul` is now async (no per-call sync). In steady-state training the caching
allocator reuses the same output addresses each step, so the launch-state cache
hits and per-call host overhead ≈ 0 even without an explicit `out=`. For the
lowest overhead, capture the step with CUDA graphs / `torch.compile` (now that
the launch is sync-free and capturable) or reuse an `out=` buffer in hot loops.
