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

## Follow-ups (rough priority)

### 1. Async `matmul` on torch's stream  *(highest value)*
`mmc.matmul` currently **blocks**: the `runtime.kernel` callable defaults to
`sync=True` and launches on the default stream 0, unlike `torch.matmul` (async).
Launch on `torch.cuda.current_stream().cuda_stream` and return **without**
syncing, so stream ordering keeps the result safe for following torch ops. This
removes a full device sync per call and matches torch semantics. Keep an opt-in
`sync=True` / `out=`.

### 2. Descriptor-cache split  *(modest; fresh-output path only)*
Cache `(fn, grid, block, shared)` + the **A & B** descriptors keyed by
`(config, M, N, K, a_ptr, b_ptr)` (stable across a loop) and re-encode **only C**
keyed by `c_ptr`. Turns a full rebuild (~21 µs) into ~one encode (~7 µs). Note:
the `CUtensorMap` is opaque, so the base pointer can't be byte-patched — a changed
output pointer requires a re-encode. Worth little when the output buffer is reused
(already cached) or at large shapes.

## Larger / future

- **Remote results cache**: upload-on-tune, query-at-runtime; shared across
  machines on the same GPU arch (the "online config registry").
- **More dtypes / layouts** beyond bf16 / (A,B K-major, C row-major).
- **Optional process isolation** for broad/experimental sweeps (a crashing kernel
  can corrupt an in-process CUDA context).

## Guidance for now
Reuse your `out=` buffer in hot loops → per-call host overhead ≈ 0.
