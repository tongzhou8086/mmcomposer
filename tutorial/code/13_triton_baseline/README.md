# Canonical Triton baseline

> 📁 **Code on GitHub:** [`tutorial/code/13_triton_baseline/`](https://github.com/tongzhou8086/mmcomposer/tree/master/tutorial/code/13_triton_baseline) — `kernel.py` + `main.py`.

We've taught the optimization ladder by hand-writing CUDA from
chapter 00 to chapter 12.  The kernel that came out the other end
runs at ~95-99 % of cuBLAS at the sweet-spot shapes.

But there's another canonical reference point in the B200 world: a
**Triton kernel** written in the idiomatic persistent + warp-spec +
FLATTEN style.  It uses *exactly the same primitives* we taught — TMA
loads, multi-stage ring buffer, warp specialization, persistent grid,
chunked CTA swizzle — just expressed in 50 lines of Python rather
than 600 lines of `.cu` + PTX inline asm.

This chapter writes that Triton kernel, autotunes it over the same
knobs ch12 tunes, and puts the three contenders side by side:

  * **ch12** — our hand-written CUDA, autotuned over `(NS, GSM)`.
  * **Triton** — this chapter's kernel, autotuned over `(BLOCK_M,
    BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps, num_stages)`.
  * **cuBLAS** — PyTorch's `A @ B` for absolute reference.

The point isn't a victory lap.  Triton is a more interesting target
than cuBLAS for "did we learn the right things":

  * cuBLAS is SASS-tuned with NVIDIA-private optimisations that no
    teaching ladder can match.  The 5 % gap to cuBLAS is the "rest of
    the iceberg."
  * Triton is what someone fluent in B200 would actually *write*.  If
    we're within a few percent of Triton, we know our hand-written
    kernel is at the same level of sophistication as a current
    production Triton kernel.

## The Triton kernel

This chapter ships the **Blackwell-standard** Triton matmul — the
pattern mirrored in Triton's `tutorials/09-persistent-matmul.py
matmul_kernel_descriptor_persistent`.  Three idioms layered on top of
the basic persistent + warp-spec + FLATTEN form:

1. **`tile_id_c` deferral** — store tile `T`'s output in the loop
   iteration that runs tile `T+1`'s K-loop.  Implemented by
   maintaining a second counter that lags `tile_id` by one outer
   iteration.  The compiler then interleaves "K-loop for T+1" with
   "epilogue store for T" → K-loop / epilogue overlap, expressed at
   the Triton expression level rather than via custom PTX.

2. **`EPILOGUE_SUBTILE`** — split the BN-wide accumulator into two
   BN/2-wide halves and store them as two separate TMA calls.  Better
   epilogue pipelining; the two halves can overlap with each other
   and with the next tile's K-loop.

3. **Wider autotune space** — BK ∈ {64, 128}, num_stages ∈ {2, 3, 4},
   num_warps ∈ {4, 8}, plus booleans for `WARP_SPECIALIZE`,
   `EPILOGUE_SUBTILE`, `FLATTEN` (84 configs after pruning the invalid
   `EPI ∧ ¬FLATTEN` corner).  Per-shape autotune picks the winner.

Stripped to the body:

```python
@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, NUM_SMS: tl.constexpr,
                  BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
                  WARP_SPECIALIZE, EPILOGUE_SUBTILE, FLATTEN):
    start_pid = tl.program_id(axis=0)
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1],
                                       [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, [K, N], [N, 1],
                                       [BLOCK_SIZE_K, BLOCK_SIZE_N])
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [N, 1],
        [BLOCK_SIZE_M, BLOCK_SIZE_N // 2 if EPILOGUE_SUBTILE else BLOCK_SIZE_N])

    tile_id_c = start_pid - NUM_SMS                              # <-- deferral
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS,
                            flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, ...)
        offs_m, offs_n = pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(tl.cdiv(K, BLOCK_SIZE_K)):
            a = a_desc.load([offs_m, ki * BLOCK_SIZE_K])
            b = b_desc.load([ki * BLOCK_SIZE_K, offs_n])
            acc = tl.dot(a, b, acc)

        tile_id_c += NUM_SMS                                      # <-- one iter behind
        pid_m, pid_n = _compute_pid(tile_id_c, ...)
        offs_cm, offs_cn = pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N
        if EPILOGUE_SUBTILE:                                      # <-- split store
            acc = tl.reshape(acc, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c_desc.store([offs_cm, offs_cn],                       acc0.to(dtype))
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2],   acc1.to(dtype))
        else:
            c_desc.store([offs_cm, offs_cn], acc.to(dtype))
```

## The mapping to ladder concepts

Every Triton construct in this kernel corresponds to something
specific we taught:

| Triton construct | Ladder chapter | What it lowers to |
|---|---|---|
| `tl.make_tensor_descriptor` | ch00, ch01 | `cuTensorMapEncodeTiled` + `cp.async.bulk.tensor.2d.shared::cluster.global` |
| `tl.dot(a, b, acc)` on sm_100a | ch02, ch08 | `tcgen05.mma.cta_group::{1,2}.kind::f16` |
| `tl.range(start_pid, num_tiles, NUM_SMS)` | persistent grid | A `for cluster_id += NUM_SMS` outer loop |
| `flatten=True` | (overlap topic) | K-loop / epilogue overlap across adjacent tiles |
| `warp_specialize=True` | ch07 | Producer/consumer warp split with mbarrier handshakes |
| `GROUP_SIZE_M` swizzle in `_compute_pid` | ch09 | The M-fast-within-chunk walk for L2 reuse |
| `triton.autotune(key=[M,N,K])` | ch12 | Per-shape config picking with measurement-driven ranking |
| `num_stages` parameter | ch04, ch08 | NS-stage K-ring buffer |

`tl.make_tensor_descriptor` deserves a note — Triton **builds the
tensormap on demand inside the kernel call**, with the strides
inferred from the tensor shape.  Our hand-written CUDA builds the
tensormap on the host once (`cuTensorMapEncodeTiled`) and passes it
in as a `__grid_constant__`.  The semantics are identical; Triton's
abstraction is just more convenient.

## Per-shape results

`M = N = K ∈ {2048, …, 12288}` (11 shapes), B200, measured via
`triton.testing.do_bench`.  All three kernels are autotuned at each
shape; Triton autotune sweeps the 84-config space above.

| shape  | ch12 cfg | ch12 TF | Triton TF | cuBLAS TF | ch12 / cuBLAS | Triton / cuBLAS |
|---|---|---|---|---|---|---|
| 2048³  | (5, 4)  |  797 |  672 |  879 |  91 % | 76 % |
| 3072³  | (6, 1)  | 1257 | 1205 | 1447 |  87 % | 83 % |
| 4096³  | (6, 1)  | 1279 | 1278 | 1413 |  90 % | 90 % |
| 5120³  | (7, 4)  | 1330 | 1292 | 1464 |  91 % | 88 % |
| 6144³  | (5, 8)  | 1376 | 1309 | 1495 |  92 % | 88 % |
| 7168³  | (6, 8)  | 1368 | 1312 | 1483 |  92 % | 88 % |
| 8192³  | (6, 8)  | 1382 | 1330 | 1461 |  95 % | 91 % |
| 9216³  | (6, 8)  | 1372 | 1318 | 1450 |  95 % | 91 % |
| 10240³ | (6, 8)  | 1402 | 1276 | 1432 |  98 % | 89 % |
| 11264³ | (7, 8)  | 1382 | 1338 | 1424 |  97 % | 94 % |
| 12288³ | (5, 8)  | 1402 | 1351 | 1489 |  94 % | 91 % |

## What the chapter teaches

Triton now lands in the **88-94 %** band across mid/large shapes,
peaking at **94 %** at 11K and matching ch12 at 4K (both 90 %).  The
hand-written ladder still wins, but by a much smaller margin (3-9
points at most shapes) than the basic Triton form gave us (10-25 pts).
Two shapes stand apart:

- **2K (76 %)** — Triton's persistent-overlap pattern needs enough
  tiles per SM to amortise the prologue; at 16×16 tiles total
  (`16² / 148 SMs ≈ 1.7 tiles/SM`) there isn't much overlap to
  recover.  ch12 doesn't pay this cost because its smaller tile
  configs (e.g. BM=128, BN=128) give more output tiles to walk.
- **10K (89 % vs 98 % for ch12)** — ch12's CTA-swizzle chunked walk
  (ch09) is unusually well-matched to the L2 working-set at this
  shape; Triton's `GROUP_SIZE_M=8` swizzle doesn't reach it.

Where the new Triton kernel makes up ground on the basic one is
exactly where you'd expect: `tile_id_c` deferral hides the epilogue
behind the next tile's K-loop, which dominates more of total runtime
at small/mid shapes — that's why the 2K→4K range improved the most
(70→76, 71→83, 80→90 from the previous run).

## When to write Triton vs hand-written PTX

The headline framing the chapter aims at:

- **Default to Triton** for new kernels.  ~80 lines of Python beats
  ~600 lines of `.cu` on engineering cost by an order of magnitude,
  and the Blackwell-standard idioms above ('tile_id_c' deferral,
  EPILOGUE_SUBTILE, wide per-shape autotune) are what closes most of
  the gap to cuBLAS.
- **Hand-write PTX when you need every percent.**  The ladder's
  technique stack — explicit cluster MMA, explicit K-major B
  descriptor, explicit warp specialization with hand-placed mbarriers,
  explicit multi-stage ring with phase tracking — picks up the
  remaining 3-9 points back to cuBLAS that Triton's compiler leaves on
  the table.  At small shapes the gap is bigger (15 pts at 2K) because
  Triton's persistent-overlap form needs many tiles per SM to amortise.

The B200 ladder ends here.  ch12 is the perf endpoint at our headline
shapes; this chapter is the mirror that grounds the comparison
against an idiomatic Triton baseline written in the Blackwell-standard
form, and says the ladder's 90-98 % of cuBLAS is genuinely past
what idiomatic Triton-on-B200 gets you today.

## Run

```bash
pip install -r ../requirements.txt
python main.py
```

A Blackwell GPU (sm_100a / B200) is required.  First run compiles
ch12's 20-variant sweep (~30 s) and triggers Triton's per-shape
autotune (~few seconds per shape).
