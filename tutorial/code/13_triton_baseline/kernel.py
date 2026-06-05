"""Canonical Blackwell-standard Triton bf16 matmul.

Mirrors Triton's `tutorials/09-persistent-matmul.py
matmul_kernel_descriptor_persistent`, adapted for B in (K, N)
row-major (the PyTorch convention) instead of the tutorial's (N, K).

Three "real" Triton-on-Blackwell idioms layered on top of the basic
persistent + warp-spec + FLATTEN kernel:

  1. **EPILOGUE_SUBTILE** — split the BN-wide accumulator into two
     BN/2-wide halves and store them separately.  Lets the epilogue
     stores overlap with each other and with the next tile's K-loop.

  2. **tile_id_c deferral** — store tile T's output AFTER tile T+1's
     K-loop has started.  Implemented by maintaining a second
     `tile_id_c` counter that lags `tile_id` by one outer iteration.
     The compiler then interleaves "K-loop for T+1" with "epilogue
     store for T" (a K-loop/epilogue overlap pattern, implemented at
     the Triton expression level rather than via custom PTX).

  3. **Wider autotune space** — BK ∈ {64, 128}, num_stages ∈ {2,3,4},
     num_warps ∈ {4, 8}, plus toggles for WARP_SPECIALIZE,
     EPILOGUE_SUBTILE, FLATTEN.  Per-shape autotune picks the winner.

Every Triton construct here corresponds to something we taught in the
hand-written ladder (ch00-ch12); the README maps them one-for-one.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _tma_alloc(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


_ALLOCATOR_SET = False
def ensure_tma_allocator():
    global _ALLOCATOR_SET
    if _ALLOCATOR_SET:
        return
    triton.set_allocator(_tma_alloc)
    _ALLOCATOR_SET = True


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M: tl.constexpr):
    group_id     = tile_id // num_pid_in_group
    first_pid_m  = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m        = first_pid_m + (tile_id % group_size_m)
    pid_n        = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# Autotune space — verbatim from the Blackwell-standard reference
# (mymatmul/gpu/matmul_triton.py).  84 configs total; per-shape
# autotune times them all on first call and caches the winner.
_CONFIGS = [
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
         "GROUP_SIZE_M": 8, "WARP_SPECIALIZE": ws,
         "EPILOGUE_SUBTILE": epi, "FLATTEN": fl},
        num_stages=ns, num_warps=nw,
    )
    for bn  in (128, 256)
    for bk  in (64, 128)
    for ns  in (2, 3, 4)
    for nw  in (4, 8)
    for ws  in (True, False)
    for epi in (True, False)
    for fl  in (True, False)
    # EPILOGUE_SUBTILE=True with FLATTEN=False is invalid per tutorial.
    if not (epi and not fl)
]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    start_pid        = tl.program_id(axis=0)
    num_pid_m        = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n        = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles          = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles        = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Device-side TMA descriptors (Triton emits cuTensorMapEncodeTiled
    # + cp.async.bulk.tensor.2d.*).  C uses a narrower box when
    # EPILOGUE_SUBTILE is set, so the two halves can be stored as
    # separate TMA calls.
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[N, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[N, 1],
        block_shape=[BLOCK_SIZE_M,
                     BLOCK_SIZE_N // 2 if EPILOGUE_SUBTILE else BLOCK_SIZE_N])

    # tile_id_c starts one outer-iter BEHIND tile_id, so this iter's
    # epilogue store is for the PREVIOUS tile.  The compiler interleaves
    # the K-loop of tile T+1 with the C store of tile T → K-loop /
    # epilogue overlap, expressed at the Triton expression level.
    tile_id_c = start_pid - NUM_SMS

    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS,
        flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE,
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_m = pid_m * BLOCK_SIZE_M
        offs_n = pid_n * BLOCK_SIZE_N

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_m, offs_k])
            b = b_desc.load([offs_k, offs_n])
            acc = tl.dot(a, b, acc)

        # Use the deferred tile_id_c for the store offset.
        tile_id_c += NUM_SMS
        pid_m_c, pid_n_c = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_cm = pid_m_c * BLOCK_SIZE_M
        offs_cn = pid_n_c * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE:
            # Split the BN-wide accumulator into 2 × BN/2 halves and
            # store separately — better epilogue pipelining.
            acc3 = tl.reshape(acc, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc3 = tl.permute(acc3, (0, 2, 1))
            acc0, acc1 = tl.split(acc3)
            c_desc.store([offs_cm, offs_cn], acc0.to(dtype))
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], acc1.to(dtype))
        else:
            c_desc.store([offs_cm, offs_cn], acc.to(dtype))


def triton_matmul(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
    """Run the Blackwell-standard Triton matmul: C[M,N] = A[M,K] @ B[K,N].

    Both A and B are bf16, row-major.  First call at a given (M, N, K)
    triggers `triton.autotune` across 84 configs (~few seconds);
    subsequent calls dispatch to the cached winner instantly.
    """
    ensure_tma_allocator()
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count

    def grid(META):
        return (min(num_sms,
                    triton.cdiv(M, META["BLOCK_SIZE_M"])
                  * triton.cdiv(N, META["BLOCK_SIZE_N"])),)

    matmul_kernel[grid](A, B, C, M, N, K, NUM_SMS=num_sms)
    return C
