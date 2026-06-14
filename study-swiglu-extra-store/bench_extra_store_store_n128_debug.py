#!/usr/bin/env python3
"""Study-only benchmark for a SwiGLU-like extra epilogue store.

The generated MMComposer kernel is patched locally so its pipelined TMA-store
epilogue writes extra output tensors.  The store-only variants keep the compute
path unchanged; the swiglu variants also compute the SwiGLU epilogue math.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import pathlib
import re
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEBUI = ROOT / "webui"
TESTS = WEBUI / "tests"
sys.path.insert(0, str(WEBUI))
sys.path.insert(0, str(TESTS))

import mvp_core as mc  # noqa: E402
import gpu_codegen_driver as gd  # noqa: E402
import torch  # noqa: E402
from cuda.bindings import driver  # noqa: E402


HERE = pathlib.Path(__file__).resolve().parent
SCRATCH = HERE / "_scratch"

CONFIG = {
    "bm": 128,
    "bn": 256,
    "bk": 64,
    "ns": 5,
    "gsm": 1,
    "nw": 4,
    "persistent": 1,
    "ld_width": 8,
    "overlap": 1,
    "split_epilogue": 0,
    "l1_no_alloc": 0,
    "tma_pipelined": 1,
    "single_tmem": 0,
}


def parse_shape(spec: str) -> tuple[int, int, int]:
    tok = spec.lower().replace(",", "x")
    if "x" in tok:
        m, n, k = (int(x) for x in tok.split("x"))
        return m, n, k
    s = int(tok)
    return s, s, s


def parse_shapes(specs: list[str]) -> list[tuple[int, int, int]]:
    out = []
    for spec in specs:
        for tok in spec.split(";"):
            tok = tok.strip()
            if tok:
                out.append(parse_shape(tok))
    return out


def parse_int_csv(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def patch_store_n(src: str, store_n: int) -> str:
    if store_n not in (64, 128):
        raise ValueError("STORE_N study currently supports only 64 or 128")
    src, n = re.subn(
        r"constexpr int STORE_N\s+= \d+;",
        f"constexpr int STORE_N          = {store_n};",
        src,
        count=1,
    )
    if n != 1:
        raise RuntimeError("could not patch STORE_N")
    src = src.replace(
        'static_assert(STORE_N == 64, "pipelined TMA store assumes STORE_N=64");',
        'static_assert(STORE_N == 64 || STORE_N == 128, '
        '"pipelined TMA store assumes STORE_N=64 or 128");',
    )
    src = src.replace(
        'static_assert(NUM_CHUNKS == 4, "half-only store assumes BN=256 split into 2x128");',
        'static_assert(NUM_CHUNKS == 2 || NUM_CHUNKS == 4, '
        '"half-only store assumes BN=256 with STORE_N=64 or 128");',
    )
    src = src.replace(
        'static_assert(NUM_CHUNKS == 4, "swiglu-half assumes BN=256 split into 2x128");',
        'static_assert(NUM_CHUNKS == 2 || NUM_CHUNKS == 4, '
        '"swiglu-half assumes BN=256 with STORE_N=64 or 128");',
    )
    src = src.replace(
        'static_assert(NUM_CHUNKS == 4, "swiglu-out assumes BN=256 split into 2x128");',
        'static_assert(NUM_CHUNKS == 2 || NUM_CHUNKS == 4, '
        '"swiglu-out assumes BN=256 with STORE_N=64 or 128");',
    )
    return src


def patch_store_stages(src: str, stages: int) -> str:
    if stages < 1:
        raise ValueError("TMA store stages must be >= 1")
    src, n = re.subn(
        r"constexpr int TMA_STORE_STAGES = \d+;",
        f"constexpr int TMA_STORE_STAGES = {stages};",
        src,
        count=1,
    )
    if n != 1:
        raise RuntimeError("could not patch TMA_STORE_STAGES")
    old = "store_stage ^= 1;"
    if old not in src:
        raise RuntimeError("could not find store_stage rotation")
    if stages == 1:
        new = "store_stage = 0;"
    elif stages == 2:
        new = old
    elif stages & (stages - 1) == 0:
        new = "store_stage = (store_stage + 1) & (TMA_STORE_STAGES - 1);"
    else:
        new = "store_stage = (store_stage + 1) % TMA_STORE_STAGES;"
    return src.replace(old, new, 1)


def patch_extra_store(src: str, mode: str) -> str:
    """Add D TMA stores.

    ``half-only`` writes only D[M, N/2] using the first half of each BN tile.
    ``extra-half`` writes D[M, N/2] using the first half of each BN tile.
    ``extra-full`` writes D[M, N] using every chunk.
    ``swiglu-half`` writes C[M, N] factors and D[M, N/2] final output.
    ``swiglu-out`` writes original C[M, N] and D[M, N/2] final output.
    ``swiglu-out-fast`` is ``swiglu-out`` with exp2.approx + rcp.approx.
    ``swiglu-out-fast-dual-b`` also uses fast math, but loads the left/gate
    B panels from separate K x (N/2) tensors.
    """
    src = src.replace(
        "const CUtensorMap* C_tmap_ptr,\n"
        "    __nv_bfloat16* __restrict__ C_ptr,",
        "const CUtensorMap* C_tmap_ptr,\n"
        "    const CUtensorMap* D_tmap_ptr,\n"
        "    __nv_bfloat16* __restrict__ C_ptr,",
        1,
    )
    src = src.replace(
        "const __grid_constant__ CUtensorMap C_tmap,\n"
        "    __nv_bfloat16* C_ptr, int M, int N, int K)",
        "const __grid_constant__ CUtensorMap C_tmap,\n"
        "    const __grid_constant__ CUtensorMap D_tmap,\n"
        "    __nv_bfloat16* C_ptr, int M, int N, int K)",
        1,
    )
    src = src.replace(
        "matmul_cluster_impl(&A_tmap, &B_tmap, &C_tmap, C_ptr, M, N, K);",
        "matmul_cluster_impl(&A_tmap, &B_tmap, &C_tmap, &D_tmap, C_ptr, M, N, K);",
        1,
    )
    old = (
        "tma_2d_store(C_tmap_ptr, src,\n"
        "                                         EPI_OUT_COL_BASE + chunk * STORE_N, EPI_OUT_ROW);\n"
        "                            tma_commit_group();"
    )
    if mode == "swiglu-half":
        return patch_swiglu_half_store(src)
    if mode in ("swiglu-out", "swiglu-out-fast"):
        return patch_swiglu_out_store(src, fast_math=(mode == "swiglu-out-fast"))
    if mode == "swiglu-out-fast-dual-b":
        return patch_dual_b_load(patch_swiglu_out_store(src, fast_math=True))
    if mode == "half-only":
        return patch_half_only_store(src)
    if mode == "extra-half":
        new = (
            "tma_2d_store(C_tmap_ptr, src,\n"
            "                                         EPI_OUT_COL_BASE + chunk * STORE_N, EPI_OUT_ROW);\n"
            "                            tma_commit_group();\n"
            "                            if (chunk < NUM_CHUNKS / 2) {\n"
            "                                tma_2d_store(D_tmap_ptr, src,\n"
            "                                             EPI_OUT_COL_BASE / 2 + chunk * STORE_N, EPI_OUT_ROW);\n"
            "                                tma_commit_group();\n"
            "                            }"
        )
    elif mode == "extra-full":
        new = (
            "tma_2d_store(C_tmap_ptr, src,\n"
            "                                         EPI_OUT_COL_BASE + chunk * STORE_N, EPI_OUT_ROW);\n"
            "                            tma_commit_group();\n"
            "                            tma_2d_store(D_tmap_ptr, src,\n"
            "                                         EPI_OUT_COL_BASE + chunk * STORE_N, EPI_OUT_ROW);\n"
            "                            tma_commit_group();"
        )
    else:
        raise ValueError(f"unknown extra store mode: {mode}")
    if old not in src:
        raise RuntimeError("could not find C TMA store block")
    return src.replace(old, new, 1)


def patch_half_only_store(src: str) -> str:
    """Patch the base TMA epilogue to write only D[M, N/2].

    The compute tile is still BN=256.  The epilogue drains both left/right
    halves from TMEM, combines them with a cheap add, and stores only one
    half-width D tensor.  This keeps both halves live while making final GMEM
    output traffic exactly 0.5x.
    """
    old = """                {
                    constexpr int LOADS_PER_CHUNK = STORE_N / 8;
                    constexpr int LOADS_PER_WARP = LOADS_PER_CHUNK / COL_GROUPS;
                    constexpr int NUM_CHUNKS = BN / STORE_N;
                    static_assert(STORE_N == 64, "pipelined TMA store assumes STORE_N=64");
                    static_assert(NUM_CHUNKS * STORE_N == BN, "BN must divide into STORE_N chunks");
                    static_assert(LOADS_PER_WARP * COL_GROUPS == LOADS_PER_CHUNK,
                                  "STORE_N/8 chunks must divide across column warp groups");
                    int store_stage = 0;

                    #pragma unroll
                    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                        if (ew == 0)
                            tma_wait_group<TMA_STORE_STAGES - 1>();

                        float t[LOADS_PER_WARP][8];
                        #pragma unroll
                        for (int n = 0; n < LOADS_PER_WARP; n++) {
                            const int local_n = col_warp * LOADS_PER_WARP + n;
                            tcgen05_ld_32x32b_x8(trow + (uint32_t)(chunk * STORE_N + local_n * 8), t[n]);
                        }
                        tcgen05_wait_ld();

                        if (chunk == NUM_CHUNKS - 1) {
                            tcgen05_fence_before_thread_sync();
                            if (ew == 0 && elect_sync())
                                EPI_TMEM_EMPTY_ARRIVE(buf);
                        }

                        asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));

                        #pragma unroll
                        for (int n = 0; n < LOADS_PER_WARP; n++) {
                            __nv_bfloat162 pk[4];
                            #pragma unroll
                            for (int i = 0; i < 4; i++)
                                pk[i] = __floats2bfloat162_rn(t[n][2 * i], t[n][2 * i + 1]);
                            const int local_n = col_warp * LOADS_PER_WARP + n;
                            const int swizzled_n = local_n ^ (my_row & 7);
                            __nv_bfloat16* write_ptr =
                                C_store + store_stage * BM * STORE_N + my_row * STORE_N + swizzled_n * 8;
                            *reinterpret_cast<int4*>(write_ptr) = *reinterpret_cast<int4*>(pk);
                        }

                        __syncwarp();
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));

                        if (ew == 0 && elect_sync()) {
                            const uint32_t src = STORE_SMEM_BASE + store_stage * STORE_BUF_BYTES;
                            tma_2d_store(C_tmap_ptr, src,
                                         EPI_OUT_COL_BASE + chunk * STORE_N, EPI_OUT_ROW);
                            tma_commit_group();
                        }

                        store_stage ^= 1;
                    }
                }"""
    new = """                {
                    constexpr int LOADS_PER_CHUNK = STORE_N / 8;
                    constexpr int LOADS_PER_WARP = LOADS_PER_CHUNK / COL_GROUPS;
                    constexpr int NUM_CHUNKS = BN / STORE_N;
                    constexpr int HALF_CHUNKS = NUM_CHUNKS / 2;
                    static_assert(STORE_N == 64, "pipelined TMA store assumes STORE_N=64");
                    static_assert(NUM_CHUNKS == 4, "half-only store assumes BN=256 split into 2x128");
                    static_assert(LOADS_PER_WARP * COL_GROUPS == LOADS_PER_CHUNK,
                                  "STORE_N/8 chunks must divide across column warp groups");
                    int store_stage = 0;

                    auto wait_for_store_buffer = [&]() {
                        if (ew == 0)
                            tma_wait_group<TMA_STORE_STAGES - 1>();
                        asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));
                    };

                    auto issue_tma_store = [&](const CUtensorMap* tmap, int out_col) {
                        __syncwarp();
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));

                        if (ew == 0 && elect_sync()) {
                            const uint32_t src = STORE_SMEM_BASE + store_stage * STORE_BUF_BYTES;
                            tma_2d_store(tmap, src, out_col, EPI_OUT_ROW);
                            tma_commit_group();
                        }

                        store_stage ^= 1;
                    };

                    #pragma unroll
                    for (int chunk = 0; chunk < HALF_CHUNKS; chunk++) {
                        float left[LOADS_PER_WARP][8];
                        float right[LOADS_PER_WARP][8];
                        #pragma unroll
                        for (int n = 0; n < LOADS_PER_WARP; n++) {
                            const int local_n = col_warp * LOADS_PER_WARP + n;
                            tcgen05_ld_32x32b_x8(
                                trow + (uint32_t)(chunk * STORE_N + local_n * 8), left[n]);
                            tcgen05_ld_32x32b_x8(
                                trow + (uint32_t)((chunk + HALF_CHUNKS) * STORE_N + local_n * 8), right[n]);
                        }
                        tcgen05_wait_ld();

                        if (chunk == HALF_CHUNKS - 1) {
                            tcgen05_fence_before_thread_sync();
                            if (ew == 0 && elect_sync())
                                EPI_TMEM_EMPTY_ARRIVE(buf);
                        }

                        wait_for_store_buffer();
                        #pragma unroll
                        for (int n = 0; n < LOADS_PER_WARP; n++) {
                            __nv_bfloat162 pk[4];
                            #pragma unroll
                            for (int i = 0; i < 4; i++) {
                                pk[i] = __floats2bfloat162_rn(
                                    left[n][2 * i] + right[n][2 * i],
                                    left[n][2 * i + 1] + right[n][2 * i + 1]);
                            }
                            const int local_n = col_warp * LOADS_PER_WARP + n;
                            const int swizzled_n = local_n ^ (my_row & 7);
                            __nv_bfloat16* write_ptr =
                                C_store + store_stage * BM * STORE_N + my_row * STORE_N + swizzled_n * 8;
                            *reinterpret_cast<int4*>(write_ptr) = *reinterpret_cast<int4*>(pk);
                        }
                        issue_tma_store(D_tmap_ptr, EPI_OUT_COL_BASE / 2 + chunk * STORE_N);
                    }
                }"""
    if old not in src:
        raise RuntimeError("could not find base TMA epilogue block for half-only")
    return src.replace(old, new, 1)


def patch_swiglu_half_store(src: str) -> str:
    """Patch the TMA epilogue to match fused SwiGLU save-factors semantics.

    For each BN=256 tile:
      left = columns [0, 128)
      gate = columns [128, 256)
      C factors [0, 128)   = silu(gate)
      C factors [128, 256) = left * silu_prime(gate)
      D output [0, 128)    = left * silu(gate)
    """
    src = src.replace(
        "const CUtensorMap* C_tmap_ptr,\n"
        "    __nv_bfloat16* __restrict__ C_ptr,",
        "const CUtensorMap* C_tmap_ptr,\n"
        "    const CUtensorMap* D_tmap_ptr,\n"
        "    __nv_bfloat16* __restrict__ C_ptr,",
        1,
    )
    src = src.replace(
        "const __grid_constant__ CUtensorMap C_tmap,\n"
        "    __nv_bfloat16* C_ptr, int M, int N, int K)",
        "const __grid_constant__ CUtensorMap C_tmap,\n"
        "    const __grid_constant__ CUtensorMap D_tmap,\n"
        "    __nv_bfloat16* C_ptr, int M, int N, int K)",
        1,
    )
    src = src.replace(
        "matmul_cluster_impl(&A_tmap, &B_tmap, &C_tmap, C_ptr, M, N, K);",
        "matmul_cluster_impl(&A_tmap, &B_tmap, &C_tmap, &D_tmap, C_ptr, M, N, K);",
        1,
    )

    old = """                {
                    constexpr int LOADS_PER_CHUNK = STORE_N / 8;
                    constexpr int LOADS_PER_WARP = LOADS_PER_CHUNK / COL_GROUPS;
                    constexpr int NUM_CHUNKS = BN / STORE_N;
                    static_assert(STORE_N == 64, "pipelined TMA store assumes STORE_N=64");
                    static_assert(NUM_CHUNKS * STORE_N == BN, "BN must divide into STORE_N chunks");
                    static_assert(LOADS_PER_WARP * COL_GROUPS == LOADS_PER_CHUNK,
                                  "STORE_N/8 chunks must divide across column warp groups");
                    int store_stage = 0;

                    #pragma unroll
                    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                        if (ew == 0)
                            tma_wait_group<TMA_STORE_STAGES - 1>();

                        float t[LOADS_PER_WARP][8];
                        #pragma unroll
                        for (int n = 0; n < LOADS_PER_WARP; n++) {
                            const int local_n = col_warp * LOADS_PER_WARP + n;
                            tcgen05_ld_32x32b_x8(trow + (uint32_t)(chunk * STORE_N + local_n * 8), t[n]);
                        }
                        tcgen05_wait_ld();

                        if (chunk == NUM_CHUNKS - 1) {
                            tcgen05_fence_before_thread_sync();
                            if (ew == 0 && elect_sync())
                                EPI_TMEM_EMPTY_ARRIVE(buf);
                        }

                        asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));

                        #pragma unroll
                        for (int n = 0; n < LOADS_PER_WARP; n++) {
                            __nv_bfloat162 pk[4];
                            #pragma unroll
                            for (int i = 0; i < 4; i++)
                                pk[i] = __floats2bfloat162_rn(t[n][2 * i], t[n][2 * i + 1]);
                            const int local_n = col_warp * LOADS_PER_WARP + n;
                            const int swizzled_n = local_n ^ (my_row & 7);
                            __nv_bfloat16* write_ptr =
                                C_store + store_stage * BM * STORE_N + my_row * STORE_N + swizzled_n * 8;
                            *reinterpret_cast<int4*>(write_ptr) = *reinterpret_cast<int4*>(pk);
                        }

                        __syncwarp();
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));

                        if (ew == 0 && elect_sync()) {
                            const uint32_t src = STORE_SMEM_BASE + store_stage * STORE_BUF_BYTES;
                            tma_2d_store(C_tmap_ptr, src,
                                         EPI_OUT_COL_BASE + chunk * STORE_N, EPI_OUT_ROW);
                            tma_commit_group();
                        }

                        store_stage ^= 1;
                    }
                }"""

    new = """                {
                    constexpr int LOADS_PER_CHUNK = STORE_N / 8;
                    constexpr int LOADS_PER_WARP = LOADS_PER_CHUNK / COL_GROUPS;
                    constexpr int NUM_CHUNKS = BN / STORE_N;
                    constexpr int HALF_CHUNKS = NUM_CHUNKS / 2;
                    static_assert(STORE_N == 64, "pipelined TMA store assumes STORE_N=64");
                    static_assert(NUM_CHUNKS == 4, "swiglu-half assumes BN=256 split into 2x128");
                    static_assert(LOADS_PER_WARP * COL_GROUPS == LOADS_PER_CHUNK,
                                  "STORE_N/8 chunks must divide across column warp groups");
                    int store_stage = 0;

                    #pragma unroll
                    for (int chunk = 0; chunk < HALF_CHUNKS; chunk++) {
                        float left[LOADS_PER_WARP][8];
                        float gate[LOADS_PER_WARP][8];
                        #pragma unroll
                        for (int n = 0; n < LOADS_PER_WARP; n++) {
                            const int local_n = col_warp * LOADS_PER_WARP + n;
                            tcgen05_ld_32x32b_x8(
                                trow + (uint32_t)(chunk * STORE_N + local_n * 8), left[n]);
                            tcgen05_ld_32x32b_x8(
                                trow + (uint32_t)((chunk + HALF_CHUNKS) * STORE_N + local_n * 8), gate[n]);
                        }
                        tcgen05_wait_ld();

                        if (chunk == HALF_CHUNKS - 1) {
                            tcgen05_fence_before_thread_sync();
                            if (ew == 0 && elect_sync())
                                EPI_TMEM_EMPTY_ARRIVE(buf);
                        }

                        #pragma unroll
                        for (int out_kind = 0; out_kind < 3; out_kind++) {
                            if (ew == 0)
                                tma_wait_group<TMA_STORE_STAGES - 1>();
                            asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));

                            #pragma unroll
                            for (int n = 0; n < LOADS_PER_WARP; n++) {
                                __nv_bfloat162 pk[4];
                                #pragma unroll
                                for (int i = 0; i < 4; i++) {
                                    const float l0 = left[n][2 * i];
                                    const float g0 = gate[n][2 * i];
                                    const float s0 = 1.0f / (1.0f + __expf(-g0));
                                    const float silu0 = g0 * s0;
                                    const float sp0 = s0 + silu0 * (1.0f - s0);
                                    const float v0 = (out_kind == 0) ? silu0
                                                   : (out_kind == 1) ? (l0 * sp0)
                                                   : (l0 * silu0);

                                    const float l1 = left[n][2 * i + 1];
                                    const float g1 = gate[n][2 * i + 1];
                                    const float s1 = 1.0f / (1.0f + __expf(-g1));
                                    const float silu1 = g1 * s1;
                                    const float sp1 = s1 + silu1 * (1.0f - s1);
                                    const float v1 = (out_kind == 0) ? silu1
                                                   : (out_kind == 1) ? (l1 * sp1)
                                                   : (l1 * silu1);
                                    pk[i] = __floats2bfloat162_rn(v0, v1);
                                }
                                const int local_n = col_warp * LOADS_PER_WARP + n;
                                const int swizzled_n = local_n ^ (my_row & 7);
                                __nv_bfloat16* write_ptr =
                                    C_store + store_stage * BM * STORE_N + my_row * STORE_N + swizzled_n * 8;
                                *reinterpret_cast<int4*>(write_ptr) = *reinterpret_cast<int4*>(pk);
                            }

                            __syncwarp();
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));

                            if (ew == 0 && elect_sync()) {
                                const uint32_t src = STORE_SMEM_BASE + store_stage * STORE_BUF_BYTES;
                                if (out_kind == 0) {
                                    tma_2d_store(C_tmap_ptr, src,
                                                 EPI_OUT_COL_BASE + chunk * STORE_N, EPI_OUT_ROW);
                                } else if (out_kind == 1) {
                                    tma_2d_store(C_tmap_ptr, src,
                                                 EPI_OUT_COL_BASE + (chunk + HALF_CHUNKS) * STORE_N, EPI_OUT_ROW);
                                } else {
                                    tma_2d_store(D_tmap_ptr, src,
                                                 EPI_OUT_COL_BASE / 2 + chunk * STORE_N, EPI_OUT_ROW);
                                }
                                tma_commit_group();
                            }

                            store_stage ^= 1;
                        }
                    }
                }"""

    if old not in src:
        raise RuntimeError("could not find TMA epilogue block for swiglu-half")
    return src.replace(old, new, 1)


def patch_swiglu_out_store(src: str, *, fast_math: bool) -> str:
    """Patch epilogue to store original preactivation plus final SwiGLU output."""
    src = src.replace(
        "const CUtensorMap* C_tmap_ptr,\n"
        "    __nv_bfloat16* __restrict__ C_ptr,",
        "const CUtensorMap* C_tmap_ptr,\n"
        "    const CUtensorMap* D_tmap_ptr,\n"
        "    __nv_bfloat16* __restrict__ C_ptr,",
        1,
    )
    src = src.replace(
        "const __grid_constant__ CUtensorMap C_tmap,\n"
        "    __nv_bfloat16* C_ptr, int M, int N, int K)",
        "const __grid_constant__ CUtensorMap C_tmap,\n"
        "    const __grid_constant__ CUtensorMap D_tmap,\n"
        "    __nv_bfloat16* C_ptr, int M, int N, int K)",
        1,
    )
    src = src.replace(
        "matmul_cluster_impl(&A_tmap, &B_tmap, &C_tmap, C_ptr, M, N, K);",
        "matmul_cluster_impl(&A_tmap, &B_tmap, &C_tmap, &D_tmap, C_ptr, M, N, K);",
        1,
    )

    old = """                {
                    constexpr int LOADS_PER_CHUNK = STORE_N / 8;
                    constexpr int LOADS_PER_WARP = LOADS_PER_CHUNK / COL_GROUPS;
                    constexpr int NUM_CHUNKS = BN / STORE_N;
                    static_assert(STORE_N == 64, "pipelined TMA store assumes STORE_N=64");
                    static_assert(NUM_CHUNKS * STORE_N == BN, "BN must divide into STORE_N chunks");
                    static_assert(LOADS_PER_WARP * COL_GROUPS == LOADS_PER_CHUNK,
                                  "STORE_N/8 chunks must divide across column warp groups");
                    int store_stage = 0;

                    #pragma unroll
                    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                        if (ew == 0)
                            tma_wait_group<TMA_STORE_STAGES - 1>();

                        float t[LOADS_PER_WARP][8];
                        #pragma unroll
                        for (int n = 0; n < LOADS_PER_WARP; n++) {
                            const int local_n = col_warp * LOADS_PER_WARP + n;
                            tcgen05_ld_32x32b_x8(trow + (uint32_t)(chunk * STORE_N + local_n * 8), t[n]);
                        }
                        tcgen05_wait_ld();

                        if (chunk == NUM_CHUNKS - 1) {
                            tcgen05_fence_before_thread_sync();
                            if (ew == 0 && elect_sync())
                                EPI_TMEM_EMPTY_ARRIVE(buf);
                        }

                        asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));

                        #pragma unroll
                        for (int n = 0; n < LOADS_PER_WARP; n++) {
                            __nv_bfloat162 pk[4];
                            #pragma unroll
                            for (int i = 0; i < 4; i++)
                                pk[i] = __floats2bfloat162_rn(t[n][2 * i], t[n][2 * i + 1]);
                            const int local_n = col_warp * LOADS_PER_WARP + n;
                            const int swizzled_n = local_n ^ (my_row & 7);
                            __nv_bfloat16* write_ptr =
                                C_store + store_stage * BM * STORE_N + my_row * STORE_N + swizzled_n * 8;
                            *reinterpret_cast<int4*>(write_ptr) = *reinterpret_cast<int4*>(pk);
                        }

                        __syncwarp();
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));

                        if (ew == 0 && elect_sync()) {
                            const uint32_t src = STORE_SMEM_BASE + store_stage * STORE_BUF_BYTES;
                            tma_2d_store(C_tmap_ptr, src,
                                         EPI_OUT_COL_BASE + chunk * STORE_N, EPI_OUT_ROW);
                            tma_commit_group();
                        }

                        store_stage ^= 1;
                    }
                }"""

    sigmoid0 = (
        "const float x0 = -g0 * 1.4426950408889634f;\n"
        "                                float e0;\n"
        "                                asm volatile(\"ex2.approx.ftz.f32 %0, %1;\" : \"=f\"(e0) : \"f\"(x0));\n"
        "                                float s0;\n"
        "                                const float d0 = 1.0f + e0;\n"
        "                                asm volatile(\"rcp.approx.ftz.f32 %0, %1;\" : \"=f\"(s0) : \"f\"(d0));"
        if fast_math else
        "const float s0 = 1.0f / (1.0f + __expf(-g0));"
    )
    sigmoid1 = (
        "const float x1 = -g1 * 1.4426950408889634f;\n"
        "                                float e1;\n"
        "                                asm volatile(\"ex2.approx.ftz.f32 %0, %1;\" : \"=f\"(e1) : \"f\"(x1));\n"
        "                                float s1;\n"
        "                                const float d1 = 1.0f + e1;\n"
        "                                asm volatile(\"rcp.approx.ftz.f32 %0, %1;\" : \"=f\"(s1) : \"f\"(d1));"
        if fast_math else
        "const float s1 = 1.0f / (1.0f + __expf(-g1));"
    )

    new = f"""                {{
                    constexpr int LOADS_PER_CHUNK = STORE_N / 8;
                    constexpr int LOADS_PER_WARP = LOADS_PER_CHUNK / COL_GROUPS;
                    constexpr int NUM_CHUNKS = BN / STORE_N;
                    constexpr int HALF_CHUNKS = NUM_CHUNKS / 2;
                    static_assert(STORE_N == 64, "pipelined TMA store assumes STORE_N=64");
                    static_assert(NUM_CHUNKS == 4, "swiglu-out assumes BN=256 split into 2x128");
                    static_assert(LOADS_PER_WARP * COL_GROUPS == LOADS_PER_CHUNK,
                                  "STORE_N/8 chunks must divide across column warp groups");
                    int store_stage = 0;

                    auto wait_for_store_buffer = [&]() {{
                        if (ew == 0)
                            tma_wait_group<TMA_STORE_STAGES - 1>();
                        asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));
                    }};

                    auto issue_tma_store = [&](const CUtensorMap* tmap, int out_col) {{
                        __syncwarp();
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));

                        if (ew == 0 && elect_sync()) {{
                            const uint32_t src = STORE_SMEM_BASE + store_stage * STORE_BUF_BYTES;
                            tma_2d_store(tmap, src, out_col, EPI_OUT_ROW);
                            tma_commit_group();
                        }}

                        store_stage ^= 1;
                    }};

                    #pragma unroll
                    for (int chunk = 0; chunk < HALF_CHUNKS; chunk++) {{
                        float left[LOADS_PER_WARP][8];
                        float gate[LOADS_PER_WARP][8];
                        #pragma unroll
                        for (int n = 0; n < LOADS_PER_WARP; n++) {{
                            const int local_n = col_warp * LOADS_PER_WARP + n;
                            tcgen05_ld_32x32b_x8(
                                trow + (uint32_t)(chunk * STORE_N + local_n * 8), left[n]);
                            tcgen05_ld_32x32b_x8(
                                trow + (uint32_t)((chunk + HALF_CHUNKS) * STORE_N + local_n * 8), gate[n]);
                        }}
                        tcgen05_wait_ld();

                        if (chunk == HALF_CHUNKS - 1) {{
                            tcgen05_fence_before_thread_sync();
                            if (ew == 0 && elect_sync())
                                EPI_TMEM_EMPTY_ARRIVE(buf);
                        }}

                        wait_for_store_buffer();
                        #pragma unroll
                        for (int n = 0; n < LOADS_PER_WARP; n++) {{
                            __nv_bfloat162 pk[4];
                            #pragma unroll
                            for (int i = 0; i < 4; i++) {{
                                pk[i] = __floats2bfloat162_rn(
                                    left[n][2 * i], left[n][2 * i + 1]);
                            }}
                            const int local_n = col_warp * LOADS_PER_WARP + n;
                            const int swizzled_n = local_n ^ (my_row & 7);
                            __nv_bfloat16* write_ptr =
                                C_store + store_stage * BM * STORE_N + my_row * STORE_N + swizzled_n * 8;
                            *reinterpret_cast<int4*>(write_ptr) = *reinterpret_cast<int4*>(pk);
                        }}
                        issue_tma_store(C_tmap_ptr, EPI_OUT_COL_BASE + chunk * STORE_N);

                        wait_for_store_buffer();
                        #pragma unroll
                        for (int n = 0; n < LOADS_PER_WARP; n++) {{
                            __nv_bfloat162 pk[4];
                            #pragma unroll
                            for (int i = 0; i < 4; i++) {{
                                pk[i] = __floats2bfloat162_rn(
                                    gate[n][2 * i], gate[n][2 * i + 1]);
                            }}
                            const int local_n = col_warp * LOADS_PER_WARP + n;
                            const int swizzled_n = local_n ^ (my_row & 7);
                            __nv_bfloat16* write_ptr =
                                C_store + store_stage * BM * STORE_N + my_row * STORE_N + swizzled_n * 8;
                            *reinterpret_cast<int4*>(write_ptr) = *reinterpret_cast<int4*>(pk);
                        }}
                        issue_tma_store(C_tmap_ptr,
                                        EPI_OUT_COL_BASE + (chunk + HALF_CHUNKS) * STORE_N);

                        wait_for_store_buffer();
                        #pragma unroll
                        for (int n = 0; n < LOADS_PER_WARP; n++) {{
                            __nv_bfloat162 pk[4];
                            #pragma unroll
                            for (int i = 0; i < 4; i++) {{
                                const float l0 = left[n][2 * i];
                                const float g0 = gate[n][2 * i];
                                {sigmoid0}
                                const float v0 = l0 * (g0 * s0);

                                const float l1 = left[n][2 * i + 1];
                                const float g1 = gate[n][2 * i + 1];
                                {sigmoid1}
                                const float v1 = l1 * (g1 * s1);

                                pk[i] = __floats2bfloat162_rn(v0, v1);
                            }}
                            const int local_n = col_warp * LOADS_PER_WARP + n;
                            const int swizzled_n = local_n ^ (my_row & 7);
                            __nv_bfloat16* write_ptr =
                                C_store + store_stage * BM * STORE_N + my_row * STORE_N + swizzled_n * 8;
                            *reinterpret_cast<int4*>(write_ptr) = *reinterpret_cast<int4*>(pk);
                        }}
                        issue_tma_store(D_tmap_ptr, EPI_OUT_COL_BASE / 2 + chunk * STORE_N);
                    }}
                }}"""

    if old not in src:
        raise RuntimeError("could not find TMA epilogue block for swiglu-out")
    return src.replace(old, new, 1)


def patch_dual_b_load(src: str) -> str:
    """Load the left/gate BN halves from separate B tensor maps.

    This is intentionally scoped to the BN=256, 2-CTA study setup: CTA rank 0
    contributes the left 128-column B panel, and CTA rank 1 contributes the
    gate 128-column B panel to the cluster MMA.
    """
    src, n = re.subn(
        r"const CUtensorMap\* B_tmap,\n"
        r"    const CUtensorMap\* C_tmap_ptr,",
        "const CUtensorMap* B_left_tmap,\n"
        "    const CUtensorMap* B_gate_tmap,\n"
        "    const CUtensorMap* C_tmap_ptr,",
        src,
        count=1,
    )
    if n != 1:
        raise RuntimeError("could not patch matmul_cluster_impl dual-B signature")

    src, n = re.subn(
        r"const int num_k = K / BK;\n"
        r"        constexpr int16_t cta_mask = \(1 << CTA_GROUP\) - 1;",
        "const int num_k = K / BK;\n"
        "        static_assert(CTA_GROUP == 2, \"dual-B study assumes 2-CTA cluster MMA\");\n"
        "        static_assert(BN == 256 && BN_LOCAL == 128,\n"
        "                      \"dual-B study assumes BN=256 split into two 128-column panels\");\n"
        "        constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;",
        src,
        count=1,
    )
    if n != 1:
        raise RuntimeError("could not add dual-B static assertions")

    old_load = """                    #pragma unroll
                    for (int n = 0; n < BN_LOCAL; n += 64) {
                        tma_2d_load_g2(B_base(slot) + n * BK * BF16_BYTES,
                                       B_tmap, local_n + n, k * BK, ready_mb_cta0);
                    }"""
    new_load = """                    const CUtensorMap* B_half_tmap =
                        (cta_rank == 0) ? B_left_tmap : B_gate_tmap;
                    const int B_half_col_base = base_n / 2;
                    #pragma unroll
                    for (int n = 0; n < BN_LOCAL; n += 64) {
                        tma_2d_load_g2(B_base(slot) + n * BK * BF16_BYTES,
                                       B_half_tmap, B_half_col_base + n, k * BK, ready_mb_cta0);
                    }"""
    if old_load not in src:
        raise RuntimeError("could not find packed-B TMA load block")
    src = src.replace(old_load, new_load, 1)

    src, n = re.subn(
        r"const __grid_constant__ CUtensorMap B_tmap,\n"
        r"    const __grid_constant__ CUtensorMap C_tmap,",
        "const __grid_constant__ CUtensorMap B_left_tmap,\n"
        "    const __grid_constant__ CUtensorMap B_gate_tmap,\n"
        "    const __grid_constant__ CUtensorMap C_tmap,",
        src,
        count=1,
    )
    if n != 1:
        raise RuntimeError("could not patch kernel dual-B wrapper signature")

    src, n = re.subn(
        r"matmul_cluster_impl\(&A_tmap, &B_tmap, &C_tmap, &D_tmap, C_ptr, M, N, K\);",
        "matmul_cluster_impl(&A_tmap, &B_left_tmap, &B_gate_tmap, "
        "&C_tmap, &D_tmap, C_ptr, M, N, K);",
        src,
        count=1,
    )
    if n != 1:
        raise RuntimeError("could not patch dual-B wrapper call")
    return src


def shared_bytes(k: dict, stages: int) -> int:
    cta_group = 2
    bn_local = k["bn"] // cta_group
    store_n = int(k.get("store_n", 64))
    a_slot = k["bm"] * k["bk"] * 2
    b_slot = bn_local * k["bk"] * 2
    slot = a_slot + b_slot
    epi = k["bm"] * store_n * 2 * stages
    return k["ns"] * slot + epi + 1024


def install_driver_hooks() -> None:
    if not hasattr(gd, "tag_for_extra_store_orig"):
        gd.tag_for_extra_store_orig = gd.tag_for
    if not hasattr(gd, "launch_spec_extra_store_orig"):
        gd.launch_spec_extra_store_orig = gd.launch_spec

    def tag_for_variant(tier, k):
        return (gd.tag_for_extra_store_orig(tier, k)
                + f"_sn{k.get('store_n', 64)}"
                + f"_ts{k.get('tma_store_stages', 2)}"
                + f"_extra{k.get('extra_store_mode', 'base').replace('-', '')}")

    def launch_spec_variant(tier, k, m, n, kval, num_sms=None):
        grid, block, _shared = gd.launch_spec_extra_store_orig(tier, k, m, n, kval, num_sms)
        return grid, block, shared_bytes(k, int(k.get("tma_store_stages", 2)))

    gd.tag_for = tag_for_variant
    gd.launch_spec = launch_spec_variant


def render_compile(tier: dict, k: dict, arch: str) -> pathlib.Path:
    src_path = gd.render_to_dir(tier, k)
    src = src_path.read_text()
    if k.get("extra_store_mode", "base") != "base":
        src = patch_extra_store(src, k["extra_store_mode"])
    src = patch_store_n(src, int(k.get("store_n", 64)))
    src = patch_store_stages(src, int(k["tma_store_stages"]))
    src_path.write_text(src)
    if k.get("extra_store_mode") in (
        "swiglu-half", "swiglu-out", "swiglu-out-fast", "swiglu-out-fast-dual-b",
    ):
        mode_name = k["extra_store_mode"].replace("-", "_")
        store_n = int(k.get("store_n", 64))
        sn_suffix = "" if store_n == 64 else f"_sn{store_n}"
        stage_path = HERE / f"fused_matmul_{mode_name}{sn_suffix}_s{k['tma_store_stages']}.cu"
        stage_path.write_text(src)
        if (k.get("extra_store_mode") == "swiglu-out-fast"
                and int(k["tma_store_stages"]) == 2 and store_n == 64):
            (HERE / "fused_matmul_swiglu.cu").write_text(src)
        elif (k.get("extra_store_mode") == "swiglu-out-fast-dual-b"
              and int(k["tma_store_stages"]) == 2 and store_n == 64):
            (HERE / "fused_matmul_swiglu_dual_b.cu").write_text(src)
        elif (k.get("extra_store_mode") == "swiglu-half"
              and int(k["tma_store_stages"]) == 1 and store_n == 64):
            (HERE / "fused_matmul_swiglu.cu").write_text(src)
    _src, rc, stderr = gd._compile_worker((str(src_path), arch))
    if rc != 0:
        raise RuntimeError(
            f"nvcc failed for stages={k['tma_store_stages']} "
            f"extra={k.get('extra_store_mode', 'base')}:\n{stderr}"
        )
    return src_path


def make_shapes(shape_list: list[tuple[int, int, int]], bn: int) -> list[dict]:
    shapes = gd.make_shapes(shape_list)
    for sh in shapes:
        m, n, _k = sh["M"], sh["N"], sh["K"]
        if n % bn != 0:
            raise ValueError(f"N={n} must be divisible by BN={bn}")
        packed_b = sh["B"].view(_k, n // bn, bn)
        sh["B_left"] = packed_b[:, :, : bn // 2].reshape(_k, n // 2).contiguous()
        sh["B_gate"] = packed_b[:, :, bn // 2:].reshape(_k, n // 2).contiguous()
        sh["D_half"] = torch.zeros(m, n // 2, dtype=torch.bfloat16, device="cuda")
        sh["D_half_ref"] = (
            sh["C_ref"].view(m, n // bn, bn)[:, :, : bn // 2]
            .reshape(m, n // 2)
            .contiguous()
        )
        sh["D_full"] = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
        sh["D_full_ref"] = sh["C_ref"]
        raw = sh["C_ref"].float().view(m, n // bn, bn)
        left = raw[:, :, : bn // 2]
        gate = raw[:, :, bn // 2:]
        sh["D_half_sum_ref"] = (left + gate).reshape(m, n // 2).to(torch.bfloat16).contiguous()
        sig = torch.sigmoid(gate)
        silu = gate * sig
        silu_prime = sig + silu * (1.0 - sig)
        factors = torch.empty_like(raw)
        factors[:, :, : bn // 2] = silu
        factors[:, :, bn // 2:] = left * silu_prime
        sh["C_swiglu_ref"] = factors.reshape(m, n).to(torch.bfloat16).contiguous()
        sh["D_swiglu_ref"] = (left * silu).reshape(m, n // 2).to(torch.bfloat16).contiguous()
    return shapes


def launch_variant(tier, k, arch, shapes, *, do_bench: bool, num_sms: int) -> dict:
    gd.load_cuda_runtime()
    src_path = str(gd.SCRATCH / gd.tag_for(tier, k) / "kernel.cu")
    cubin_path = src_path[:-3] + f"_{arch}.cubin"
    res = {"tier": tier["dir"], "two_cta": int(tier["cluster"]), **k,
           "launched": False, "correct": False, "error": None, "perf": {}}
    mod = None
    try:
        with open(cubin_path, "rb") as f:
            cubin = f.read()
        mod = gd.rt.cu(driver.cuModuleLoadData(cubin))
        kernel = gd.rt.cu(driver.cuModuleGetFunction(mod, tier["symbol"].encode()))
        overall = True
        for sh in shapes:
            m, n, kval = sh["M"], sh["N"], sh["K"]
            if not gd.shape_compatible(tier, k, m, n, kval):
                continue
            store_n = int(k.get("store_n", 64))
            grid, block, shared = gd.launch_spec(tier, k, m, n, kval, num_sms)
            gd.rt.cu(driver.cuFuncSetAttribute(
                kernel,
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared,
            ))
            a_tmap = gd.rt.encode_tensor_map(
                dtype=gd.rt.TMA_BFLOAT16, rank=2, gptr=sh["A"].data_ptr(),
                global_dim=[kval, m], global_strides=[kval * 2],
                box_dim=[k["bk"], k["bm"]], element_strides=[1, 1],
                swizzle=gd.rt.TMA_SWIZZLE_128B,
            )
            mode = k.get("extra_store_mode", "base")
            dual_b = mode == "swiglu-out-fast-dual-b"
            if dual_b:
                b_left_tmap = gd.rt.encode_tensor_map(
                    dtype=gd.rt.TMA_BFLOAT16, rank=2, gptr=sh["B_left"].data_ptr(),
                    global_dim=[n // 2, kval], global_strides=[(n // 2) * 2],
                    box_dim=[64, k["bk"]], element_strides=[1, 1],
                    swizzle=gd.rt.TMA_SWIZZLE_128B,
                )
                b_gate_tmap = gd.rt.encode_tensor_map(
                    dtype=gd.rt.TMA_BFLOAT16, rank=2, gptr=sh["B_gate"].data_ptr(),
                    global_dim=[n // 2, kval], global_strides=[(n // 2) * 2],
                    box_dim=[64, k["bk"]], element_strides=[1, 1],
                    swizzle=gd.rt.TMA_SWIZZLE_128B,
                )
            else:
                b_tmap = gd.rt.encode_tensor_map(
                    dtype=gd.rt.TMA_BFLOAT16, rank=2, gptr=sh["B"].data_ptr(),
                    global_dim=[n, kval], global_strides=[n * 2],
                    box_dim=[64, k["bk"]], element_strides=[1, 1],
                    swizzle=gd.rt.TMA_SWIZZLE_128B,
                )
            c_tmap = gd.rt.encode_tensor_map(
                dtype=gd.rt.TMA_BFLOAT16, rank=2, gptr=sh["C"].data_ptr(),
                global_dim=[n, m], global_strides=[n * 2],
                box_dim=[store_n, k["bm"]], element_strides=[1, 1],
                swizzle=gd.rt.TMA_SWIZZLE_128B,
            )
            args = [(ctypes.c_byte * 128).from_buffer_copy(a_tmap.tobytes())]
            if dual_b:
                args += [
                    (ctypes.c_byte * 128).from_buffer_copy(b_left_tmap.tobytes()),
                    (ctypes.c_byte * 128).from_buffer_copy(b_gate_tmap.tobytes()),
                ]
            else:
                args.append((ctypes.c_byte * 128).from_buffer_copy(b_tmap.tobytes()))
            args.append((ctypes.c_byte * 128).from_buffer_copy(c_tmap.tobytes()))
            if mode != "base":
                if mode in (
                    "half-only", "extra-half", "swiglu-half", "swiglu-out",
                    "swiglu-out-fast", "swiglu-out-fast-dual-b",
                ):
                    d_tensor = sh["D_half"]
                    d_width = n // 2
                elif mode == "extra-full":
                    d_tensor = sh["D_full"]
                    d_width = n
                else:
                    raise ValueError(f"unknown extra store mode: {mode}")
                d_tmap = gd.rt.encode_tensor_map(
                    dtype=gd.rt.TMA_BFLOAT16, rank=2, gptr=d_tensor.data_ptr(),
                    global_dim=[d_width, m], global_strides=[d_width * 2],
                    box_dim=[store_n, k["bm"]], element_strides=[1, 1],
                    swizzle=gd.rt.TMA_SWIZZLE_128B,
                )
                args.append((ctypes.c_byte * 128).from_buffer_copy(d_tmap.tobytes()))
            args += [ctypes.c_void_p(sh["C"].data_ptr()),
                     ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(kval)]
            sh["C"].zero_()
            sh["D_half"].zero_()
            sh["D_full"].zero_()
            gd.rt.launch(kernel, grid=grid, block=block, shared=shared, args=args)
            res["launched"] = True
            if mode == "half-only":
                c_rel = None
                correct = True
            else:
                c_ref = sh["C_swiglu_ref"] if mode == "swiglu-half" else sh["C_ref"]
                c_rel = ((sh["C"].float() - c_ref.float()).abs().max().item()
                         / c_ref.float().abs().max().item())
                correct = c_rel < 5e-2
            d_rel = None
            if mode != "base":
                if mode in ("swiglu-half", "swiglu-out", "swiglu-out-fast", "swiglu-out-fast-dual-b"):
                    d_tensor = sh["D_half"]
                    d_ref = sh["D_swiglu_ref"]
                elif mode == "half-only":
                    d_tensor = sh["D_half"]
                    d_ref = sh["D_half_sum_ref"]
                elif mode == "extra-half":
                    d_tensor = sh["D_half"]
                    d_ref = sh["D_half_ref"]
                else:
                    d_tensor = sh["D_full"]
                    d_ref = sh["D_full_ref"]
                d_rel = ((d_tensor.float() - d_ref.float()).abs().max().item()
                         / d_ref.float().abs().max().item())
                correct &= d_rel < 5e-2
            overall &= correct
            entry = {"rel_err": c_rel, "d_rel_err": d_rel, "correct": correct,
                     "us": None, "tflops": None}
            if do_bench and correct:
                us = gd.rt.time_kernel_us(
                    lambda: gd.rt.launch(kernel, grid=grid, block=block,
                                         shared=shared, args=args, sync=False),
                    warmup_ms=gd.BENCH_WARMUP_MS,
                    rep_ms=gd.BENCH_REP_MS,
                )
                entry["us"] = us
                entry["tflops"] = (2.0 * m * n * kval) / (us * 1e-6) / 1e12
            res["perf"][mc.shape_key(m, n, kval)] = entry
        res["correct"] = overall
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        try:
            driver.cuCtxSynchronize()
        except Exception:
            pass
    finally:
        if mod is not None:
            try:
                gd.rt.cu(driver.cuModuleUnload(mod))
            except Exception:
                pass
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", action="append", default=None,
                    help="shape to benchmark; may be repeated or semicolon-separated")
    ap.add_argument("--stages", default="1,2", help="comma-separated TMA store stage counts")
    ap.add_argument("--store-ns", default="64", help="comma-separated TMA store chunk widths: 64,128")
    ap.add_argument("--variants", default="base,extra-half,extra-full",
                    help=("comma-separated variants: half-only,base,extra-half,extra-full,"
                          "swiglu-half,swiglu-out,swiglu-out-fast,"
                          "swiglu-out-fast-dual-b"))
    ap.add_argument("--warmup-ms", type=int, default=1000)
    ap.add_argument("--rep-ms", type=int, default=1000)
    ap.add_argument("--cublas-warmup-samples", type=int, default=1)
    ap.add_argument("--cublas-samples", type=int, default=10)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.warmup_ms <= 0 or args.rep_ms <= 0:
        ap.error("--warmup-ms and --rep-ms must be positive")
    if args.cublas_warmup_samples < 0 or args.cublas_samples <= 0:
        ap.error("--cublas-warmup-samples must be non-negative and --cublas-samples must be positive")

    shape_list = parse_shapes(args.shape or ["32768x4608x768"])
    stages_list = parse_int_csv(args.stages)
    store_n_list = parse_int_csv(args.store_ns)
    bad_store_n = sorted(set(store_n_list) - {64, 128})
    if bad_store_n:
        ap.error(f"unsupported --store-ns values: {bad_store_n}")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    allowed = {
        "half-only", "base", "extra-half", "extra-full", "swiglu-half", "swiglu-out",
        "swiglu-out-fast", "swiglu-out-fast-dual-b",
    }
    unknown = sorted(set(variants) - allowed)
    if unknown:
        ap.error(f"unknown variants: {unknown}")

    gd.SCRATCH = SCRATCH
    install_driver_hooks()
    gd.BENCH_WARMUP_MS = args.warmup_ms
    gd.BENCH_REP_MS = args.rep_ms
    gd.CBLAS_WARMUP_SAMPLES = args.cublas_warmup_samples
    gd.CBLAS_MEASURE_SAMPLES = args.cublas_samples

    tier = mc.tier_for(True, True)
    if tier is None:
        raise RuntimeError("missing tier3 cluster tier")

    gd.load_cuda_runtime()
    device, _ctx = gd.rt.init_cuda()
    arch = gd.rt.compute_arch(device)
    num_sms = gd.rt.cu(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device))
    max_smem = gd.rt.cu(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device))

    shape_msg = ", ".join(f"{m}x{n}x{k}" for m, n, k in shape_list)
    print(f"# shapes={shape_msg} arch={arch} sms={num_sms}", flush=True)
    print(f"# config={CONFIG}", flush=True)
    print(f"# store_ns={store_n_list}", flush=True)
    print(f"# do_bench warmup={args.warmup_ms}ms rep={args.rep_ms}ms", flush=True)
    print(f"# cuBLAS samples: warmup={args.cublas_warmup_samples} measured={args.cublas_samples} median", flush=True)
    print(f"# max opt-in dynamic shared memory per block: {max_smem} B", flush=True)

    shapes = make_shapes(shape_list, CONFIG["bn"])
    cublas = {}
    cublas_samples = {}
    for sh in shapes:
        m, n, kval = sh["M"], sh["N"], sh["K"]
        key = mc.shape_key(m, n, kval)
        tf, samples = gd.measure_cublas_tflops(sh["A"], sh["B"], m, n, kval)
        cublas[key] = tf
        cublas_samples[key] = samples
        print("# cuBLAS "
              f"{key}: {tf:.1f} TFLOPS "
              f"(samples {', '.join(f'{x:.1f}' for x in samples)})",
              flush=True)

    rows = []
    for variant in variants:
        for store_n in store_n_list:
            for stages in stages_list:
                k = dict(CONFIG)
                k["store_n"] = store_n
                k["tma_store_stages"] = stages
                k["extra_store_mode"] = variant
                shared = shared_bytes(k, stages)
                print(f"\n# variant={variant} store_n={store_n} stages={stages} "
                      f"shared={shared} B ({shared / 1024:.1f} KiB)", flush=True)
                if shared > max_smem:
                    print(f"SKIP variant={variant} store_n={store_n} stages={stages}: "
                          "shared memory exceeds opt-in limit", flush=True)
                    rows.append({
                        "variant": variant,
                        "store_n": store_n,
                        "stages": stages,
                        "shared_bytes": shared,
                        "skipped": True,
                    })
                    continue
                src_path = render_compile(tier, k, arch)
                start = time.time()
                res = launch_variant(tier, k, arch, shapes, do_bench=True, num_sms=num_sms)
                elapsed = time.time() - start
                ok = bool(res.get("correct"))
                err = res.get("error")
                shape_results = {}
                for sh in shapes:
                    m, n, kval = sh["M"], sh["N"], sh["K"]
                    key = mc.shape_key(m, n, kval)
                    perf = (res.get("perf") or {}).get(key, {})
                    tf = perf.get("tflops")
                    rel = perf.get("rel_err")
                    d_rel = perf.get("d_rel_err")
                    us = perf.get("us")
                    ratio = (tf / cublas[key]) if (tf and cublas.get(key)) else None
                    shape_results[key] = {
                        "tflops": tf,
                        "vs_cublas": ratio,
                        "us": us,
                        "rel_err": rel,
                        "d_rel_err": d_rel,
                        "correct": perf.get("correct"),
                    }
                    d_msg = f" d_rel_err={d_rel:.6g}" if d_rel is not None else ""
                    print(
                        f"variant={variant} store_n={store_n} stages={stages} shape={key} "
                        f"correct={perf.get('correct')} "
                        f"tflops={(f'{tf:.1f}' if tf else 'n/a')} "
                        f"vs_cublas={(f'{ratio:.1%}' if ratio else 'n/a')} "
                        f"us={(f'{us:.3f}' if us else 'n/a')} "
                        f"rel_err={(f'{rel:.6g}' if rel else 'n/a')}{d_msg} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                if err:
                    print(f"error: {err}", flush=True)
                rows.append({
                    "variant": variant,
                    "store_n": store_n,
                    "stages": stages,
                    "shared_bytes": shared,
                    "source": str(src_path.relative_to(ROOT)),
                    "correct": ok,
                    "perf": shape_results,
                    "error": err,
                })
    out = {
        "shapes": [list(s) for s in shape_list],
        "config": CONFIG,
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "cublas_tflops": cublas,
        "cublas_samples": cublas_samples,
        "results": rows,
    }
    out_path = pathlib.Path(args.json) if args.json else SCRATCH / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
