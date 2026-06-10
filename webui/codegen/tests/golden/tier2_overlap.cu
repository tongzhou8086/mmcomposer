#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

// ── User-tunable constants (the webui substitutes these six) ────────
constexpr int BM           = 128;
constexpr int BN           = 256;
constexpr int BK           = 64;
constexpr int NS           = 5;       // multi-stage SMEM ring depth
constexpr int GROUP_SIZE_M = 4;       // CTA-swizzle chunk (1 = no swizzle)
constexpr int NUM_WARPS    = 8;       // total warps per CTA
constexpr int TMA_STORE    = 0;       // epilogue Phase 2: 0 = int4 stores, 1 = async TMA store
constexpr int TCGEN05_LD_WIDTH = 8;  // TMEM->reg epilogue load width: 8 or 16 (32-bit elems per lane)
constexpr int EPILOGUE_OVERLAP = 1;  // 1 = persistent + overlap epilogue with next tile's K-loop (Step B)
constexpr int EPILOGUE_SPLIT   = 0;  // 1 = split overlapped int4 writeback into two half-BN passes (cluster only today)

// ── Derived constants (do not edit) ─────────────────────────────────
constexpr int MMA_K   = 16;
constexpr int K_MMAS  = BK / MMA_K;          // 4

constexpr int BF16_BYTES        = 2;        // byte size of the operand element
constexpr int SWIZZLE_ROW_BYTES = 128;      // one 128B-swizzle atom row

// ── SWIZZLE_128B constraint on K-major B ────────────────────────────
//
// TMA with SWIZZLE_128B requires the box's INNERMOST dimension to be
// exactly SWIZZLE_ROW_BYTES (= 128 B) — one swizzle atom.  For native
// (K, N) row-major B, N is the contiguous dim, so the inner box covers
// N and must be capped at SWIZZLE_ROW_BYTES / BF16_BYTES = 64 BF16.
// Loading the full BN requires BN / 64 sub-tile TMA calls, each
// writing into a different SMEM offset.  (Ch05 dodged this by
// host-transposing B so K — also 64 BF16 — was the inner dim.)
//
// The literal 64 below is therefore not a free parameter; it's
// `SWIZZLE_ROW_BYTES / BF16_BYTES`.

constexpr int A_SLOT_BYTES = BM * BK * BF16_BYTES;     // 16 KB
constexpr int B_SLOT_BYTES = BN * BK * BF16_BYTES;     // 32 KB
constexpr int SLOT_BYTES   = A_SLOT_BYTES + B_SLOT_BYTES;          // 48 KB / slot

constexpr int WARP_SIZE = 32;
constexpr int THREADS   = NUM_WARPS * WARP_SIZE;


// ── helpers (identical to ch05) ─────────────────────────────────────
__device__ __forceinline__ bool elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t .reg .pred px;\n\t"
        "elect.sync _|px, %1;\n\t"
        "@px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred) : "r"(0xFFFFFFFF));
    return pred;
}

__device__ __forceinline__ void tma_2d_load(
    uint32_t smem_dst, const void* tmap, int x, int y, uint32_t mbar
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar) : "memory");
}

// A's descriptor — unchanged from earlier chapters (MN-major).
__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    uint64_t b = ((SBO)              >> 4) & 0x3FFFULL;
    return a | (b << 32) | (1ULL << 46) | (2ULL << 61);
}

// NEW: K-major B descriptor.  The only additional field versus make_desc
// is the LBO (leading byte offset) at bits [16, 29] — telling the tensor
// cores how far apart consecutive N-sub-tiles sit in SMEM.
__device__ __forceinline__ uint64_t make_desc_K_major(
    uint32_t smem_addr, int lbo_bytes
) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a   = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    uint64_t lbo = ((uint64_t)lbo_bytes >> 4) & 0x3FFFULL;
    uint64_t b   = ((SBO)               >> 4) & 0x3FFFULL;
    return a | (lbo << 16) | (b << 32) | (1ULL << 46) | (2ULL << 61);
}

// idesc with bit 16 = 1: B is K-major.
__device__ __forceinline__ uint32_t make_idesc_bf16_kmajor_b(int m, int n) {
    uint32_t d = 0;
    d |= (1u << 4);                                    // c_format = F32
    d |= (1u << 7);                                    // a_format = BF16
    d |= (1u << 10);                                   // b_format = BF16
    d |= (1u << 16);                                   // B is K-major  ← new
    d |= (((uint32_t)(n >> 3) & 0x3F) << 17);
    d |= (((uint32_t)(m >> 4) & 0x1F) << 24);
    return d;
}

__device__ __forceinline__ void tcgen05_alloc(uint32_t smem_dst, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_dst), "r"(n_cols) : "memory");
}
__device__ __forceinline__ void tcgen05_dealloc(uint32_t taddr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(taddr), "r"(n_cols) : "memory");
}
__device__ __forceinline__ void tcgen05_mma(
    uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
    uint32_t idesc, bool enable_d
) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "setp.ne.b32 P, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, P;\n\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
           "r"((uint32_t)enable_d) : "memory");
}
__device__ __forceinline__ void tcgen05_commit(uint32_t smem_bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(smem_bar) : "memory");
}
__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}
__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
// ── tcgen05.ld width helpers (building block) ───────────────────────
// mvp_core splices these at the TCGEN05_LD marker in every tier, so the
// TMEM->register load width (TCGEN05_LD_WIDTH = 8/16 32-bit elems per lane)
// is one knob with the asm in a single place.  Wider = fewer ld + fewer
// wait_ld syncs (more registers, but we're SMEM-occupancy-bound so it's free).
// The epilogue picks the variant via `#if` (resolved at generation time).

__device__ __forceinline__ void tcgen05_ld_32x32b_x8(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        :
          "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
          "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7])
        : "r"(taddr));
}

__device__ __forceinline__ void tcgen05_ld_32x32b_x16(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        :
          "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
          "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7]),
          "=f"(out[8]), "=f"(out[9]), "=f"(out[10]), "=f"(out[11]),
          "=f"(out[12]), "=f"(out[13]), "=f"(out[14]), "=f"(out[15])
        : "r"(taddr));
}

__device__ __forceinline__ void mbarrier_init(uint32_t mb, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mb), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive_no_tx(uint32_t mb) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(mb) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint32_t mb, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(mb), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbarrier_wait_phase(uint32_t mb, uint32_t phase) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE_%=;\n\t bra WAIT_%=;\n\t DONE_%=:\n\t }"
        :: "r"(mb), "r"(phase) : "memory");
}


// ── Kernel ──────────────────────────────────────────────────────────
// ── TMA store helpers (epilogue Phase 2 when TMA_STORE=1) ───────────
__device__ __forceinline__ void tma_2d_store(
    const void* tmap, uint32_t smem_src, int x, int y
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
        "[%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_src) : "memory");
}
__device__ __forceinline__ void tma_commit_group() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void tma_wait_group() {
    asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory");
}

// MMA-issue building block (shared fragment).  MMA_ISSUE picks the
// single-CTA g1 instruction; the cluster tier supplies the g2 variant.
#define MMA_ISSUE(t, a, b, i, e) tcgen05_mma((t), (a), (b), (i), (e))
// ── MMA-issue chain (building block) ────────────────────────────────
// Issues the K_MMAS tcgen05 MMAs for one K-tile (slot) into the
// accumulator at `taddr`.  mvp_core stitches this into every tier at the
// MMA-chain marker, so the descriptor math + K-step loop live in exactly
// one place.  The only per-tier variation is the MMA instruction
// itself, supplied just before the marker as:
//   MMA_ISSUE(taddr, a_desc, b_desc, idesc, enable_d)
// → tcgen05_mma (single-CTA) or tcgen05_mma_g2 (2-CTA cluster).
__device__ __forceinline__ void issue_mma_chain(
    uint32_t taddr, uint32_t a_base_slot, uint32_t b_base_slot,
    uint32_t idesc, bool first_k_tile)
{
    #pragma unroll
    for (int kk = 0; kk < K_MMAS; kk++) {
        const uint64_t a_desc = make_desc(a_base_slot + kk * MMA_K * BF16_BYTES);
        const uint64_t b_desc = make_desc_K_major(
            b_base_slot + kk * MMA_K * SWIZZLE_ROW_BYTES, BK * SWIZZLE_ROW_BYTES);
        const bool first_ever = first_k_tile && (kk == 0);
        MMA_ISSUE(taddr, a_desc, b_desc, idesc, !first_ever);
    }
}
#undef MMA_ISSUE

extern "C" __global__ void matmul_coalesced_epilogue(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ CUtensorMap C_tmap,
    __nv_bfloat16* __restrict__ C_ptr,
    int M, int N, int K
) {
    extern __shared__ __align__(1024) char smem[];
    const uint32_t SMEM_BASE = (uint32_t)__cvta_generic_to_shared(smem);
    auto A_base = [SMEM_BASE](int s) -> uint32_t {
        return SMEM_BASE + s * SLOT_BYTES;
    };
    // B_base points at sub-tile 0 of slot s.  Sub-tile at N-offset n
    // (n in {0, 64, 128, 192}) sits at B_base(s) + n * BK * BF16_BYTES.
    auto B_base = [SMEM_BASE](int s) -> uint32_t {
        return SMEM_BASE + s * SLOT_BYTES + A_SLOT_BYTES;
    };

    __shared__ uint64_t tile_ready[NS];
    __shared__ uint64_t mma_done[NS];
    __shared__ uint64_t all_mmas_done;
    __shared__ uint32_t tmem_addr_holder[1];

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane    = tid % WARP_SIZE;

    {
        // ── Step B: persistent grid + epilogue/K-loop OVERLAP ───────────
        // Continuous TMA-producer (warp0) + MMA-consumer (warp1) stream into a
        // 2-buffer TMEM accumulator (taddr + buf*BN); 4 dedicated epilogue
        // warps (4..7) drain the *other* buffer concurrently — handed off via
        // tmem_full / tmem_empty.  Epilogue staging is DISJOINT from the ring
        // (at NS*SLOT).  Requires the persistent launch grid and NW>=8.
        __shared__ uint64_t tmem_full[2];
        __shared__ uint64_t tmem_empty[2];
        constexpr int EPI_STAGE_COLS = EPILOGUE_SPLIT ? (BN / 2) : BN;
        auto C_sh = reinterpret_cast<__nv_bfloat16(*)[EPI_STAGE_COLS + 8]>(smem + NS * SLOT_BYTES);

        if (warp_id == 0)
            tcgen05_alloc((uint32_t)__cvta_generic_to_shared(tmem_addr_holder), 2 * BN);
        if (warp_id == 0 && elect_sync()) {
            #pragma unroll
            for (int s = 0; s < NS; s++) {
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&tile_ready[s]), 1);
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mma_done[s]), 1);
                mbarrier_arrive_no_tx((uint32_t)__cvta_generic_to_shared(&mma_done[s]));
            }
            #pragma unroll
            for (int b = 0; b < 2; b++) {
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&tmem_full[b]), 1);
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&tmem_empty[b]), 1);
                mbarrier_arrive_no_tx((uint32_t)__cvta_generic_to_shared(&tmem_empty[b]));
            }
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        __syncthreads();
        const uint32_t taddr = tmem_addr_holder[0];
        const uint32_t idesc = make_idesc_bf16_kmajor_b(BM, BN);
        const int num_k = K / BK;
        const int grid_m = M / BM, grid_n = N / BN;
        const int num_block_in_group = GROUP_SIZE_M * grid_n;
        const int num_tiles = grid_m * grid_n;
        const int num_my = ((int)blockIdx.x >= num_tiles) ? 0
                         : (num_tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;

        // GSM-swizzled tile-index (this CTA's ti-th tile) -> (off_m, off_n).
        auto map_off = [&](int ti, int& off_m, int& off_n) {
            int tile = (int)blockIdx.x + ti * (int)gridDim.x;
            int group_id = tile / num_block_in_group;
            int first = group_id * GROUP_SIZE_M;
            int gsm = min(grid_m - first, GROUP_SIZE_M);
            off_m = (first + (tile % gsm)) * BM;
            off_n = ((tile % num_block_in_group) / gsm) * BN;
        };

        if (warp_id == 0 && elect_sync()) {                 // TMA producer (continuous)
            uint32_t ph[NS] = {}; long gk = 0;
            for (int ti = 0; ti < num_my; ti++) {
                int off_m, off_n; map_off(ti, off_m, off_n);
                for (int k = 0; k < num_k; k++) {
                    int slot = gk % NS;
                    uint32_t mb = (uint32_t)__cvta_generic_to_shared(&tile_ready[slot]);
                    mbarrier_wait_phase((uint32_t)__cvta_generic_to_shared(&mma_done[slot]), ph[slot]);
                    tma_2d_load(A_base(slot), &A_tmap, k * BK, off_m, mb);
                    #pragma unroll
                    for (int n = 0; n < BN; n += 64)
                        tma_2d_load(B_base(slot) + n * BK * BF16_BYTES, &B_tmap, off_n + n, k * BK, mb);
                    mbarrier_arrive_expect_tx(mb, SLOT_BYTES);
                    ph[slot] ^= 1; gk++;
                }
            }
        } else if (warp_id == 1 && elect_sync()) {          // MMA consumer -> TMEM double-buffer
            uint32_t ph[NS] = {}, emp[2] = {}; long gk = 0;
            for (int ti = 0; ti < num_my; ti++) {
                int buf = ti & 1; uint32_t d_tmem = taddr + buf * BN;
                mbarrier_wait_phase((uint32_t)__cvta_generic_to_shared(&tmem_empty[buf]), emp[buf]); emp[buf] ^= 1;
                for (int k = 0; k < num_k; k++) {
                    int slot = gk % NS;
                    mbarrier_wait_phase((uint32_t)__cvta_generic_to_shared(&tile_ready[slot]), ph[slot]);
                    tcgen05_fence_after_thread_sync();
                    issue_mma_chain(d_tmem, A_base(slot), B_base(slot), idesc, /*first_k_tile=*/ k == 0);
                    tcgen05_commit((uint32_t)__cvta_generic_to_shared(&mma_done[slot]));
                    ph[slot] ^= 1; gk++;
                }
                tcgen05_commit((uint32_t)__cvta_generic_to_shared(&tmem_full[buf]));
            }
        } else if (warp_id >= 4 && warp_id < NUM_WARPS + 4) {   // NUM_WARPS epilogue warps (drain prev buffer)
            // Epilogue runs in its OWN warpgroup(s): the 2 stream warps (TMA=0,
            // MMA=1) sit in warpgroup 0, so the epilogue starts at warp 4 (warps
            // 2,3 idle).  Sharing warpgroup 0 between the MMA warp and the
            // tcgen05.ld epilogue warps corrupts the TMEM reads.  Block =
            // (NUM_WARPS + 4) warps; NUM_WARPS still scales the epilogue.
            // Contract for the shared overlap-drain fragment: single-CTA tier
            // writes off_m / off_n and releases TMEM with a plain mbarrier arrive.
#define EPI_OUT_ROW                 off_m
#define EPI_OUT_COL_BASE            off_n
#define EPI_TMEM_EMPTY_ARRIVE(buf)  mbarrier_arrive_no_tx((uint32_t)__cvta_generic_to_shared(&tmem_empty[buf]))
            constexpr int ROW_STRIPS    = BM / 32;                 // 4
            constexpr int COL_GROUPS    = NUM_WARPS / ROW_STRIPS;  // warps per row-strip
            constexpr int COLS_PER_WARP = BN / COL_GROUPS;
            constexpr int EPI_THREADS   = NUM_WARPS * 32;
            const int ew = warp_id - 4;                            // 0..NUM_WARPS-1
            const int row_warp = ew % ROW_STRIPS, col_warp = ew / ROW_STRIPS;
            const int my_row = row_warp * 32 + lane, col_base = col_warp * COLS_PER_WARP;
            const int etid = ew * 32 + lane;
            uint32_t full[2] = {};
            for (int ti = 0; ti < num_my; ti++) {
                int buf = ti & 1, off_m, off_n; map_off(ti, off_m, off_n);
                mbarrier_wait_phase((uint32_t)__cvta_generic_to_shared(&tmem_full[buf]), full[buf]); full[buf] ^= 1;
                tcgen05_fence_after_thread_sync();
                uint32_t trow = (taddr + buf * BN) + ((uint32_t)(row_warp * 32) << 16);
                constexpr int LDW = TCGEN05_LD_WIDTH;

                // ── Overlap epilogue drain (TMEM → SMEM → GMEM), shared ─────
                // Spliced into the overlap epilogue-warp loop of every warp-spec
                // tier's drain marker, right after `trow` (the tier-specific TMEM
                // lane base) and `LDW` are in scope.  The skeleton supplies three
                // contract macros for the per-tier bits:
                //   EPI_OUT_ROW                  this CTA's GMEM row base
                //   EPI_OUT_COL_BASE             this CTA's GMEM column base
                //   EPI_TMEM_EMPTY_ARRIVE(buf)   release the drained TMEM buffer
                // EPILOGUE_SPLIT (constexpr) picks the two-pass half-BN writeback,
                // which stages one BN/2 column panel at a time (EPI_STAGE_COLS=BN/2)
                // so the epilogue SMEM shrinks enough for an extra K-loop stage.
                if constexpr (EPILOGUE_SPLIT) {
                    static_assert((BN / 2) % 8 == 0, "split epilogue needs int4-aligned columns");
                    static_assert((BN / 2) % COL_GROUPS == 0,
                                  "split epilogue panel must divide across column warp groups");
                    constexpr int SPLIT_COLS_PER_WARP = (BN / 2) / COL_GROUPS;
                    static_assert(SPLIT_COLS_PER_WARP % LDW == 0,
                                  "split epilogue per-warp column span must divide by LDW");
                    #pragma unroll
                    for (int split = 0; split < 2; split++) {
                        const int split_base = split * EPI_STAGE_COLS;
                        const int panel_col_base = split_base + col_warp * SPLIT_COLS_PER_WARP;
                        #pragma unroll
                        for (int n = panel_col_base; n < panel_col_base + SPLIT_COLS_PER_WARP; n += LDW) {
                            float t[LDW];
                            if constexpr (LDW == 8) tcgen05_ld_32x32b_x8 (trow + (uint32_t)n, t);
                            else                    tcgen05_ld_32x32b_x16(trow + (uint32_t)n, t);
                            tcgen05_wait_ld();
                            __nv_bfloat162 pk[LDW / 2];
                            #pragma unroll
                            for (int i = 0; i < LDW / 2; i++)
                                pk[i] = __floats2bfloat162_rn(t[2 * i], t[2 * i + 1]);
                            #pragma unroll
                            for (int c = 0; c < LDW; c += 8)
                                *reinterpret_cast<int4*>(&C_sh[my_row][n - split_base + c]) =
                                    *reinterpret_cast<int4*>(&pk[c / 2]);
                        }
                        asm volatile("bar.sync 1, %0;" :: "n"(NUM_WARPS * 32));

                        // Once split 1 is staged, this tile's TMEM buffer is no
                        // longer needed.  Release before the split-1 GMEM store so
                        // the MMA warp can start the next tile earlier.
                        if (split == 1 && ew == 0 && elect_sync())
                            EPI_TMEM_EMPTY_ARRIVE(buf);

                        constexpr int CHUNKS = (BN / 2) / 8;
                        constexpr int STORES = BM * (BN / 2) / (EPI_THREADS * 8);
                        static_assert(STORES * EPI_THREADS * 8 == BM * (BN / 2),
                                      "split epilogue tile must divide across epilogue threads");
                        #pragma unroll
                        for (int s = 0; s < STORES; s++) {
                            const int flat = etid + s * EPI_THREADS;
                            const int row = flat / CHUNKS;
                            const int col = (flat % CHUNKS) * 8;
                            *reinterpret_cast<int4*>(&C_ptr[(EPI_OUT_ROW + row) * N + EPI_OUT_COL_BASE + split_base + col]) =
                                *reinterpret_cast<const int4*>(&C_sh[row][col]);
                        }
                        asm volatile("bar.sync 1, %0;" :: "n"(NUM_WARPS * 32));
                    }
                } else {
                    #pragma unroll
                    for (int n = col_base; n < col_base + COLS_PER_WARP; n += LDW) {
                        float t[LDW];
                        if constexpr (LDW == 8) tcgen05_ld_32x32b_x8 (trow + (uint32_t)n, t);
                        else                    tcgen05_ld_32x32b_x16(trow + (uint32_t)n, t);
                        tcgen05_wait_ld();
                        __nv_bfloat162 pk[LDW / 2];
                        #pragma unroll
                        for (int i = 0; i < LDW / 2; i++)
                            pk[i] = __floats2bfloat162_rn(t[2 * i], t[2 * i + 1]);
                        #pragma unroll
                        for (int c = 0; c < LDW; c += 8)
                            *reinterpret_cast<int4*>(&C_sh[my_row][n + c]) =
                                *reinterpret_cast<int4*>(&pk[c / 2]);
                    }
                    asm volatile("bar.sync 1, %0;" :: "n"(NUM_WARPS * 32));
                    if (ew == 0 && elect_sync())
                        EPI_TMEM_EMPTY_ARRIVE(buf);
                    constexpr int CHUNKS = BN / 8;
                    constexpr int STORES = BM * BN / (EPI_THREADS * 8);
                    #pragma unroll
                    for (int s = 0; s < STORES; s++) {
                        int flat = etid + s * EPI_THREADS;
                        int row = flat / CHUNKS;
                        int col = (flat % CHUNKS) * 8;
                        *reinterpret_cast<int4*>(&C_ptr[(EPI_OUT_ROW + row) * N + EPI_OUT_COL_BASE + col]) =
                            *reinterpret_cast<const int4*>(&C_sh[row][col]);
                    }
                    asm volatile("bar.sync 1, %0;" :: "n"(NUM_WARPS * 32));
                }
            }
#undef EPI_OUT_ROW
#undef EPI_OUT_COL_BASE
#undef EPI_TMEM_EMPTY_ARRIVE
        }
        __syncthreads();
        if (warp_id == 0 && elect_sync()) tcgen05_dealloc(taddr, 2 * BN);
        return;
    }
}
