#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_bf16.h>

/*
 * Hopper warp-specialized dual-B SwiGLU kernels.
 *
 * Fixed config: BM128 / internal BN256 / BK64 / WG2 / NS4 / GM8.  Each output
 * tile covers 128 H columns.  The internal BN256 accumulator is laid out as
 * [left 128 | gate 128].  One entry writes only D = left * silu(gate); the
 * store-preact entry also writes packed C = [all-left | all-gate].
 */

#ifndef LB_MIN_BLOCKS
#define LB_MIN_BLOCKS 1
#endif

// ── wgmma helpers ─────────────────────────────────────────────────────────────
__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
// wgmma drain helpers kept for reference; pipeline uses WAIT_MMA(n) macro instead
__device__ __forceinline__ void wgmma_wait_0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// -- TMA-store helpers --------------------------------------------------------

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
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory");
}


__device__ __forceinline__ void tma_2d_load(
    uint32_t smem_dst, const void* tmap, int x, int y, uint32_t mbar
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar) : "memory");
}

__device__ __forceinline__ void mbarrier_init(uint32_t mb, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mb), "r"(count));
}


__device__ __forceinline__ void mbarrier_arrive_no_tx(uint32_t mb) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(mb) : "memory");
}

__device__ __forceinline__ void math_barrier() {
    asm volatile("bar.sync 1, 256;" ::: "memory");
}

__device__ __forceinline__ void signal_on_bytes_loaded(uint32_t mb, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(mb), "r"(bytes) : "memory");
}

__device__ __forceinline__ void wait_phase(uint32_t mb, uint32_t phase) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE_%=;\n\t bra WAIT_%=;\n\t DONE_%=:\n\t }"
        :: "r"(mb), "r"(phase) : "memory");
}

__device__ __forceinline__ float swiglu_value(float left, float gate) {
    const float x = -gate * 1.4426950408889634f;
    float e;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(x));
    const float denom = 1.0f + e;
    float sig;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(sig) : "f"(denom));
    return left * (gate * sig);
}

// ── GmmaDescriptors (unchanged from h2_s6) ───────────────────────────────────

template<int BK>
__device__ __forceinline__ uint64_t make_wgmma_a_desc(uint32_t smem_addr, int kk) {
    constexpr uint64_t layout = (BK == 64) ? 1ULL : (BK == 32) ? 2ULL : 3ULL;
    constexpr uint64_t sbo = (uint64_t)BK;
    uint64_t start = ((uint64_t)(smem_addr >> 4) & 0x3FFFULL) + (uint64_t)(kk * 2);
    return start | (sbo << 32) | (layout << 62);
}

template<int BN, int BK>
__device__ __forceinline__ uint64_t make_wgmma_b_desc(uint32_t smem_addr) {
    constexpr uint64_t LAYOUT_B128 = 1ULL << 62;
    constexpr int n_atoms = BN / 64;
    constexpr uint64_t lbo = (n_atoms <= 1) ? 0ULL : (uint64_t)(8 * BK);
    constexpr uint64_t sbo = 64ULL;
    uint64_t start = (uint64_t)(smem_addr >> 4) & 0x3FFFULL;
    return start | (lbo << 16) | (sbo << 32) | LAYOUT_B128;
}

// ── wgmma SS wrappers (identical to h2_s6) ────────────────────────────────────

__device__ __forceinline__
void wgmma_ss_m64n64k16(float d[32], uint64_t a, uint64_t b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "%32,%33,1,1,1,0,1;\n"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
         "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
         "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
         "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
        :"l"(a),"l"(b));
}

__device__ __forceinline__
void wgmma_ss_m64n128k16(float d[64], uint64_t a, uint64_t b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        "%64,%65,1,1,1,0,1;\n"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
         "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
         "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
         "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),
         "+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),
         "+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),
         "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),
         "+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
        :"l"(a),"l"(b));
}

__device__ __forceinline__
void wgmma_ss_m64n256k16(float d[128], uint64_t a, uint64_t b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,"
        "%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,"
        "%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,"
        "%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,"
        "%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127},"
        "%128,%129,1,1,1,0,1;\n"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
         "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
         "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
         "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),
         "+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),
         "+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),
         "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),
         "+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]),
         "+f"(d[64]),"+f"(d[65]),"+f"(d[66]),"+f"(d[67]),"+f"(d[68]),"+f"(d[69]),"+f"(d[70]),"+f"(d[71]),
         "+f"(d[72]),"+f"(d[73]),"+f"(d[74]),"+f"(d[75]),"+f"(d[76]),"+f"(d[77]),"+f"(d[78]),"+f"(d[79]),
         "+f"(d[80]),"+f"(d[81]),"+f"(d[82]),"+f"(d[83]),"+f"(d[84]),"+f"(d[85]),"+f"(d[86]),"+f"(d[87]),
         "+f"(d[88]),"+f"(d[89]),"+f"(d[90]),"+f"(d[91]),"+f"(d[92]),"+f"(d[93]),"+f"(d[94]),"+f"(d[95]),
         "+f"(d[96]),"+f"(d[97]),"+f"(d[98]),"+f"(d[99]),"+f"(d[100]),"+f"(d[101]),"+f"(d[102]),"+f"(d[103]),
         "+f"(d[104]),"+f"(d[105]),"+f"(d[106]),"+f"(d[107]),"+f"(d[108]),"+f"(d[109]),"+f"(d[110]),"+f"(d[111]),
         "+f"(d[112]),"+f"(d[113]),"+f"(d[114]),"+f"(d[115]),"+f"(d[116]),"+f"(d[117]),"+f"(d[118]),"+f"(d[119]),
         "+f"(d[120]),"+f"(d[121]),"+f"(d[122]),"+f"(d[123]),"+f"(d[124]),"+f"(d[125]),"+f"(d[126]),"+f"(d[127])
        :"l"(a),"l"(b));
}

template<int BN>
__device__ __forceinline__
void wgmma_ss_call(float* acc, uint64_t desc_a, uint64_t desc_b) {
    if constexpr      (BN ==  64) wgmma_ss_m64n64k16 (acc, desc_a, desc_b);
    else if constexpr (BN == 128) wgmma_ss_m64n128k16(acc, desc_a, desc_b);
    else if constexpr (BN == 256) wgmma_ss_m64n256k16(acc, desc_a, desc_b);
}

// -- Warp-specialized Hopper kernel: producer TMA load + math WGMMA -----------

template<int GROUP_M>
__device__ __forceinline__ void swiglu_tile_coords_static(
    int tile_id, int num_pid_m, int num_pid_n, int& pid_m, int& pid_n
) {
    if constexpr (GROUP_M <= 1) {
        pid_m = tile_id / num_pid_n;
        pid_n = tile_id - pid_m * num_pid_n;
    } else {
        const int tiles_per_group = GROUP_M * num_pid_n;
        const int group_id = tile_id / tiles_per_group;
        const int first_m = group_id * GROUP_M;
        const int rem_m = num_pid_m - first_m;
        const int group_m = rem_m < GROUP_M ? rem_m : GROUP_M;
        const int idx = tile_id - group_id * tiles_per_group;
        pid_m = first_m + (idx % group_m);
        pid_n = idx / group_m;
    }
}

__device__ __forceinline__ void swiglu_tile_coords_runtime(
    int tile_id, int num_pid_m, int num_pid_n, int group_m_arg, int& pid_m, int& pid_n
) {
    if (group_m_arg <= 1) {
        pid_m = tile_id / num_pid_n;
        pid_n = tile_id - pid_m * num_pid_n;
    } else {
        const int tiles_per_group = group_m_arg * num_pid_n;
        const int group_id = tile_id / tiles_per_group;
        const int first_m = group_id * group_m_arg;
        const int rem_m = num_pid_m - first_m;
        const int group_m = rem_m < group_m_arg ? rem_m : group_m_arg;
        const int idx = tile_id - group_id * tiles_per_group;
        pid_m = first_m + (idx % group_m);
        pid_n = idx / group_m;
    }
}

template<int BM, int BN, int BK, int NUM_WG, int NUM_STAGES, int GROUP_M, bool STORE_PREACT>
__device__ __forceinline__ void hopper_swiglu_dual_b_impl(
    const CUtensorMap* A_tmap,
    const CUtensorMap* B_left_tmap,
    const CUtensorMap* B_gate_tmap,
    const CUtensorMap* C_tmap,
    const CUtensorMap* D_tmap,
    int M, int K, int H, int runtime_group_m = GROUP_M
) {
    static_assert(BM == 128 && BN == 256 && BK == 64, "first cut is fixed to BM128 BN256 BK64");
    static_assert(NUM_WG == 2 && NUM_STAGES == 4, "first cut is fixed to WG2 NS4");
    static_assert(BM % (NUM_WG * 64) == 0, "BM must be multiple of NUM_WG*64");
    static_assert(BN % 64 == 0, "BN must be multiple of 64");

    constexpr int NS         = NUM_STAGES;
    constexpr int M_ITERS    = BM / (NUM_WG * 64);
    constexpr int M_PER_WG   = BM / NUM_WG;
    constexpr int D          = BN / 2;
    constexpr int N_SUBTILES = BN / 64;
    constexpr int OUT_N      = BN / 2;

    constexpr int MATH_THREADS     = NUM_WG * 128;
    constexpr int PRODUCER_TID     = MATH_THREADS;
    constexpr int BF16_BYTES       = 2;
    constexpr int STORE_N          = 64;
    constexpr int OUT_CHUNKS       = OUT_N / STORE_N;
    constexpr int TMA_STORE_STAGES = 2;
    constexpr int STORE_BUF_BYTES  = BM * STORE_N * BF16_BYTES;
    constexpr int A_BYTES          = NS * BM * BK * BF16_BYTES;
    constexpr int B_BYTES          = NS * BK * BN * BF16_BYTES;
    constexpr int A_TILE_BYTES     = BM * BK * BF16_BYTES;
    constexpr int B_TILE_BYTES     = BK * BN * BF16_BYTES;
    constexpr int SLOT_BYTES       = A_TILE_BYTES + B_TILE_BYTES;
    constexpr int STORE_SMEM_OFF   = A_BYTES + B_BYTES;
    constexpr int K_STEP_BYTES     = 16 * 64 * BF16_BYTES;

    static_assert(MATH_THREADS == 256, "math barrier assumes 256 math threads");
    static_assert(STORE_BUF_BYTES == 16 * 1024, "expected 16 KiB store buffer");
    static_assert(STORE_SMEM_OFF + TMA_STORE_STAGES * STORE_BUF_BYTES == 224 * 1024,
                  "expected 224 KiB total dynamic SMEM");

    const int tid        = threadIdx.x;
    const bool is_math   = tid < MATH_THREADS;
    const int wg_id      = tid / 128;
    const int local_warp = (tid % 128) / 32;
    const int lane       = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    auto A_sh = reinterpret_cast<__nv_bfloat16 (*)[BM][BK]>(smem_raw);
    auto B_sh = reinterpret_cast<__nv_bfloat16 (*)[N_SUBTILES][BK][64]>(smem_raw + A_BYTES);
    auto C_store = reinterpret_cast<__nv_bfloat16*>(smem_raw + STORE_SMEM_OFF);

    __shared__ uint64_t mbar_tma_ready[NS];
    __shared__ uint64_t mbar_slot_free[NS];
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < NS; s++) {
            mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_tma_ready[s]), 1);
            mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_slot_free[s]), 1);
            mbarrier_arrive_no_tx((uint32_t)__cvta_generic_to_shared(&mbar_slot_free[s]));
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    const int num_k_tiles = K / BK;
    const int num_pid_m = (M + BM - 1) / BM;
    const int num_pid_n = H / OUT_N;
    const int total_tiles = num_pid_m * num_pid_n;

    if (!is_math) {
        if (tid == PRODUCER_TID) {
            uint32_t slot_free_phase[NS] = {};
            for (int tile_id = (int)blockIdx.x; tile_id < total_tiles; tile_id += (int)gridDim.x) {
                int pid_m, pid_n;
                if constexpr (GROUP_M > 0) {
                    swiglu_tile_coords_static<GROUP_M>(tile_id, num_pid_m, num_pid_n, pid_m, pid_n);
                } else {
                    swiglu_tile_coords_runtime(tile_id, num_pid_m, num_pid_n, runtime_group_m, pid_m, pid_n);
                }
                const int block_row = pid_m * BM;
                const int block_col = pid_n * OUT_N;

                for (int ktile = 0; ktile < num_k_tiles; ktile++) {
                    const int slot = ktile % NS;
                    const uint32_t free_mb = (uint32_t)__cvta_generic_to_shared(&mbar_slot_free[slot]);
                    const uint32_t ready_mb = (uint32_t)__cvta_generic_to_shared(&mbar_tma_ready[slot]);
                    const uint32_t abase = (uint32_t)__cvta_generic_to_shared(&A_sh[slot][0][0]);
                    const uint32_t bbase = (uint32_t)__cvta_generic_to_shared(&B_sh[slot][0][0][0]);

                    wait_phase(free_mb, slot_free_phase[slot]);                                        
                    tma_2d_load(abase, A_tmap, ktile * BK, block_row, ready_mb);                    
                    #pragma unroll
                    for (int nst = 0; nst < N_SUBTILES; nst++) {
                        const CUtensorMap* half_tmap =
                            (nst < N_SUBTILES / 2) ? B_left_tmap : B_gate_tmap;
                        const int half_col = block_col + (nst % (N_SUBTILES / 2)) * STORE_N;
                        tma_2d_load(bbase + nst * BK * STORE_N * BF16_BYTES,
                                    half_tmap, half_col, ktile * BK, ready_mb);
                    }
                    signal_on_bytes_loaded(ready_mb, SLOT_BYTES);
                    slot_free_phase[slot] ^= 1;
                }
            }
        }
        return;
    }

    float acc[M_ITERS][D];
    uint32_t tma_ready_phase[NS] = {};

#define WAIT_SMEM(slot_)                                                            \
    do {                                                                            \
        const uint32_t _mb = (uint32_t)__cvta_generic_to_shared(                    \
            &mbar_tma_ready[(slot_)]);                                              \
        wait_phase(_mb, tma_ready_phase[(slot_)]);                                  \
        tma_ready_phase[(slot_)] ^= 1;                                              \
    } while (0)

#define SIGNAL_SLOT_FREE(slot_)                                                     \
    do {                                                                            \
        if (tid == 0) {                                                             \
            mbarrier_arrive_no_tx((uint32_t)__cvta_generic_to_shared(               \
                &mbar_slot_free[(slot_)]));                                         \
        }                                                                           \
    } while (0)

#define COMPUTE_TILE(slot_)                                                         \
    do {                                                                            \
        wgmma_fence();                                                              \
        _Pragma("unroll")                                                           \
        for (int _kk = 0; _kk < BK / 16; _kk++) {                                  \
            const uint32_t _bb = (uint32_t)__cvta_generic_to_shared(                \
                                     &B_sh[(slot_)][0][0][0]);                      \
            const uint64_t _db = make_wgmma_b_desc<BN, BK>(                         \
                                     (uint32_t)(_bb + _kk * K_STEP_BYTES));         \
            _Pragma("unroll")                                                       \
            for (int _m = 0; _m < M_ITERS; _m++) {                                  \
                const int _mrow = wg_id * M_PER_WG + _m * 64;                       \
                const uint32_t _aa = (uint32_t)__cvta_generic_to_shared(            \
                                         &A_sh[(slot_)][_mrow][0]);                 \
                const uint64_t _da = make_wgmma_a_desc<BK>(_aa, _kk);               \
                wgmma_ss_call<BN>((float*)acc[_m], _da, _db);                       \
            }                                                                       \
        }                                                                           \
        wgmma_commit();                                                             \
    } while (0)

#define WAIT_MMA(n_) \
    asm volatile("wgmma.wait_group.sync.aligned " #n_ ";\n" ::: "memory")

    for (int tile_id = (int)blockIdx.x; tile_id < total_tiles; tile_id += (int)gridDim.x) {
        int pid_m, pid_n;
        if constexpr (GROUP_M > 0) {
            swiglu_tile_coords_static<GROUP_M>(tile_id, num_pid_m, num_pid_n, pid_m, pid_n);
        } else {
            swiglu_tile_coords_runtime(tile_id, num_pid_m, num_pid_n, runtime_group_m, pid_m, pid_n);
        }

        const int block_row = pid_m * BM;
        const int block_col = pid_n * OUT_N;

        #pragma unroll
        for (int _m = 0; _m < M_ITERS; _m++) {
            #pragma unroll
            for (int _d = 0; _d < D; _d++) {
                acc[_m][_d] = 0.0f;
            }
        }

        for (int k = 0; k < num_k_tiles; k++) {
            const int slot = k % NS;
            WAIT_SMEM(slot);
            math_barrier();
            COMPUTE_TILE(slot);
            WAIT_MMA(1);
            if (k >= 1) {
                SIGNAL_SLOT_FREE((k - 1) % NS);
            }
        }

        WAIT_MMA(0);
        math_barrier();
        SIGNAL_SLOT_FREE((num_k_tiles - 1) % NS);

        constexpr int CHUNKS = OUT_CHUNKS;
        constexpr int J_PER_CHUNK = STORE_N / 8;
        const int base_col = (lane % 4) * 2;
        const int base_row = lane / 4;

        int store_stage = 0;

#define WAIT_STORE_BUFFER()                                                        \
    do {                                                                            \
        if (tid == 0) {                                                             \
            tma_wait_group<TMA_STORE_STAGES - 1>();                                 \
        }                                                                           \
        math_barrier();                                                             \
    } while (0)

#define ISSUE_STORE(tmap_, col_)                                                    \
    do {                                                                            \
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");              \
        math_barrier();                                                             \
        if (tid == 0) {                                                             \
            const uint32_t src = (uint32_t)__cvta_generic_to_shared(                \
                C_store + store_stage * STORE_BUF_BYTES / BF16_BYTES);              \
            tma_2d_store((tmap_), src, (col_), block_row);                          \
            tma_commit_group();                                                     \
        }                                                                           \
        store_stage ^= 1;                                                           \
    } while (0)

        #pragma unroll
        for (int chunk = 0; chunk < CHUNKS; chunk++) {
            if constexpr (STORE_PREACT) {
                WAIT_STORE_BUFFER();
                #pragma unroll
                for (int m = 0; m < M_ITERS; m++) {
                    const int row0 = wg_id * M_PER_WG + m * 64 + local_warp * 16 + base_row;
                    const int row8 = row0 + 8;
                    #pragma unroll
                    for (int jj = 0; jj < J_PER_CHUNK; jj++) {
                        const int j = chunk * J_PER_CHUNK + jj;
                        const int local_col = jj * 8 + base_col;
                        const int swz0 = ((local_col / 8) ^ (row0 & 7)) * 8 + (local_col & 7);
                        const int swz8 = ((local_col / 8) ^ (row8 & 7)) * 8 + (local_col & 7);
                        __nv_bfloat16* ptr0 = C_store + store_stage * STORE_BUF_BYTES / BF16_BYTES
                                            + row0 * STORE_N + swz0;
                        __nv_bfloat16* ptr8 = C_store + store_stage * STORE_BUF_BYTES / BF16_BYTES
                                            + row8 * STORE_N + swz8;
                        *reinterpret_cast<__nv_bfloat162*>(ptr0) =
                            __floats2bfloat162_rn(acc[m][j*4+0], acc[m][j*4+1]);
                        *reinterpret_cast<__nv_bfloat162*>(ptr8) =
                            __floats2bfloat162_rn(acc[m][j*4+2], acc[m][j*4+3]);
                    }
                }
                ISSUE_STORE(C_tmap, block_col + chunk * STORE_N);

                WAIT_STORE_BUFFER();
                #pragma unroll
                for (int m = 0; m < M_ITERS; m++) {
                    const int row0 = wg_id * M_PER_WG + m * 64 + local_warp * 16 + base_row;
                    const int row8 = row0 + 8;
                    #pragma unroll
                    for (int jj = 0; jj < J_PER_CHUNK; jj++) {
                        const int j = (chunk + OUT_CHUNKS) * J_PER_CHUNK + jj;
                        const int local_col = jj * 8 + base_col;
                        const int swz0 = ((local_col / 8) ^ (row0 & 7)) * 8 + (local_col & 7);
                        const int swz8 = ((local_col / 8) ^ (row8 & 7)) * 8 + (local_col & 7);
                        __nv_bfloat16* ptr0 = C_store + store_stage * STORE_BUF_BYTES / BF16_BYTES
                                            + row0 * STORE_N + swz0;
                        __nv_bfloat16* ptr8 = C_store + store_stage * STORE_BUF_BYTES / BF16_BYTES
                                            + row8 * STORE_N + swz8;
                        *reinterpret_cast<__nv_bfloat162*>(ptr0) =
                            __floats2bfloat162_rn(acc[m][j*4+0], acc[m][j*4+1]);
                        *reinterpret_cast<__nv_bfloat162*>(ptr8) =
                            __floats2bfloat162_rn(acc[m][j*4+2], acc[m][j*4+3]);
                    }
                }
                ISSUE_STORE(C_tmap, H + block_col + chunk * STORE_N);
            }

            WAIT_STORE_BUFFER();
            #pragma unroll
            for (int m = 0; m < M_ITERS; m++) {
                const int row0 = wg_id * M_PER_WG + m * 64 + local_warp * 16 + base_row;
                const int row8 = row0 + 8;
                #pragma unroll
                for (int jj = 0; jj < J_PER_CHUNK; jj++) {
                    const int left_j = chunk * J_PER_CHUNK + jj;
                    const int gate_j = (chunk + OUT_CHUNKS) * J_PER_CHUNK + jj;
                    const int local_col = jj * 8 + base_col;
                    const int swz0 = ((local_col / 8) ^ (row0 & 7)) * 8 + (local_col & 7);
                    const int swz8 = ((local_col / 8) ^ (row8 & 7)) * 8 + (local_col & 7);
                    __nv_bfloat16* ptr0 = C_store + store_stage * STORE_BUF_BYTES / BF16_BYTES
                                        + row0 * STORE_N + swz0;
                    __nv_bfloat16* ptr8 = C_store + store_stage * STORE_BUF_BYTES / BF16_BYTES
                                        + row8 * STORE_N + swz8;
                    *reinterpret_cast<__nv_bfloat162*>(ptr0) = __floats2bfloat162_rn(
                        swiglu_value(acc[m][left_j*4+0], acc[m][gate_j*4+0]),
                        swiglu_value(acc[m][left_j*4+1], acc[m][gate_j*4+1]));
                    *reinterpret_cast<__nv_bfloat162*>(ptr8) = __floats2bfloat162_rn(
                        swiglu_value(acc[m][left_j*4+2], acc[m][gate_j*4+2]),
                        swiglu_value(acc[m][left_j*4+3], acc[m][gate_j*4+3]));
                }
            }
            ISSUE_STORE(D_tmap, block_col + chunk * STORE_N);
        }

#undef WAIT_STORE_BUFFER
#undef ISSUE_STORE
    }

    if (tid == 0) {
        tma_wait_group<0>();
    }

#undef WAIT_SMEM
#undef SIGNAL_SLOT_FREE
#undef COMPUTE_TILE
#undef WAIT_MMA
}

// -- Kernel entry points -------------------------------------------------------

#define MAKE_LAUNCHER(GM_)                                                        \
extern "C" __global__ __launch_bounds__(9 * 32, LB_MIN_BLOCKS)                   \
void matmul_hopper_swiglu_dual_b_bm128_bn256_bk64_wg2_ns4_gm##GM_(              \
    const __grid_constant__ CUtensorMap A_tmap,                                  \
    const __grid_constant__ CUtensorMap B_left_tmap,                             \
    const __grid_constant__ CUtensorMap B_gate_tmap,                             \
    const __grid_constant__ CUtensorMap D_tmap,                                  \
    int M, int K, int H)                                                         \
{                                                                                \
    hopper_swiglu_dual_b_impl<128, 256, 64, 2, 4, GM_, false>(                   \
        &A_tmap, &B_left_tmap, &B_gate_tmap, &D_tmap, &D_tmap, M, K, H);         \
}                                                                                \
extern "C" __global__ __launch_bounds__(9 * 32, LB_MIN_BLOCKS)                   \
void matmul_hopper_swiglu_dual_b_store_preact_bm128_bn256_bk64_wg2_ns4_gm##GM_( \
    const __grid_constant__ CUtensorMap A_tmap,                                  \
    const __grid_constant__ CUtensorMap B_left_tmap,                             \
    const __grid_constant__ CUtensorMap B_gate_tmap,                             \
    const __grid_constant__ CUtensorMap C_tmap,                                  \
    const __grid_constant__ CUtensorMap D_tmap,                                  \
    int M, int K, int H)                                                         \
{                                                                                \
    hopper_swiglu_dual_b_impl<128, 256, 64, 2, 4, GM_, true>(                    \
        &A_tmap, &B_left_tmap, &B_gate_tmap, &C_tmap, &D_tmap, M, K, H);         \
}

MAKE_LAUNCHER(8)

extern "C" __global__ __launch_bounds__(9 * 32, LB_MIN_BLOCKS)
void matmul_hopper_swiglu_dual_b_bm128_bn256_bk64_wg2_ns4_runtime_gm(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_left_tmap,
    const __grid_constant__ CUtensorMap B_gate_tmap,
    const __grid_constant__ CUtensorMap D_tmap,
    int M, int K, int H, int group_m)
{
    hopper_swiglu_dual_b_impl<128, 256, 64, 2, 4, 0, false>(
        &A_tmap, &B_left_tmap, &B_gate_tmap, &D_tmap, &D_tmap, M, K, H, group_m);
}

#undef MAKE_LAUNCHER
