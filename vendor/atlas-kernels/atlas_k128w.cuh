// Packaging notice (2026-10-15): modified deployment snapshot; see vendor/MODIFICATIONS.md.
// Original copyright/license notices below remain applicable.
// SPDX-License-Identifier: AGPL-3.0-only
// Derived from Enntity/atlas 72588541a466f53cb9d38afd74c515e14ab601db; see NOTICE and build_port.py.
#include "qad_helpers.cuh"
// Per-expert M64-tile prefix for the K128W grid below: prefix[e] = number of
// row tiles of the local experts before e (remote experts, NULL weights, have
// none); prefix[num_experts] = the total. One block; num_experts <= 1024.
extern "C" __global__ void __launch_bounds__(1024) moe_mtile_prefix(
    const int* __restrict__ expert_offsets,
    const unsigned long long* __restrict__ B_packed_ptrs,
    int* __restrict__ prefix,
    unsigned int num_experts
) {
    __shared__ int s[1024];
    const unsigned int t = threadIdx.x;
    int v = 0;
    if (t < num_experts && B_packed_ptrs[t] != 0) {
        const int rows = expert_offsets[t + 1] - expert_offsets[t];
        v = rows > 0 ? (rows + M_TILE - 1) / M_TILE : 0;
    }
    s[t] = v;
    __syncthreads();
    for (unsigned int d = 1; d < blockDim.x; d <<= 1) {
        const int add = t >= d ? s[t - d] : 0;
        __syncthreads();
        s[t] += add;
        __syncthreads();
    }
    if (t < num_experts) prefix[t + 1] = s[t];
    if (t == 0) prefix[0] = 0;
}

// ── Prequant native-FP4 grouped GEMM, 64 x 256 tiles (prefill) ───────────
// The MMA sequence of moe_w4a4_grouped_gemm_prequant_t_k128 per output
// element (same m16n8k64 operands and scales, k64 slices in K order), so the
// outputs match it bit for bit, with 40% of its instructions (ncu: 153M vs
// 375M on a 4K-chunk gate; at ~97% of GB10's measured read+write DRAM
// bandwidth in moe_fp4_prefill_bench):
//   * 8 warps of 32 rows x 64 columns: each A/B fragment feeds two MMAs;
//   * the [K/2, N] weight tile needs no transpose pass: ldmatrix.m16n16.trans
//     .b8 reads a 16 kp x 16 n byte block straight into the b0/b1 fragments
//     of two n8 subtiles, and the same load over the [K/16, N] scale rows
//     leaves slice s's four scales in quad thread s (the MMA's B scale
//     thread selector), so shared memory holds only the raw tiles;
//   * weight and scale rows are XOR-swizzled in 16-byte chunks (no padding);
//   * CTAs cover only the local experts' row tiles: grid (N/256, bound, 1)
//     with bound >= prefix[num_experts] (moe_mtile_prefix), each CTA y a
//     global row tile, so the dense grid's empty expert CTAs never launch.
// Two stages at two CTAs/SM measured faster than 3-4 stages at one CTA/SM
// and than 64 x 128 tiles with 3 stages. Requires K % 128 == 0, N % 256 == 0.
#define PQ2_ARGS \
    const unsigned char* __restrict__ A_packed, \
    const unsigned char* __restrict__ A_scale, \
    const unsigned long long* __restrict__ B_packed_ptrs, \
    const unsigned long long* __restrict__ B_scale_ptrs, \
    const float* __restrict__ scale2_vals, \
    __nv_bfloat16* __restrict__ C, \
    const int* __restrict__ expert_offsets, \
    const int* __restrict__ sorted_token_ids, \
    unsigned int num_experts, unsigned int N, unsigned int K

__device__ __forceinline__ void pqw_ldsm_t2(unsigned int (&d)[4], unsigned int addr) {
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(addr));
}

__device__ __forceinline__ void pqw_ldsm_t1(unsigned int (&d)[2], unsigned int addr) {
    asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0,%1}, [%2];"
                 : "=r"(d[0]), "=r"(d[1]) : "r"(addr));
}

// The up projection and NVFP4 outputs of the fused gate/up variant.
struct PqwGateUp {
    const unsigned long long* __restrict__ packed_ptrs;
    const unsigned long long* __restrict__ scale_ptrs;
    const float* __restrict__ scale2_vals;
    unsigned char* __restrict__ out_packed;   // [rows, N/2] E2M1
    unsigned char* __restrict__ out_scale;    // [rows, N/16] E4M3
    const float* __restrict__ down_global_scale;
};

// One K128W tile is 64 rows x 256 B columns (GATE_UP: 128 gate columns and
// the same 128 up columns, with silu_mul_quant_nvfp4 applied in registers),
// computed by 256 threads in 8 warps of 32 rows x 64 columns. The pieces
// below are shared by the one-tile-per-CTA grid (pqw_impl) and the
// persistent grid (pqp_impl).
#define PQW_NT 256
#define PQW_NB (PQW_NT / 4 / 16)   // 16-column ldmatrix blocks per warp
// [kp][n] and [group][n] shared tiles: 16-byte chunk c of row r at c ^ (r & 7).
typedef unsigned char PqwA[M_TILE][PQ2_AP];
typedef unsigned char PqwAs[M_TILE][PQ2_KS / GROUP_SIZE];
typedef unsigned char PqwB[PQW_NT][PQ2_KP];
typedef unsigned char PqwS[PQW_NT][PQ2_KS / GROUP_SIZE];
typedef float PqwAcc[2][2 * PQW_NB][4];

// The local expert owning global row tile `tile` of mtile_prefix.
__device__ __forceinline__ unsigned int pqw_tile_expert(
    int tile, const int* __restrict__ mtile_prefix, unsigned int num_experts
) {
    unsigned int lo = 0, hi = num_experts;
    while (hi - lo > 1) {
        const unsigned int mid = (lo + hi) >> 1;
        if (mtile_prefix[mid] <= tile) lo = mid; else hi = mid;
    }
    return lo;
}

// Threads t < M_TILE: the source row of the tile's row t (its sorted token,
// or the row itself).
__device__ __forceinline__ void pqw_gather_rows(
    int* sTok, unsigned int t, const int* __restrict__ sorted_token_ids,
    unsigned int cta_m, int cta_m_local, int M_expert
) {
    if (t < M_TILE) {
        const bool live = (cta_m_local + (int)t) < M_expert;
        sTok[t] = (sorted_token_ids && live) ? sorted_token_ids[cta_m + t] : (int)(cta_m + t);
    }
}

__device__ __forceinline__ void pqw_zero_acc(PqwAcc& acc) {
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int i = 0; i < 2 * PQW_NB; i++) acc[mi][i][0] = acc[mi][i][1] = acc[mi][i][2] = acc[mi][i][3] = 0.0f;
}

// Tile 16-byte column chunk c: whether it reads the up matrix, and its column.
template<bool GATE_UP>
__device__ __forceinline__ bool pqw_chunk_up(unsigned int c) { return GATE_UP && c >= PQW_NT / 32; }
template<bool GATE_UP>
__device__ __forceinline__ unsigned int pqw_chunk_col(unsigned int cta_n, unsigned int c) {
    return cta_n + ((GATE_UP ? c % (PQW_NT / 32) : c) << 4);
}

// Thread t's cp.async loads of K stage kb into one set of shared tiles.
template<bool GATE_UP>
__device__ __forceinline__ void pqw_issue(
    unsigned int t, PqwA& sA, PqwAs& sAs, PqwB& sB, PqwS& sS, const int* sTok,
    const unsigned char* __restrict__ A_packed, const unsigned char* __restrict__ A_scale,
    const unsigned char* B_expert, const unsigned char* S_expert,
    const unsigned char* U_expert, const unsigned char* US_expert,
    int cta_m_local, unsigned int M_eff, unsigned int cta_n, unsigned int N, unsigned int K, unsigned int kb, unsigned int expert_id
) {
    {
        const unsigned int row = t >> 2, col = (t & 3) << 4;
        const bool valid = (cta_m_local + row) < M_eff;
        const unsigned int a_row = (unsigned int)sTok[row];
        moe_cp_async_pred_16(&sA[row][col],
            &A_packed[(unsigned long long)a_row * (K / 2) + kb / 2 + col], valid);
    }
    if (t < M_TILE) {
        const bool valid = (cta_m_local + t) < M_eff;
        const unsigned int a_row = (unsigned int)sTok[t];
        moe_cp_async_pred_8(&sAs[t][0],
            &A_scale[(unsigned long long)a_row * (K / GROUP_SIZE) + kb / GROUP_SIZE], valid);
    }
    // QAD keeps the existing tile-major bytes used by decode. Each 128x64
    // box is contiguous; copy four 16-byte pieces per weight row directly.
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const unsigned int j=t+r*256, row=j/4, col16=(j%4)*16;
        const bool is_up=GATE_UP && row>=128;
        const unsigned int nc=cta_n+(GATE_UP ? row%128 : row);
        const unsigned long long tile=GATE_UP ? ((nc/128*2+is_up)*(K/128)+kb/128)
            : (kb/128*(N/128)+nc/128);
        const auto* ptr=is_up ? U_expert : B_expert;
        moe_cp_async_pred_16(&sB[row][col16^(((row>>1)&3)*16)],ptr+tile*8192+(nc%128)*64+col16,true);
    }
    // Two losslessly decoded four-byte CSF words per logical weight row.
    const bool is_up=GATE_UP && t>=128;
    const unsigned int nc=cta_n+(GATE_UP ? t%128 : t);
    const unsigned int logical_row=GATE_UP ? nc+(is_up ? 0 : N) : nc;
    const unsigned int all_rows=GATE_UP ? 2*N : N;
    const auto* scales=is_up ? US_expert : S_expert;
    *(unsigned int*)&sS[t][0]=csf_word(scales,expert_id,all_rows,K/16,logical_row,kb/16);
    *(unsigned int*)&sS[t][4]=csf_word(scales,expert_id,all_rows,K/16,logical_row,kb/16+4);

}

// Warp warp_id's MMAs of one K128 stage held in shared memory.
template<bool GATE_UP>
__device__ __forceinline__ void pqw_mma_stage(
    PqwAcc& acc, const PqwA& sA, const PqwAs& sAs, const PqwB& sB, const PqwS& sS,
    unsigned int warp_id, unsigned int lane_id
) {
    constexpr int NB = PQW_NB;
    const unsigned int warp_m_offset = (warp_id & 1) * 32;
    const unsigned int warp_n_offset = (warp_id >> 1) * (PQW_NT / 4);
    // This warp's nb-th chunk (GATE_UP: gate chunks 2wn, 2wn+1, then up's).
    auto warp_chunk = [&](int nb) -> unsigned int {
        return GATE_UP ? (nb >> 1) * (PQW_NT / 32) + (warp_id >> 1) * 2 + (nb & 1)
                       : (warp_n_offset >> 4) + nb;
    };
    // Per-lane ldmatrix row: B (x2) rows kp = lane (matrix 0: k bytes 0-15 of
    // a k64 slice, matrix 1: bytes 16-31); scales (x1) rows g = lane & 7.
    const unsigned int kp_l = lane_id, g_l = lane_id & 7;
    const unsigned int sfa_m = (lane_id & 1) * 8 + (lane_id >> 2);
    const unsigned short tid_a = 0, bid_a = 0, bid_b = 0;
    #pragma unroll
    for (int sl = 0; sl < 2; ++sl) {
        const unsigned int ko = sl * 32;
        const unsigned short tid_b = 0;
        unsigned int a[2][4], sfa[2];
        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const unsigned int j = lane_id >> 3, r = lane_id & 7;
            const unsigned int addr = __cvta_generic_to_shared(
                &sA[warp_m_offset + mi * 16 + r + (j & 1) * 8][ko + (j >> 1) * 16]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a[mi][0]), "=r"(a[mi][1]), "=r"(a[mi][2]), "=r"(a[mi][3]) : "r"(addr));
            sfa[mi] = *(const unsigned int*)&sAs[warp_m_offset + mi * 16 + sfa_m][sl * 4];
        }
        #pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
            const unsigned int c = warp_chunk(nb);
            const unsigned int nc=c*16+(lane_id>>2);
            const unsigned int kc=ko+(lane_id&3)*4;
            unsigned int b[4];
            b[0]=*(const unsigned int*)&sB[nc][kc^(((nc>>1)&3)*16)];
            b[1]=*(const unsigned int*)&sB[nc+8][kc^((((nc+8)>>1)&3)*16)];
            b[2]=*(const unsigned int*)&sB[nc][(kc+16)^(((nc>>1)&3)*16)];
            b[3]=*(const unsigned int*)&sB[nc+8][(kc+16)^((((nc+8)>>1)&3)*16)];
            const unsigned int sf0=*(const unsigned int*)&sS[nc][sl*4];
            const unsigned int sf1=*(const unsigned int*)&sS[nc+8][sl*4];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int nt = nb * 2 + h;
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    asm volatile(
                        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
                        "{%14},{%15,%16},{%17},{%18,%19};"
                        :"=f"(acc[mi][nt][0]),"=f"(acc[mi][nt][1]),"=f"(acc[mi][nt][2]),"=f"(acc[mi][nt][3])
                        :"r"(a[mi][0]),"r"(a[mi][1]),"r"(a[mi][2]),"r"(a[mi][3]),"r"(b[h]),"r"(b[2 + h]),
                         "f"(acc[mi][nt][0]),"f"(acc[mi][nt][1]),"f"(acc[mi][nt][2]),"f"(acc[mi][nt][3]),
                         "r"(sfa[mi]),"h"(bid_a),"h"(tid_a),"r"(h ? sf1 : sf0),"h"(bid_b),"h"(tid_b));
                }
            }
        }
    }
}

// Warp warp_id's stores of a finished tile: BF16 C, or (GATE_UP) the packed
// NVFP4 SiLU·mul outputs.
template<bool GATE_UP>
__device__ __forceinline__ void pqw_epilogue(
    const PqwAcc& acc, unsigned int warp_id, unsigned int lane_id,
    unsigned int expert_id, float scale2, unsigned int cta_m, int cta_m_local, int M_expert,
    unsigned int cta_n, unsigned int N, __nv_bfloat16* __restrict__ C, const PqwGateUp& up
) {
    constexpr int NB = PQW_NB;
    const unsigned int warp_m_offset = (warp_id & 1) * 32;
    const unsigned int warp_n_offset = (warp_id >> 1) * (PQW_NT / 4);
    const unsigned int group_id = lane_id >> 2, tid = lane_id & 3;
    if constexpr (GATE_UP) {
        // Gate subtile j pairs with up subtile j + 4 (same columns); each
        // 16-column group spans two subtiles, i.e. one quad of threads.
        const float scale2_up = up.scale2_vals[expert_id];
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            const unsigned int rl = warp_m_offset + mi * 16 + group_id;
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                const unsigned int row = cta_m + rl + half * 8;
                const bool valid = (int)(rl + half * 8 + cta_m_local) < M_expert;
                #pragma unroll
                for (int q = 0; q < 2; q++) {
                    float v[4], group_max = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        const int j = q * 2 + (i >> 1), e = half * 2 + (i & 1);
                        v[i] = silu_nvfp4_act(
                            acc[mi][j][e] * scale2,
                            acc[mi][j + 4][e] * scale2_up);
                        group_max = fmaxf(group_max, fabsf(v[i]));
                    }
                    group_max = fmaxf(group_max, __shfl_xor_sync(0xffffffffu, group_max, 1));
                    group_max = fmaxf(group_max, __shfl_xor_sync(0xffffffffu, group_max, 2));
                    float inv;
                    const unsigned char sc = qad_group_scale(group_max, up.down_global_scale[expert_id], &inv);
                    const unsigned int col = cta_n + (warp_id >> 1) * 32 + q * 16;
                    if (!valid) continue;
                    if (tid == 0) up.out_scale[(unsigned long long)row * (N / 16) + col / 16] = sc;
                    #pragma unroll
                    for (int hh = 0; hh < 2; hh++) {
                        const unsigned int q0 = qad_fp4(v[hh * 2] * inv);
                        const unsigned int q1 = qad_fp4(v[hh * 2 + 1] * inv);
                        up.out_packed[(unsigned long long)row * (N / 2) + (col + hh * 8) / 2 + tid] =
                            (unsigned char)((q1 << 4) | q0);
                    }
                }
            }
        }
    } else {
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            const unsigned int rl = warp_m_offset + mi * 16 + group_id;
            const bool r0v = (int)(rl + cta_m_local) < M_expert;
            const bool r1v = (int)(rl + 8 + cta_m_local) < M_expert;
            const unsigned int r0 = cta_m + rl, r1 = r0 + 8;
            #pragma unroll
            for (int nt = 0; nt < 2 * NB; nt++) {
                const unsigned int c0 = cta_n + warp_n_offset + nt * 8 + tid * 2;
                if (r0v)
                    *(__nv_bfloat162*)&C[r0 * N + c0] = __floats2bfloat162_rn(
                        acc[mi][nt][0] * scale2, acc[mi][nt][1] * scale2);
                if (r1v)
                    *(__nv_bfloat162*)&C[r1 * N + c0] = __floats2bfloat162_rn(
                        acc[mi][nt][2] * scale2, acc[mi][nt][3] * scale2);
            }
        }
    }
}

template<bool GATE_UP>
__device__ __forceinline__ void pqw_impl(
    PQ2_ARGS,
    const int* __restrict__ mtile_prefix,
    const PqwGateUp up
) {
    constexpr int STAGES = 2;
    const int tile = blockIdx.y;
    if (tile >= mtile_prefix[num_experts]) return;
    const unsigned int expert_id = pqw_tile_expert(tile, mtile_prefix, num_experts);
    const int m_start = expert_offsets[expert_id];
    const int M_expert = expert_offsets[expert_id + 1] - m_start;
    const int cta_m_local = (tile - mtile_prefix[expert_id]) * M_TILE;
    if (cta_m_local >= M_expert) return;
    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    if (B_expert == 0) return;
    const float scale2 = scale2_vals[expert_id];
    const unsigned char* U_expert = GATE_UP ? (const unsigned char*)up.packed_ptrs[expert_id] : nullptr;
    const unsigned char* US_expert = GATE_UP ? (const unsigned char*)up.scale_ptrs[expert_id] : nullptr;
    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * (GATE_UP ? PQW_NT / 2 : PQW_NT);

    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t / 32, lane_id = t % 32;

    __shared__ __align__(16) PqwA sA[STAGES];
    __shared__ __align__(16) PqwAs sAs[STAGES];
    __shared__ __align__(16) PqwB sB[STAGES];
    __shared__ __align__(16) PqwS sS[STAGES];
    __shared__ int sTok[M_TILE];

    pqw_gather_rows(sTok, t, sorted_token_ids, cta_m, cta_m_local, M_expert);
    __syncthreads();

    const unsigned int M_eff = (unsigned int)M_expert;
    auto issue = [&](int buf, unsigned int kb) {
        pqw_issue<GATE_UP>(t, sA[buf], sAs[buf], sB[buf], sS[buf], sTok, A_packed, A_scale,
            B_expert, S_expert, U_expert, US_expert, cta_m_local, M_eff, cta_n, N, K, kb, expert_id);
    };

    PqwAcc acc;
    pqw_zero_acc(acc);

    const unsigned int stages = K / PQ2_KS;
    #pragma unroll
    for (int s0 = 0; s0 < STAGES - 1; ++s0) {
        if (s0 < stages) issue(s0, s0 * PQ2_KS);
        moe_cp_async_commit();
    }
    for (unsigned int st = 0; st < stages; ++st) {
        const int buf = st % STAGES;
        asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2));
        __syncthreads();   // stage `st` landed; every warp finished stage st-1
        const unsigned int nxt = st + STAGES - 1;
        if (nxt < stages) issue(nxt % STAGES, nxt * PQ2_KS);
        moe_cp_async_commit();
        pqw_mma_stage<GATE_UP>(acc, sA[buf], sAs[buf], sB[buf], sS[buf], warp_id, lane_id);
    }
    pqw_epilogue<GATE_UP>(acc, warp_id, lane_id, expert_id, scale2, cta_m, cta_m_local, M_expert,
        cta_n, N, C, up);
}

extern "C" __global__ void __launch_bounds__(256, 2) moe_w4a4_grouped_gemm_prequant_t_k128w_compact(
    PQ2_ARGS,
    const int* __restrict__ mtile_prefix
) {
    pqw_impl<false>(A_packed, A_scale, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,
        expert_offsets, sorted_token_ids, num_experts, N, K, mtile_prefix, PqwGateUp{});
}

// Gate (B_*) and up projections of the same 128 intermediate columns per CTA,
// grid (N/128, bound, 1), with silu_mul_quant_nvfp4 applied in the epilogue:
// writes its packed E2M1 [rows, N/2] + E4M3 [rows, N/16] bytes for the local
// experts' rows exactly (C unused), skipping both BF16 intermediates.
extern "C" __global__ void __launch_bounds__(256, 2) moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w(
    PQ2_ARGS,
    const int* __restrict__ mtile_prefix,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    unsigned char* __restrict__ out_packed,
    unsigned char* __restrict__ out_scale,
    const float* __restrict__ down_global_scale
) {
    pqw_impl<true>(A_packed, A_scale, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,
        expert_offsets, sorted_token_ids, num_experts, N, K, mtile_prefix,
        PqwGateUp{up_packed_ptrs, up_scale_ptrs, up_scale2_vals, out_packed, out_scale, down_global_scale});
}

// ── Persistent K128W (ATLAS_GLM_MOE_PREFILL_PERSIST) ─────────────────────
// The K128W tiles, loads, MMA sequence and epilogue above, so the output
// bytes are identical; only the scheduling changes. Two CTAs per SM take
// (row tile, N tile) work items w = n + N_tiles * row, in order, from the
// zeroed counter `next_work`, which keeps the resident tiles as close
// together as the K128W grid's in-order dispatch does (the row tiles of one
// expert share its weights in L2; a static stride lets the CTAs drift apart
// and re-read them from DRAM). Nothing launches past the live tiles (the
// K128W grid launches the host's bound, most of it exiting at once), and
// each CTA gathers its next tile's token rows and issues its first K stage
// before the current tile's epilogue. Every tile below
// mtile_prefix[num_experts] holds rows of a local expert. An item whose
// weight table entry is null (the prefix came from another table) loads
// and stores nothing, as the grid kernel's CTA returns, but still passes
// the K loop's barriers with the CTA, its MMAs reading stale shared memory.
template<bool GATE_UP>
__device__ __forceinline__ void pqp_impl(
    PQ2_ARGS,
    const int* __restrict__ mtile_prefix,
    int* __restrict__ next_work,
    const PqwGateUp up
) {
    const unsigned int n_tiles = N / (GATE_UP ? PQW_NT / 2 : PQW_NT);
    const unsigned int work = (unsigned int)mtile_prefix[num_experts] * n_tiles;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t / 32, lane_id = t % 32;

    __shared__ __align__(16) PqwA sA[2];
    __shared__ __align__(16) PqwAs sAs[2];
    __shared__ __align__(16) PqwB sB[2];
    __shared__ __align__(16) PqwS sS[2];
    __shared__ int sTok[M_TILE];
    __shared__ unsigned int sNext;

    // One work item: its expert, rows, column tile and weight tables.
    struct Tile {
        unsigned int expert_id, cta_m, cta_n;
        int M_expert, cta_m_local;
        const unsigned char *B, *S, *U, *US;
    };
    auto decode = [&](unsigned int w) {
        Tile d;
        const int tl = (int)(w / n_tiles);
        d.cta_n = (w % n_tiles) * (GATE_UP ? PQW_NT / 2 : PQW_NT);
        d.expert_id = pqw_tile_expert(tl, mtile_prefix, num_experts);
        const int m_start = expert_offsets[d.expert_id];
        d.M_expert = expert_offsets[d.expert_id + 1] - m_start;
        d.cta_m_local = (tl - mtile_prefix[d.expert_id]) * M_TILE;
        d.cta_m = m_start + d.cta_m_local;
        d.B = (const unsigned char*)B_packed_ptrs[d.expert_id];
        d.S = (const unsigned char*)B_scale_ptrs[d.expert_id];
        d.U = GATE_UP ? (const unsigned char*)up.packed_ptrs[d.expert_id] : nullptr;
        d.US = GATE_UP ? (const unsigned char*)up.scale_ptrs[d.expert_id] : nullptr;
        return d;
    };
    auto issue = [&](const Tile& d, int buf, unsigned int kb) {
        if (d.B == 0) return;
        pqw_issue<GATE_UP>(t, sA[buf], sAs[buf], sB[buf], sS[buf], sTok, A_packed, A_scale,
            d.B, d.S, d.U, d.US, d.cta_m_local, (unsigned int)d.M_expert, d.cta_n, N, K, kb, d.expert_id);
    };
    // Gather the item's token rows and issue its first K stage. Every
    // thread's reads of sTok and sNext and of stage buffer 0 for the
    // previous item precede the last __syncthreads of its K loop or this one.
    auto start = [&](const Tile& d) {
        pqw_gather_rows(sTok, t, sorted_token_ids, d.cta_m, d.cta_m_local, d.M_expert);
        __syncthreads();
        issue(d, 0, 0);
        moe_cp_async_commit();
    };

    if (t == 0) sNext = atomicAdd(next_work, 1);
    __syncthreads();
    unsigned int w = sNext;
    if (w >= work) return;
    const unsigned int stages = K / PQ2_KS;
    Tile cur = decode(w);
    start(cur);
    while (true) {
        // Claim the next item now; the K loop's barriers publish it.
        if (t == 0) sNext = atomicAdd(next_work, 1);
        PqwAcc acc;
        pqw_zero_acc(acc);
        for (unsigned int st = 0; st < stages; ++st) {
            const int buf = st % 2;
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();   // stage `st` landed; every warp finished stage st-1
            if (st + 1 < stages) issue(cur, buf ^ 1, (st + 1) * PQ2_KS);
            moe_cp_async_commit();
            pqw_mma_stage<GATE_UP>(acc, sA[buf], sAs[buf], sB[buf], sS[buf], warp_id, lane_id);
        }
        w = sNext;
        const Tile done = cur;
        const bool more = w < work;
        if (more) {
            cur = decode(w);
            start(cur);
        }
        if (done.B != 0)
            pqw_epilogue<GATE_UP>(acc, warp_id, lane_id, done.expert_id, scale2_vals[done.expert_id],
                done.cta_m, done.cta_m_local, done.M_expert, done.cta_n, N, C, up);
        if (!more) break;
    }
}

// Grid (2 x SMs, 1, 1), 256 threads; arguments as
// moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w plus the zeroed work
// counter.
extern "C" __global__ void __launch_bounds__(256, 2) moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w_persist(
    PQ2_ARGS,
    const int* __restrict__ mtile_prefix,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    unsigned char* __restrict__ out_packed,
    unsigned char* __restrict__ out_scale,
    const float* __restrict__ down_global_scale,
    int* __restrict__ next_work
) {
    pqp_impl<true>(A_packed, A_scale, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,
        expert_offsets, sorted_token_ids, num_experts, N, K, mtile_prefix, next_work,
        PqwGateUp{up_packed_ptrs, up_scale_ptrs, up_scale2_vals, out_packed, out_scale, down_global_scale});
}

// Grid (2 x SMs, 1, 1), 256 threads; arguments as
// moe_w4a4_grouped_gemm_prequant_t_k128w_compact plus the zeroed work counter.
extern "C" __global__ void __launch_bounds__(256, 2) moe_w4a4_grouped_gemm_prequant_t_k128w_compact_persist(
    PQ2_ARGS,
    const int* __restrict__ mtile_prefix,
    int* __restrict__ next_work
) {
    pqp_impl<false>(A_packed, A_scale, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,
        expert_offsets, sorted_token_ids, num_experts, N, K, mtile_prefix, next_work, PqwGateUp{});
}

