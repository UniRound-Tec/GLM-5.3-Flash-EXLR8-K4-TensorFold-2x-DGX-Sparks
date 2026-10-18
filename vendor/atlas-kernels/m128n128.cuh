// Packaging notice (2026-10-15): modified deployment snapshot; see vendor/MODIFICATIONS.md.
// Original copyright/license notices below remain applicable.
// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
namespace m128n128 {
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
constexpr int NT=128, MW=4, MI=2, MT=128;
constexpr int NBLOCK=4; //   // 16-column ldmatrix blocks per warp
// [kp][n] and [group][n] shared tiles: 16-byte chunk c of row r at c ^ (r & 7).
typedef unsigned char PqwA[MT][PQ2_AP];
typedef unsigned char PqwAs[MT][PQ2_KS / GROUP_SIZE];
typedef unsigned char PqwB[NT][PQ2_KP];
typedef unsigned char PqwS[NT][PQ2_KS / GROUP_SIZE];
typedef float PqwAcc[MI][2 * NBLOCK][4];

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

// Threads t < MT: the source row of the tile's row t (its sorted token,
// or the row itself).
__device__ __forceinline__ void pqw_gather_rows(
    int* sTok, unsigned int t, const int* __restrict__ sorted_token_ids,
    unsigned int cta_m, int cta_m_local, int M_expert
) {
    if (t < MT) {
        const bool live = (cta_m_local + (int)t) < M_expert;
        sTok[t] = (sorted_token_ids && live) ? sorted_token_ids[cta_m + t] : (int)(cta_m + t);
    }
}

__device__ __forceinline__ void pqw_zero_acc(PqwAcc& acc) {
    #pragma unroll
    for (int mi = 0; mi < MI; mi++)
        #pragma unroll
        for (int i = 0; i < 2 * NBLOCK; i++) acc[mi][i][0] = acc[mi][i][1] = acc[mi][i][2] = acc[mi][i][3] = 0.0f;
}

// Tile 16-byte column chunk c: whether it reads the up matrix, and its column.
template<bool GATE_UP>
__device__ __forceinline__ bool pqw_chunk_up(unsigned int c) { return GATE_UP && c >= NT / 32; }
template<bool GATE_UP>
__device__ __forceinline__ unsigned int pqw_chunk_col(unsigned int cta_n, unsigned int c) {
    return cta_n + ((GATE_UP ? c % (NT / 32) : c) << 4);
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
    #pragma unroll
    for (int ar=0; ar<2; ++ar) {
        const unsigned int at=t+ar*256;
        const unsigned int row = at >> 2, col = (at & 3) << 4;
        const bool valid = (cta_m_local + row) < M_eff;
        const unsigned int a_row = (unsigned int)sTok[row];
        moe_cp_async_pred_16(&sA[row][col],
            &A_packed[(unsigned long long)a_row * (K / 2) + kb / 2 + col], valid);
    }
    if (t < MT) {
        const bool valid = (cta_m_local + t) < M_eff;
        const unsigned int a_row = (unsigned int)sTok[t];
        moe_cp_async_pred_8(&sAs[t][0],
            &A_scale[(unsigned long long)a_row * (K / GROUP_SIZE) + kb / GROUP_SIZE], valid);
    }
    // QAD keeps the existing tile-major bytes used by decode. Each 128x64
    // box is contiguous; copy four 16-byte pieces per weight row directly.
    #pragma unroll
    for (int r = 0; r < NT/64; ++r) {
        const unsigned int j=t+r*256, row=j/4, col16=(j%4)*16;
        const bool is_up=GATE_UP && row>=NT/2;
        const unsigned int nc=cta_n+(GATE_UP ? row%(NT/2) : row);
        const unsigned long long tile=GATE_UP ? ((nc/128*2+is_up)*(K/128)+kb/128)
            : (kb/128*(N/128)+nc/128);
        const auto* ptr=is_up ? U_expert : B_expert;
        moe_cp_async_pred_16(&sB[row][col16^(((row>>1)&3)*16)],ptr+tile*8192+(nc%128)*64+col16,true);
    }
    // Two losslessly decoded four-byte CSF words per logical weight row.
    if (t < NT) {
    const bool is_up=GATE_UP && t>=NT/2;
    const unsigned int nc=cta_n+(GATE_UP ? t%(NT/2) : t);
    const unsigned int logical_row=GATE_UP ? nc+(is_up ? 0 : N) : nc;
    const unsigned int all_rows=GATE_UP ? 2*N : N;
    const auto* scales=is_up ? US_expert : S_expert;
    *(unsigned int*)&sS[t][0]=csf_word(scales,expert_id,all_rows,K/16,logical_row,kb/16);
    *(unsigned int*)&sS[t][4]=csf_word(scales,expert_id,all_rows,K/16,logical_row,kb/16+4);
    }
}

// Warp warp_id's MMAs of one K128 stage held in shared memory.
template<bool GATE_UP>
__device__ __forceinline__ void pqw_mma_stage(
    PqwAcc& acc, const PqwA& sA, const PqwAs& sAs, const PqwB& sB, const PqwS& sS,
    unsigned int warp_id, unsigned int lane_id
) {
    constexpr int NB = NBLOCK;
    const unsigned int warp_m_offset = (warp_id % MW) * (MI * 16);
    const unsigned int warp_n_offset = (warp_id / MW) * (NT / (8 / MW));
    // This warp's nb-th chunk (GATE_UP: gate chunks 2wn, 2wn+1, then up's).
    auto warp_chunk = [&](int nb) -> unsigned int {
        return GATE_UP ? (nb >> 1) * (NT / 32) + (warp_id / MW) * 2 + (nb & 1)
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
        unsigned int a[MI][4], sfa[MI];
        #pragma unroll
        for (int mi = 0; mi < MI; ++mi) {
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
                for (int mi = 0; mi < MI; ++mi) {
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
    constexpr int NB = NBLOCK;
    const unsigned int warp_m_offset = (warp_id % MW) * (MI * 16);
    const unsigned int warp_n_offset = (warp_id / MW) * (NT / (8 / MW));
    const unsigned int group_id = lane_id >> 2, tid = lane_id & 3;
    if constexpr (GATE_UP) {
        // Gate subtile j pairs with up subtile j + 4 (same columns); each
        // 16-column group spans two subtiles, i.e. one quad of threads.
        const float scale2_up = up.scale2_vals[expert_id];
        #pragma unroll
        for (int mi = 0; mi < MI; mi++) {
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
                    const unsigned int col = cta_n + (warp_id / MW) * 32 + q * 16;
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
        for (int mi = 0; mi < MI; mi++) {
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
    const int cta_m_local = (tile - mtile_prefix[expert_id]) * MT;
    if (cta_m_local >= M_expert) return;
    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    if (B_expert == 0) return;
    const float scale2 = scale2_vals[expert_id];
    const unsigned char* U_expert = GATE_UP ? (const unsigned char*)up.packed_ptrs[expert_id] : nullptr;
    const unsigned char* US_expert = GATE_UP ? (const unsigned char*)up.scale_ptrs[expert_id] : nullptr;
    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * (GATE_UP ? NT / 2 : NT);

    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t / 32, lane_id = t % 32;

    __shared__ __align__(16) PqwA sA[STAGES];
    __shared__ __align__(16) PqwAs sAs[STAGES];
    __shared__ __align__(16) PqwB sB[STAGES];
    __shared__ __align__(16) PqwS sS[STAGES];
    __shared__ int sTok[MT];

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

__global__ void __launch_bounds__(256, 2) moe_w4a4_grouped_gemm_prequant_t_k128w_compact(
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
__global__ void __launch_bounds__(256, 2) moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w(
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


}
