// Packaging notice (2026-10-15): modified deployment snapshot; see vendor/MODIFICATIONS.md.
// Original copyright/license notices below remain applicable.
// SPDX-License-Identifier: AGPL-3.0-only
// Atlas 72588541 M16 arithmetic; QAD/CSF/tile-major adaptation. See NOTICE.
#pragma once
#define PQD_M 16
typedef unsigned char PqdA[PQD_M][PQ2_AP];
typedef unsigned char PqdAs[PQD_M][PQ2_KS / GROUP_SIZE];
typedef float PqdAcc[4][4];
template<bool GATE_UP>
__device__ __forceinline__ unsigned int pqd_warp_chunk(unsigned int w,int nb) {
    return GATE_UP ? nb*(PQW_NT/32)+w : w*2+nb;
}
// Warp warp_id's MMAs of one K128 stage: pqw_mma_stage's fragments for the
// one row slab and this warp's two column chunks.
template<bool GATE_UP>
__device__ __forceinline__ void pqd_mma_stage(
    PqdAcc& acc, const PqdA& sA, const PqdAs& sAs, const PqwB& sB, const PqwS& sS,
    unsigned int warp_id, unsigned int lane_id
) {
    const unsigned int kp_l = lane_id, g_l = lane_id & 7;
    const unsigned int sfa_m = (lane_id & 1) * 8 + (lane_id >> 2);
    const unsigned short tid_a = 0, bid_a = 0, bid_b = 0;
    #pragma unroll
    for (int sl = 0; sl < 2; ++sl) {
        const unsigned int ko = sl * 32;
        const unsigned short tid_b = 0;
        unsigned int a[4];
        {
            const unsigned int j = lane_id >> 3, r = lane_id & 7;
            const unsigned int addr = __cvta_generic_to_shared(&sA[r + (j & 1) * 8][ko + (j >> 1) * 16]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
        }
        const unsigned int sfa = *(const unsigned int*)&sAs[sfa_m][sl * 4];
        #pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
            const unsigned int c = pqd_warp_chunk<GATE_UP>(warp_id, nb);
            const unsigned int nc=c*16+(lane_id>>2), kc=ko+(lane_id&3)*4;
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
                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
                    "{%14},{%15,%16},{%17},{%18,%19};"
                    :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3])
                    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[h]),"r"(b[2 + h]),
                     "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]),
                     "r"(sfa),"h"(bid_a),"h"(tid_a),"r"(h ? sf1 : sf0),"h"(bid_b),"h"(tid_b));
            }
        }
    }
}

// Warp warp_id's stores of a finished tile, as pqw_epilogue: BF16 C, or
// (GATE_UP) the packed NVFP4 SiLU·mul of its 16 gate/up columns.
template<bool GATE_UP>
__device__ __forceinline__ void pqd_epilogue(
    const PqdAcc& acc, unsigned int warp_id, unsigned int lane_id,
    unsigned int expert_id, float scale2, unsigned int cta_m, int M_expert,
    unsigned int cta_n, unsigned int N, __nv_bfloat16* __restrict__ C, const PqwGateUp& up
) {
    const unsigned int group_id = lane_id >> 2, tid = lane_id & 3;
    if constexpr (GATE_UP) {
        const float scale2_up = up.scale2_vals[expert_id];
        const unsigned int col = cta_n + warp_id * 16;
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            const unsigned int row = cta_m + group_id + half * 8;
            const bool valid = (int)(group_id + half * 8) < M_expert;
            float v[4], group_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int j = i >> 1, e = half * 2 + (i & 1);
                v[i] = silu_nvfp4_act(
                    acc[j][e] * scale2,
                    acc[j + 2][e] * scale2_up);
                group_max = fmaxf(group_max, fabsf(v[i]));
            }
            group_max = fmaxf(group_max, __shfl_xor_sync(0xffffffffu, group_max, 1));
            group_max = fmaxf(group_max, __shfl_xor_sync(0xffffffffu, group_max, 2));
            float inv;
            const unsigned char sc = qad_group_scale(group_max, up.down_global_scale[expert_id], &inv);
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
    } else {
        const bool r0v = (int)group_id < M_expert, r1v = (int)(group_id + 8) < M_expert;
        const unsigned int r0 = cta_m + group_id, r1 = r0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            const unsigned int c0 = cta_n + warp_id * 32 + nt * 8 + tid * 2;
            if (r0v)
                *(__nv_bfloat162*)&C[r0 * N + c0] = __floats2bfloat162_rn(
                    acc[nt][0] * scale2, acc[nt][1] * scale2);
            if (r1v)
                *(__nv_bfloat162*)&C[r1 * N + c0] = __floats2bfloat162_rn(
                    acc[nt][2] * scale2, acc[nt][3] * scale2);
        }
    }
}

