// Packaging notice (2026-10-15): modified deployment snapshot; see vendor/MODIFICATIONS.md.
// Original copyright/license notices below remain applicable.
// SPDX-License-Identifier: AGPL-3.0-only
// Atlas stream schedule, adapted to contiguous tile-major QAD weights.
#pragma once

template<bool GU>
__device__ __forceinline__ const unsigned char* weight_piece(const unsigned char* b,
    const unsigned char* u,unsigned row,unsigned col,unsigned cta_n,unsigned N,unsigned K,unsigned kb) {
    bool up=GU && row>=128;
    unsigned nc=cta_n+(GU ? row%128 : row);
    unsigned long long tile=GU ? ((nc/128*2+up)*(K/128)+kb/128)
        : (kb/128*(N/128)+nc/128);
    return (up ? u : b)+tile*8192+(nc%128)*64+col;
}

template<bool GU,int MI>
__device__ __forceinline__ void stream_load(unsigned t,
    unsigned char (*sa)[PQ2_AP],unsigned char (*sas)[8],PqwB& sb,PqwS& ss,
    const int* tok,const unsigned char* ap,const unsigned char* as,
    const unsigned char* bp,const unsigned char* bsp,const unsigned char* up,const unsigned char* usp,
    unsigned effective,unsigned cn,unsigned N,unsigned K,unsigned kb,unsigned expert,const unsigned char* live) {
    // Issue the whole CTA's streaming weight loads before shared stores.
    uint4 v[4];
    #pragma unroll
    for(int q=0;q<4;q++) {
        unsigned j=t+q*256,row=j/4,col=(j%4)*16;
        v[q]=(!live || live[kb/32+col/16])
            ? __ldcs((const uint4*)weight_piece<GU>(bp,up,row,col,cn,N,K,kb))
            : make_uint4(0,0,0,0);
    }
    bool isup=GU && t>=128;
    unsigned nc=cn+(GU ? t%128 : t), logical=GU ? nc+(isup ? 0 : N) : nc;
    unsigned allrows=GU ? 2*N : N;
    auto sp=isup ? usp : bsp;
    unsigned sf0=csf_word(sp,expert,allrows,K/16,logical,kb/16);
    unsigned sf1=csf_word(sp,expert,allrows,K/16,logical,kb/16+4);
    #pragma unroll
    for(int q=0;q<4;q++) {
        unsigned j=t+q*256,row=j/4,col=(j%4)*16;
        *(uint4*)&sb[row][col^(((row>>1)&3)*16)]=v[q];
    }
    *(unsigned*)&ss[t][0]=sf0;*(unsigned*)&ss[t][4]=sf1;
    if(t<PQD_M*MI*4) {
        unsigned r=t/4,c=(t%4)*16;
        *(uint4*)&sa[r][c]=r<effective ? *(const uint4*)&ap[(size_t)tok[r]*(K/2)+kb/2+c]
            : make_uint4(0,0,0,0);
    }
    if(t<PQD_M*MI)
        *(unsigned long long*)&sas[t][0]=t<effective
            ? *(const unsigned long long*)&as[(size_t)tok[t]*(K/16)+kb/16] : 0;
}

template<bool GU>
__device__ __forceinline__ void stream_prefetch(unsigned t,const unsigned char* bp,const unsigned char* up,
    unsigned cn,unsigned N,unsigned K,unsigned kb,const unsigned char* live) {
    // Each CTA owns two contiguous 8KiB boxes; prefetch every 128-byte line once.
    if(t<128) {
        unsigned box=t/64,line=t%64;
        // A line contains two complete 64-byte rows in tile-major storage.
        const auto* p=weight_piece<GU>(bp,up,box*128+line*2,0,cn,N,K,kb);
        asm volatile("prefetch.global.L2 [%0];"::"l"(p));
    }
}

template<bool GU,bool ZERO,int MI,bool PF>
__device__ __forceinline__ void stream_impl(PQ2_ARGS,const unsigned* work,const PqwGateUp q) {
    if(blockIdx.y>=work[0])return;
    unsigned e=work[4+blockIdx.y],t=threadIdx.x;
    if(e>=num_experts)return;
    unsigned cm=expert_offsets[e],effective=expert_offsets[e+1]-cm;
    if(!effective || effective>PQD_M*MI)return;
    const auto* bp=(const unsigned char*)B_packed_ptrs[e];
    const auto* sp=(const unsigned char*)B_scale_ptrs[e];
    const auto* up=GU ? (const unsigned char*)q.packed_ptrs[e] : nullptr;
    const auto* us=GU ? (const unsigned char*)q.scale_ptrs[e] : nullptr;
    unsigned cn=blockIdx.x*(GU ? 128 : 256),w=t/32,l=t%32;
    __shared__ __align__(16) unsigned char sa[2][PQD_M*MI][PQ2_AP];
    __shared__ __align__(16) unsigned char sas[2][PQD_M*MI][8];
    __shared__ __align__(16) PqwB sb[2];
    __shared__ __align__(16) PqwS ss[2];
    __shared__ int tok[PQD_M*MI];
    __shared__ unsigned char live[ZERO ? 64 : 1];
    if(t<PQD_M*MI)tok[t]=cm+t;
    __syncthreads();
    // Tile-major keeps each output row contiguous. Skip a 16-byte load only
    // when all 32 activation values are FP4 +/-zero in every expert row.
    if constexpr(ZERO) {
        if(t<K/32) {
            uint4 any=make_uint4(0,0,0,0);
            for(unsigned r=0;r<effective;r++) {
                uint4 a=*(const uint4*)&A_packed[(size_t)tok[r]*(K/2)+t*16];
                any.x|=a.x;any.y|=a.y;any.z|=a.z;any.w|=a.w;
            }
            live[t]=((any.x|any.y|any.z|any.w)&0x77777777u)!=0;
        }
        __syncthreads();
    }
    const auto* mask=ZERO ? live : nullptr;
    auto load=[&](int b,unsigned kb){stream_load<GU,MI>(t,sa[b],sas[b],sb[b],ss[b],tok,
        A_packed,A_scale,bp,sp,up,us,effective,cn,N,K,kb,e,mask);};
    unsigned stages=K/128;
    auto pf=[&](unsigned st){if constexpr(PF)if(st<stages)stream_prefetch<GU>(t,bp,up,cn,N,K,st*128,mask);};
    for(unsigned st=0;st<=2;st++)pf(st);
    load(0,0);__syncthreads();
    PqdAcc acc[MI];
    #pragma unroll
    for(int mi=0;mi<MI;mi++)
        #pragma unroll
        for(int j=0;j<4;j++)acc[mi][j][0]=acc[mi][j][1]=acc[mi][j][2]=acc[mi][j][3]=0;
    for(unsigned st=0;st<stages;st++) {
        unsigned b=st&1;pf(st+3);
        if(st+1<stages)load(b^1,(st+1)*128);
        #pragma unroll
        for(int mi=0;mi<MI;mi++)pqd_mma_stage<GU>(acc[mi],*(const PqdA*)&sa[b][mi*16],
            *(const PqdAs*)&sas[b][mi*16],sb[b],ss[b],w,l);
        __syncthreads();
    }
    #pragma unroll
    for(int mi=0;mi<MI;mi++)pqd_epilogue<GU>(acc[mi],w,l,e,scale2_vals[e],cm+mi*16,
        int(effective)-mi*16,cn,N,C,q);
}

template<bool GU,bool ZERO,int MI,bool PF>
__global__ void __launch_bounds__(256,2) qad_stream(PQ2_ARGS,const unsigned* work,PqwGateUp q) {
    stream_impl<GU,ZERO,MI,PF>(A_packed,A_scale,B_packed_ptrs,B_scale_ptrs,scale2_vals,C,
        expert_offsets,sorted_token_ids,num_experts,N,K,work,q);
}

__global__ void reduce_vec8(const __nv_bfloat16* down,const int* inverse,const float* weights,
    __nv_bfloat16* out,int rows) {
    int pos=blockIdx.x*blockDim.x+threadIdx.x;
    if(pos>=rows*4096/8)return;
    int token=pos/(4096/8),c=pos%(4096/8)*8;float sum[8]={};
    #pragma unroll
    for(int k=0;k<8;k++) {
        int p=token*8+k;uint4 v=*(const uint4*)&down[(size_t)inverse[p]*4096+c];
        unsigned bits[4]={v.x,v.y,v.z,v.w};
        #pragma unroll
        for(int j=0;j<8;j++) {
            float f=__uint_as_float(((bits[j/2]>>(16*(j&1)))&65535u)<<16)*weights[p];
            sum[j]+=__bfloat162float(__float2bfloat16(f));
        }
    }
    unsigned b[4];
    #pragma unroll
    for(int j=0;j<4;j++)b[j]=__bfloat16_as_ushort(__float2bfloat16(sum[2*j]))
        | ((unsigned)__bfloat16_as_ushort(__float2bfloat16(sum[2*j+1]))<<16);
    *(uint4*)&out[(size_t)token*4096+c]=make_uint4(b[0],b[1],b[2],b[3]);
}
