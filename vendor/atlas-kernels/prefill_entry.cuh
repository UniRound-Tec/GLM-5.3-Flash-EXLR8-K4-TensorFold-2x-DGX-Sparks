// Packaging notice (2026-10-15): modified deployment snapshot; see vendor/MODIFICATIONS.md.
// Original copyright/license notices below remain applicable.
// SPDX-License-Identifier: AGPL-3.0-only
// Modes: 256 + GU*16 + DN; 0=M64N256, 1=M128N128, 2=M128N256.
__global__ void scan_routes_both(const int* counts,int* offsets,int* cursor,int* prefix) {
    __shared__ int s[512],a[512],b[512];int t=threadIdx.x;
    int v=t<E?counts[t]:0;s[t]=v;a[t]=(v+63)/64;b[t]=(v+127)/128;__syncthreads();
    for(int d=1;d<512;d*=2){int x=t>=d?s[t-d]:0,y=t>=d?a[t-d]:0,z=t>=d?b[t-d]:0;
        __syncthreads();s[t]+=x;a[t]+=y;b[t]+=z;__syncthreads();}
    if(t<E){offsets[t+1]=s[t];cursor[t]=s[t]-v;prefix[t+1]=a[t];prefix[E+2+t]=b[t];}
    if(!t){offsets[0]=0;prefix[0]=0;prefix[E+1]=0;}
}
extern "C" int atlas_run_prefill(const uint64_t* p,int rows,uint64_t stream_bits,int mode) {
    if(rows<1||rows>4096||mode<256)return -1;
    int gu=(mode-256)/16,dn=(mode-256)%16;
    if(gu>2||dn>2)return -2;
    auto s=(cudaStream_t)stream_bits;int r=rows*TOP;
    auto x=(const __nv_bfloat16*)p[0];auto ids=(const int*)p[1];auto probs=(const float*)p[2];auto out=(__nv_bfloat16*)p[3];
    auto w13=(const unsigned long long*)p[4];auto sf13=(const unsigned long long*)p[5];auto a13=(const float*)p[6];
    auto w2=(const unsigned long long*)p[7];auto sf2=(const unsigned long long*)p[8];auto a2=(const float*)p[9];
    auto gs1=(const float*)p[10];auto gs2=(const float*)p[11];
    auto counts=(int*)p[12];auto offsets=(int*)p[13];auto cursor=(int*)p[14];auto pairs=(int*)p[15];
    auto inverse=(int*)p[16];auto experts=(int*)p[17];auto prefix=(int*)p[18];
    auto ap=(unsigned char*)p[19];auto as=(unsigned char*)p[20];auto mp=(unsigned char*)p[21];auto ms=(unsigned char*)p[22];
    auto down=(__nv_bfloat16*)p[23];
    cudaError_t error=cudaMemsetAsync(counts,0,E*sizeof(int),s);if(error!=cudaSuccess)return error;
    count_routes<<<(r+255)/256,256,0,s>>>(ids,counts,r);
    scan_routes_both<<<1,512,0,s>>>(counts,offsets,cursor,prefix);
    scatter_routes<<<(r+255)/256,256,0,s>>>(ids,cursor,pairs,inverse,experts,r);
    quant_routes<<<r,256,0,s>>>(x,pairs,experts,gs1,ap,as,r);
    int b64=(r+63)/64+E,b128=(r+127)/128+E;auto p128=prefix+E+1;
    #define GU(NS,NC,BND,PF) NS moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w<<<dim3(I/(NC/2),BND),256,0,s>>>(ap,as,w13,sf13,a13,nullptr,offsets,nullptr,E,I,H,PF,w13,sf13,a13,mp,ms,gs2)
    if(gu==0){GU(,256,b64,prefix);}else if(gu==1){GU(m128n128::,128,b128,p128);}else{GU(m128n256::,256,b128,p128);}
    #undef GU
    #define DN(NS,NC,BND,PF) NS moe_w4a4_grouped_gemm_prequant_t_k128w_compact<<<dim3(H/NC,BND),256,0,s>>>(mp,ms,w2,sf2,a2,down,offsets,nullptr,E,H,I,PF)
    if(dn==0){DN(,256,b64,prefix);}else if(dn==1){DN(m128n128::,128,b128,p128);}else{DN(m128n256::,256,b128,p128);}
    #undef DN
    reduce_routes<<<(rows*H+255)/256,256,0,s>>>(down,inverse,probs,out,rows);
    return cudaGetLastError();
}
