// Packaging notice (2026-10-15): modified deployment snapshot; see vendor/MODIFICATIONS.md.
// Original copyright/license notices below remain applicable.
// SPDX-License-Identifier: AGPL-3.0-only
#include "atlas_k128w.cuh"
#include "atlas_m16.cuh"
#include "stream_qad.cuh"
constexpr int E=288,H=4096,I=1024,TOP=8;
__global__ void count_routes(const int* ids,int* counts,int total) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p<total) atomicAdd(counts+ids[p],1);
}
__global__ void scan_routes(const int* counts,int* offsets,int* cursor,int* prefix,bool decode=false) {
    __shared__ int s[512],tiles[512];int t=threadIdx.x;
    int v=t<E?counts[t]:0;s[t]=v;tiles[t]=decode ? (v>0) : (v+63)/64;__syncthreads();
    for(int d=1;d<512;d*=2){int a=t>=d?s[t-d]:0,b=t>=d?tiles[t-d]:0;__syncthreads();s[t]+=a;tiles[t]+=b;__syncthreads();}
    __syncthreads();
    if(t<E){offsets[t+1]=s[t];cursor[t]=s[t]-v;if(!decode)prefix[t+1]=tiles[t];
        else if(v)prefix[4+tiles[t]-1]=t;}
    if(!t){offsets[0]=0;prefix[0]=decode ? tiles[511] : 0;}
}
__global__ void scatter_routes(const int* ids,int* cursor,int* pairs,int* inverse,int* experts,int total) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p<total){int e=ids[p],r=atomicAdd(cursor+e,1);pairs[r]=p;inverse[p]=r;experts[r]=e;}
}
__global__ void quant_routes(const __nv_bfloat16* x,const int* pairs,const int* experts,
        const float* gs,unsigned char* ap,unsigned char* as,int total) {
    int row=blockIdx.x,group=threadIdx.x; // 256 groups, one complete routed row.
    if(row>=total)return;
    float v[16],mx=0;
    int token=pairs[row]/TOP;
    #pragma unroll
    for(int j=0;j<16;j++){v[j]=__bfloat162float(x[(size_t)token*H+group*16+j]);mx=fmaxf(mx,fabsf(v[j]));}
    float inv;as[(size_t)row*(H/16)+group]=qad_group_scale(mx,gs[experts[row]],&inv);
    unsigned long long packed=0;
    #pragma unroll
    for(int j=0;j<16;j++)packed|=(unsigned long long)qad_fp4(v[j]*inv)<<(4*j);
    *(unsigned long long*)&ap[(size_t)row*(H/2)+group*8]=packed;
}
__global__ void reduce_routes(const __nv_bfloat16* down,const int* inverse,const float* weights,
        __nv_bfloat16* out,int rows) {
    int pos=blockIdx.x*blockDim.x+threadIdx.x;
    if(pos>=rows*H)return;
    int token=pos/H,c=pos%H;float sum=0;
    #pragma unroll
    for(int k=0;k<TOP;k++) {
        int p=token*TOP+k;
        // B12X keeps a BF16 route contribution before the top-k sum.
        float v=__bfloat162float(down[(size_t)inverse[p]*H+c])*weights[p];
        sum+=__bfloat162float(__float2bfloat16(v));
    }
    out[pos]=__float2bfloat16(sum);
}
extern "C" int atlas_run_control(const uint64_t* p,int rows,uint64_t stream_bits) {
    if(rows<1||rows>8192)return -1;
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
    scan_routes<<<1,512,0,s>>>(counts,offsets,cursor,prefix);
    scatter_routes<<<(r+255)/256,256,0,s>>>(ids,cursor,pairs,inverse,experts,r);
    quant_routes<<<r,256,0,s>>>(x,pairs,experts,gs1,ap,as,r);
    int bound=(r+63)/64+E;
    moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w<<<dim3(I/128,bound),256,0,s>>>(
        ap,as,w13,sf13,a13,nullptr,offsets,nullptr,E,I,H,prefix,w13,sf13,a13,mp,ms,gs2);
    moe_w4a4_grouped_gemm_prequant_t_k128w_compact<<<dim3(H/256,bound),256,0,s>>>(
        mp,ms,w2,sf2,a2,down,offsets,nullptr,E,H,I,prefix);
    reduce_routes<<<(rows*H+255)/256,256,0,s>>>(down,inverse,probs,out,rows);
    return cudaGetLastError();
}

extern "C" int atlas_run_persistent(const uint64_t* p,int rows,uint64_t stream_bits) {
    if(rows<1||rows>8192)return -1;
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
    scan_routes<<<1,512,0,s>>>(counts,offsets,cursor,prefix);
    scatter_routes<<<(r+255)/256,256,0,s>>>(ids,cursor,pairs,inverse,experts,r);
    quant_routes<<<r,256,0,s>>>(x,pairs,experts,gs1,ap,as,r);
    int* next=prefix+289;
    cudaMemsetAsync(next,0,sizeof(int),s);
    moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w_persist<<<96,256,0,s>>>(
        ap,as,w13,sf13,a13,nullptr,offsets,nullptr,E,I,H,prefix,w13,sf13,a13,mp,ms,gs2,next);
    cudaMemsetAsync(next,0,sizeof(int),s);
    moe_w4a4_grouped_gemm_prequant_t_k128w_compact_persist<<<96,256,0,s>>>(
        mp,ms,w2,sf2,a2,down,offsets,nullptr,E,H,I,prefix,next);
    reduce_routes<<<(rows*H+255)/256,256,0,s>>>(down,inverse,probs,out,rows);
    return cudaGetLastError();
}

// Validation entry: full logical CSF scale planes, independently CPU-decoded
// from the checkpoint by integration.py. Not called in the serving path.
__global__ void export_scales(const unsigned char* src,unsigned int* dst,int rows,int cols) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x,total=E*rows*(cols/4);
    if(i<total){unsigned c=i%(cols/4),r=i/(cols/4)%rows,e=i/((cols/4)*rows);
        dst[i]=csf_word(src,e,rows,cols,r,c*4);}
}
extern "C" int atlas_export_scales(uint64_t src,uint64_t dst,int rows,int cols,uint64_t stream) {
    export_scales<<<(E*rows*(cols/4)+255)/256,256,0,(cudaStream_t)stream>>>((const unsigned char*)src,(unsigned int*)dst,rows,cols);
    return cudaGetLastError();
}

#include "entry.cuh"

#include "m128n128.cuh"
#include "m128n256.cuh"
#include "prefill_entry.cuh"
#include "sampling_pack.cuh"
