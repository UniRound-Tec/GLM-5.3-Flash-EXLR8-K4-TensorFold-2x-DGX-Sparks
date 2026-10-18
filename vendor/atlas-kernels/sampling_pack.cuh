// Packaging notice (2026-10-15): modified deployment snapshot; see vendor/MODIFICATIONS.md.
// Original copyright/license notices below remain applicable.
// Candidate values and token IDs in the unchanged float32 transport format.
__global__ void pack_sampling_kernel(const float* values,const long long* ids,float* out,
                                    int count,int n,int offset) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count){int row=i/n,c=i%n;out[row*2*n+c]=values[i];
        ((int*)out)[row*2*n+n+c]=(int)(ids[i]+offset);}
}
extern "C" int atlas_pack_sampling(uint64_t vals,uint64_t ids,uint64_t out,
                                   int count,int n,int offset,uint64_t stream) {
    if(count<=0||n<=0||count%n)return -1;
    pack_sampling_kernel<<<(count+255)/256,256,0,(cudaStream_t)stream>>>(
        (const float*)vals,(const long long*)ids,(float*)out,count,n,offset);
    return cudaGetLastError();
}
