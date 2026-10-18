// Native SM121 warp MMA. Fragment/scale layout follows NVIDIA PTX block-scaling
// documentation; scale-row convention cross-checked with b12x (Apache-2.0)
// attention/_shared/mla/decode_math.py. No TMA, no weight requantization.
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ void mx(float (&d)[4],uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,
                                  uint32_t b0,uint32_t b1,uint32_t sa,uint32_t sb) {
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"h"((uint16_t)0),"h"((uint16_t)0),
      "r"(sb),"h"((uint16_t)0),"h"((uint16_t)0));
}

template<bool TILED>
__global__ void __launch_bounds__(128) kernel(const uint8_t* X,const uint8_t* XS,const uint8_t* W,const uint16_t* WS,
                                             float* O,int M,int N,int K,int SK) {
  int lane=threadIdx.x&31,warp=threadIdx.x>>5,q=lane>>2,c=lane&3;
  int nb=blockIdx.x*64+warp*16,mb=blockIdx.y*8,split=blockIdx.z;
  int r0=nb+q,r1=r0+8,br=mb+q;
  float d[4]={0,0,0,0};
  for(int ki=split*(K/SK);ki<(split+1)*(K/SK);ki+=32){
    int ko=ki+4*c;
    const uint32_t* wp=(const uint32_t*)W+((size_t)(nb/16)*(K/32)+ki/32)*128+lane;
    uint32_t a0=TILED?__ldg(wp):(r0<N?__ldg((const uint32_t*)(W+(size_t)r0*K+ko)):0);
    uint32_t a1=TILED?__ldg(wp+32):(r1<N?__ldg((const uint32_t*)(W+(size_t)r1*K+ko)):0);
    uint32_t a2=TILED?__ldg(wp+64):(r0<N?__ldg((const uint32_t*)(W+(size_t)r0*K+ko+16)):0);
    uint32_t a3=TILED?__ldg(wp+96):(r1<N?__ldg((const uint32_t*)(W+(size_t)r1*K+ko+16)):0);
    uint32_t b0=br<M?__ldg((const uint32_t*)(X+br*K+ko)):0;
    uint32_t b1=br<M?__ldg((const uint32_t*)(X+br*K+ko+16)):0;
    int sr=nb+q+8*(lane&1);
    uint32_t sa=TILED?__ldg((const uint8_t*)WS+((size_t)(nb/16)*(K/32)+ki/32)*32+lane):(sr<N?__ldg(WS+(size_t)sr*(K/32)+ki/32):127);
    uint32_t sb=br<M?__ldg(XS+br*(K/32)+ki/32):127;
    mx(d,a0,a1,a2,a3,b0,b1,sa,sb);
  }
  int m0=mb+2*c,m1=m0+1;
  size_t base=(size_t)split*M*N;
  if(m0<M&&r0<N)O[base+m0*N+r0]=d[0];
  if(m1<M&&r0<N)O[base+m1*N+r0]=d[1];
  if(m0<M&&r1<N)O[base+m0*N+r1]=d[2];
  if(m1<M&&r1<N)O[base+m1*N+r1]=d[3];
}
extern "C" int launch(const void* X,const void* XS,const void* W,const void* WS,void* O,int M,int N,int K,int SK,void* stream){
  kernel<false><<<dim3((N+63)/64,(M+7)/8,SK),128,0,(cudaStream_t)stream>>>(
    (const uint8_t*)X,(const uint8_t*)XS,(const uint8_t*)W,(const uint16_t*)WS,(float*)O,M,N,K,SK);
  return (int)cudaGetLastError();
}
extern "C" int launch_tiled(const void* X,const void* XS,const void* W,const void* WS,void* O,int M,int N,int K,int SK,void* stream){
  kernel<true><<<dim3((N+63)/64,(M+7)/8,SK),128,0,(cudaStream_t)stream>>>(
    (const uint8_t*)X,(const uint8_t*)XS,(const uint8_t*)W,(const uint16_t*)WS,(float*)O,M,N,K,SK);
  return (int)cudaGetLastError();
}
