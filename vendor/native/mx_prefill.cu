// GLM large-window MXFP8 adapter. b12x AOT kernels: Apache-2.0.
// Weight repack is lossless; activation quantization is intentionally lossy.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "aot/mx_0.h"
#include "aot/mx_3.h"
#include "aot/mx_6.h"
static mx_0_Kernel_Module_t quant;
static mx_3_Kernel_Module_t kda;
static mx_6_Kernel_Module_t dsa;
static bool loaded=false;
static size_t align256(size_t x){return (x+255)&~size_t(255);}
extern "C" int mxp_load(){
 if(!loaded){mx_0_Kernel_Module_Load(&quant);mx_3_Kernel_Module_Load(&kda);mx_6_Kernel_Module_Load(&dsa);loaded=true;}
 return cudaGetLastError();
}
__global__ void repack(const uint32_t* w,const uint8_t* s,uint32_t* dst,uint8_t* sf,int n,int np){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=np*1024)return;
 int row=i/1024,col=(i%1024)*4;
 uint32_t value=0;
 int tile=(row/16)*128+col/32;
 if(row<n){int reg=(row%16)/8+2*((col%32)/16),lane=(row%8)*4+(col%16)/4;value=w[tile*128+reg*32+lane];}
 dst[i]=value;
 if((col%32)==0){
  int group=col/32;
  size_t idx=((size_t(row/128)*32+group/4)*32+row%32)*16+(row%128)/32*4+group%4;
  sf[idx]=row<n?s[tile*32+(row%8)*4+(row%16)/8]:127;
 }
}
__global__ void copyout(const __nv_bfloat16* src,__nv_bfloat16* dst,int m,int n,int np){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<m*n)dst[i]=src[(i/n)*np+i%n];
}
struct Space{uint8_t *w,*s,*aq,*ar,*am,*y,*alpha;size_t bytes;};
static Space space(uint64_t scratch,int m,int n){
 int np=(n+127)/128*128,mp=(m+127)/128*128;size_t off=0;auto p=(uint8_t*)scratch;Space s;
 #define TAKE(field,bytes) s.field=p+off;off+=align256(bytes)
 TAKE(w,size_t(np)*4096);TAKE(s,size_t(np)*128);
 TAKE(aq,size_t(mp)*4096);TAKE(ar,size_t(mp)*128);TAKE(am,size_t(mp)*128);
 TAKE(y,size_t(mp)*np*2);TAKE(alpha,256);
 #undef TAKE
 s.bytes=off;return s;
}
extern "C" uint64_t mxp_bytes(int m,int n){return space(0,m,n).bytes;}
extern "C" int mxp_repack(uint64_t w,uint64_t sf,uint64_t dst,uint64_t dsf,int n,uint64_t stream){
 int np=(n+127)/128*128;
 repack<<<(np*1024+255)/256,256,0,(cudaStream_t)stream>>>((uint32_t*)w,(uint8_t*)sf,(uint32_t*)dst,(uint8_t*)dsf,n,np);
 return cudaGetLastError();
}
extern "C" int mxp_run(uint64_t x,uint64_t w,uint64_t sf,uint64_t out,uint64_t scratch,uint64_t capacity,int m,int n,uint64_t stream){
 if(!loaded||m<33||m>4096||(n!=12576&&n!=2048))return -101;
 auto st=(cudaStream_t)stream;Space b=space(scratch,m,n);if(b.bytes>capacity)return -102;
 int err=mxp_repack(w,sf,(uint64_t)b.w,(uint64_t)b.s,n,stream);if(err)return err;
 // alpha is a device scalar; it is ignored by the unscaled output specialization.
 err=cudaMemsetAsync(b.alpha,0,256,st);if(err)return err;
 int grid=(m*32+3)/4;
 err=cute_dsl_mx_0_wrapper(&quant,(void*)x,b.aq,b.ar,b.am,m,4096,grid,st);if(err)return err;
 #define ARGS b.aq,b.w,b.am,b.s,b.y,b.y,b.y,b.y,b.alpha,m,st
 err=n==12576?cute_dsl_mx_3_wrapper(&kda,ARGS):cute_dsl_mx_6_wrapper(&dsa,ARGS);
 #undef ARGS
 if(err)return err;
 copyout<<<(m*n+255)/256,256,0,st>>>((__nv_bfloat16*)b.y,(__nv_bfloat16*)out,m,n,(n+127)/128*128);
 return cudaGetLastError();
}
