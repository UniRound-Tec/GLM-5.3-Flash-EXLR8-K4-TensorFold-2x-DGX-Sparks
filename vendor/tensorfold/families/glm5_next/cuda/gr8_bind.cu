#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "adapted/tr3.cuh"
__device__ __forceinline__ int acquire(const int* p){int v;asm volatile("ld.acquire.gpu.global.s32 %0, [%1];":"=r"(v):"l"(p):"memory");return v;}
__global__ __launch_bounds__(512) void gr8_fused(K1Args a) {
 extern __shared__ u32 sm[];__shared__ int item,last;
 int n1=8*a.B*2*a.KS1,n2=8*a.B*a.NS2;
 for(;;){
  if(threadIdx.x==0)item=atomicAdd(a.fctl,1);__syncthreads();int u=item;if(u>=n1+n2)break;
  if(u<n1){
   int slot=u/(a.KS1*2*a.B),b=(u/(a.KS1*2))%a.B,mat=(u/a.KS1)&1;
   auto tr=a.t_tab+6*a.slot_expert[slot];
   auto mw=(mat?a.tf_up:a.t_fc1)+tr[0]+b*8*128;
   tr3_fc1_body<4,true,false>(a,0,u,tr,mw,1,sm);
   __threadfence();__syncthreads();if(threadIdx.x==0)atomicAdd(a.fctl+1+slot*a.B+b,1);
  }else{
   u-=n1;int slot=u/(a.NS2*a.B),b=(u/a.NS2)%a.B;
   if(threadIdx.x==0)while(acquire(a.fctl+1+slot*a.B+b)<2*a.KS1){}
   __syncthreads();tr3_fc2_body<4,true>(a,0,u,a.t_tab+6*a.slot_expert[slot],1,sm,true,10.f);
  }__syncthreads();
 }
 if(threadIdx.x==0){__threadfence();last=atomicAdd(a.fctl+FCTL_N-1,1)==gridDim.x-1;}
 __syncthreads();if(last){for(int i=threadIdx.x;i<FCTL_N;i+=blockDim.x)a.fctl[i]=0;__threadfence();}
}
__global__ void gr8_finalize(K1Args a){tr3_fin_body<false>(a,0);}
void run(std::vector<torch::Tensor> t){
 TORCH_CHECK(t.size()==19);
 for(auto &v:t)TORCH_CHECK(v.is_cuda() && v.is_contiguous());
 K1Args a{};a.M=1;a.B=8;a.KS1=2;a.WG=2;a.NS2=4;a.n_slots=8;a.smax=8;a.out_bf16=2;
 a.x=(__half*)t[0].data_ptr();a.slot_expert=(int*)t[1].data_ptr();a.tok_w=(float*)t[2].data_ptr();
 a.t_tab=(long long*)t[3].data_ptr();a.t_fc1=(uint8_t*)t[4].data_ptr();a.tf_up=(uint8_t*)t[5].data_ptr();a.t_fc2=(uint8_t*)t[6].data_ptr();
 a.t_v1=(__half*)t[7].data_ptr();a.tf_suhu=(__half*)t[8].data_ptr();a.t_v2=(__half*)t[9].data_ptr();a.t_v3=(__half*)t[10].data_ptr();
 a.slot_tok=(int*)t[11].data_ptr();a.tok_slot=(int*)t[12].data_ptr();a.tok_col=(int*)t[11].data_ptr();
 a.gpart=(float*)t[13].data_ptr();a.z=(__half*)t[14].data_ptr();a.ypart=(__half*)t[15].data_ptr();a.out=(__half*)t[16].data_ptr();a.fctl=(int*)t[17].data_ptr();
 auto s=at::cuda::getCurrentCUDAStream();gr8_fused<<<48,512,32768,s>>>(a);gr8_finalize<<<32,128,0,s>>>(a);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
