// Lane matmul: acc = fma(xs, b, fma(P, s, acc)) per group in order, K slices set by shape, so no row affects another.

#include <ATen/ATen.h>
#include <algorithm>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "qmm_frag.cuh"

namespace {

using namespace qmm_frag;

// Tile shapes: BM rows by BN columns a block, WM x WN warps, each warp (BM / WM) x (BN / WN).
template <int GS, int BM, int BN, int WM, int WN, int STAGES>
struct Tile {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;               // m16 tiles a warp
    static constexpr int NT = BN / WN / 8;                // n8 tiles a warp
    static constexpr int ROW = GS * 2;                    // bytes of one input row a group
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;                    // stage bytes: inputs,
    static constexpr int W = BN * GS / 2;                 // weights,
    static constexpr int S = BN * 2;                      // scales, biases (bf16),
    static constexpr int XS = BM * 4;                     // and input sums (fp32)
    static constexpr int STAGE = X + W + 2 * S + XS;
    static constexpr int PARTIALS = MT * NT * 4 * THREADS * 4;   // a K slice's partial, parked for the cluster sum
    static constexpr int SMEM = STAGES * STAGE > PARTIALS ? STAGES * STAGE : PARTIALS;
};

template <int GS, int BM, int BN, int WM, int WN, int STAGES, bool F32, bool CLUSTER, bool PIPE = false>
__global__ void __launch_bounds__(WM * WN * 32) paired_kernel(
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const uint32_t* __restrict__ w,
        const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases,
        void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad, int ldx, int group,
        const __nv_bfloat16* x1, const float* xs1, const uint32_t* w1,
        const __nv_bfloat16* scales1, const __nv_bfloat16* biases1, void* out1) {
    if (blockIdx.y) { x=x1; xs=xs1; w=w1; scales=scales1; biases=biases1; out=out1; }
    using T = Tile<GS, BM, BN, WM, WN, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, per = KG / SK;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y, slice = blockIdx.z, g0 = slice * per;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    // one group's inputs, weights, scales, biases and input sums into stage s
    auto load = [&](int s, int g, int m0) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8,
                  m0 + r < M);
        }
        unsigned char* pw = p + T::X;
        constexpr int TILE_BYTES = 64 * GS / 2;           // one stored 64-column tile's group block
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            if constexpr (BN >= 64) {
                const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
                const size_t tile = static_cast<size_t>(n0 / 64 + t) * KG + g;
                cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + tile * TILE_BYTES + off * 16);
            } else {                                      // part of a stored tile: its n8 tiles are contiguous
                const size_t tile = static_cast<size_t>(n0 / 64) * KG + g;
                cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + tile * TILE_BYTES + (n0 % 64) * GS / 2 +
                                  c * 16);
            }
        }
        unsigned char* ps = pw + T::W;
        for (int c = tid; c < 2 * (T::S / 16); c += T::THREADS) {
            const int which = c / (T::S / 16), off = c % (T::S / 16);
            const __nv_bfloat16* src = (which ? biases : scales) + static_cast<size_t>(g) * npad + n0 + off * 8;
            cp16(ps + which * T::S + off * 16, src);
        }
        float* px = reinterpret_cast<float*>(ps + 2 * T::S);
        for (int r = tid; r < BM; r += T::THREADS)
            cp4(px + r, xs + static_cast<size_t>(min(m0 + r, M - 1)) * KG + g);
    };

    {
        float acc[T::MT][T::NT][4];
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
        // acc = fma(xs, b, fma(p, s, acc)) per output, the order every row gets
        auto epilogue = [&](float (&acc_)[T::MT][T::NT][4], const float (&d_)[T::MT][T::NT][4],
                            const float (&s_)[T::NT][2], const float (&b_)[T::NT][2], const float (&x_)[T::MT][2]) {
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int i = 0; i < T::MT; ++i)
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        acc_[i][j][e] = __fmaf_rn(x_[i][e >> 1], b_[j][e & 1], __fmaf_rn(d_[i][j][e], s_[j][e & 1],
                                                                                       acc_[i][j][e]));
        };
        float dp[T::MT][T::NT][4], sp[T::NT][2], bp[T::NT][2], xp[T::MT][2];
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < per) load(s, g0 + s, m0);
            commit();
        }
        for (int it = 0; it < per; ++it) {
            wait<STAGES - 2>();
            __syncthreads();
            const int next = it + STAGES - 1;
            if (next < per) load(next % STAGES, g0 + next, m0);
            commit();
            const unsigned char* p = stage(it % STAGES);
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
            const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::W);
            const float* px = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S);
            uint32_t words[T::NT][GS / 32];
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int v = 0; v < GS / 32; ++v)
                    words[j][v] = pw[((wn * T::NT + j) * 32 + lane) * (GS / 32) + v];
            float d[T::MT][T::NT][4];
#pragma unroll
            for (int kt = 0; kt < GS / 16; ++kt) {
                uint32_t a[T::MT][4];
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int ch = kt * 2 + (lane >> 4);
                    ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
                }
#pragma unroll
                for (int j = 0; j < T::NT; ++j) {
                    const uint32_t b0 = pair(words[j][kt / 2], (kt & 1) * 8);
                    const uint32_t b1 = pair(words[j][kt / 2], (kt & 1) * 8 + 4);
#pragma unroll
                    for (int i = 0; i < T::MT; ++i) {
                        if (kt == 0) mma0(d[i][j], a[i], b0, b1);
                        else mma(d[i][j], a[i], b0, b1);
                    }
                }
                if (PIPE && kt == 0 && it > 0) epilogue(acc, dp, sp, bp, xp);      // the previous group's, between MMAs
            }
            float sv[T::NT][2], bv[T::NT][2], xv[T::MT][2];
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const int col = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
                const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + col);
                const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(ps + BN + col);
                sv[j][0] = __low2float(s2); sv[j][1] = __high2float(s2);
                bv[j][0] = __low2float(b2); bv[j][1] = __high2float(b2);
            }
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int row = wm * (BM / WM) + i * 16 + (lane >> 2);
                xv[i][0] = px[row];
                xv[i][1] = px[row + 8];
            }
            if constexpr (PIPE) {
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
#pragma unroll
                    for (int j = 0; j < T::NT; ++j)
#pragma unroll
                        for (int e = 0; e < 4; ++e) dp[i][j][e] = d[i][j][e];
                    xp[i][0] = xv[i][0]; xp[i][1] = xv[i][1];
                }
#pragma unroll
                for (int j = 0; j < T::NT; ++j) {
                    sp[j][0] = sv[j][0]; sp[j][1] = sv[j][1]; bp[j][0] = bv[j][0]; bp[j][1] = bv[j][1];
                }
            } else {
                epilogue(acc, d, sv, bv, xv);
            }
        }
        if (PIPE && per > 0) epilogue(acc, dp, sp, bp, xp);
        wait<0>();
        __syncthreads();
        if constexpr (CLUSTER) {
#if __CUDA_ARCH__ < 900
            __trap();                                     // no clusters before sm_90: the host never launches this
#else
            // K slices of a tile form a cluster: slice 0 adds peers' partials in slice order, as reduce_kernel does
            constexpr int E = T::MT * T::NT * 4;
            auto cluster = cooperative_groups::this_cluster();
            float* mine = reinterpret_cast<float*>(buf);
            if (slice != 0) {
#pragma unroll
                for (int i = 0; i < T::MT; ++i)
#pragma unroll
                    for (int j = 0; j < T::NT; ++j)
#pragma unroll
                        for (int e = 0; e < 4; ++e) mine[((i * T::NT + j) * 4 + e) * T::THREADS + tid] = acc[i][j][e];
            }
            cluster.sync();
            if (slice == 0) {
                for (int peer = 1; peer < SK; ++peer) {
                    const float* theirs = cluster.map_shared_rank(mine, peer);
#pragma unroll
                    for (int i = 0; i < T::MT; ++i)
#pragma unroll
                        for (int j = 0; j < T::NT; ++j)
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                                acc[i][j][e] = acc[i][j][e] + theirs[((i * T::NT + j) * 4 + e) * T::THREADS + tid];
                }
            }
            cluster.sync();                               // peers keep their memory until slice 0 has read it
            if (slice != 0) return;
#endif
        }
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                    if (row >= M) continue;
                    const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                    if (SK > 1 && !CLUSTER) {
                        float* dst = part + (static_cast<size_t>(slice) * M + row) * N + col;
                        if (col < N) dst[0] = v0;
                        if (col + 1 < N) dst[1] = v1;
                    } else if (F32) {
                        float* dst = reinterpret_cast<float*>(out) + static_cast<size_t>(row) * N + col;
                        if (col < N) dst[0] = v0;
                        if (col + 1 < N) dst[1] = v1;
                    } else {
                        __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
                        if (col + 1 < N && (N & 1) == 0)
                            *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                        else {
                            if (col < N) dst[0] = __float2bfloat16_rn(v0);
                            if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                        }
                    }
                }
            }
    }
}


template <int BM>
void paired_launch(const at::Tensor& x0,const at::Tensor& x1,const at::Tensor& xs0,const at::Tensor& xs1,
 const at::Tensor& w0,const at::Tensor& s0,const at::Tensor& b0,const at::Tensor& w1,const at::Tensor& s1,
 const at::Tensor& b1,at::Tensor& y0,at::Tensor& y1,int N) {
 using T=Tile<64,BM,64,1,4,4>;
 auto kernel=paired_kernel<64,BM,64,1,4,4,false,false>;
 static bool configured=false;
 if(!configured){C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,T::SMEM));configured=true;}
 const int M=x0.size(0),K=x0.size(1),rows=(M+BM-1)/BM;
 cudaLaunchConfig_t config={};config.gridDim=dim3(rows*((N+63)/64),2,1);config.blockDim=dim3(T::THREADS);
 config.dynamicSmemBytes=T::SMEM;config.stream=at::cuda::getCurrentCUDAStream();
 C10_CUDA_CHECK(cudaLaunchKernelEx(&config,kernel,
  reinterpret_cast<const __nv_bfloat16*>(x0.data_ptr()),xs0.data_ptr<float>(),
  reinterpret_cast<const uint32_t*>(w0.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(s0.data_ptr()),
  reinterpret_cast<const __nv_bfloat16*>(b0.data_ptr()),y0.data_ptr(),static_cast<float*>(nullptr),
  M,N,K,1,static_cast<int>(s0.stride(0)),M==1?K:static_cast<int>(x0.stride(0)),rows,
  reinterpret_cast<const __nv_bfloat16*>(x1.data_ptr()),xs1.data_ptr<float>(),
  reinterpret_cast<const uint32_t*>(w1.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(s1.data_ptr()),
  reinterpret_cast<const __nv_bfloat16*>(b1.data_ptr()),y1.data_ptr()));
}
} // namespace
void paired_cuda(const at::Tensor& x0,const at::Tensor& x1,const at::Tensor& xs0,const at::Tensor& xs1,
 const at::Tensor& w0,const at::Tensor& s0,const at::Tensor& b0,const at::Tensor& w1,const at::Tensor& s1,
 const at::Tensor& b1,at::Tensor& y0,at::Tensor& y1,int N) {
 TORCH_CHECK(x0.is_cuda()&&x1.is_cuda()&&x0.scalar_type()==at::kBFloat16&&x1.scalar_type()==at::kBFloat16);
 TORCH_CHECK(x0.sizes()==x1.sizes()&&x0.size(1)==128&&x0.size(0)>=1&&x0.size(0)<=32);
 TORCH_CHECK(x0.stride(0)==x1.stride(0)&&x0.stride(1)==1&&x1.stride(1)==1);
 TORCH_CHECK(y0.is_contiguous()&&y1.is_contiguous()&&y0.scalar_type()==at::kBFloat16&&y1.scalar_type()==at::kBFloat16);
 TORCH_CHECK(y0.size(0)==x0.size(0)&&y1.sizes()==y0.sizes()&&y0.size(1)==N&&N%64==0);
 TORCH_CHECK(s0.stride(0)==s1.stride(0)&&s0.sizes()==s1.sizes()&&w0.sizes()==w1.sizes());
 c10::cuda::CUDAGuard guard(x0.device());
 if(x0.size(0)<=16) paired_launch<16>(x0,x1,xs0,xs1,w0,s0,b0,w1,s1,b1,y0,y1,N);
 else paired_launch<32>(x0,x1,xs0,xs1,w0,s0,b0,w1,s1,b1,y0,y1,N);
}
