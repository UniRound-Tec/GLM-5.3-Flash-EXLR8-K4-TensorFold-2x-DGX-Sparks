// One-shot RoCE all-gather between DGX Sparks: the GPU side of b12x's "RoCEnante" protocol
// (https://github.com/local-inference-lab/b12x, b12x/comm/roce/_allgather_cute.py at commit
// 8a99d639410e39d5f39cb4037675331beceea1d4, Copyright the b12x contributors, Apache License 2.0),
// reimplemented in CUDA C++ from its CuTe DSL kernel with the same stages and memory orderings.
//
// The pinned host region (roce_proxy.c) holds, per rank: recv[src][slot], flag[src][slot][hca], send[slot] and a
// control record {seq, nbytes, error seq, missing peer, nbytes of slot 0, nbytes of slot 1, missing hca}. One launch:
//   1. every block stages its share of the local shard into send[seq & 1];
//   2. the last block to finish staging (a free-running arrival counter) publishes nbytes and rings ctrl.seq after
//      a system fence; the proxy thread RDMA-writes the payload to every peer, then a flag per HCA stripe;
//   3. one thread per (peer, HCA) spins on that stripe's flag with system-scope acquire loads (a limit poisons);
//   4. every block copies the shards in rank order: its own from the input, the peers' from the NIC-written slots;
//   5. the last block to finish advances the device epoch, so CUDA graphs replay the exchange.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ uint32_t ld_relaxed_gpu(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t ld_relaxed_sys(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint4 ld_relaxed_sys_v4(const uint4* p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_relaxed_sys(uint32_t* p, uint32_t v) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void st_release_gpu(uint32_t* p, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t atom_add_relaxed_gpu(uint32_t* p, uint32_t v) {
    uint32_t old;
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}
__device__ __forceinline__ void fence_sc_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }
__device__ __forceinline__ void fence_sc_gpu() { asm volatile("fence.sc.gpu;" ::: "memory"); }

// ctrl record, in 32-bit words
constexpr int C_SEQ = 0, C_NBYTES = 1, C_ERROR = 2, C_PEER = 3, C_SLOT_NBYTES = 4, C_HCA = 6;

__global__ void gather_kernel(const uint4* __restrict__ in, uint4* __restrict__ out, int shard_packs, uint32_t nbytes,
                              int row_packs, char* recv_base, char* flag_base, char* send_base, uint32_t* ctrl,
                              long long slot_bytes, uint32_t* epoch, uint32_t* stage_ctr, uint32_t* tail_ctr,
                              uint32_t* poison, uint32_t spin_limit, int world, int rank, int slots, int flag_stride,
                              int n_hca) {
    const int tid = threadIdx.x;
    const uint32_t seq = ld_relaxed_gpu(epoch) + 1u;
    const uint32_t slot = seq & 1u;
    uint4* send_slot = reinterpret_cast<uint4*>(send_base + (long long)slot * slot_bytes);
    const int index = blockIdx.x * blockDim.x + tid, stride = gridDim.x * blockDim.x;
    // a recorded timeout poisons the runtime: later launches do nothing until the host raises
    if (ld_relaxed_gpu(poison) != 0u) return;

    for (int i = index; i < shard_packs; i += stride) send_slot[i] = in[i];                         // 1. stage
    __syncthreads();
    if (tid == 0) {                                                                                 // 2. doorbell
        fence_sc_sys();
        const uint32_t prior = atom_add_relaxed_gpu(stage_ctr, 1u);
        if ((prior + 1u) % gridDim.x == 0u) {
            st_relaxed_sys(ctrl + C_NBYTES, nbytes);
            st_relaxed_sys(ctrl + C_SLOT_NBYTES + slot, nbytes);
            fence_sc_sys();
            st_relaxed_sys(ctrl + C_SEQ, seq);
        }
    }
    if (tid < world * n_hca) {                                                                      // 3. wait
        const int peer = tid / n_hca, hca = tid - peer * n_hca;
        if (peer != rank) {
            const uint32_t* flag = reinterpret_cast<const uint32_t*>(
                flag_base + (((long long)peer * slots + slot) * n_hca + hca) * flag_stride);
            uint32_t polls = 0;
            while (ld_acquire_sys(flag) != seq) {
                if (++polls >= spin_limit) {
                    st_relaxed_sys(ctrl + C_PEER, (uint32_t)peer);
                    st_relaxed_sys(ctrl + C_HCA, (uint32_t)hca);
                    st_relaxed_sys(ctrl + C_ERROR, seq);
                    st_release_gpu(poison, seq);
                    break;
                }
            }
        }
    }
    __syncthreads();
    if (ld_relaxed_gpu(poison) == 0u) {                                                             // 4. copy out
        const int out_row_packs = world * row_packs;
        for (int source = 0; source < world; ++source) {
            const uint4* peer_slot = reinterpret_cast<const uint4*>(
                recv_base + ((long long)source * slots + slot) * slot_bytes);
            for (int i = index; i < shard_packs; i += stride) {
                const int row = i / row_packs, col = i - row * row_packs;
                const uint4 w = source == rank ? in[i] : ld_relaxed_sys_v4(peer_slot + i);
                out[(long long)row * out_row_packs + (long long)source * row_packs + col] = w;
            }
        }
    }
    fence_sc_gpu();                                                                                 // 5. epoch
    __syncthreads();
    if (tid == 0) {
        const uint32_t prior = atom_add_relaxed_gpu(tail_ctr, 1u);
        if ((prior + 1u) % gridDim.x == 0u) {
            fence_sc_gpu();
            if (ld_relaxed_sys(ctrl + C_ERROR) == 0u) st_release_gpu(epoch, seq);
        }
    }
}

}  // namespace

void roce_gather_cuda(const at::Tensor& in, at::Tensor& out, int64_t shard_packs, int64_t nbytes, int64_t row_packs,
                      int64_t recv_base, int64_t flag_base, int64_t send_base, int64_t ctrl, int64_t slot_bytes,
                      int64_t epoch, int64_t stage_ctr, int64_t tail_ctr, int64_t poison, int64_t spin_limit,
                      int64_t world, int64_t rank, int64_t slots, int64_t flag_stride, int64_t n_hca, int64_t grid,
                      int64_t threads) {
    auto stream = at::cuda::getCurrentCUDAStream();
    gather_kernel<<<(unsigned)grid, (unsigned)threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(in.data_ptr()), reinterpret_cast<uint4*>(out.data_ptr()), (int)shard_packs,
        (uint32_t)nbytes, (int)row_packs, reinterpret_cast<char*>(recv_base), reinterpret_cast<char*>(flag_base),
        reinterpret_cast<char*>(send_base), reinterpret_cast<uint32_t*>(ctrl), (long long)slot_bytes,
        reinterpret_cast<uint32_t*>(epoch), reinterpret_cast<uint32_t*>(stage_ctr),
        reinterpret_cast<uint32_t*>(tail_ctr), reinterpret_cast<uint32_t*>(poison), (uint32_t)spin_limit, (int)world,
        (int)rank, (int)slots, (int)flag_stride, (int)n_hca);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
