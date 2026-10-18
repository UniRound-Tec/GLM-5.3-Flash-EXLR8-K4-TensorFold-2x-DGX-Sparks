// L2 prefetch of weight ranges for GLM-5.3-Flash's decode windows (l2pf.py): a small kernel on a side stream asks
// the memory system to bring the next kernels' weights into L2 while the main stream waits on an all-gather or runs
// latency-bound glue. It only loads or prefetches (no store, atomic or reduction), so no result can change.
//
// TABLE [n, 2] int64: (address, bytes) pieces, each 16-byte aligned and at most a few hundred KiB (the host splits).
// MODE 0 ("bulk"): one cp.async.bulk.prefetch.L2 a piece (the TMA unit walks it; sm_90+).
// MODE 1 ("lines"): prefetch.global.L2::evict_last on every 128-byte line of a piece, a warp's lanes on adjacent lines.
// MODE 2 ("touch"): ld.global.cg of every line's first 16 bytes (a real load: the kernel lasts as long as the reads).
//
// Adapted from jayleaton/glm53-tensorfold-spark patches/0460 (l2pf.cu; Apache-2.0, Copyright 2026 Jay Leaton).
// Changes: one kernel for the three modes over a table of pre-split pieces (the host cuts ranges into pieces, so no
// per-thread chunk loop), lines mode a warp a piece with prefetch.global.L2::evict_last, touch mode as loads here.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

namespace {

__global__ void l2pf_kernel(const long long* __restrict__ table, int n, int mode, unsigned* __restrict__ sink) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int warps = (gridDim.x * blockDim.x) >> 5;
    if (mode == 0) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
            const unsigned long long a = (unsigned long long)table[2 * i];
            const unsigned bytes = (unsigned)table[2 * i + 1];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(a), "r"(bytes) : "memory");
#else
            for (unsigned o = 0; o < bytes; o += 128)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(a + o));
#endif
        }
        return;
    }
    unsigned acc = 0;
    for (int i = warp; i < n; i += warps) {             // a warp a piece, its lanes on adjacent lines
        const unsigned long long a = (unsigned long long)table[2 * i];
        const unsigned bytes = (unsigned)table[2 * i + 1];
        for (unsigned o = lane * 128u; o < bytes; o += 32u * 128u) {
            if (mode == 1) {
                asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(a + o));
            } else {
                unsigned x, y, z, w;
                asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(a + o));
                acc ^= x ^ y ^ z ^ w;
            }
        }
    }
    if (mode == 2 && acc == 0x9e3779b9u && sink != nullptr) sink[0] = acc;   // keeps the loads; never true in practice
}

}  // namespace

void l2pf_cuda(const at::Tensor& table, int64_t first, int64_t count, int64_t mode, int64_t blocks, int64_t threads,
               const at::Tensor& sink) {
    if (count <= 0) return;
    const long long* t = reinterpret_cast<const long long*>(table.data_ptr<int64_t>()) + 2 * first;
    l2pf_kernel<<<(unsigned)blocks, (unsigned)threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        t, (int)count, (int)mode, reinterpret_cast<unsigned*>(sink.data_ptr<int>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
