#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void roce_gather_cuda(const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                      int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                      int64_t, int64_t);

void gather(const at::Tensor& in, at::Tensor out, int64_t shard_packs, int64_t nbytes, int64_t row_packs,
            int64_t recv_base, int64_t flag_base, int64_t send_base, int64_t ctrl, int64_t slot_bytes, int64_t epoch,
            int64_t stage_ctr, int64_t tail_ctr, int64_t poison, int64_t spin_limit, int64_t world, int64_t rank,
            int64_t slots, int64_t flag_stride, int64_t n_hca, int64_t grid, int64_t threads) {
    TORCH_CHECK(in.is_cuda() && out.is_cuda() && in.is_contiguous() && out.is_contiguous(),
                "roce gather: contiguous CUDA tensors");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(in.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "roce gather: 16-byte aligned tensors");
    const at::cuda::CUDAGuard guard(in.device());
    roce_gather_cuda(in, out, shard_packs, nbytes, row_packs, recv_base, flag_base, send_base, ctrl, slot_bytes, epoch,
                     stage_ctr, tail_ctr, poison, spin_limit, world, rank, slots, flag_stride, n_hca, grid, threads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather", &gather, "one-shot RoCE all-gather (b12x's RoCEnante protocol)");
}
