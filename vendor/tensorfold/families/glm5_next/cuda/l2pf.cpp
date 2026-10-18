#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void l2pf_cuda(const at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, const at::Tensor&);

// Prefetch TABLE[first : first + count] ((address, bytes) int64 pieces) into L2; see l2pf.cu.
void prefetch(const at::Tensor& table, int64_t first, int64_t count, int64_t mode, int64_t blocks, int64_t threads,
              const at::Tensor& sink) {
    TORCH_CHECK(table.is_cuda() && table.scalar_type() == at::kLong && table.is_contiguous() && table.dim() == 2 &&
                table.size(1) == 2, "l2pf: table [n, 2] int64 on the GPU");
    TORCH_CHECK(first >= 0 && count >= 0 && first + count <= table.size(0), "l2pf: pieces out of the table");
    TORCH_CHECK(mode >= 0 && mode <= 2 && blocks >= 1 && threads >= 32 && threads % 32 == 0, "l2pf: launch");
    TORCH_CHECK(sink.is_cuda() && sink.scalar_type() == at::kInt, "l2pf: sink int32 on the GPU");
    c10::cuda::CUDAGuard guard(table.device());
    l2pf_cuda(table, first, count, mode, blocks, threads, sink);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prefetch", &prefetch);
}
