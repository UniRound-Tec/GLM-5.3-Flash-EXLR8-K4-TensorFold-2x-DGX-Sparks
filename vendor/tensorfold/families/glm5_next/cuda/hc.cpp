#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void hc_partial_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t);

// PART [rows, 16, 32] fp32 = glue._hc_partial's output for x [rows, 16384] bf16 and fn [24, 16384] bf16.
void hc_partial(const at::Tensor& x, const at::Tensor& fn, at::Tensor part, int64_t rows) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.size(1) == 16384,
                "hc_partial: x [rows, 16384] bf16, contiguous");
    TORCH_CHECK(fn.is_cuda() && fn.scalar_type() == at::kBFloat16 && fn.is_contiguous() && fn.size(0) == 24 &&
                fn.size(1) == 16384, "hc_partial: fn [24, 16384] bf16");
    TORCH_CHECK(part.is_cuda() && part.scalar_type() == at::kFloat && part.is_contiguous() &&
                part.numel() >= rows * 16 * 32, "hc_partial: part [rows, 16, 32] fp32");
    TORCH_CHECK(rows >= 1 && x.size(0) >= rows, "hc_partial: rows");
    c10::cuda::CUDAGuard guard(x.device());
    hc_partial_cuda(x, fn, part, rows);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hc_partial", &hc_partial);
}
