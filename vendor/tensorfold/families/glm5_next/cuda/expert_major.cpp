#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
void exlr8_dec8_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                   const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t,
                   int64_t, int64_t, int64_t, int64_t, bool, const at::Tensor&, const at::Tensor&,
                   const at::Tensor&, at::Tensor&, double, at::Tensor&, int64_t);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), name,
                ": expected a contiguous CUDA tensor of the right dtype");
}

void dec(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& T0, const at::Tensor& T1, const at::Tensor& items,
         const at::Tensor& counts, const at::Tensor& members, at::Tensor Z, int64_t mats, int64_t K, int64_t N,
         int64_t P, int64_t SK, int64_t max_items, int64_t slots, int64_t fuse, bool xrow,
         const at::Tensor& sv0, const at::Tensor& sv1, const at::Tensor& su2, at::Tensor out, double limit,
         at::Tensor done, int64_t ld) {
    const int64_t E = T0.size(0);
    for (auto* t : {&X0, &X1, &sv0, &sv1, &su2}) check(*t, at::kHalf, "dec: fp16 input");
    for (auto* t : {&T0, &T1, &items, &counts, &members}) check(*t, at::kInt, "dec: int32 input");
    check(done, at::kInt, "done");
    check(Z, at::kFloat, "Z");
    TORCH_CHECK(fuse == 1 || Z.numel() >= mats * SK * P * N, "Z too small");
    TORCH_CHECK(fuse != 1 || (out.scalar_type() == at::kFloat && out.numel() >= P * N), "dec: Y fp32 [P, N]");
    TORCH_CHECK(fuse != 2 || (out.scalar_type() == at::kHalf && out.numel() >= P * N &&
                              done.numel() >= max_items * (N / 128)), "dec: Xd fp16 [P, N] and done counts");
    // loads 1..3 read every block's item row before its count check (one round trip): items sized for the grid
    TORCH_CHECK(ld == 0 || items.numel() >= 3 * max_items, "dec: items [max_items, 3] for the 16-byte load path");
    c10::cuda::CUDAGuard guard(X0.device());
    exlr8_dec8_cuda(X0, X1, T0, T1, items, counts, members, Z, mats, K, N, P, SK, max_items, E, slots, fuse, xrow,
                  sv0, sv1, su2, out, limit, done, ld);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("dec", &dec); }
