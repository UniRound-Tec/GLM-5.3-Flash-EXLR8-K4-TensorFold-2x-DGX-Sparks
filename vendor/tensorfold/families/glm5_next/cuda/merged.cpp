#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void exl3_grouped_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                       const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t,
                       int64_t, int64_t, int64_t, int64_t, bool, bool, bool);
void exl3_rot_in_cuda(const at::Tensor&, int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                      at::Tensor&, int64_t, int64_t, int64_t);
void exl3_gateup_epilogue_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                               const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, double);
void exl3_down_epilogue_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t,
                             int64_t, int64_t, int64_t);

void exl3_prompt_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                      const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                      const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t,
                      double, int64_t, int64_t, bool, const c10::optional<at::Tensor>&);
void exl3_rot_rows_cuda(const at::Tensor&, int64_t, const at::Tensor&, at::Tensor&, int64_t, int64_t);
void exl3_dec_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                   const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t,
                   int64_t, int64_t, int64_t, int64_t, bool, const at::Tensor&, const at::Tensor&,
                   const at::Tensor&, at::Tensor&, double, at::Tensor&, int64_t);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), name,
                ": expected a contiguous CUDA tensor of the right dtype");
}

void grouped(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& T0, const at::Tensor& T1,
             const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members, at::Tensor Z, int64_t mats,
             int64_t K, int64_t N, int64_t P, int64_t SK, int64_t max_items, int64_t nt, int64_t warps, bool nfirst,
             bool pf, bool seq) {
    const int64_t E = T0.size(0);
    check(X0, at::kHalf, "X0");
    check(X1, at::kHalf, "X1");
    check(T0, at::kInt, "T0");
    check(T1, at::kInt, "T1");
    check(items, at::kInt, "items");
    check(counts, at::kInt, "counts");
    check(members, at::kInt, "members");
    check(Z, at::kFloat, "Z");
    TORCH_CHECK(Z.numel() >= mats * SK * P * N, "Z too small");
    c10::cuda::CUDAGuard guard(X0.device());
    exl3_grouped_cuda(X0, X1, T0, T1, items, counts, members, Z, mats, K, N, P, SK, max_items, nt, warps, E, nfirst, pf, seq);
}

void rot_in(const at::Tensor& x, int64_t x_stride, const at::Tensor& pick, const at::Tensor& suh0,
            const at::Tensor& suh1, at::Tensor out0, at::Tensor out1, int64_t rows, int64_t K, int64_t slots) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16, "x: bf16 CUDA");
    check(pick, at::kInt, "pick");
    check(suh0, at::kHalf, "suh0");
    check(suh1, at::kHalf, "suh1");
    check(out0, at::kHalf, "out0");
    check(out1, at::kHalf, "out1");
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    c10::cuda::CUDAGuard guard(x.device());
    exl3_rot_in_cuda(x, x_stride, pick, suh0, suh1, out0, out1, rows, K, slots);
}

// Xh [rows, K] = rot_in's input rotation once a row, for a layer whose experts all share one suh for gate and up.
void rot_rows(const at::Tensor& x, int64_t x_stride, const at::Tensor& suh, at::Tensor out, int64_t rows, int64_t K) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16, "x: bf16 CUDA");
    check(suh, at::kHalf, "suh");
    check(out, at::kHalf, "out");
    TORCH_CHECK(K % 128 == 0 && suh.numel() == K && out.numel() >= rows * K, "rot_rows: shapes");
    c10::cuda::CUDAGuard guard(x.device());
    exl3_rot_rows_cuda(x, x_stride, suh, out, rows, K);
}

void gateup_epilogue(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_g, const at::Tensor& svh_u,
                     const at::Tensor& suh_d, at::Tensor xd, int64_t rows, int64_t P, int64_t N, int64_t SK,
                     int64_t slots, double limit) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_g, at::kHalf, "svh_g");
    check(svh_u, at::kHalf, "svh_u");
    check(suh_d, at::kHalf, "suh_d");
    check(xd, at::kHalf, "xd");
    TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128");
    c10::cuda::CUDAGuard guard(Z.device());
    exl3_gateup_epilogue_cuda(Z, pick, svh_g, svh_u, suh_d, xd, rows, P, N, SK, slots, limit);
}

void down_epilogue(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor y, int64_t rows,
                   int64_t P, int64_t D, int64_t SK, int64_t slots) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_d, at::kHalf, "svh_d");
    check(y, at::kFloat, "y");
    TORCH_CHECK(D % 128 == 0, "D must be a multiple of 128");
    c10::cuda::CUDAGuard guard(Z.device());
    exl3_down_epilogue_cuda(Z, pick, svh_d, y, rows, P, D, SK, slots);
}

void prompt(const at::Tensor& Xg, const at::Tensor& Xu, const at::Tensor& Tg, const at::Tensor& Tu,
            const at::Tensor& Td, const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members,
            const at::Tensor& svh_g, const at::Tensor& svh_u, const at::Tensor& suh_d, const at::Tensor& svh_d,
            at::Tensor xd, at::Tensor y, int64_t D, int64_t N, int64_t E, int64_t max_items, double limit,
            int64_t pass, int64_t slots, bool shx, c10::optional<at::Tensor> order) {
    const at::cuda::CUDAGuard guard(Xg.device());
    for (auto* t : {&Xg, &Xu, &svh_g, &svh_u, &suh_d, &svh_d}) check(*t, at::kHalf, "prompt: fp16 input");
    for (auto* t : {&Tg, &Tu, &Td, &items, &counts, &members}) check(*t, at::kInt, "prompt: int32 input");
    check(xd, at::kHalf, "prompt: xd");
    check(y, at::kFloat, "prompt: y");
    TORCH_CHECK(!shx || Xg.data_ptr() == Xu.data_ptr(), "prompt: shared rows take one Xh for gate and up");
    TORCH_CHECK(slots > 0, "prompt: slots");
    if (order.has_value()) check(*order, at::kInt, "prompt: order");
    exl3_prompt_cuda(Xg, Xu, Tg, Tu, Td, items, counts, members, svh_g, svh_u, suh_d, svh_d, xd, y, D, N, E,
                     max_items, limit, pass, slots, shx, order);
}

// Decode windows: the grouped kernel's sums; fuse 1 writes Y (down, its epilogue fused), 2 writes Xd
// (gate/up, the epilogue run by each (item, n block)'s last block; ``done`` zeroed ints, reset by the kernel).
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
    exl3_dec_cuda(X0, X1, T0, T1, items, counts, members, Z, mats, K, N, P, SK, max_items, E, slots, fuse, xrow,
                  sv0, sv1, su2, out, limit, done, ld);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped", &grouped);
    m.def("prompt", &prompt);
    m.def("rot_in", &rot_in);
    m.def("rot_rows", &rot_rows);
    m.def("gateup_epilogue", &gateup_epilogue);
    m.def("down_epilogue", &down_epilogue);
    m.def("dec", &dec);
}
