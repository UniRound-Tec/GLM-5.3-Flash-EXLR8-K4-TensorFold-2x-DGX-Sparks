#include <torch/extension.h>
void exl3_atomic_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                      const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                      const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t,
                      double, int64_t, int64_t, bool, const c10::optional<at::Tensor>&, const at::Tensor&);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("prompt_atomic", &exl3_atomic_cuda);}
