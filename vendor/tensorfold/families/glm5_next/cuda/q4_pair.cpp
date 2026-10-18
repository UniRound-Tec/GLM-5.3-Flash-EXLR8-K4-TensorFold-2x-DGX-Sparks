#include <torch/extension.h>
void paired_cuda(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,at::Tensor&,at::Tensor&,int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("pair",&paired_cuda);}
