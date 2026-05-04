/***************************************************************************************************
 * Python bindings for CUTLASS FP16 Stream-K GEMM
 **************************************************************************************************/

#include <torch/extension.h>

torch::Tensor streamk_gemm(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("streamk_gemm", &streamk_gemm,
          "FP16 Stream-K GEMM on Ampere (A100) with load-balanced scheduling - C = A * B");
}
