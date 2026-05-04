/***************************************************************************************************
 * Python bindings for CUTLASS TF32 Tensor Core GEMM
 **************************************************************************************************/

#include <torch/extension.h>

// Forward declaration of the CUDA function
torch::Tensor ampere_tf32_gemm(torch::Tensor A, torch::Tensor B);

// Python module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ampere_tf32_gemm", &ampere_tf32_gemm,
          "TF32 Tensor Core GEMM on Ampere (A100) - C = A * B");
}
