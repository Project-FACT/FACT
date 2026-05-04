/***************************************************************************************************
 * Python bindings for CUTLASS TF32 Stream-K Batched GEMM
 **************************************************************************************************/

#include <torch/extension.h>

torch::Tensor batched_streamk_tf32_gemm(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_streamk_tf32_gemm", &batched_streamk_tf32_gemm,
          "TF32 Stream-K Batched GEMM on Ampere (A100) — C[b] = A[b] * B[b]");
}
