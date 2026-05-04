/***************************************************************************************************
 * PyTorch bindings for MLP GEMM Fusion with GELU Activation
 **************************************************************************************************/
#include <torch/extension.h>

// Forward declarations from the CUDA file
torch::Tensor mlp_fc_gelu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias);

torch::Tensor mlp_proj(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias);

torch::Tensor mlp_forward(
    torch::Tensor input,
    torch::Tensor fc_weight,
    torch::Tensor fc_bias,
    torch::Tensor proj_weight,
    torch::Tensor proj_bias);

///////////////////////////////////////////////////////////////////////////////////////////////////
// PyBind11 module definition
///////////////////////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mlp_fc_gelu", &mlp_fc_gelu,
        "MLP first layer GEMM with fused GELU activation",
        py::arg("input"), py::arg("weight"), py::arg("bias"));

  m.def("mlp_proj", &mlp_proj,
        "MLP projection layer GEMM (no activation)",
        py::arg("input"), py::arg("weight"), py::arg("bias"));

  m.def("mlp_forward", &mlp_forward,
        "Complete MLP forward pass: c_fc -> GELU -> c_proj",
        py::arg("input"), py::arg("fc_weight"), py::arg("fc_bias"),
        py::arg("proj_weight"), py::arg("proj_bias"));
}
