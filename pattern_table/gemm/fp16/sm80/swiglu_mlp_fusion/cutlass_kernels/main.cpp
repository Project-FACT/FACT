/***************************************************************************************************
 * PyBind11 bindings for SwiGLU MLP Fusion CUTLASS kernels
 *
 * Exports the following Python functions:
 *   - swiglu_gate_up: gate_proj + SiLU + up_proj + elementwise multiply
 *   - swiglu_down: down_proj GEMM
 *   - swiglu_mlp_forward: complete SwiGLU MLP forward pass
 **************************************************************************************************/
#include <torch/extension.h>

// Forward declarations from swiglu_mlp_fusion.cu
torch::Tensor swiglu_gate_up(
    torch::Tensor input,
    torch::Tensor gate_weight,
    torch::Tensor up_weight);

torch::Tensor swiglu_down(
    torch::Tensor input,
    torch::Tensor weight);

torch::Tensor swiglu_mlp_forward(
    torch::Tensor input,
    torch::Tensor gate_weight,
    torch::Tensor up_weight,
    torch::Tensor down_weight);

///////////////////////////////////////////////////////////////////////////////////////////////////
// PyBind11 module definition
///////////////////////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUTLASS FP16 Tensor Core kernels for SwiGLU MLP fusion";

  // Function: gate_proj + SiLU + up_proj + elementwise multiply
  m.def(
      "swiglu_gate_up",
      &swiglu_gate_up,
      py::arg("input"),
      py::arg("gate_weight"),
      py::arg("up_weight"),
      "Computes SiLU(input @ gate_weight.T) * (input @ up_weight.T)"
  );

  // Function: down_proj GEMM
  m.def(
      "swiglu_down",
      &swiglu_down,
      py::arg("input"),
      py::arg("weight"),
      "Computes input @ weight.T (down projection)"
  );

  // Function: complete SwiGLU MLP forward pass
  m.def(
      "swiglu_mlp_forward",
      &swiglu_mlp_forward,
      py::arg("input"),
      py::arg("gate_weight"),
      py::arg("up_weight"),
      py::arg("down_weight"),
      "Computes SwiGLU MLP: gate_proj + SiLU + up_proj + multiply + down_proj"
  );
}
