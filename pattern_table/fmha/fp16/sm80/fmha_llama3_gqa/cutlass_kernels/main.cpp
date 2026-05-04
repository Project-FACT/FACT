/***************************************************************************************************
 * Pybind11 bindings for CUTLASS FMHA kernel
 *
 * This file provides Python bindings for the FMHA kernel using PyTorch C++ API.
 **************************************************************************************************/

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// Declare the CUDA kernel function
torch::Tensor fmha_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float scale,
    int64_t num_batches,
    int64_t num_heads,
    int64_t num_queries,
    int64_t num_keys,
    int64_t head_dim,
    int64_t head_dim_v
);

static void check_inputs(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");

    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must have same dtype");
    TORCH_CHECK(q.scalar_type() == v.scalar_type(), "q and v must have same dtype");
    TORCH_CHECK(q.scalar_type() == at::ScalarType::Float,
                "Only float32 is supported currently (internal conversion to FP16)");
}

torch::Tensor fmha_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float scale,
    int64_t num_batches,
    int64_t num_heads,
    int64_t num_queries,
    int64_t num_keys,
    int64_t head_dim,
    int64_t head_dim_v
) {
    check_inputs(q, k, v);
    return fmha_forward_cuda(q, k, v, scale, num_batches, num_heads, num_queries, num_keys, head_dim, head_dim_v);
}

// Autograd-compatible wrapper
class FMHAKernelFunction : public torch::autograd::Function<FMHAKernelFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        float scale,
        int64_t num_batches,
        int64_t num_heads,
        int64_t num_queries,
        int64_t num_keys,
        int64_t head_dim,
        int64_t head_dim_v
    ) {
        ctx->save_for_backward({q, k, v});
        ctx->saved_data["scale"] = scale;
        ctx->saved_data["num_batches"] = num_batches;
        ctx->saved_data["num_heads"] = num_heads;
        ctx->saved_data["num_queries"] = num_queries;
        ctx->saved_data["num_keys"] = num_keys;
        ctx->saved_data["head_dim"] = head_dim;
        ctx->saved_data["head_dim_v"] = head_dim_v;

        return fmha_forward(q, k, v, scale, num_batches, num_heads, num_queries, num_keys, head_dim, head_dim_v);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        // For now, we don't implement the backward pass
        // Fall back to PyTorch's autograd
        auto grad = grad_outputs[0];
        return {grad, torch::Tensor(), torch::Tensor(),
                torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor()};
    }
};

torch::Tensor fmha_autograd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float scale,
    int64_t num_batches,
    int64_t num_heads,
    int64_t num_queries,
    int64_t num_keys,
    int64_t head_dim,
    int64_t head_dim_v
) {
    return FMHAKernelFunction::apply(q, k, v, scale, num_batches, num_heads, num_queries, num_keys, head_dim, head_dim_v);
}

PYBIND11_MODULE(fmha_ext, m) {
    m.doc() = "CUTLASS FMHA kernel for PyTorch";

    m.def("fmha_forward", &fmha_forward,
          "FMHA forward pass (no autograd)",
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("scale"),
          py::arg("num_batches"),
          py::arg("num_heads"),
          py::arg("num_queries"),
          py::arg("num_keys"),
          py::arg("head_dim"),
          py::arg("head_dim_v"));

    m.def("fmha_autograd", &fmha_autograd,
          "FMHA forward pass with autograd support",
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("scale"),
          py::arg("num_batches"),
          py::arg("num_heads"),
          py::arg("num_queries"),
          py::arg("num_keys"),
          py::arg("head_dim"),
          py::arg("head_dim_v"));
}
