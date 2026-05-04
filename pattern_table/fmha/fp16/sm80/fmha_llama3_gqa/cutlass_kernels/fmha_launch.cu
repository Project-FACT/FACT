/***************************************************************************************************
 * CUTLASS FMHA Kernel Wrapper for PyTorch Extension
 *
 * Wraps the CUTLASS Fused Multi-Head Attention kernel for use with PyTorch.
 * Based on CUTLASS example 41_fused_multi_head_attention.
 *
 * Adapted for Llama 3 8B:
 * - head_dim = 128 (vs 96 in original)
 * - 32 heads, 2048 sequence length
 * - FP16 computation with FP32 accumulation
 *
 * GQA is handled via repeat_interleave BEFORE the FMHA kernel call,
 * so the kernel receives standard MHA inputs (32 Q heads, 32 K/V heads).
 **************************************************************************************************/

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// CUTLASS library headers - MUST be included first to define CUDA_STD_HEADER macro
#include "cutlass/cutlass.h"
#include "cutlass/half.h"

// Local FMHA example headers (copied from CUTLASS example 41)
// NOTE: These local headers expect CUDA_STD_HEADER to be defined above
#include "kernel_forward.h"

// Use FP16 for CUTLASS FMHA kernel (required for SM80 support)
using scalar_t = cutlass::half_t;
using accum_t = float;
using ArchTag = cutlass::arch::Sm80;

// Tile configuration for head_dim_v = 128 (Llama 3 8B)
// Based on CUTLASS example 41 logic:
// - If head_size_v > 64: kQueriesPerBlock=32, kKeysPerBlock=128
// - If head_size_v <= 64: kQueriesPerBlock=64, kKeysPerBlock=64
constexpr bool kIsAligned = true;       // Aligned memory (required by example)
constexpr int kQueriesPerBlock = 32;    // For head_dim_v > 64
constexpr int kKeysPerBlock = 128;      // For head_dim_v > 64
constexpr int kMaxK = 128;              // Max head_dim_v for Llama 3 8B (128)

using FMHAKernel = AttentionKernel<
    scalar_t,
    ArchTag,
    kIsAligned,
    kQueriesPerBlock,
    kKeysPerBlock,
    kMaxK
>;

torch::Tensor fmha_forward_cuda(
    torch::Tensor q,   // FP32 input: (B, T, nh, hs)
    torch::Tensor k,   // FP32 input: (B, T, nh, hs)
    torch::Tensor v,   // FP32 input: (B, T, nh, hs_v)
    float scale,
    int64_t num_batches,
    int64_t num_heads,
    int64_t num_queries,
    int64_t num_keys,
    int64_t head_dim,
    int64_t head_dim_v
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(k.scalar_type() == torch::kFloat32, "k must be float32");
    TORCH_CHECK(v.scalar_type() == torch::kFloat32, "v must be float32");

    TORCH_CHECK(q.dim() == 4, "q must be 4D");
    TORCH_CHECK(k.dim() == 4, "k must be 4D");
    TORCH_CHECK(v.dim() == 4, "v must be 4D");

    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();

    // Convert FP32 inputs to FP16 for CUTLASS FMHA
    auto q_fp16 = q.to(torch::kFloat16);
    auto k_fp16 = k.to(torch::kFloat16);
    auto v_fp16 = v.to(torch::kFloat16);

    // Create FP16 output tensor
    auto output_fp16 = torch::empty_like(v_fp16);

    // Setup FMHA kernel parameters
    typename FMHAKernel::Params params;

    // PyTorch data_ptr expects standard types, so we cast from void*
    params.query_ptr = reinterpret_cast<scalar_t*>(q_fp16.data_ptr());
    params.key_ptr = reinterpret_cast<scalar_t*>(k_fp16.data_ptr());
    params.value_ptr = reinterpret_cast<scalar_t*>(v_fp16.data_ptr());
    params.output_ptr = reinterpret_cast<scalar_t*>(output_fp16.data_ptr());
    params.output_accum_ptr = nullptr;
    params.logsumexp_ptr = nullptr;

    params.scale = scale;

    params.num_batches = num_batches;
    params.num_heads = num_heads;
    params.num_queries = num_queries;
    params.num_keys = num_keys;
    params.head_dim = head_dim;
    params.head_dim_value = head_dim_v;

    // Strides for BMHK format (Batch, M, Heads, K)
    params.q_strideM = num_heads * head_dim;
    params.k_strideM = num_heads * head_dim;
    params.v_strideM = num_heads * head_dim_v;
    params.o_strideM = num_heads * head_dim_v;

    params.q_strideH = head_dim;
    params.k_strideH = head_dim;
    params.v_strideH = head_dim_v;

    params.q_strideB = num_queries * params.q_strideM;
    params.k_strideB = num_keys * params.k_strideM;
    params.v_strideB = num_keys * params.v_strideM;

    // No causal masking (bi-directional attention for this use case)
    params.custom_mask_type = FMHAKernel::NoCustomMask;

    // Check if kernel supports these parameters
    if (!FMHAKernel::check_supported(params)) {
        std::cerr << "fmha_forward_cuda: Kernel does not support these inputs" << std::endl;
        std::cerr << "  num_batches: " << num_batches << ", num_heads: " << num_heads << std::endl;
        std::cerr << "  num_queries: " << num_queries << ", num_keys: " << num_keys << std::endl;
        std::cerr << "  head_dim: " << head_dim << ", head_dim_v: " << head_dim_v << std::endl;
        std::cerr << "  Tile config: QueriesPerBlock=" << kQueriesPerBlock
                  << ", KeysPerBlock=" << kKeysPerBlock << ", MaxK=" << kMaxK << std::endl;
        TORCH_CHECK(false, "CUTLASS FMHA kernel does not support these inputs");
    }

    // Launch kernel
    constexpr auto kernel_fn = attention_kernel_batched_impl<FMHAKernel>;
    int smem_bytes = sizeof(typename FMHAKernel::SharedStorage);

    if (smem_bytes > 0) {
        cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    dim3 grid = params.getBlocksGrid();
    dim3 block = params.getThreadsGrid();

    kernel_fn<<<grid, block, smem_bytes>>>(params);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "fmha_forward_cuda: Kernel launch failed: " << cudaGetErrorString(error) << std::endl;
        TORCH_CHECK(false, "CUTLASS FMHA kernel launch failed: ", cudaGetErrorString(error));
    }

    // Convert FP16 output back to FP32
    auto output = output_fp16.to(torch::kFloat32);

    return output;
}
