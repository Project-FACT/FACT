/***************************************************************************************************
 * CUTLASS FP16 Tensor Core GEMM with GELU Activation for MLP Fusion
 * Pattern: MLP GEMM Fusion with GELU Activation (c_fc -> GELU -> c_proj)
 * Target: Ampere (A100) SM80
 *
 * This implements the first GEMM (c_fc) with fused GELU activation:
 *   Output = GELU(Input @ Weight.T + Bias)
 *
 * The second GEMM (c_proj) uses standard epilogue (no activation).
 **************************************************************************************************/
#include <torch/extension.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"
#include "cutlass/epilogue/thread/activation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
// CUTLASS GEMM configuration for FP16 tensor cores on Ampere with GELU epilogue
///////////////////////////////////////////////////////////////////////////////////////////////////

// Data type definitions for FP16 with FP32 accumulation
using ElementAccumulator = float;        // Accumulator data type (FP32 for stability)
using ElementInputA = cutlass::half_t;   // Input matrix A (FP16)
using ElementInputB = cutlass::half_t;   // Input matrix B (FP16)
using ElementOutput = float;             // Output matrix C (FP32 to match PyTorch)

// Layout definitions
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;  // Weights are stored transposed
using LayoutOutput = cutlass::layout::RowMajor;

// Use tensor cores on Ampere
using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

// Tile sizes for optimal performance on A100
// Threadblock: 128x128x32, Warp: 64x64x32, Instruction: 16x8x16 (FP16 MMA shape)
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// Threadblock swizzling
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Epilogue 1: GELU activation (for c_fc)
// Uses GELU_taylor approximation which matches PyTorch's NewGELU
///////////////////////////////////////////////////////////////////////////////////////////////////

using EpilogueOpGELU = cutlass::epilogue::thread::LinearCombinationGELU<
    ElementOutput,                               // Output data type
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // Elements per vector access
    ElementAccumulator,                          // Accumulator data type
    ElementAccumulator,                          // Compute data type
    cutlass::epilogue::thread::ScaleType::NoBetaScaling,  // No beta scaling (alpha * AB + bias)
    cutlass::FloatRoundStyle::round_to_nearest>;

// GEMM with GELU epilogue
using GemmWithGELU = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOpGELU,
    SwizzleThreadBlock,
    3>;  // 3 pipeline stages for Ampere

///////////////////////////////////////////////////////////////////////////////////////////////////
// Epilogue 2: Standard linear combination (for c_proj)
// No activation, just alpha * AB + beta * C
///////////////////////////////////////////////////////////////////////////////////////////////////

using EpilogueOpStandard = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator>;

// GEMM with standard epilogue
using GemmStandard = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOpStandard,
    SwizzleThreadBlock,
    3>;  // 3 pipeline stages

namespace {
// LibTorch exports data_ptr<c10::Half>() but not data_ptr<cutlass::half_t>(); types are ABI-compatible.
inline ElementInputA* as_cutlass_half_ptr(torch::Tensor const& t) {
  static_assert(sizeof(c10::Half) == sizeof(ElementInputA), "FP16 layout mismatch");
  return reinterpret_cast<ElementInputA*>(t.data_ptr<c10::Half>());
}
}  // namespace

///////////////////////////////////////////////////////////////////////////////////////////////////
// GEMM 1: c_fc with GELU activation
// Input: [M, K_in], Weight: [K_out, K_in], Output: [M, K_out]
///////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor mlp_fc_gelu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

  // Input validation
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "Input and weight must be 2D");
  TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");

  int64_t M = input.size(0);
  int64_t K_in = input.size(1);
  TORCH_CHECK(weight.size(1) == K_in, "Input K dimension mismatch with weight");
  int64_t K_out = weight.size(0);
  TORCH_CHECK(bias.size(0) == K_out, "Bias size mismatch");

  // Ensure contiguous memory layout
  input = input.contiguous();
  weight = weight.contiguous();

  // Convert input/weight to FP16 if needed
  auto input_fp16 = input.to(torch::kFloat16);
  auto weight_fp16 = weight.to(torch::kFloat16);

  // Allocate output tensor (FP32)
  auto output = torch::empty({M, K_out}, input.options().dtype(torch::kFloat32));

  // Create problem size
  cutlass::gemm::GemmCoord problem_size(static_cast<int>(M), static_cast<int>(K_out), static_cast<int>(K_in));

  // Epilogue parameters (alpha = 1, beta is implicit for bias)
  ElementAccumulator alpha = ElementAccumulator(1);

  // Leading dimensions
  int lda = static_cast<int>(K_in);
  int ldb = static_cast<int>(K_in);  // Weight is stored K_out x K_in, used as K_in x K_out (transposed)
  int ldc = static_cast<int>(K_out);

  // Prepare GEMM arguments
  typename GemmWithGELU::Arguments arguments{
      problem_size,
      {as_cutlass_half_ptr(input_fp16), lda},
      {as_cutlass_half_ptr(weight_fp16), ldb},
      {bias.data_ptr<ElementOutput>(), 0},  // Bias vector (stride 0 for broadcasting)
      {output.data_ptr<ElementOutput>(), ldc},
      {alpha}};

  // Instantiate and run GEMM
  GemmWithGELU gemm_op;

  // Check if problem is supported
  cutlass::Status status = gemm_op.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM::can_implement failed");

  // Get workspace size
  size_t workspace_size = GemmWithGELU::get_workspace_size(arguments);

  // Allocate workspace
  torch::Tensor workspace = torch::empty({static_cast<int64_t>(workspace_size)}, torch::kUInt8);

  // Initialize GEMM
  status = gemm_op.initialize(arguments, workspace.data_ptr<uint8_t>());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM::initialize failed");

  // Run GEMM
  status = gemm_op();
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM::run failed");

  return output;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// GEMM 2: c_proj (standard, no activation)
// Input: [M, K_in], Weight: [K_out, K_in], Output: [M, K_out]
///////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor mlp_proj(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

  // Input validation
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "Input and weight must be 2D");
  TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");

  int64_t M = input.size(0);
  int64_t K_in = input.size(1);
  TORCH_CHECK(weight.size(1) == K_in, "Input K dimension mismatch with weight");
  int64_t K_out = weight.size(0);
  TORCH_CHECK(bias.size(0) == K_out, "Bias size mismatch");

  // Ensure contiguous memory layout
  input = input.contiguous();
  weight = weight.contiguous();

  // Convert input/weight to FP16 if needed
  auto input_fp16 = input.to(torch::kFloat16);
  auto weight_fp16 = weight.to(torch::kFloat16);

  // Allocate output tensor (FP32)
  auto output = torch::empty({M, K_out}, input.options().dtype(torch::kFloat32));

  // Create problem size
  cutlass::gemm::GemmCoord problem_size(static_cast<int>(M), static_cast<int>(K_out), static_cast<int>(K_in));

  // Epilogue parameters (alpha = 1, beta = 1 for bias addition)
  ElementAccumulator alpha = ElementAccumulator(1);
  ElementAccumulator beta = ElementAccumulator(1);

  // Leading dimensions
  int lda = static_cast<int>(K_in);
  int ldb = static_cast<int>(K_in);
  int ldc = static_cast<int>(K_out);

  // Prepare GEMM arguments
  typename GemmStandard::Arguments arguments{
      problem_size,
      {as_cutlass_half_ptr(input_fp16), lda},
      {as_cutlass_half_ptr(weight_fp16), ldb},
      {bias.data_ptr<ElementOutput>(), 0},  // Bias vector
      {output.data_ptr<ElementOutput>(), ldc},
      {alpha, beta}};

  // Instantiate and run GEMM
  GemmStandard gemm_op;

  // Check if problem is supported
  cutlass::Status status = gemm_op.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM::can_implement failed");

  // Get workspace size
  size_t workspace_size = GemmStandard::get_workspace_size(arguments);

  // Allocate workspace
  torch::Tensor workspace = torch::empty({static_cast<int64_t>(workspace_size)}, torch::kUInt8);

  // Initialize GEMM
  status = gemm_op.initialize(arguments, workspace.data_ptr<uint8_t>());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM::initialize failed");

  // Run GEMM
  status = gemm_op();
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM::run failed");

  return output;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Combined MLP forward pass (c_fc -> GELU -> c_proj)
///////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor mlp_forward(
    torch::Tensor input,
    torch::Tensor fc_weight,
    torch::Tensor fc_bias,
    torch::Tensor proj_weight,
    torch::Tensor proj_bias) {

  // First GEMM: c_fc with GELU
  auto hidden = mlp_fc_gelu(input, fc_weight, fc_bias);

  // Second GEMM: c_proj (standard)
  auto output = mlp_proj(hidden, proj_weight, proj_bias);

  return output;
}
