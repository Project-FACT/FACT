/***************************************************************************************************
 * CUTLASS FP16 Tensor Core GEMM with GELU Activation for MLP Fusion (autotuned tile/stages)
 * Pattern: MLP GEMM Fusion with GELU Activation (c_fc -> GELU -> c_proj)
 * Target: Ampere (A100) SM80
 *
 * Tile / warp / pipeline stages match autotune_results.json "best_config":
 *   tile [128, 256, 32], warp [64, 64, 32], stages 4
 *
 * Drop-in sibling of mlp_gemm_fusion.cu — link one or the other, not both (same symbols).
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

using ElementAccumulator = float;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = float;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

// Autotuned threadblock / warp / instruction (best_config + CUTLASS default instruction shape)
using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Epilogue 1: GELU (c_fc)
///////////////////////////////////////////////////////////////////////////////////////////////////

using EpilogueOpGELU = cutlass::epilogue::thread::LinearCombinationGELU<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling,
    cutlass::FloatRoundStyle::round_to_nearest>;

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
    4>;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Epilogue 2: standard (c_proj)
///////////////////////////////////////////////////////////////////////////////////////////////////

using EpilogueOpStandard = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator>;

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
    4>;

namespace {
inline ElementInputA* as_cutlass_half_ptr(torch::Tensor const& t) {
  static_assert(sizeof(c10::Half) == sizeof(ElementInputA), "FP16 layout mismatch");
  return reinterpret_cast<ElementInputA*>(t.data_ptr<c10::Half>());
}
}  // namespace

///////////////////////////////////////////////////////////////////////////////////////////////////
// GEMM 1: c_fc with GELU
///////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor mlp_fc_gelu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "Input and weight must be 2D");
  TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");

  int64_t M = input.size(0);
  int64_t K_in = input.size(1);
  TORCH_CHECK(weight.size(1) == K_in, "Input K dimension mismatch with weight");
  int64_t K_out = weight.size(0);
  TORCH_CHECK(bias.size(0) == K_out, "Bias size mismatch");

  input = input.contiguous();
  weight = weight.contiguous();

  auto input_fp16 = input.to(torch::kFloat16);
  auto weight_fp16 = weight.to(torch::kFloat16);

  auto output = torch::empty({M, K_out}, input.options().dtype(torch::kFloat32));

  cutlass::gemm::GemmCoord problem_size(static_cast<int>(M), static_cast<int>(K_out), static_cast<int>(K_in));

  ElementAccumulator alpha = ElementAccumulator(1);

  int lda = static_cast<int>(K_in);
  int ldb = static_cast<int>(K_in);
  int ldc = static_cast<int>(K_out);

  typename GemmWithGELU::Arguments arguments{
      problem_size,
      {as_cutlass_half_ptr(input_fp16), lda},
      {as_cutlass_half_ptr(weight_fp16), ldb},
      {bias.data_ptr<ElementOutput>(), 0},
      {output.data_ptr<ElementOutput>(), ldc},
      {alpha}};

  GemmWithGELU gemm_op;

  cutlass::Status status = gemm_op.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM::can_implement failed");

  size_t workspace_size = GemmWithGELU::get_workspace_size(arguments);
  torch::Tensor workspace = torch::empty({static_cast<int64_t>(workspace_size)}, torch::kUInt8);

  status = gemm_op.initialize(arguments, workspace.data_ptr<uint8_t>());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM::initialize failed");

  status = gemm_op();
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM::run failed");

  return output;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// GEMM 2: c_proj
///////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor mlp_proj(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "Input and weight must be 2D");
  TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");

  int64_t M = input.size(0);
  int64_t K_in = input.size(1);
  TORCH_CHECK(weight.size(1) == K_in, "Input K dimension mismatch with weight");
  int64_t K_out = weight.size(0);
  TORCH_CHECK(bias.size(0) == K_out, "Bias size mismatch");

  input = input.contiguous();
  weight = weight.contiguous();

  auto input_fp16 = input.to(torch::kFloat16);
  auto weight_fp16 = weight.to(torch::kFloat16);

  auto output = torch::empty({M, K_out}, input.options().dtype(torch::kFloat32));

  cutlass::gemm::GemmCoord problem_size(static_cast<int>(M), static_cast<int>(K_out), static_cast<int>(K_in));

  ElementAccumulator alpha = ElementAccumulator(1);
  ElementAccumulator beta = ElementAccumulator(1);

  int lda = static_cast<int>(K_in);
  int ldb = static_cast<int>(K_in);
  int ldc = static_cast<int>(K_out);

  typename GemmStandard::Arguments arguments{
      problem_size,
      {as_cutlass_half_ptr(input_fp16), lda},
      {as_cutlass_half_ptr(weight_fp16), ldb},
      {bias.data_ptr<ElementOutput>(), 0},
      {output.data_ptr<ElementOutput>(), ldc},
      {alpha, beta}};

  GemmStandard gemm_op;

  cutlass::Status status = gemm_op.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM::can_implement failed");

  size_t workspace_size = GemmStandard::get_workspace_size(arguments);
  torch::Tensor workspace = torch::empty({static_cast<int64_t>(workspace_size)}, torch::kUInt8);

  status = gemm_op.initialize(arguments, workspace.data_ptr<uint8_t>());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM::initialize failed");

  status = gemm_op();
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM::run failed");

  return output;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Combined MLP forward
///////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor mlp_forward(
    torch::Tensor input,
    torch::Tensor fc_weight,
    torch::Tensor fc_bias,
    torch::Tensor proj_weight,
    torch::Tensor proj_bias) {

  auto hidden = mlp_fc_gelu(input, fc_weight, fc_bias);
  auto output = mlp_proj(hidden, proj_weight, proj_bias);
  return output;
}
