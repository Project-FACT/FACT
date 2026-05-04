/***************************************************************************************************
 * CUTLASS FP16 Tensor Core GEMM with SwiGLU Activation for MLP Fusion
 * Pattern: SwiGLU MLP Fusion (gate_proj + SiLU + up_proj + multiply + down_proj)
 * Target: Ampere (A100) SM80
 *
 * This implements the SwiGLU MLP from Llama 3 8B:
 *   1. gate_proj: GEMM with SiLU activation
 *   2. up_proj: GEMM (no activation)
 *   3. Elementwise multiply: SiLU(gate) * up
 *   4. down_proj: GEMM (no activation)
 *
 * Strategy:
 *   - Kernel 1 (swiglu_gate_up): gate_proj + SiLU + up_proj + elementwise multiply
 *   - Kernel 2 (down_proj): down_proj GEMM
 **************************************************************************************************/
#include <torch/extension.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "cutlass/epilogue/thread/linear_combination.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
// CUTLASS GEMM configuration for FP16 tensor cores on Ampere
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
// Epilogue 1: SiLU activation (for gate_proj)
// SiLU(x) = x * sigmoid(x)
///////////////////////////////////////////////////////////////////////////////////////////////////

using EpilogueOpSiLU = cutlass::epilogue::thread::LinearCombinationSilu<
    ElementOutput,                               // Output data type
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // Elements per vector access
    ElementAccumulator,                          // Accumulator data type
    ElementAccumulator,                          // Compute data type
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling,  // Only alpha scaling (no bias)
    cutlass::FloatRoundStyle::round_to_nearest>;

// GEMM with SiLU epilogue
using GemmWithSiLU = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOpSiLU,
    SwizzleThreadBlock,
    3>;  // 3 pipeline stages for Ampere

///////////////////////////////////////////////////////////////////////////////////////////////////
// Epilogue 2: Standard linear combination (for up_proj and down_proj)
// No activation, just alpha * AB
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
// Kernel 1: gate_proj + up_proj with SiLU and elementwise multiply
//
// This computes: SiLU(input @ gate_weight.T) * (input @ up_weight.T)
// Input: [M, K_in], gate_weight: [K_out, K_in], up_weight: [K_out, K_in]
// Output: [M, K_out]
///////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor swiglu_gate_up(
    torch::Tensor input,
    torch::Tensor gate_weight,
    torch::Tensor up_weight) {

  // Input validation
  TORCH_CHECK(input.is_cuda() && gate_weight.is_cuda() && up_weight.is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(input.dim() == 2 && gate_weight.dim() == 2 && up_weight.dim() == 2,
              "Input and weights must be 2D");

  int64_t M = input.size(0);
  int64_t K_in = input.size(1);
  TORCH_CHECK(gate_weight.size(1) == K_in, "Input K dimension mismatch with gate_weight");
  TORCH_CHECK(up_weight.size(1) == K_in, "Input K dimension mismatch with up_weight");
  int64_t K_out = gate_weight.size(0);
  TORCH_CHECK(up_weight.size(0) == K_out, "gate_weight and up_weight output dimension mismatch");

  // Ensure contiguous memory layout
  input = input.contiguous();
  gate_weight = gate_weight.contiguous();
  up_weight = up_weight.contiguous();

  // Convert to FP16 if needed
  auto input_fp16 = input.to(torch::kFloat16);
  auto gate_weight_fp16 = gate_weight.to(torch::kFloat16);
  auto up_weight_fp16 = up_weight.to(torch::kFloat16);

  // Allocate intermediate tensors (FP32)
  auto gate_silu = torch::empty({M, K_out}, input.options().dtype(torch::kFloat32));
  auto up_output = torch::empty({M, K_out}, input.options().dtype(torch::kFloat32));

  // Create problem size
  cutlass::gemm::GemmCoord problem_size(static_cast<int>(M), static_cast<int>(K_out), static_cast<int>(K_in));

  // Epilogue parameters (alpha = 1)
  ElementAccumulator alpha = ElementAccumulator(1);

  // Leading dimensions
  int lda = static_cast<int>(K_in);
  int ldb = static_cast<int>(K_in);  // Weights are stored K_out x K_in, used as K_in x K_out (transposed)
  int ldc = static_cast<int>(K_out);

  ////////////////////////////////////////////////////////////
  // GEMM 1: gate_proj with SiLU activation
  ////////////////////////////////////////////////////////////

  // Prepare GEMM arguments
  typename GemmWithSiLU::Arguments arguments_gate{
      problem_size,
      {as_cutlass_half_ptr(input_fp16), lda},
      {as_cutlass_half_ptr(gate_weight_fp16), ldb},
      {gate_silu.data_ptr<ElementOutput>(), ldc},  // Output (no bias)
      {gate_silu.data_ptr<ElementOutput>(), ldc},
      {alpha}};

  // Instantiate and run GEMM
  GemmWithSiLU gemm_gate;

  // Check if problem is supported
  cutlass::Status status = gemm_gate.can_implement(arguments_gate);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM::can_implement failed for gate_proj");

  // Get workspace size
  size_t workspace_size = GemmWithSiLU::get_workspace_size(arguments_gate);

  // Allocate workspace
  torch::Tensor workspace = torch::empty({static_cast<int64_t>(workspace_size)}, torch::kUInt8);

  // Initialize GEMM
  status = gemm_gate.initialize(arguments_gate, workspace.data_ptr<uint8_t>());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM::initialize failed for gate_proj");

  // Run GEMM
  status = gemm_gate();
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM::run failed for gate_proj");

  ////////////////////////////////////////////////////////////
  // GEMM 2: up_proj (standard, no activation)
  ////////////////////////////////////////////////////////////

  // Prepare GEMM arguments
  typename GemmStandard::Arguments arguments_up{
      problem_size,
      {as_cutlass_half_ptr(input_fp16), lda},
      {as_cutlass_half_ptr(up_weight_fp16), ldb},
      {up_output.data_ptr<ElementOutput>(), ldc},  // Output (no bias)
      {up_output.data_ptr<ElementOutput>(), ldc},
      {alpha}};

  // Instantiate and run GEMM
  GemmStandard gemm_up;

  // Check if problem is supported
  status = gemm_up.can_implement(arguments_up);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM::can_implement failed for up_proj");

  // Initialize GEMM
  status = gemm_up.initialize(arguments_up, workspace.data_ptr<uint8_t>());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM::initialize failed for up_proj");

  // Run GEMM
  status = gemm_up();
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM::run failed for up_proj");

  ////////////////////////////////////////////////////////////
  // Elementwise multiply: SiLU(gate) * up
  ////////////////////////////////////////////////////////////

  auto output = gate_silu * up_output;

  return output;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel 2: down_proj GEMM
// Input: [M, K_in], Weight: [K_out, K_in], Output: [M, K_out]
///////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor swiglu_down(
    torch::Tensor input,
    torch::Tensor weight) {

  // Input validation
  TORCH_CHECK(input.is_cuda() && weight.is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "Input and weight must be 2D");

  int64_t M = input.size(0);
  int64_t K_in = input.size(1);
  TORCH_CHECK(weight.size(1) == K_in, "Input K dimension mismatch with weight");
  int64_t K_out = weight.size(0);

  // Ensure contiguous memory layout
  input = input.contiguous();
  weight = weight.contiguous();

  // Convert to FP16 if needed (input is already FP32 from previous step)
  auto input_fp16 = input.to(torch::kFloat16);
  auto weight_fp16 = weight.to(torch::kFloat16);

  // Allocate output tensor (FP32)
  auto output = torch::empty({M, K_out}, input.options().dtype(torch::kFloat32));

  // Create problem size
  cutlass::gemm::GemmCoord problem_size(static_cast<int>(M), static_cast<int>(K_out), static_cast<int>(K_in));

  // Epilogue parameters (alpha = 1)
  ElementAccumulator alpha = ElementAccumulator(1);

  // Leading dimensions
  int lda = static_cast<int>(K_in);
  int ldb = static_cast<int>(K_in);
  int ldc = static_cast<int>(K_out);

  // Prepare GEMM arguments
  typename GemmStandard::Arguments arguments{
      problem_size,
      {as_cutlass_half_ptr(input_fp16), lda},
      {as_cutlass_half_ptr(weight_fp16), ldb},
      {output.data_ptr<ElementOutput>(), ldc},  // Output (no bias)
      {output.data_ptr<ElementOutput>(), ldc},
      {alpha}};

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
// Combined SwiGLU MLP forward pass (gate + up + multiply + down)
///////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor swiglu_mlp_forward(
    torch::Tensor input,
    torch::Tensor gate_weight,
    torch::Tensor up_weight,
    torch::Tensor down_weight) {

  // SwiGLU: gate_proj + SiLU + up_proj + elementwise multiply
  auto hidden = swiglu_gate_up(input, gate_weight, up_weight);

  // Down projection
  auto output = swiglu_down(hidden, down_weight);

  return output;
}
