/***************************************************************************************************
 * CUTLASS TF32 Tensor Core GEMM for PyTorch Extension
 * Based on CUTLASS example 14_ampere_tf32_tensorop_gemm
 * Minimal implementation for square matrix multiplication (C = A * B)
 *
 * Uses TF32 tensor cores on Ampere (A100) for optimal performance
 **************************************************************************************************/

#include <torch/extension.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
// CUTLASS GEMM configuration for TF32 tensor cores on Ampere
///////////////////////////////////////////////////////////////////////////////////////////////////

// Data type definitions (using float for TF32 support)
using ElementAccumulator = float;        // Accumulator data type
using ElementInputA = float;             // Input matrix A (FP32, internally converted to TF32)
using ElementInputB = float;             // Input matrix B (FP32, internally converted to TF32)
using ElementOutput = float;             // Output matrix C (FP32)

// Layout definitions (all RowMajor for square matrices)
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// Use tensor cores on Ampere
using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

// Tile sizes for optimal performance on A100
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

// Threadblock swizzling
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Epilogue (linear combination: C = alpha * AB + beta * C)
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator>;

// Number of pipeline stages
constexpr int NumStages = 4;

// GEMM operator type
using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    SwizzleThreadBlock,
    NumStages>;

///////////////////////////////////////////////////////////////////////////////////////////////////
// PyTorch binding function
///////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor ampere_tf32_gemm(torch::Tensor A, torch::Tensor B) {
  // Input validation
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
  TORCH_CHECK(A.scalar_type() == torch::kFloat && B.scalar_type() == torch::kFloat,
              "float32 only");

  int64_t M = A.size(0);
  int64_t K = A.size(1);
  TORCH_CHECK(B.size(0) == K, "Inner dimension mismatch");
  int64_t N = B.size(1);

  // Ensure contiguous memory layout
  A = A.contiguous();
  B = B.contiguous();

  // Allocate output tensor
  auto C = torch::empty({M, N}, A.options());

  // Create problem size
  cutlass::gemm::GemmCoord problem_size(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));

  // Epilogue parameters (alpha and beta)
  ElementAccumulator alpha = ElementAccumulator(1);
  ElementAccumulator beta = ElementAccumulator(0);

  // Leading dimensions
  int lda = static_cast<int>(K);
  int ldb = static_cast<int>(N);
  int ldc = static_cast<int>(N);

  // Prepare GEMM arguments
  typename Gemm::Arguments arguments{
      problem_size,
      {A.data_ptr<ElementInputA>(), lda},
      {B.data_ptr<ElementInputB>(), ldb},
      {C.data_ptr<ElementOutput>(), ldc},
      {C.data_ptr<ElementOutput>(), ldc},
      {alpha, beta}};

  // Instantiate and run GEMM
  Gemm gemm_op;

  // Check if problem is supported
  cutlass::Status status = gemm_op.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Gemm::can_implement failed");

  // Get workspace size
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace
  torch::Tensor workspace = torch::empty({static_cast<int64_t>(workspace_size)}, torch::kUInt8);

  // Initialize GEMM
  status = gemm_op.initialize(arguments, workspace.data_ptr<uint8_t>());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Gemm::initialize failed");

  // Run GEMM
  status = gemm_op();
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Gemm::run failed");

  return C;
}
