/***************************************************************************************************
 * CUTLASS FP16 Stream-K GEMM for PyTorch Extension
 * Based on CUTLASS example 47_ampere_gemm_universal_streamk
 *
 * Stream-K scheduling for improved load balancing on large-K matrix multiplication.
 * Target: A100 (SM80), C = A @ B where K dimension is very large (524288).
 *
 * FP16 inputs, FP32 accumulator, FP32 output (HMMA.16816.F16.F32).
 **************************************************************************************************/

#include <torch/extension.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
// CUTLASS Stream-K GEMM configuration for FP16 tensor cores on Ampere
///////////////////////////////////////////////////////////////////////////////////////////////////

using ElementA = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::half_t;
using LayoutB = cutlass::layout::RowMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = float;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm80;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
constexpr int NumStages = 4;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    AlignmentC,
    ElementAccumulator,
    ElementAccumulator>;

using DeviceGemmStreamK = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
    NumStages,
    AlignmentA,
    AlignmentB>;

///////////////////////////////////////////////////////////////////////////////////////////////////
// PyTorch binding function
///////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor streamk_gemm(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
  TORCH_CHECK(A.scalar_type() == torch::kFloat16 && B.scalar_type() == torch::kFloat16,
              "float16 only");

  int64_t M = A.size(0);
  int64_t K = A.size(1);
  int64_t N = B.size(1);

  A = A.contiguous();
  B = B.contiguous();

  auto C = torch::zeros({M, N}, A.options().dtype(torch::kFloat32));

  cutlass::gemm::GemmCoord problem_size(
      static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));

  int split_k_factor = 1;
  int avail_sms = -1;

  ElementAccumulator alpha = ElementAccumulator(1);
  ElementAccumulator beta = ElementAccumulator(0);

  int lda = static_cast<int>(K);
  int ldb = static_cast<int>(N);
  int ldc = static_cast<int>(N);

  auto* ptr_A = reinterpret_cast<ElementA*>(A.data_ptr<at::Half>());
  auto* ptr_B = reinterpret_cast<ElementB*>(B.data_ptr<at::Half>());
  auto* ptr_C = C.data_ptr<float>();

  typename DeviceGemmStreamK::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      split_k_factor,
      {alpha, beta},
      ptr_A,
      ptr_B,
      ptr_C,
      ptr_C,
      problem_size.mk().product(),
      problem_size.nk().product(),
      problem_size.mn().product(),
      problem_size.mn().product(),
      lda, ldb, ldc, ldc,
      avail_sms};

  DeviceGemmStreamK gemm_op;

  cutlass::Status status = gemm_op.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "can_implement failed");

  size_t workspace_size = DeviceGemmStreamK::get_workspace_size(arguments);

  void* workspace_ptr = nullptr;
  cudaError_t cuda_err = cudaMalloc(&workspace_ptr, workspace_size);
  TORCH_CHECK(cuda_err == cudaSuccess, "cudaMalloc failed for workspace");

  status = gemm_op.initialize(arguments, static_cast<uint8_t*>(workspace_ptr));
  TORCH_CHECK(status == cutlass::Status::kSuccess, "initialize failed");

  status = gemm_op();
  cudaFree(workspace_ptr);

  TORCH_CHECK(status == cutlass::Status::kSuccess, "run failed");
  return C;
}
