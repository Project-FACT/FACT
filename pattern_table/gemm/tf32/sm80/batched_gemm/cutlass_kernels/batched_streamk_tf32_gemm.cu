/***************************************************************************************************
 * CUTLASS TF32 Stream-K Batched GEMM for PyTorch Extension
 * Based on CUTLASS examples 47 (Stream-K) and 41 (TF32 tensor cores on Ampere)
 *
 * Batched GEMM: C[b] = A[b] * B[b] for b in [0, batch_size)
 * Uses GemmUniversal with ThreadblockSwizzleStreamK for Stream-K scheduling
 * Target: Ampere (SM80, A100)
 **************************************************************************************************/

#include <torch/extension.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
// CUTLASS Batched GEMM configuration — TF32 tensor cores on Ampere with Stream-K
///////////////////////////////////////////////////////////////////////////////////////////////////

using ElementAccumulator = float;
using ElementInputA      = float;
using ElementInputB      = float;
using ElementOutput      = float;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAOp   = cutlass::arch::OpClassTensorOp;
using SmArch  = cutlass::arch::Sm80;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 32>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementInputB>::value;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator>;

constexpr int NumStages = 3;

using SwizzleThreadBlock = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;

using BatchedGemmStreamK = cutlass::gemm::device::GemmUniversal<
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
    NumStages,
    AlignmentA,
    AlignmentB>;

///////////////////////////////////////////////////////////////////////////////////////////////////
// PyTorch binding
///////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor batched_streamk_tf32_gemm(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
  TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "A and B must be 3D (batch, M/K, K/N)");
  TORCH_CHECK(A.scalar_type() == torch::kFloat && B.scalar_type() == torch::kFloat,
              "float32 only");
  TORCH_CHECK(A.size(0) == B.size(0), "Batch dimension mismatch");
  TORCH_CHECK(A.size(2) == B.size(1), "Inner dimension K mismatch");

  int64_t batch = A.size(0);
  int64_t M = A.size(1);
  int64_t K = A.size(2);
  int64_t N = B.size(2);

  A = A.contiguous();
  B = B.contiguous();

  auto C = torch::empty({batch, M, N}, A.options());

  cutlass::gemm::GemmCoord problem_size(static_cast<int>(M),
                                        static_cast<int>(N),
                                        static_cast<int>(K));

  int64_t batch_stride_A = static_cast<int64_t>(M) * K;
  int64_t batch_stride_B = static_cast<int64_t>(K) * N;
  int64_t batch_stride_C = static_cast<int64_t>(M) * N;
  int64_t batch_stride_D = static_cast<int64_t>(M) * N;

  typename BatchedGemmStreamK::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kBatched,
      problem_size,
      static_cast<int>(batch),
      {ElementAccumulator(1), ElementAccumulator(0)},
      A.data_ptr<ElementInputA>(),
      B.data_ptr<ElementInputB>(),
      C.data_ptr<ElementOutput>(),
      C.data_ptr<ElementOutput>(),
      static_cast<int>(batch_stride_A),
      static_cast<int>(batch_stride_B),
      static_cast<int>(batch_stride_C),
      static_cast<int>(batch_stride_D),
      static_cast<int>(K),
      static_cast<int>(N),
      static_cast<int>(N),
      static_cast<int>(N),
      -1};

  BatchedGemmStreamK gemm_op;

  cutlass::Status status = gemm_op.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GemmUniversal::can_implement failed");

  size_t workspace_size = BatchedGemmStreamK::get_workspace_size(arguments);
  torch::Tensor workspace = torch::empty({static_cast<int64_t>(workspace_size)}, torch::kUInt8);

  status = gemm_op.initialize(arguments, workspace.data_ptr<uint8_t>());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GemmUniversal::initialize failed");

  status = gemm_op();
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GemmUniversal::run failed");

  return C;
}
