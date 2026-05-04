/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*!
    \file
    \brief Hopper Stream-K GEMM example derived from example 48 structure.
*/

#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

using ElementA = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::half_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = cutlass::half_t;
using LayoutC = cutlass::layout::ColumnMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using TileShape = Shape<_128,_128,_64>;
using ClusterShape = Shape<_2,_1,_1>;
using StageCountType = cutlass::gemm::collective::StageCountAuto;
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::StreamKScheduler
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using DeviceGemmReference = cutlass::reference::device::Gemm<
  ElementA, LayoutA,
  ElementB, LayoutB,
  ElementC, LayoutC,
  ElementAccumulator,
  ElementAccumulator>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

struct Options {
  bool help;
  float alpha, beta;
  int iterations;
  int m, n, k, l;

  Options():
    help(false),
    m(256), n(256), k(131072 * 4), l(1),
    alpha(1.f), beta(0.f),
    iterations(20)
  {}

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);
    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  std::ostream& print_usage(std::ostream &out) const {
    out << "90_hopper_streamk_gemm\n\n"
      << "  Hopper Stream-K GEMM (fp16 inputs, fp32 accumulate/output).\n\n"
      << "Options:\n\n"
      << "  --help                      Display this usage statement\n\n"
      << "  --m=<int>                   Sets M\n"
      << "  --n=<int>                   Sets N\n"
      << "  --k=<int>                   Sets K\n"
      << "  --l=<int>                   Sets batch count (currently must be 1)\n"
      << "  --alpha=<f32>               Epilogue alpha\n"
      << "  --beta=<f32>                Epilogue beta\n"
      << "  --iterations=<int>          Timed iterations\n\n";
    return out;
  }

  bool valid() const {
    return l == 1;
  }

  double gflops(double runtime_s) const {
    uint64_t flop = uint64_t(2) * uint64_t(m) * uint64_t(n) * uint64_t(k) * uint64_t(l);
    return (double(flop) / 1.0e9) / runtime_s;
  }
};

struct Result {
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(double avg_runtime_ms = 0,
         double gflops = 0,
         cutlass::Status status = cutlass::Status::kSuccess,
         cudaError_t error = cudaSuccess)
  : avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false) {}
};

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

template <class Element>
bool initialize_block(cutlass::DeviceAllocation<Element>& block, uint64_t seed = 2023) {
  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = Element(2);
    scope_min = Element(0);
  } else if (bits_input <= 8) {
    scope_max = Element(2);
    scope_min = Element(-2);
  } else {
    scope_max = Element(8);
    scope_min = Element(-8);
  }

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);
  return true;
}

void initialize(const Options &options) {
  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, options.l});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, options.l});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, options.l});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, options.l});

  int64_t size_a = int64_t(options.m) * options.k * options.l;
  int64_t size_b = int64_t(options.k) * options.n * options.l;
  int64_t size_c = int64_t(options.m) * options.n * options.l;
  block_A.reset(size_a);
  block_B.reset(size_b);
  block_C.reset(size_c);
  block_D.reset(size_c);
  block_ref_D.reset(size_c);

  initialize_block(block_A, seed + 2023);
  initialize_block(block_B, seed + 2022);
  initialize_block(block_C, seed + 2021);
}

template <typename GemmT>
typename GemmT::Arguments args_from_options(const Options &options) {
  int device_id = 0;
  cutlass::KernelHardwareInfo kernel_hw_info =
    cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename GemmT::GemmKernel>(device_id);

  typename GemmT::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, options.l},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D},
    kernel_hw_info
  };
  return arguments;
}

bool verify(const Options &options) {
  cutlass::TensorRef ref_A(block_A.get(), Gemm::LayoutA::packed({options.m, options.k}));
  cutlass::TensorRef ref_B(block_B.get(), Gemm::LayoutB::packed({options.k, options.n}));
  cutlass::TensorRef ref_C(block_C.get(), Gemm::LayoutC::packed({options.m, options.n}));
  cutlass::TensorRef ref_D(block_ref_D.get(), Gemm::LayoutD::packed({options.m, options.n}));

  DeviceGemmReference gemm_reference;
  gemm_reference(
    {options.m, options.n, options.k},
    ElementAccumulator(options.alpha),
    ref_A,
    ref_B,
    ElementAccumulator(options.beta),
    ref_C,
    ref_D);

  CUDA_CHECK(cudaDeviceSynchronize());
  bool passed = cutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(), block_D.size());
  return passed;
}

template <typename GemmT>
int run(Options &options) {
  initialize(options);
  GemmT gemm;
  auto arguments = args_from_options<GemmT>(options);

  size_t workspace_size = GemmT::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm.run());

  Result result;
  result.passed = verify(options);
  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;
  if (!result.passed) {
    return -1;
  }

  if (options.iterations > 0) {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  if (props.major != 9 || props.minor != 0) {
    std::cerr << "This example requires Hopper (sm90).\n";
    return 0;
  }

  Options options;
  options.parse(argc, args);
  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }
  if (!options.valid()) {
    std::cerr << "Only --l=1 is supported in this harness.\n";
    return -1;
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  return run<Gemm>(options);
#else
  return 0;
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////
