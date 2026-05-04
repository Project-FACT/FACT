/***************************************************************************************************
 * FMHA Configuration Header
 *
 * This header must be included first in all FMHA kernel files.
 * It ensures CUTLASS library headers are included before local FMHA headers
 * that depend on macros defined in cutlass/cutlass.h (like CUDA_STD_HEADER).
 **************************************************************************************************/

#pragma once

// Include CUTLASS library headers FIRST to define required macros
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/numeric_types.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/vector.h"
#include "cutlass/matrix.h"
#include "cutlass/tensor_ref.h"

// Ensure CUDA_STD_HEADER macro is available for downstream headers
#ifndef CUDA_STD_HEADER
#define CUDA_STD_HEADER(header) <cuda/std/header>
#endif
