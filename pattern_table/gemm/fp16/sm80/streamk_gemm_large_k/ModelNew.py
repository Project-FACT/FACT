"""
ModelNew: CUTLASS FP16 Stream-K GEMM for large-K matrix multiplication (M=256, K=524288, N=256)
Uses Stream-K scheduling via ThreadblockSwizzleStreamK for improved load balancing.

FP32 inputs are converted to FP16 for the CUTLASS kernel, then converted back to FP32.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

from kernelbench.cutlass_cpp.runtime import (
    cutlass_include_paths,
    default_extra_cuda_cflags,
    require_cutlass_root,
)

_cutlass_ext: Any | None = None


def _get_cutlass_ext():
    global _cutlass_ext
    if _cutlass_ext is None:
        root = require_cutlass_root()
        cuda_dir = os.path.join(os.path.dirname(__file__), 'cutlass_kernels')

        _cutlass_ext = load(
            name="kernelbench_streamk_gemm_module",
            sources=[
                os.path.join(cuda_dir, 'streamk_gemm.cu'),
                os.path.join(cuda_dir, 'main.cpp'),
            ],
            extra_include_paths=cutlass_include_paths(root),
            extra_cuda_cflags=default_extra_cuda_cflags(),
            verbose=os.environ.get("KERNELBENCH_CUTLASS_VERBOSE", "").lower()
            in ("1", "true", "yes"),
        )
    return _cutlass_ext


class ModelNew(nn.Module):
    """
    CUTLASS Stream-K GEMM with FP16 computation.
    Converts FP32 inputs to FP16, runs Stream-K kernel, converts back to FP32.
    """
    def __init__(self) -> None:
        super().__init__()
        torch.backends.cuda.matmul.allow_tf32 = True

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # FP32 -> FP16 for CUTLASS kernel
        A_half = A.half().contiguous()
        B_half = B.half().contiguous()

        # Run FP16 Stream-K GEMM
        C_half = _get_cutlass_ext().streamk_gemm(A_half, B_half)

        # FP16 -> FP32 to match baseline output type
        return C_half.float()


# Problem dimensions (must match Model.py)
M = 256
K = 524288
N = 256


def get_inputs():
    A = torch.rand(M, K, dtype=torch.float32)
    B = torch.rand(K, N, dtype=torch.float32)
    return [A, B]


def get_init_inputs():
    return []
