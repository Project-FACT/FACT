"""
ModelNew: CUTLASS TF32 Stream-K Batched GEMM for batched matrix multiplication (Problem 3, Level 1)
Uses CUTLASS from nvidia-cutlass/3.8.0.0 module with TF32 tensor cores and Stream-K scheduling on Ampere (A100).
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
            name="kernelbench_batched_streamk_tf32_module",
            sources=[
                os.path.join(cuda_dir, 'batched_streamk_tf32_gemm.cu'),
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
    CUTLASS-accelerated batched GEMM using TF32 tensor cores with Stream-K scheduling.
    """
    def __init__(self) -> None:
        super().__init__()
        torch.backends.cuda.matmul.allow_tf32 = True

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        return _get_cutlass_ext().batched_streamk_tf32_gemm(A, B)


batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4


def get_inputs():
    A = torch.rand(batch_size, m, k)
    B = torch.rand(batch_size, k, n)
    return [A, B]


def get_init_inputs():
    return []
