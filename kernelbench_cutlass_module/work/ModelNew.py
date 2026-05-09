"""
ModelNew: CUTLASS TF32 Tensor Core GEMM for square matrix multiplication (Problem 1, Level 1)
Uses CUTLASS from nvidia-cutlass/3.8.0.0 module with TF32 tensor cores on Ampere (A100).

This version uses the cluster's nvidia-cutlass module instead of local CUTLASS installation.
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

# Global variable to cache the loaded extension
_cutlass_ext: Any | None = None


def _get_cutlass_ext():
    """
    Load the CUTLASS extension (cached after first load).
    Loads from separate CUDA files using nvidia-cutlass module.
    Requires nvidia-cutlass/3.8.0.0 module to be loaded.
    """
    global _cutlass_ext
    if _cutlass_ext is None:
        # Get CUTLASS_ROOT from module (should be set by module load)
        root = require_cutlass_root()

        # Get the directory containing the CUDA source files
        cuda_dir = os.path.join(os.path.dirname(__file__), 'cutlass_kernels')

        _cutlass_ext = load(
            name="kernelbench_cutlass_tf32_gemm_module",
            sources=[
                os.path.join(cuda_dir, 'ampere_tf32_gemm.cu'),
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
    CUTLASS-accelerated model for square matrix multiplication using TF32 tensor cores.
    This implementation targets Ampere (A100) GPUs using the cluster's nvidia-cutlass module.
    """
    def __init__(self) -> None:
        super().__init__()
        # Enable TF32 tensor cores for consistent behavior with PyTorch
        torch.backends.cuda.matmul.allow_tf32 = True

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        """
        Forward pass using CUTLASS TF32 tensor core GEMM kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return _get_cutlass_ext().ampere_tf32_gemm(A, B)


# Problem size (should match Model.py)
N = 2048 * 2


def get_inputs():
    """Get test inputs for the model"""
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]


def get_init_inputs():
    """No special initialization inputs needed"""
    return []
