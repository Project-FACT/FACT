"""
Baseline Model: PyTorch matmul for square matrix multiplication (Problem 1, Level 1)
Reference implementation from KernelBench
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    Uses TF32 tensor cores on Ampere GPUs for optimal performance.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Enable TF32 tensor cores for faster matmul on A100
        torch.backends.cuda.matmul.allow_tf32 = True

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return torch.matmul(A, B)


# Problem size from KernelBench
N = 2048 * 2


def get_inputs():
    """Get test inputs for the model"""
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]


def get_init_inputs():
    """No special initialization inputs needed"""
    return []
