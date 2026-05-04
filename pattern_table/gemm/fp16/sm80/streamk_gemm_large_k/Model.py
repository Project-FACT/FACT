"""
Baseline Model: PyTorch matmul for large-K matrix multiplication (M=256, K=524288, N=256)
Reference implementation from KernelBench.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Baseline model: C = A @ B with large K dimension.
    A: (M, K), B: (K, N), C: (M, N) where K=524288.
    """
    def __init__(self):
        super().__init__()
        torch.backends.cuda.matmul.allow_tf32 = True

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)


# Problem dimensions from KernelBench model
M = 256
K = 524288
N = 256


def get_inputs():
    """Get test inputs for the model"""
    A = torch.rand(M, K, dtype=torch.float32)
    B = torch.rand(K, N, dtype=torch.float32)
    return [A, B]


def get_init_inputs():
    return []
