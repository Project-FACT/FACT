"""
ModelNew: CUTLASS-accelerated MiniGPT Transformer Block
Pattern: MLP GEMM Fusion with GELU Activation (c_fc -> GELU -> c_proj)
Target: Ampere (A100) SM80

This implementation:
- Uses PyTorch baseline for Attention (CausalSelfAttention)
- Uses CUTLASS FP16 tensor cores for MLP with fused GELU activation
- Includes timing instrumentation for MLP isolation
"""
from __future__ import annotations

import os
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
    Requires nvidia-cutlass module to be loaded.
    """
    global _cutlass_ext
    if _cutlass_ext is None:
        # Get CUTLASS_ROOT from module (should be set by module load)
        root = require_cutlass_root()

        # Get the directory containing the CUDA source files
        cuda_dir = os.path.join(os.path.dirname(__file__), 'cutlass_kernels')

        _cutlass_ext = load(
            name="mlp_gemm_fusion_gelu_module",
            sources=[
                os.path.join(cuda_dir, 'mlp_gemm_fusion.cu'),
                os.path.join(cuda_dir, 'main.cpp'),
            ],
            extra_include_paths=cutlass_include_paths(root),
            extra_cuda_cflags=default_extra_cuda_cflags(),
            verbose=os.environ.get("KERNELBENCH_CUTLASS_VERBOSE", "").lower()
            in ("1", "true", "yes"),
        )
    return _cutlass_ext


# =============================================================================
# PyTorch Baseline Components
# =============================================================================

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(NewGELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    PyTorch baseline implementation (same as 44_MiniGPTBlock.py).
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


# =============================================================================
# CUTLASS MLP Block
# =============================================================================

class MLPBlockNew(nn.Module):
    """
    MLP block using CUTLASS FP16 tensor cores with fused GELU activation.

    This implementation:
    1. Converts inputs to FP16 for the GEMM computation
    2. Uses tensor cores for efficient matrix multiplication
    3. Fuses GELU activation into the first GEMM epilogue
    4. Accumulates in FP32 for numerical stability
    5. Returns FP32 output matching PyTorch baseline
    6. Includes timing for MLP isolation
    """
    def __init__(self, n_embd, resid_pdrop):
        super().__init__()
        self.n_embd = n_embd

        # Linear layers (for weight storage and PyTorch fallback)
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.act = NewGELU()
        self.dropout = nn.Dropout(resid_pdrop)

        # PyTorch fallback lambda
        self.mlpf = lambda x: self.dropout(self.c_proj(self.act(self.c_fc(x))))

        # Enable TF32 for consistency
        torch.backends.cuda.matmul.allow_tf32 = True

        # Timing storage
        self.mlp_time_ms = 0.0

    def forward(self, x, use_cutlass=True):
        """
        Forward pass using CUTLASS FP16 tensor core GEMM kernels.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, n_embd]
            use_cutlass (bool): If True, use CUTLASS; otherwise use PyTorch fallback

        Returns:
            torch.Tensor: Output tensor of shape [B, T, n_embd]
        """
        B, T, C = x.shape
        x_2d = x.view(B * T, C)

        if use_cutlass:
            # Time CUTLASS MLP forward
            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.autograd.profiler.record_function("mlp_cutlass"):
                output = _get_cutlass_ext().mlp_forward(
                    x_2d,
                    self.c_fc.weight,
                    self.c_fc.bias,
                    self.c_proj.weight,
                    self.c_proj.bias
                )

            torch.cuda.synchronize()
            self.mlp_time_ms = (time.perf_counter() - start) * 1000

            # Reshape back to 3D
            return output.view(B, T, C)
        else:
            # PyTorch fallback
            return self.mlpf(x_2d).view(B, T, C)


# =============================================================================
# Full Transformer Block
# =============================================================================

class ModelNew(nn.Module):
    """
    CUTLASS-accelerated MiniGPT Transformer Block.

    Architecture:
    - Attention: PyTorch baseline (CausalSelfAttention)
    - MLP: CUTLASS FP16 tensor cores with fused GELU
    - Includes timing instrumentation for MLP isolation
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.n_embd = n_embd

        # LayerNorm layers
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)

        # Attention block (PyTorch baseline)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)

        # MLP block (CUTLASS accelerated)
        self.mlp = MLPBlockNew(n_embd, resid_pdrop)

        # Enable TF32 for consistency
        torch.backends.cuda.matmul.allow_tf32 = True

    def forward(self, x, use_cutlass_mlp=True):
        """
        Forward pass with optional CUTLASS MLP.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, n_embd]
            use_cutlass_mlp (bool): If True, use CUTLASS for MLP; otherwise use PyTorch

        Returns:
            torch.Tensor: Output tensor of shape [B, T, n_embd]
        """
        # Attention block with residual connection (PyTorch)
        x = x + self.attn(self.ln_1(x))

        # MLP block with residual connection (CUTLASS or PyTorch)
        x = x + self.mlp(self.ln_2(x), use_cutlass=use_cutlass_mlp)

        return x

    def get_mlp_time_ms(self):
        """Get the last MLP forward pass time in milliseconds."""
        return self.mlp.mlp_time_ms


# =============================================================================
# Model Configuration (matches 44_MiniGPTBlock.py)
# =============================================================================

batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0


def get_inputs():
    """Get test inputs matching the original model dimensions."""
    return [torch.rand(batch_size, seq_len, n_embd)]


def get_init_inputs():
    """Initialization inputs for the model."""
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
