"""
Ablation Model: MLP Only (PyTorch FMHA)

This model isolates the MLP GEMM Fusion pattern speedup:
- FMHA: PyTorch baseline fallback
- MLP: CUTLASS optimized kernel

Purpose: Measure individual MLP contribution to overall speedup.
"""

import importlib.util
import math
import os
import sys
from pathlib import Path


def _ensure_cutlass_agent_runtime_paths() -> None:
    """Ensure kernelbench is importable and CUTLASS_ROOT is set."""
    # FACT/examples/whole_model/miniGPT/<this_file> -> FACT root = parent^3
    repo_root = Path(__file__).resolve().parent.parent.parent
    kb_src = repo_root / "kernelbench_cutlass_module" / "src"
    if kb_src.is_dir():
        s = str(kb_src)
        if s not in sys.path:
            sys.path.insert(0, s)
    if not os.environ.get("CUTLASS_ROOT", "").strip():
        cutlass = repo_root / "cutlass"
        if (cutlass / "include" / "cutlass").is_dir():
            os.environ["CUTLASS_ROOT"] = str(cutlass)


_ensure_cutlass_agent_runtime_paths()

import torch
import torch.nn as nn
import torch.nn.functional as F

# Path calculations
_FACT_ROOT = Path(__file__).resolve().parent.parent.parent
mlp_pattern_dir = str(_FACT_ROOT / "pattern_table/gemm/tf32/sm80/mlp_gemm_fusion_gelu")


# =============================================================================
# Extension Loading (MLP only)
# =============================================================================

def _load_pattern_modelnew(directory: str, unique_name: str):
    """Load a pattern's ModelNew.py from disk."""
    path = Path(directory) / "ModelNew.py"
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load MLP extension only
try:
    _mlp_mod = _load_pattern_modelnew(mlp_pattern_dir, "whole_model_mlp_pattern_modelnew")
    _mlp_ext = _mlp_mod._get_cutlass_ext()
    _mlp_available = True if _mlp_ext else False
    print("✓ MLP extension loaded (MLP-only mode)")
except Exception as e:
    print(f"Warning: Failed to load MLP extension: {e}")
    _mlp_available = False
    _mlp_ext = None

# FMHA is disabled in this ablation model
_fmha_available = False
_fmha_ext = None


# =============================================================================
# Model Components
# =============================================================================

class NewGELU(nn.Module):
    """GELU activation function matching PyTorch's NewGELU."""
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    CausalSelfAttention - EXACT baseline implementation (no CUTLASS in MLP-only mode).
    Matches 44_MiniGPTBlock.py exactly.
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


class MLPBlock(nn.Module):
    """
    MLP block with CUTLASS GEMM Fusion (PyTorch FMHA fallback).
    """

    def __init__(self, n_embd, resid_pdrop):
        super().__init__()
        self.n_embd = n_embd

        # Linear layers (weights are used by CUTLASS kernel)
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.act = NewGELU()
        self.dropout = nn.Dropout(resid_pdrop)
        # For baseline-compatible fallback
        self.mlpf = lambda x: self.dropout(self.c_proj(self.act(self.c_fc(x))))

    def forward(self, x):
        """Forward pass with CUTLASS MLP GEMM fusion."""
        # Reshape input for GEMM: (B, T, C) -> (B*T, C)
        B, T, C = x.shape
        x_2d = x.view(B * T, C)

        if _mlp_available:
            # Use CUTLASS MLP GEMM fusion kernel
            y_2d = self._cutlass_mlp_forward(x_2d)
        else:
            # Fallback to exact baseline implementation
            y_2d = self.mlpf(x_2d)

        # Reshape back: (B*T, C) -> (B, T, C)
        y = y_2d.view(B, T, C)
        return y

    def _cutlass_mlp_forward(self, x):
        """CUTLASS MLP GEMM fusion kernel."""
        output = _mlp_ext.mlp_forward(
            x,
            self.c_fc.weight, self.c_fc.bias,
            self.c_proj.weight, self.c_proj.bias
        )
        return output


class Model(nn.Module):
    """
    MiniGPT Transformer Block - MLP Only Ablation.

    FMHA: PyTorch baseline
    MLP: CUTLASS optimized
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()

        # LayerNorm layers
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)

        # Attention block (PyTorch)
        self.attn = CausalSelfAttention(
            n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen
        )

        # MLP block (CUTLASS)
        self.mlp = MLPBlock(n_embd, resid_pdrop)

    def forward(self, x):
        """Forward pass with MLP-only optimization."""
        # Attention block with residual connection
        x = x + self.attn(self.ln_1(x))

        # MLP block with residual connection
        x = x + self.mlp(self.ln_2(x))

        return x

    def get_pattern_status(self):
        """Get pattern availability status."""
        return {
            'fmha_available': False,  # Always False in MLP-only mode
            'mlp_available': _mlp_available if _mlp_ext else False,
            'mode': 'mlp_only',
            'fmha_config': {
                'queries_per_block': 64,
                'keys_per_block': 128,
                'aligned': True
            },
            'mlp_config': {
                'tile': [128, 256, 32],
                'warp': [64, 64, 32],
                'stages': 4
            }
        }


# =============================================================================
# KernelBench Test Configuration
# =============================================================================

batch_size = 128
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0
max_seqlen = 1024


def get_inputs():
    """Return input tensors for the model."""
    return [torch.rand(batch_size, seq_len, n_embd)]


def get_init_inputs():
    """Initialization arguments for the model."""
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
