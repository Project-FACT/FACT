import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Llama uses RMSNorm instead of LayerNorm."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    """Llama uses SwiGLU activation instead of GELU.
    SwiGLU(x) = Swish(xW) ⊗ (xV) where Swish(x) = x · σ(x)
    """
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        # Llama uses hidden_dim = 2/3 * 4 * dim, rounded to multiple_of
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary positional embedding to query and key."""
    # The first two dimensions are batch and head, rest are seq_len and head_dim
    cos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim//2]
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaAttention(nn.Module):
    """Llama multi-head self-attention with Grouped-Query Attention (GQA) and RoPE."""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_seqlen: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.repeat_kv_heads = n_heads // n_kv_heads

        # QKV projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim // n_heads * n_kv_heads, bias=False)
        self.v_proj = nn.Linear(dim, dim // n_heads * n_kv_heads, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # RoPE parameters (simplified - using fixed inv_freq)
        self.max_seqlen = max_seqlen
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids=None):
        B, T, C = x.shape

        if position_ids is None:
            position_ids = torch.arange(T, device=x.device).unsqueeze(0)

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE (simplified version - in practice, cos/sin would be precomputed)
        # For simplicity, we skip RoPE here since it doesn't affect the fusion pattern
        # In production implementation, cos/sin would be precomputed based on position_ids

        # GQA: repeat KV heads to match Q heads
        if self.repeat_kv_heads > 1:
            k = k.repeat_interleave(self.repeat_kv_heads, dim=1)
            v = v.repeat_interleave(self.repeat_kv_heads, dim=1)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.o_proj(out)


class LlamaDecoderLayer(nn.Module):
    """Single Llama transformer decoder block (one layer)."""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, intermediate_dim: int,
                 rms_norm_eps: float = 1e-5):
        super().__init__()
        self.self_attn = LlamaAttention(dim, n_heads, n_kv_heads, max_seqlen=8192)
        self.mlp = SwiGLU(dim, intermediate_dim)
        self.input_layernorm = RMSNorm(dim, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(dim, eps=rms_norm_eps)

    def forward(self, x):
        # Pre-norm architecture (standard in Llama)
        # Attention block with residual connection
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x

        # MLP block with residual connection
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class Model(nn.Module):
    """Wrapper for KernelBench compatibility - single Llama decoder layer."""
    def __init__(self, dim: int = 4096, n_heads: int = 32, n_kv_heads: int = 8,
                 intermediate_dim: int = 14336, rms_norm_eps: float = 1e-5):
        super().__init__()
        self.decoder_layer = LlamaDecoderLayer(dim, n_heads, n_kv_heads,
                                               intermediate_dim, rms_norm_eps)
        # Enable TF32 for A100 tensor cores (fair comparison with CUTLASS)
        torch.backends.cuda.matmul.allow_tf32 = True

    def forward(self, x):
        return self.decoder_layer(x)


# ============================================================================
# KernelBench Test Configuration - Llama 3 8B Single Block
# ============================================================================

# Llama 3 8B parameters
batch_size = 16        # Typical batch size for inference
seq_len = 2048         # Typical sequence length
dim = 4096             # hidden_size
n_heads = 32           # num_attention_heads
n_kv_heads = 8         # num_key_value_heads (GQA)
intermediate_dim = 14336  # intermediate_size (~3.5x expansion)
rms_norm_eps = 1e-5    # rms_norm_eps


def get_inputs():
    """Return input tensors for the model."""
    return [torch.rand(batch_size, seq_len, dim)]


def get_init_inputs():
    """Return initialization arguments for the model."""
    return [dim, n_heads, n_kv_heads, intermediate_dim, rms_norm_eps]
