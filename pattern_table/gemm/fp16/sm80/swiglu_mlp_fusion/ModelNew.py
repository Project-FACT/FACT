"""
ModelNew: CUTLASS-accelerated Llama 3 8B Single Transformer Block
Pattern: SwiGLU MLP Fusion (gate_proj + SiLU + up_proj + multiply + down_proj)
Target: Ampere (A100) SM80

This implementation uses CUTLASS FP16 tensor cores for the MLP portion,
while attention remains in PyTorch (baseline).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from typing import Optional, Tuple


# Load CUTLASS extension
def _load_cutlass_extension():
    """JIT-load the CUTLASS extension."""
    # Get the directory containing this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_dir = os.path.join(current_dir, 'cutlass_kernels')

    # Set CUTLASS_ROOT if not already set
    if 'CUTLASS_ROOT' not in os.environ:
        # Try to find CUTLASS from the project structure
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        cutlass_root = os.path.join(project_root, 'cutlass')
        if os.path.exists(cutlass_root):
            os.environ['CUTLASS_ROOT'] = cutlass_root

    # JIT load the extension
    from torch.utils.cpp_extension import load

    # Get CUDA architecture
    cuda_arch = os.environ.get('GPU_ARCH', 'sm_80')
    arch_list = cuda_arch.replace('sm_', '').split(';')

    extension = load(
        name='swiglu_mlp_fusion',
        sources=[
            os.path.join(kernel_dir, 'swiglu_mlp_fusion.cu'),
            os.path.join(kernel_dir, 'main.cpp')
        ],
        extra_cuda_cflags=[
            f'-gencode=arch=compute_{arch_list[0]},code=sm_{arch_list[0]}',
            '-lineinfo',
            f'-I{os.environ.get("CUTLASS_ROOT", "/usr/local/cuda")}/include',
            '-DNDEBUG',
            '-O3'
        ],
        extra_cflags=['-O3'],
        with_cuda=True,
        verbose=os.environ.get('KERNELBENCH_CUTLASS_VERBOSE', '0') == '1'
    )

    return extension


# Lazy load the extension
_cutlass_module = None


def get_cutlass_module():
    """Get or lazily load the CUTLASS extension."""
    global _cutlass_module
    if _cutlass_module is None:
        _cutlass_module = _load_cutlass_extension()
    return _cutlass_module


class RMSNorm(nn.Module):
    """Llama uses RMSNorm instead of LayerNorm."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLUMLPNew(nn.Module):
    """
    CUTLASS-accelerated SwiGLU MLP implementation.

    This uses custom CUTLASS kernels for:
    - gate_proj + SiLU + up_proj + elementwise multiply (fused)
    - down_proj (separate kernel)

    The CUTLASS implementation uses FP16 tensor cores with FP32 accumulation
    for maximum throughput while maintaining numerical stability.
    """
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        # Llama uses hidden_dim = 2/3 * 4 * dim, rounded to multiple_of
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # Store dimensions
        self.dim = dim
        self.hidden_dim = hidden_dim

        # Create linear layers (for compatibility with state_dict)
        # These will be used for weight storage only
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

        # Get CUTLASS module
        self.cutlass = get_cutlass_module()

    def forward(self, x):
        """
        Forward pass using CUTLASS kernels.

        Args:
            x: Input tensor [M, dim] where M = batch_size * seq_len

        Returns:
            Output tensor [M, dim]
        """
        # Reshape if needed: [batch, seq_len, dim] -> [batch*seq_len, dim]
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, x.size(-1))

        M = x.size(0)

        # Ensure contiguous layout
        x = x.contiguous()

        # Get weights (transposed for GEMM: [out_dim, in_dim])
        # PyTorch Linear stores weights as [out_dim, in_dim]
        # CUTLASS expects column-major, which matches this layout
        gate_weight = self.gate_proj.weight  # [hidden_dim, dim]
        up_weight = self.up_proj.weight      # [hidden_dim, dim]
        down_weight = self.down_proj.weight  # [dim, hidden_dim]

        # Call CUTLASS kernel for complete SwiGLU MLP
        # Input: [M, dim], Output: [M, dim]
        output = self.cutlass.swiglu_mlp_forward(
            x,
            gate_weight,
            up_weight,
            down_weight
        )

        # Restore original shape if needed
        if len(orig_shape) == 3:
            output = output.view(orig_shape)

        return output


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary positional embedding to query and key."""
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaAttentionNew(nn.Module):
    """
    Llama multi-head self-attention with Grouped-Query Attention (GQA) and RoPE.

    This version uses PyTorch baseline for attention (not optimized).
    """
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

    def forward(self, x, position_ids=None):
        B, T, C = x.shape

        if position_ids is None:
            position_ids = torch.arange(T, device=x.device).unsqueeze(0)

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

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


class LlamaDecoderLayerNew(nn.Module):
    """
    Single Llama transformer decoder block (one layer).

    Hybrid optimization:
    - Attention: PyTorch baseline (not optimized)
    - MLP: CUTLASS FP16 tensor cores (optimized)
    """
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, intermediate_dim: int,
                 rms_norm_eps: float = 1e-5):
        super().__init__()
        self.self_attn = LlamaAttentionNew(dim, n_heads, n_kv_heads, max_seqlen=8192)
        self.mlp = SwiGLUMLPNew(dim, intermediate_dim)
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


class ModelNew(nn.Module):
    """
    CUTLASS-accelerated Llama 3 8B Single Transformer Block.

    Hybrid optimization strategy:
    - Full Attention: PyTorch baseline
    - SwiGLU MLP: CUTLASS FP16 tensor cores

    This matches the original Model.py structure but optimizes
    only the MLP portion with CUTLASS.
    """
    def __init__(self, dim: int = 4096, n_heads: int = 32, n_kv_heads: int = 8,
                 intermediate_dim: int = 14336, rms_norm_eps: float = 1e-5):
        super().__init__()
        self.decoder_layer = LlamaDecoderLayerNew(dim, n_heads, n_kv_heads,
                                                       intermediate_dim, rms_norm_eps)

    def forward(self, x):
        return self.decoder_layer(x)

    def copy_weights_from_baseline(self, baseline_model):
        """
        Copy weights from baseline Model for fair comparison.

        This ensures both models have identical weights for correctness testing.
        """
        # Copy RMSNorm weights
        self.decoder_layer.input_layernorm.weight.data.copy_(
            baseline_model.decoder_layer.input_layernorm.weight.data)
        self.decoder_layer.post_attention_layernorm.weight.data.copy_(
            baseline_model.decoder_layer.post_attention_layernorm.weight.data)

        # Copy attention weights
        self.decoder_layer.self_attn.q_proj.weight.data.copy_(
            baseline_model.decoder_layer.self_attn.q_proj.weight.data)
        self.decoder_layer.self_attn.k_proj.weight.data.copy_(
            baseline_model.decoder_layer.self_attn.k_proj.weight.data)
        self.decoder_layer.self_attn.v_proj.weight.data.copy_(
            baseline_model.decoder_layer.self_attn.v_proj.weight.data)
        self.decoder_layer.self_attn.o_proj.weight.data.copy_(
            baseline_model.decoder_layer.self_attn.o_proj.weight.data)

        # Copy MLP weights
        self.decoder_layer.mlp.gate_proj.weight.data.copy_(
            baseline_model.decoder_layer.mlp.gate_proj.weight.data)
        self.decoder_layer.mlp.up_proj.weight.data.copy_(
            baseline_model.decoder_layer.mlp.up_proj.weight.data)
        self.decoder_layer.mlp.down_proj.weight.data.copy_(
            baseline_model.decoder_layer.mlp.down_proj.weight.data)
