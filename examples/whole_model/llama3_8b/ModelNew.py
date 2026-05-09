"""
Composed ModelNew: Llama 3 8B Block with Multi-Pattern Optimization
====================================================================

Pattern Composition:
  - p1: FMHA (Fused Multi-Head Attention)
  - p2: SwiGLU MLP Fusion

Target GPU: A100 (SM80)
Target Model: KernelBench/KernelBench/level3/51_Llama3_8B_Block.py
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# =============================================================================
# Load FMHA Extension
# =============================================================================

_fmha_ext = None
_fmha_available = False


def _load_fmha_extension():
    """Load FMHA extension from pattern directory."""
    global _fmha_ext, _fmha_available

    if _fmha_ext is not None:
        return _fmha_ext

    try:
        # Get the pattern directory
        # FACT/examples/whole_model/llama3_8b/<this_file> -> FACT root = dirname^3
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fact_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        fmha_pattern_dir = os.path.join(
            fact_root,
            "pattern_table/fmha/fp16/sm80/fmha_llama3_gqa"
        )

        # Add pattern directory to sys.path
        import sys
        if fmha_pattern_dir not in sys.path:
            sys.path.insert(0, fmha_pattern_dir)

        # Import the FMHA extension
        import importlib.util
        fmha_module_name = "fmha_llama3_fp16_gqa_module"
        if fmha_module_name in sys.modules:
            fmha_module = sys.modules[fmha_module_name]
        else:
            # Load the ModelNew which has the extension embedded
            spec = importlib.util.spec_from_file_location(
                fmha_module_name,
                os.path.join(fmha_pattern_dir, "ModelNew.py")
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load FMHA module from {fmha_pattern_dir}")
            fmha_module = importlib.util.module_from_spec(spec)
            sys.modules[fmha_module_name] = fmha_module
            spec.loader.exec_module(fmha_module)

        # Get the extension
        _fmha_ext = fmha_module._get_cutlass_ext()
        _fmha_available = True if _fmha_ext else False

    except Exception as e:
        print(f"Warning: Failed to load FMHA extension: {e}")
        _fmha_available = False
        _fmha_ext = False

    return _fmha_ext


def get_fmha_extension():
    """Get FMHA extension."""
    if _fmha_ext is None:
        _load_fmha_extension()
    return _fmha_ext


# =============================================================================
# Load SwiGLU Extension
# =============================================================================

_swiglu_ext = None
_swiglu_available = False


def _load_swiglu_extension():
    """Load SwiGLU extension from pattern directory."""
    global _swiglu_ext, _swiglu_available

    if _swiglu_ext is not None:
        return _swiglu_ext

    try:
        # Get the pattern directory
        # FACT/examples/whole_model/llama3_8b/<this_file> -> FACT root = dirname^3
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fact_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        swiglu_pattern_dir = os.path.join(
            fact_root,
            "pattern_table/gemm/fp16/sm80/swiglu_mlp_fusion"
        )

        # Add pattern directory to sys.path
        import sys
        if swiglu_pattern_dir not in sys.path:
            sys.path.insert(0, swiglu_pattern_dir)

        # Import the SwiGLU extension
        import importlib.util
        swiglu_module_name = "swiglu_mlp_fusion_fp16_module"
        if swiglu_module_name in sys.modules:
            swiglu_module = sys.modules[swiglu_module_name]
        else:
            # Load the ModelNew which has the extension embedded
            spec = importlib.util.spec_from_file_location(
                swiglu_module_name,
                os.path.join(swiglu_pattern_dir, "ModelNew.py")
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load SwiGLU module from {swiglu_pattern_dir}")
            swiglu_module = importlib.util.module_from_spec(spec)
            sys.modules[swiglu_module_name] = swiglu_module
            spec.loader.exec_module(swiglu_module)

        # Get the extension
        _swiglu_ext = swiglu_module.get_cutlass_module()
        _swiglu_available = True if _swiglu_ext else False

    except Exception as e:
        print(f"Warning: Failed to load SwiGLU extension: {e}")
        _swiglu_available = False
        _swiglu_ext = False

    return _swiglu_ext


def get_swiglu_extension():
    """Get SwiGLU extension."""
    if _swiglu_ext is None:
        _load_swiglu_extension()
    return _swiglu_ext


# =============================================================================
# Model Components
# =============================================================================

class RMSNorm(nn.Module):
    """Llama uses RMSNorm instead of LayerNorm."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLUBaseline(nn.Module):
    """PyTorch baseline SwiGLU MLP."""
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class SwiGLUCutlass(nn.Module):
    """CUTLASS-accelerated SwiGLU MLP."""
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dim = dim
        self.hidden_dim = hidden_dim

        # Create linear layers (for weight storage)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

        # Get CUTLASS module
        self.cutlass = get_swiglu_extension()

    def forward(self, x):
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, x.size(-1))

        M = x.size(0)
        x = x.contiguous()

        # Get weights
        gate_weight = self.gate_proj.weight
        up_weight = self.up_proj.weight
        down_weight = self.down_proj.weight

        # Call CUTLASS kernel
        output = self.cutlass.swiglu_mlp_forward(
            x, gate_weight, up_weight, down_weight
        )

        if len(orig_shape) == 3:
            output = output.view(orig_shape)

        return output


class LlamaAttention(nn.Module):
    """Llama multi-head self-attention with GQA."""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_seqlen: int,
                 use_cutlass: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.repeat_kv_heads = n_heads // n_kv_heads
        self.use_cutlass = use_cutlass

        # QKV projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim // n_heads * n_kv_heads, bias=False)
        self.v_proj = nn.Linear(dim, dim // n_heads * n_kv_heads, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, position_ids=None):
        B, T, C = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: repeat KV heads
        if self.repeat_kv_heads > 1:
            k = k.repeat_interleave(self.repeat_kv_heads, dim=1)
            v = v.repeat_interleave(self.repeat_kv_heads, dim=1)

        if self.use_cutlass:
            out = self._cutlass_forward(q, k, v)
        else:
            out = self._pytorch_forward(q, k, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)

    def _cutlass_forward(self, q, k, v):
        """CUTLASS FMHA kernel."""
        B, nh, T, hs = q.shape

        q_bmhk = q.transpose(1, 2).contiguous()
        k_bmhk = k.transpose(1, 2).contiguous()
        v_bmhk = v.transpose(1, 2).contiguous()

        scale = 1.0 / math.sqrt(hs)

        ext = get_fmha_extension()
        output = ext.fmha_forward(
            q_bmhk, k_bmhk, v_bmhk,
            scale, B, nh, T, T, hs, hs
        )

        output = output.view(B, T, nh, hs).transpose(1, 2).contiguous()
        return output

    def _pytorch_forward(self, q, k, v):
        """PyTorch baseline."""
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        attn = F.softmax(scores, dim=-1)
        return attn @ v


class LlamaDecoderLayer(nn.Module):
    """Single Llama transformer decoder block with pattern switches."""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, intermediate_dim: int,
                 rms_norm_eps: float = 1e-5,
                 enable_fmha: bool = True,
                 enable_swiglu: bool = True):
        super().__init__()

        self.enable_fmha = enable_fmha
        self.enable_swiglu = enable_swiglu

        # Determine which implementations to use
        use_fmha_cutlass = enable_fmha and get_fmha_extension()
        use_swiglu_cutlass = enable_swiglu and get_swiglu_extension()

        # Attention
        self.self_attn = LlamaAttention(
            dim, n_heads, n_kv_heads, max_seqlen=8192,
            use_cutlass=use_fmha_cutlass
        )

        # MLP
        if use_swiglu_cutlass:
            self.mlp = SwiGLUCutlass(dim, intermediate_dim)
        else:
            self.mlp = SwiGLUBaseline(dim, intermediate_dim)

        # Normalization
        self.input_layernorm = RMSNorm(dim, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(dim, eps=rms_norm_eps)

    def forward(self, x):
        # Pre-norm architecture
        # Attention block
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x

        # MLP block
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class Model(nn.Module):
    """
    Composed Llama 3 8B Block with multi-pattern optimization.

    Args:
        dim: Hidden dimension (default: 4096 for Llama 3 8B)
        n_heads: Number of attention heads (default: 32)
        n_kv_heads: Number of key-value heads for GQA (default: 8)
        intermediate_dim: MLP intermediate dimension (default: 14336)
        rms_norm_eps: RMSNorm epsilon (default: 1e-5)
        enable_fmha: Enable FMHA pattern (default: True)
        enable_swiglu: Enable SwiGLU MLP pattern (default: True)
    """
    def __init__(self, dim: int = 4096, n_heads: int = 32, n_kv_heads: int = 8,
                 intermediate_dim: int = 14336, rms_norm_eps: float = 1e-5,
                 enable_fmha: bool = True, enable_swiglu: bool = True):
        super().__init__()

        self.enable_fmha = enable_fmha
        self.enable_swiglu = enable_swiglu

        # Load extensions
        _load_fmha_extension()
        _load_swiglu_extension()

        self.decoder_layer = LlamaDecoderLayer(
            dim, n_heads, n_kv_heads, intermediate_dim, rms_norm_eps,
            enable_fmha=enable_fmha,
            enable_swiglu=enable_swiglu
        )

    def forward(self, x):
        return self.decoder_layer(x)

    def get_pattern_status(self):
        """Get pattern availability and enabled status."""
        fmha_ext = get_fmha_extension()
        swiglu_ext = get_swiglu_extension()

        return {
            'fmha_available': fmha_ext is not False,
            'swiglu_available': swiglu_ext is not False,
            'fmha_enabled': self.enable_fmha,
            'swiglu_enabled': self.enable_swiglu,
            'fmha_config': {
                'queries_per_block': 32,
                'keys_per_block': 128,
                'aligned': True
            },
            'swiglu_config': {
                'tile': [128, 128, 32],
                'warp': [64, 64, 32],
                'stages': 3
            }
        }


# =============================================================================
# KernelBench Test Configuration
# =============================================================================

batch_size = 16
seq_len = 2048
dim = 4096
n_heads = 32
n_kv_heads = 8
intermediate_dim = 14336
rms_norm_eps = 1e-5


def get_inputs():
    """Return input tensors for the model."""
    return [torch.rand(batch_size, seq_len, dim)]


def get_init_inputs():
    """Return initialization arguments for the model."""
    return [dim, n_heads, n_kv_heads, intermediate_dim, rms_norm_eps]
