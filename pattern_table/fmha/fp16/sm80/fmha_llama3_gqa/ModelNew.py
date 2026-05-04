import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.utils.cpp_extension import load


def _build_ext():
    """Load CUTLASS FMHA kernel as PyTorch extension."""
    cutlass_kernels_dir = os.path.join(os.path.dirname(__file__), "cutlass_kernels")

    sources = [
        os.path.join(cutlass_kernels_dir, "fmha_launch.cu"),
        os.path.join(cutlass_kernels_dir, "main.cpp"),
    ]

    from kernelbench.cutlass_cpp.runtime import cutlass_include_paths, default_extra_cuda_cflags

    # Add local FMHA headers to include path
    fmha_include = os.path.join(cutlass_kernels_dir, "fmha")

    extra_cuda_cflags = default_extra_cuda_cflags(gpu_arch="80") + [
        f"-I{fmha_include}",  # Include local FMHA headers
    ]

    ext = load(
        name="fmha_ext",
        sources=sources,
        extra_include_paths=cutlass_include_paths() + [fmha_include],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=os.environ.get("KERNELBENCH_CUTLASS_VERBOSE", "0") == "1",
    )
    return ext


_ext = None
_cutlass_available = None  # None=unknown, True=available, False=not available


def _get_cutlass_ext():
    global _ext, _cutlass_available
    if _ext is None:
        try:
            _ext = _build_ext()
            _cutlass_available = True  # Only set to True on successful load
        except Exception as e:
            print(f"Warning: CUTLASS FMHA kernel compilation failed: {e}")
            print("Falling back to PyTorch implementation")
            _ext = False
            _cutlass_available = False
    return _ext


# Property for backward compatibility
def __getattr__(name):
    """Lazy accessor for CUTLASS_AVAILABLE."""
    if name == "CUTLASS_AVAILABLE":
        global _cutlass_available
        if _cutlass_available is None:
            # Try loading the extension to determine availability
            _get_cutlass_ext()
        return _cutlass_available
    raise AttributeError("module '" + name + "' has no attribute '" + name + "'")


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


class LlamaAttention(nn.Module):
    """Llama multi-head self-attention with Grouped-Query Attention (GQA).
    CUTLASS FMHA kernel replaces the core attention computation.
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

        self.ext = _get_cutlass_ext()

    def forward(self, x, position_ids=None):
        B, T, C = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: repeat KV heads to match Q heads BEFORE FMHA kernel
        if self.repeat_kv_heads > 1:
            k = k.repeat_interleave(self.repeat_kv_heads, dim=1)
            v = v.repeat_interleave(self.repeat_kv_heads, dim=1)

        # Now we have standard MHA inputs (n_heads Q, n_heads K, n_heads V after GQA repetition)
        if self.ext:
            out = self._cutlass_fmha_forward(q, k, v)
        else:
            out = self._pytorch_forward(q, k, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)

    def _cutlass_fmha_forward(self, q, k, v):
        """CUTLASS FMHA kernel for core attention computation."""
        B, nh, T, hs = q.shape

        # Contiguous tensors in BMHK format expected by CUTLASS
        q_bmhk = q.transpose(1, 2).contiguous()  # (B, T, nh, hs)
        k_bmhk = k.transpose(1, 2).contiguous()
        v_bmhk = v.transpose(1, 2).contiguous()

        scale = 1.0 / math.sqrt(hs)

        output = self.ext.fmha_forward(
            q_bmhk,
            k_bmhk,
            v_bmhk,
            scale,
            B,
            nh,
            T,
            T,
            hs,
            hs
        )

        output = output.view(B, T, nh, hs).transpose(1, 2).contiguous()
        return output

    def _pytorch_forward(self, q, k, v):
        """PyTorch fallback implementation."""
        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        return out


class LlamaDecoderLayer(nn.Module):
    """Single Llama transformer decoder block (one layer) with CUTLASS FMHA."""
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
    """Wrapper for KernelBench compatibility - single Llama decoder layer with CUTLASS FMHA."""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, intermediate_dim: int,
                 rms_norm_eps: float = 1e-5):
        super().__init__()
        self.decoder_layer = LlamaDecoderLayer(dim, n_heads, n_kv_heads,
                                               intermediate_dim, rms_norm_eps)

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
