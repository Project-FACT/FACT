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


class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.max_seqlen = max_seqlen

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        self.ext = _get_cutlass_ext()

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.ext:
            y = self._cutlass_fmha_forward(q, k, v)
        else:
            y = self._pytorch_forward(q, k, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

    def _cutlass_fmha_forward(self, q, k, v):
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
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        causal_mask = torch.tril(torch.ones(q.size(2), q.size(2), device=q.device, dtype=torch.bool))
        att = att.masked_fill(~causal_mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        return y


class Model(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc = nn.Linear(n_embd, 4 * n_embd),
            c_proj = nn.Linear(4 * n_embd, n_embd),
            act = NewGELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0


def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd)]


def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
