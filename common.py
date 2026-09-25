"""Common modules and helper classes for Anima IP-Adapter."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.ops

SIGLIP_HIDDEN_SIZE = 768
SIGLIP_NUM_TOKENS = 1024
SIGLIP_IMAGE_SIZE = 512


def _make_linear(in_features, out_features, bias=True, device=None, dtype=None):
    """Linear layer that casts stored weights to the runtime activation dtype."""
    if dtype is None or not dtype.is_floating_point:
        dtype = torch.bfloat16
    layer = comfy.ops.manual_cast.Linear(
        in_features, out_features, bias=bias, device=device, dtype=dtype
    )
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer.weight is None:
        layer.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs),
            requires_grad=False,
        )
    if bias and layer.bias is None:
        layer.bias = nn.Parameter(
            torch.empty(out_features, **factory_kwargs),
            requires_grad=False,
        )
    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
    if layer.bias is not None:
        bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
        nn.init.uniform_(layer.bias, -bound, bound)
    return layer


def _make_shared_proj(dim, inner_dim, device, dtype):
    """Shared projection module: Linear -> GELU -> Linear."""
    if dtype is None or not dtype.is_floating_point:
        dtype = torch.bfloat16
    class _Proj(nn.Module):
        def __init__(self):
            super().__init__()
            self.expand = _make_linear(dim, inner_dim, dtype=dtype)
            self.act = nn.GELU()
            self.project = _make_linear(inner_dim, inner_dim, dtype=dtype)

        def forward(self, x):
            x = self.expand(x)
            x = self.act(x)
            x = self.project(x)
            return x

    return _Proj().to(device=device, dtype=dtype)


def _make_shared_proj_ref(shared_module):
    """Callable reference to shared module, auto-moves to input device."""
    class _Ref:
        def __call__(self, x):
            shared_module.to(x.device)
            return shared_module(x)

    return _Ref()


class _LoRALinear(nn.Module):
    """Minimal LoRA wrapper for inference."""
    def __init__(self, base: nn.Linear, lora_A: torch.Tensor, lora_B: torch.Tensor, scale: float):
        super().__init__()
        self.base = base
        base_dtype = lora_A.dtype if lora_A.dtype.is_floating_point else torch.bfloat16
        base_device = base.weight.device if hasattr(base, 'weight') and base.weight is not None else None
        self.lora_A = nn.Parameter(lora_A.to(device=base_device, dtype=base_dtype))
        self.lora_B = nn.Parameter(lora_B.to(device=base_device, dtype=base_dtype))
        self.scale = scale

    def forward(self, x):
        lora_A = self.lora_A.to(device=x.device, dtype=x.dtype)
        lora_B = self.lora_B.to(device=x.device, dtype=x.dtype)
        return self.base(x) + (x @ lora_A.T @ lora_B.T) * self.scale


class _SigLIPCompressorLayer(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm_self = nn.LayerNorm(dim)
        self.self_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.self_out = nn.Linear(dim, dim, bias=False)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, q, kv):
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        Q = self._heads(self.q_proj(q_norm))
        K = self._heads(self.k_proj(kv_norm))
        V = self._heads(self.v_proj(kv_norm))
        out = F.scaled_dot_product_attention(Q, K, V)
        q = q + self.out_proj(self._unheads(out))
        qn = self.norm_self(q)
        qkv = self.self_qkv(qn).chunk(3, dim=-1)
        Q2, K2, V2 = [self._heads(t) for t in qkv]
        s_out = F.scaled_dot_product_attention(Q2, K2, V2)
        q = q + self.self_out(self._unheads(s_out))
        q = q + self.ffn(self.norm_ffn(q))
        return q

    def _heads(self, x):
        B, S, D = x.shape
        return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def _unheads(self, x):
        B, H, S, D = x.shape
        return x.transpose(1, 2).reshape(B, S, H * D)


class SigLIPCompressor(nn.Module):
    def __init__(self, dim=768, num_queries=64, num_layers=2):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.layers = nn.ModuleList([
            _SigLIPCompressorLayer(dim) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        B = x.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, x)
        return self.final_norm(q)


class IPSelfAttn(nn.Module):
    """1-layer self-attention with SDPA (Flash Attention) for O(N) memory."""
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(dim * 4, dim, bias=False),
        )

    def forward(self, x):
        B, S, D = x.shape
        normed = self.norm1(x)
        qkv = self.qkv(normed).chunk(3, dim=-1)
        q = qkv[0].view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = qkv[1].view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = qkv[2].view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, S, D)
        x = x + self.out_proj(attn)
        x = x + self.ffn(self.norm2(x))
        return x
