"""
AGT-v3 POC — Grouped Query Attention (GQA)

Causal self-attention with grouped query heads and
RoPE applied to Q/K.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norms import RMSNorm
from .rope import precompute_rope_frequencies, apply_rope


class GQA(nn.Module):
    """
    Grouped Query Attention.

    Queries have more heads than keys/values.
    Key/value heads are repeated across query groups.

    Input:
        x: [batch, seq_len, dim]

    Output:
        [batch, seq_len, dim]
    """

    def __init__(
        self,
        dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        rope_base: float = 10000.0,
        bias: bool = False,
    ):
        super().__init__()

        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                "num_query_heads must be divisible by num_kv_heads."
            )

        self.dim = dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        self.q_proj = nn.Linear(
            dim,
            num_query_heads * head_dim,
            bias=bias,
        )

        self.k_proj = nn.Linear(
            dim,
            num_kv_heads * head_dim,
            bias=bias,
        )

        self.v_proj = nn.Linear(
            dim,
            num_kv_heads * head_dim,
            bias=bias,
        )

        self.o_proj = nn.Linear(
            num_query_heads * head_dim,
            dim,
            bias=bias,
        )

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        cos, sin = precompute_rope_frequencies(
            head_dim,
            max_seq_len,
            base=rope_base,
        )

        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, T, heads, head_dim]
        q = q.view(
            B, T,
            self.num_query_heads,
            self.head_dim,
        )

        k = k.view(
            B, T,
            self.num_kv_heads,
            self.head_dim,
        )

        v = v.view(
            B, T,
            self.num_kv_heads,
            self.head_dim,
        )

        # Normalize per head
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q = apply_rope(
            q,
            self.rope_cos,
            self.rope_sin,
        )

        k = apply_rope(
            k,
            self.rope_cos,
            self.rope_sin,
        )

        # Convert to [B, heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat K/V heads across query groups
        groups = self.num_query_heads // self.num_kv_heads

        if groups > 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

        # Causal attention
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )

        # [B, heads, T, head_dim] -> [B, T, heads * head_dim]
        y = y.transpose(1, 2).contiguous()

        y = y.view(
            B,
            T,
            self.num_query_heads * self.head_dim,
        )

        return self.o_proj(y)
