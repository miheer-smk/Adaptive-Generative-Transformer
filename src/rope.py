"""
AGT-v3 POC — Rotary Positional Embedding (RoPE)
"""

import torch


def precompute_rope_frequencies(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device=None,
):
    """
    Precompute cosine and sine tables for RoPE.

    Args:
        dim: Head dimension. Must be even.
        max_seq_len: Maximum sequence length.
        base: RoPE frequency base.

    Returns:
        cos: [max_seq_len, dim // 2]
        sin: [max_seq_len, dim // 2]
    """

    if dim % 2 != 0:
        raise ValueError("RoPE dimension must be even.")

    inv_freq = 1.0 / (
        base ** (
            torch.arange(0, dim, 2, device=device).float() / dim
        )
    )

    positions = torch.arange(
        max_seq_len,
        device=device,
        dtype=torch.float32,
    )

    freqs = torch.outer(positions, inv_freq)

    return freqs.cos(), freqs.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    """
    Apply rotary position embedding.

    Expected x shape:
        [..., seq_len, dim]

    cos/sin shape:
        [seq_len, dim // 2]
    """

    seq_len = x.shape[-2]
    dim = x.shape[-1]

    if dim % 2 != 0:
        raise ValueError("RoPE dimension must be even.")

    cos = cos[:seq_len].to(dtype=x.dtype, device=x.device)
    sin = sin[:seq_len].to(dtype=x.dtype, device=x.device)

    # Split even/odd dimensions
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # Rotation
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    # Interleave even/odd dimensions again
    out = torch.stack(
        (rotated_even, rotated_odd),
        dim=-1,
    ).flatten(-2)

    return out
