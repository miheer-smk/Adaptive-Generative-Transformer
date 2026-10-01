"""
AGT-v3 POC — SwiGLU Feed-Forward Network
"""

import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward block.

    GLU formulation:

        y = SiLU(x W_gate) * (x W_up)

    followed by:

        output = y W_down
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        bias: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.gate_proj = nn.Linear(
            dim,
            hidden_dim,
            bias=bias,
        )

        self.up_proj = nn.Linear(
            dim,
            hidden_dim,
            bias=bias,
        )

        self.down_proj = nn.Linear(
            hidden_dim,
            dim,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(
            self.gate_proj(x)
        )

        up = self.up_proj(x)

        return self.down_proj(gate * up)
