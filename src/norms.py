"""
AGT-v3 POC — Normalization Layers

RMSNorm implementation used throughout the architecture.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes the last dimension without subtracting the mean:

        x_hat = x / sqrt(mean(x^2) + eps)
        y     = weight * x_hat
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.dim = dim
        self.eps = eps

        # Learnable gain
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS over the final dimension
        rms = torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )

        return x * rms * self.weight
