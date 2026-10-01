"""
AGT-v3 POC — Manifold-Constrained Hyper-Connections (mHC)

mHC maintains a learned residual mixing matrix while
constraining the effective mixing through row/column
normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MHC(nn.Module):
    """
    Manifold-constrained residual mixing.

    Input:
        x: [B, T, D]

    Output:
        [B, T, D]

    The mixing matrix is learned in unconstrained form and
    projected to a stable doubly-stochastic-like form.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.dim = dim
        self.eps = eps

        # Unconstrained residual mixing logits.
        self.logits = nn.Parameter(
            torch.zeros(dim, dim)
        )

        # Learnable residual scale.
        self.scale = nn.Parameter(
            torch.ones(dim)
        )

    def mixing_matrix(self):
        """
        Construct a positive row-normalized mixing matrix.

        Sinkhorn-style alternating normalization gives a
        stable doubly-stochastic approximation.
        """

        matrix = torch.exp(
            self.logits.clamp(-10.0, 10.0)
        )

        for _ in range(5):
            matrix = matrix / (
                matrix.sum(dim=-1, keepdim=True)
                + self.eps
            )

            matrix = matrix / (
                matrix.sum(dim=-2, keepdim=True)
                + self.eps
            )

        return matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply constrained residual mixing.
        """

        matrix = self.mixing_matrix()

        # Mix the final feature dimension.
        y = torch.matmul(x, matrix.T)

        return y * self.scale
