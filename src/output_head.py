
import torch
import torch.nn as nn

from .norms import RMSNorm


class OutputHead(nn.Module):
    """
    AGT-v3 next-token prediction head.

    Macro boundary:

        H_t -> p(x_{t+1} | H_t)
    """

    def __init__(
        self,
        dim: int = 256,
        vocab_size: int = 50257,
    ):
        super().__init__()

        self.dim = dim
        self.vocab_size = vocab_size

        self.proj = nn.Linear(
            dim,
            vocab_size,
            bias=False,
        )

    def forward(self, H_t):
        """
        H_t:
            [B, D]

        returns:
            logits:
                [B, vocab_size]
        """

        return self.proj(H_t)

    def probabilities(self, H_t):
        """
        Convert logits to next-token probabilities.
        """

        return torch.softmax(
            self.forward(H_t),
            dim=-1,
        )
