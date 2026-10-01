
import torch
import torch.nn as nn

from .norms import RMSNorm


class TerminalCommit(nn.Module):
    """
    AGT-v3 terminal commit.

    Locked macro boundary:

        h_terminal -> H_t

    Persistent state is updated only after adaptive execution
    terminates. In-flight execution must not mutate H.
    """

    def __init__(
        self,
        dim: int = 256,
    ):
        super().__init__()

        self.dim = dim

        # Provisional decision:
        # s_t = RMSNorm_f(h_terminal)
        self.state_norm = RMSNorm(dim)

    def forward(
        self,
        h_terminal,
    ):
        """
        h_terminal:
            [B, D]

        returns:
            H_t:
                [B, D]
        """

        H_t = self.state_norm(
            h_terminal
        )

        return H_t

    @staticmethod
    def commit_once(
        persistent_state,
        terminal_state,
    ):
        """
        Functional single-commit operation.

        Returns a new persistent state without mutating the original.
        """

        return terminal_state.clone()
