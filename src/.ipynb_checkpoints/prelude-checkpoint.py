
import torch
import torch.nn as nn

from .norms import RMSNorm


class Prelude(nn.Module):
    """
    AGT-v3 Prelude.

    Constructs the initial live stream h_{t,0} from:
      - persistent state H_{t-1}
      - current token embedding x_t
      - persistent memory M_t
      - controller state C_{t-1}

    The detailed adaptive execution and terminal commit are handled
    by later modules.
    """

    def __init__(
        self,
        dim: int = 256,
    ):
        super().__init__()

        self.dim = dim

        self.input_norm = RMSNorm(dim)
        self.state_norm = RMSNorm(dim)
        self.memory_norm = RMSNorm(dim)
        self.controller_norm = RMSNorm(dim)

        self.input_proj = nn.Linear(dim, dim, bias=False)
        self.state_proj = nn.Linear(dim, dim, bias=False)
        self.memory_proj = nn.Linear(dim, dim, bias=False)
        self.controller_proj = nn.Linear(dim, dim, bias=False)

        self.output_norm = RMSNorm(dim)

    def forward(
        self,
        x_t,
        H_prev,
        memory_read,
        C_prev,
    ):
        """
        All inputs are [B, D].

        Returns:
            h0: [B, D]
        """

        x = self.input_proj(
            self.input_norm(x_t)
        )

        h = self.state_proj(
            self.state_norm(H_prev)
        )

        m = self.memory_proj(
            self.memory_norm(memory_read)
        )

        c = self.controller_proj(
            self.controller_norm(C_prev)
        )

        h0 = self.output_norm(
            x + h + m + c
        )

        return h0
