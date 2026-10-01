"""
AGT-v3 POC — Gated Delta Network (GDN)

Stateful sequence mixer.

The recurrent state is explicitly separated from the token
activation so that later executor logic can freeze/restore it.
"""

import torch
import torch.nn as nn


class GDN(nn.Module):
    """
    Minimal gated delta-rule sequence mixer.

    Input:
        x: [batch, seq_len, dim]

    Output:
        y: [batch, seq_len, dim]

    State:
        recurrent state [batch, dim]

    The state is returned explicitly rather than hidden inside
    the module, which is important for AGT-v3 frozen-state
    execution semantics.
    """

    def __init__(
        self,
        dim: int,
        state_dim: int = None,
        bias: bool = False,
    ):
        super().__init__()

        if state_dim is None:
            state_dim = dim

        self.dim = dim
        self.state_dim = state_dim

        self.input_proj = nn.Linear(
            dim,
            3 * state_dim,
            bias=bias,
        )

        self.output_proj = nn.Linear(
            state_dim,
            dim,
            bias=bias,
        )

        # Learned decay gate.
        self.decay = nn.Parameter(
            torch.zeros(state_dim)
        )

    def initial_state(
        self,
        batch_size: int,
        device=None,
        dtype=None,
    ):
        """Create a zero recurrent state."""
        return torch.zeros(
            batch_size,
            self.state_dim,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor = None,
    ):
        """
        Process a sequence while explicitly carrying state.

        Returns:
            y, final_state
        """

        B, T, _ = x.shape

        if state is None:
            state = self.initial_state(
                B,
                device=x.device,
                dtype=x.dtype,
            )

        projected = self.input_proj(x)

        key, value, gate = projected.chunk(
            3,
            dim=-1,
        )

        decay = torch.sigmoid(self.decay)

        outputs = []

        for t in range(T):
            # Gated delta-style state update.
            delta = key[:, t] * value[:, t]

            state = (
                decay * state
                + (1.0 - decay) * delta
            )

            g = torch.sigmoid(gate[:, t])

            outputs.append(
                self.output_proj(
                    g * state
                )
            )

        y = torch.stack(outputs, dim=1)

        return y, state
