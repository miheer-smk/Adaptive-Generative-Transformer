
import torch
import torch.nn as nn

from .norms import RMSNorm


class Controller(nn.Module):
    """
    AGT-v3 unified execution controller.

    Locked action set:
        ADVANCE(j')
        REPEAT
        RESTART
        HALT

    This module produces action logits. The executor will later
    interpret those logits and enforce the execution/mask rules.
    """

    ACTIONS = (
        "ADVANCE",
        "REPEAT",
        "RESTART",
        "HALT",
    )

    def __init__(
        self,
        dim: int = 256,
        num_blocks: int = 8,
    ):
        super().__init__()

        self.dim = dim
        self.num_blocks = num_blocks

        self.norm = RMSNorm(dim)

        # One shared controller representation.
        self.state_proj = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        # ADVANCE chooses the next block j'.
        self.advance_head = nn.Linear(
            dim,
            num_blocks,
        )

        self.repeat_head = nn.Linear(
            dim,
            1,
        )

        self.restart_head = nn.Linear(
            dim,
            1,
        )

        self.halt_head = nn.Linear(
            dim,
            1,
        )

    def forward(self, state):
        """
        state:
            [B, D]

        Returns:
            dictionary containing logits for each action.
        """

        h = self.state_proj(
            self.norm(state)
        )

        return {
            "advance": self.advance_head(h),
            "repeat": self.repeat_head(h).squeeze(-1),
            "restart": self.restart_head(h).squeeze(-1),
            "halt": self.halt_head(h).squeeze(-1),
        }

    def action_logits(self, state):
        """
        Return a single unified action-logit tensor.

        Layout:
            [ADVANCE(j'=0..N-1), REPEAT, RESTART, HALT]
        """

        outputs = self.forward(state)

        return torch.cat(
            [
                outputs["advance"],
                outputs["repeat"].unsqueeze(-1),
                outputs["restart"].unsqueeze(-1),
                outputs["halt"].unsqueeze(-1),
            ],
            dim=-1,
        )

    @property
    def num_actions(self):
        return self.num_blocks + 3
