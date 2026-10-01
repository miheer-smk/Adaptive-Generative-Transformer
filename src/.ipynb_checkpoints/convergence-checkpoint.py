import torch
import torch.nn as nn


class ConvergenceGate(nn.Module):
    """
    AGT-v3 deterministic, parameter-free convergence gate.

    Convergence is based on the normalized change in execution state.
    """

    def __init__(
        self,
        dim: int = 256,
        epsilon: float = 1e-6,
    ):
        super().__init__()

        self.dim = dim
        self.epsilon = epsilon

    def forward(self, prev_state, state):
        """
        prev_state:
            [B, D]

        state:
            [B, D]

        returns:
            delta:
                [B], normalized state change
        """

        denom = prev_state.norm(dim=-1).clamp_min(self.epsilon)

        delta = (
            (state - prev_state).norm(dim=-1)
            / denom
        )

        return delta

    def converged(self, prev_state, state):
        """
        Deterministic convergence decision.
        """

        delta = self.forward(prev_state, state)

        return delta < self.epsilon


class HaltShadowRollout:
    """
    AMB-4 provisional halt shadow rollout.

    The executor can use this helper to evaluate one REPEAT of the
    current block under stop-gradient before accepting HALT.
    """

    @staticmethod
    def detach(state):
        return state.detach()