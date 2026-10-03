import torch
import torch.nn as nn

from .norms import RMSNorm


class Controller(nn.Module):
    """
    AGT-v3 Stage-3 pilot controller.

    D64 pilot input:
        h                  : 64
        memory_read        : 64
        memory_mean        : 256
        pointer_one_hot    : 9
        budget              : 2
        --------------------------------
        total              : 395

    Controller MLP:
        395 -> 256 -> 11

    Action layout:
        0 ... N-1 : ADVANCE(0 ... N-1)
        N         : REPEAT
        N+1       : RESTART
        N+2       : HALT

    The D64 dimensionality is a pilot implementation contract, not a
    permanent declaration about the final AGT-v3 scaling.
    """

    def __init__(
        self,
        dim: int = 64,
        num_blocks: int = 8,
        memory_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()

        if dim <= 0:
            raise ValueError("dim must be positive.")

        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive.")

        if memory_dim <= 0:
            raise ValueError("memory_dim must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.dim = dim
        self.num_blocks = num_blocks
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim

        self.pointer_dim = num_blocks + 1
        self.budget_dim = 2

        self.input_dim = (
            dim
            + dim
            + memory_dim
            + self.pointer_dim
            + self.budget_dim
        )

        self.norm_h = RMSNorm(dim)
        self.norm_memory_read = RMSNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_blocks + 3),
        )

    def _validate_tensor(
        self,
        x: torch.Tensor,
        *,
        name: str,
        width: int,
    ):
        if not torch.is_tensor(x):
            raise TypeError(f"{name} must be a torch.Tensor.")

        if x.ndim != 2:
            raise ValueError(
                f"{name} must have shape [B, {width}], got {tuple(x.shape)}."
            )

        if x.shape[-1] != width:
            raise ValueError(
                f"{name} width {x.shape[-1]} != expected {width}."
            )

    def _prepare_inputs(
        self,
        h: torch.Tensor,
        memory_read: torch.Tensor,
        memory_mean: torch.Tensor,
        pointer_one_hot: torch.Tensor,
        budget: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_tensor(
            h,
            name="h",
            width=self.dim,
        )
        self._validate_tensor(
            memory_read,
            name="memory_read",
            width=self.dim,
        )
        self._validate_tensor(
            memory_mean,
            name="memory_mean",
            width=self.memory_dim,
        )
        self._validate_tensor(
            pointer_one_hot,
            name="pointer_one_hot",
            width=self.pointer_dim,
        )
        self._validate_tensor(
            budget,
            name="budget",
            width=self.budget_dim,
        )

        batch_size = h.shape[0]

        for name, tensor in (
            ("memory_read", memory_read),
            ("memory_mean", memory_mean),
            ("pointer_one_hot", pointer_one_hot),
            ("budget", budget),
        ):
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"{name} batch size {tensor.shape[0]} != h batch size {batch_size}."
                )

        # Controller receives normalized token state and normalized
        # memory read; persistent memory slot summary remains in its
        # supplied memory-space representation.
        h_n = self.norm_h(h)
        memory_read_n = self.norm_memory_read(memory_read)

        return torch.cat(
            [
                h_n,
                memory_read_n,
                memory_mean,
                pointer_one_hot,
                budget,
            ],
            dim=-1,
        )

    def forward(
        self,
        state=None,
        *,
        h=None,
        memory_read=None,
        memory_mean=None,
        pointer_one_hot=None,
        budget=None,
    ):
        """
        Canonical Stage-3 call:

            controller(
                h=h,
                memory_read=memory_read,
                memory_mean=memory_mean,
                pointer_one_hot=pointer_one_hot,
                budget=budget,
            )

        Legacy compatibility:

            controller(h)

        In legacy mode, the additional controller inputs are represented
        by zero tensors so existing pre-Gate-C executor tests continue to
        exercise the controller without forcing executor integration yet.
        """
        legacy_mode = (
            h is None
            and memory_read is None
            and memory_mean is None
            and pointer_one_hot is None
            and budget is None
            and state is not None
        )

        if h is None:
            if state is None:
                raise ValueError(
                    "Provide h or the legacy positional state argument."
                )
            h = state

        if memory_read is None:
            memory_read = torch.zeros_like(h)

        if memory_mean is None:
            memory_mean = torch.zeros(
                h.shape[0],
                self.memory_dim,
                device=h.device,
                dtype=h.dtype,
            )

        if pointer_one_hot is None:
            pointer_one_hot = torch.zeros(
                h.shape[0],
                self.pointer_dim,
                device=h.device,
                dtype=h.dtype,
            )
            pointer_one_hot[:, 0] = 1.0

        if budget is None:
            budget = torch.zeros(
                h.shape[0],
                self.budget_dim,
                device=h.device,
                dtype=h.dtype,
            )

        features = self._prepare_inputs(
            h,
            memory_read,
            memory_mean,
            pointer_one_hot,
            budget,
        )

        logits = self.mlp(features)

        return {
            "logits": logits,
            "advance": logits[:, : self.num_blocks],
            "repeat": logits[:, self.num_blocks],
            "restart": logits[:, self.num_blocks + 1],
            "halt": logits[:, self.num_blocks + 2],
            "legacy_mode": legacy_mode,
        }

    def action_logits(self, state=None, **kwargs):
        """
        Return the unified action-logit vector.

        Kept as a compatibility helper for existing code/tests.
        """
        return self.forward(
            state,
            **kwargs,
        )["logits"]

    @property
    def num_actions(self):
        return self.num_blocks + 3
