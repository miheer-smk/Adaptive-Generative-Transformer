
import torch
import torch.nn as nn

from .norms import RMSNorm


class PersistentMemory(nn.Module):
    """
    AGT-v3 persistent memory bank.

    Stores M memory slots that can be read by the current token state
    and updated explicitly by the execution mechanism.

    This module provides the memory representation and read/write
    primitives; the exact controller-driven update semantics are handled
    later by the executor/commit stages.
    """

    def __init__(
        self,
        dim: int = 256,
        num_slots: int = 16,
    ):
        super().__init__()

        self.dim = dim
        self.num_slots = num_slots

        # Learnable initial memory state.
        self.initial_memory = nn.Parameter(
            torch.zeros(num_slots, dim)
        )

        # Parameter-free RMSNorm for memory-read query,
        # following the provisional AMB-8 decision.
        self.query_norm = RMSNorm(dim)

        # Projections used for memory addressing.
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)

    def initial_state(
        self,
        batch_size: int,
        device=None,
        dtype=None,
    ):
        """
        Return a fresh persistent memory state for a batch.
        """
        memory = self.initial_memory

        if device is not None or dtype is not None:
            memory = memory.to(
                device=device if device is not None else memory.device,
                dtype=dtype if dtype is not None else memory.dtype,
            )

        return memory.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        ).clone()

    def read(self, query, memory):
        """
        Read from persistent memory.

        query:
            [B, D]

        memory:
            [B, S, D]

        returns:
            read_value: [B, D]
            attention:  [B, S]
        """
        q = self.query_proj(self.query_norm(query))
        k = self.key_proj(memory)
        v = self.value_proj(memory)

        scale = self.dim ** -0.5

        scores = torch.einsum(
            "bd,bsd->bs",
            q,
            k,
        ) * scale

        attention = torch.softmax(scores, dim=-1)

        read_value = torch.einsum(
            "bs,bsd->bd",
            attention,
            v,
        )

        return read_value, attention

    def write(
        self,
        memory,
        slot_index,
        value,
    ):
        """
        Functional memory write.

        Returns a new memory tensor rather than mutating the supplied
        persistent state in-place. Controller/commit logic will later
        determine when this operation is allowed.

        memory:
            [B, S, D]

        slot_index:
            [B] integer slot indices

        value:
            [B, D]
        """
        updated = memory.clone()

        batch_indices = torch.arange(
            memory.shape[0],
            device=memory.device,
        )

        updated[
            batch_indices,
            slot_index,
        ] = value

        return updated
