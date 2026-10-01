
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

        # Persistent memory slot width is d_M=256, while the model stream
        # remains d=self.dim. This is the interface required by the
        # terminal MemoryWrite module.
        self.mem_dim = 256

        # Learnable initial memory state: [S, d_M].
        self.initial_memory = nn.Parameter(
            torch.zeros(num_slots, self.mem_dim)
        )

        # Query is formed from the model-width stream.
        self.query_norm = RMSNorm(dim)

        # Addressing lives in memory space d_M.
        self.query_proj = nn.Linear(dim, self.mem_dim, bias=False)
        self.key_proj = nn.Linear(self.mem_dim, self.mem_dim, bias=False)

        # Read value is projected back to model width for the Prelude.
        self.value_proj = nn.Linear(self.mem_dim, dim, bias=False)

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

    def memory_keys(self, memory):
        """
        Return K_M for the current persistent memory state.

        memory:
            [B, S, d_M]

        returns:
            keys:
                [B, S, d_M]
        """
        if memory.ndim != 3:
            raise ValueError(
                "memory must have shape [B, S, d_M]."
            )
        if memory.shape[-1] != self.mem_dim:
            raise ValueError(
                f"memory width {memory.shape[-1]} != mem_dim={self.mem_dim}."
            )
        return self.key_proj(memory)

    def read(self, query, memory):
        """
        Read from persistent memory.

        query:
            [B, d]
        memory:
            [B, S, d_M]

        returns:
            read_value: [B, d]
            attention:  [B, S]
        """
        if query.ndim != 2 or query.shape[-1] != self.dim:
            raise ValueError(
                f"query must be [B, {self.dim}]."
            )
        if memory.ndim != 3 or memory.shape[-1] != self.mem_dim:
            raise ValueError(
                f"memory must be [B, S, {self.mem_dim}]."
            )

        q = self.query_proj(self.query_norm(query))
        k = self.memory_keys(memory)
        v = self.value_proj(memory)

        scale = self.mem_dim ** -0.5

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
        Functional legacy write primitive.

        value is now a memory-space vector [B, d_M].
        """
        if memory.ndim != 3 or memory.shape[-1] != self.mem_dim:
            raise ValueError(
                f"memory must be [B, S, {self.mem_dim}]."
            )
        if value.ndim != 2 or value.shape[-1] != self.mem_dim:
            raise ValueError(
                f"value must be [B, {self.mem_dim}]."
            )

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
