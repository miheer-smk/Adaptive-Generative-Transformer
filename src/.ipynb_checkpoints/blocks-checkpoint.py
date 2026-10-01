
import torch
import torch.nn as nn

from .norms import RMSNorm
from .gqa import GQA
from .gdn import GDN
from .mhc import MHC


class AGTBlock(nn.Module):
    """
    Single heterogeneous AGT-v3 block.

    Supported species:
        GQA
        GDN
        mHC

    MEMORY is handled by the persistent-memory pathway rather than
    as an ordinary token-mixing block.
    """

    def __init__(
        self,
        dim: int = 256,
        species: str = "GQA",
    ):
        super().__init__()

        self.dim = dim
        self.species = species.upper()

        self.norm = RMSNorm(dim)

        if self.species == "GQA":
            self.core = GQA(
                dim=dim,
                num_query_heads=8,
                num_kv_heads=2,
                head_dim=32,
                max_seq_len=256,
            )

        elif self.species == "GDN":
            self.core = GDN(dim=dim)

        elif self.species == "MHC":
            self.core = MHC(dim=dim)

        elif self.species == "MEMORY":
            self.core = nn.Identity()

        else:
            raise ValueError(
                f"Unknown AGT block species: {species}"
            )

    def forward(self, x, **kwargs):
        """
        Residual block interface.

        Species-specific recurrent/cache arguments are forwarded
        through kwargs.
        """

        residual = x
        h = self.norm(x)

        if self.species == "GQA":
            h = self.core(h, **kwargs)

        elif self.species == "GDN":
            h, state = self.core(h, **kwargs)
            h = residual + h
            return h, state

        else:
            h = self.core(h)

        return residual + h


class HeterogeneousBlockBank(nn.Module):
    """
    Fixed heterogeneous bank with per-block parameters.

    No per-step parameters are introduced here.
    """

    def __init__(
        self,
        dim: int = 256,
        species_order=None,
    ):
        super().__init__()

        if species_order is None:
            species_order = [
                "GQA",
                "GQA",
                "GDN",
                "GDN",
                "MHC",
                "MHC",
                "MEMORY",
                "MEMORY",
            ]

        self.species_order = list(species_order)

        self.blocks = nn.ModuleList([
            AGTBlock(
                dim=dim,
                species=species,
            )
            for species in self.species_order
        ])

    def __len__(self):
        return len(self.blocks)

    def forward(self, index, x, **kwargs):
        return self.blocks[index](x, **kwargs)
