"""
AGT-v3 POC — General Utilities
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    """Return the number of trainable parameters."""
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def count_all_parameters(model):
    """Return the total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def tensor_is_finite(x: torch.Tensor) -> bool:
    """Check whether every element of a tensor is finite."""
    return bool(torch.isfinite(x).all().item())


def ensure_dir(path: str):
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)
