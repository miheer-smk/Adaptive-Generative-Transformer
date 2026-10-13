"""Isolated reproduction of the original, noncausal seed3401 monitor."""
import copy
import random
import torch
from experiments.stage3_measurements import load_model


@torch.no_grad()
def reconstruct(device):
    python_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            original = load_model(3401, torch.device(device))
            monitor = copy.deepcopy(original.controller).eval()
            for p in monitor.parameters():
                p.requires_grad_(False)
            return monitor
    finally:
        random.setstate(python_state)


@torch.no_grad()
def observe(monitor, record, device):
    before = torch.tensor(record['h_before'], device=device, dtype=torch.float32)
    onehot = before.new_zeros(1, 9)
    onehot[:, record['pointer_before']] = 1
    features = dict(h=before,
        memory_read=torch.tensor(record['memory_read'], device=device, dtype=torch.float32),
        memory_mean=torch.tensor(record['memory_mean'], device=device, dtype=torch.float32),
        pointer_one_hot=onehot, budget=before.new_zeros(1, 2))
    return monitor(**features)['logits'].cpu().tolist()
