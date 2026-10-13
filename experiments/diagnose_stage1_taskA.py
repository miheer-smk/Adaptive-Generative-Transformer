from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from data.task_a import TaskAParity
from src.agt import AGTv3


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_sequence(model: AGTv3, x: torch.Tensor):
    state = model.initial_persistent_state(
        batch_size=x.shape[0],
        device=x.device,
        dtype=x.dtype,
    )

    C_prev = torch.zeros(
        x.shape[0],
        model.dim,
        device=x.device,
        dtype=x.dtype,
    )

    logits = None
    for t in range(x.shape[1]):
        out = model.step(x[:, t, :], state, C_prev)
        state = out["state"]
        logits = out["logits"]

    return logits, state


def train_last_bit_probe(
    dim: int,
    device: torch.device,
) -> None:
    """
    Plumbing diagnostic only.

    Same AGT execution path, but target is the final input bit rather
    than parity. This is NOT a Task-A experiment.
    """
    set_seed(3201)

    task = TaskAParity(dim=dim, device=device)

    bits = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [1, 1, 0, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 1, 1, 1],
            [0, 0, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 1],
            [1, 1, 0, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 1, 1, 0, 1],
            [0, 1, 0, 0, 1, 0, 1, 0],
            [1, 1, 1, 0, 1, 0, 0, 1],
        ],
        dtype=torch.long,
        device=device,
    )

    x = task.token_features[bits]
    target = bits[:, -1]

    model = AGTv3(
        dim=dim,
        num_memory_slots=16,
        vocab_size=2,
    ).to(device)

    for p in model.controller.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.0,
    )

    for step in range(1, 401):
        logits, _ = run_sequence(model, x)

        loss = F.cross_entropy(logits, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits, _ = run_sequence(model, x)
        pred = logits.argmax(dim=-1)
        acc = (pred == target).float().mean().item()
        ce = F.cross_entropy(logits, target).item()

    print("\n=== LAST-BIT PLUMBING DIAGNOSTIC ===")
    print(f"ce={ce:.6f}")
    print(f"accuracy={acc:.4f}")


def main() -> None:
    set_seed(3200)
    torch.set_num_threads(4)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dim = 64
    task = TaskAParity(dim=dim, device=device)

    model = AGTv3(
        dim=dim,
        num_memory_slots=16,
        vocab_size=2,
    ).to(device)

    for p in model.controller.parameters():
        p.requires_grad_(False)

    bits = torch.tensor(
        [
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1],
            [1,0,1,0,1,0,1,0],
            [1,1,0,1,1,0,0,1],
        ],
        dtype=torch.long,
        device=device,
    )

    x = task.token_features[bits]

    with torch.no_grad():
        logits, states = run_sequence(model, x)

    print("=== STAGE 1 DATAFLOW DIAGNOSTIC ===")
    print(f"device={device}")
    print(f"input_feature_delta=",
          (x[0, 0] - x[1, 0]).abs().sum().item())

    print("\nfinal-logit pairwise differences:")
    for i in range(bits.shape[0]):
        for j in range(i + 1, bits.shape[0]):
            delta = (logits[i] - logits[j]).abs().sum().item()
            print(f"  pair {i}-{j}: {delta:.6e}")

    print("\nflip-one-position sensitivity:")
    base = bits[0].unsqueeze(0).repeat(8, 1)

    for pos in range(8):
        flipped = base.clone()
        flipped[:, pos] = 1

        # Compare all-zero sequence against sequence with exactly one 1.
        with torch.no_grad():
            base_logits, _ = run_sequence(
                model,
                task.token_features[base],
            )
            flip_logits, _ = run_sequence(
                model,
                task.token_features[flipped],
            )

        delta = (
            flip_logits - base_logits
        ).abs().sum(dim=-1).mean().item()

        print(f"  position {pos}: {delta:.6e}")

    train_last_bit_probe(dim=dim, device=device)


if __name__ == "__main__":
    main()
