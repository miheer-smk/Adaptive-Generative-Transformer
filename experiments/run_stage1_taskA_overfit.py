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


def run_sequence(model: AGTv3, x: torch.Tensor) -> torch.Tensor:
    batch_size = x.shape[0]

    state = model.initial_persistent_state(
        batch_size=batch_size,
        device=x.device,
        dtype=x.dtype,
    )

    C_prev = torch.zeros(
        batch_size,
        model.dim,
        device=x.device,
        dtype=x.dtype,
    )

    logits = None

    for t in range(x.shape[1]):
        out = model.step(x[:, t, :], state, C_prev)
        state = out["state"]
        logits = out["logits"]

    assert logits is not None
    return logits


def main() -> None:
    set_seed(3101)
    torch.set_num_threads(4)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dim = 64
    length = 8
    subset_size = 16
    steps = 2000
    lr = 1e-4

    task = TaskAParity(
        dim=dim,
        device=device,
    )

    # Fixed tiny subset: the exact same examples are trained repeatedly.
    subset = task.make_fixed_set(
        split="train",
        length=length,
        size=subset_size,
    )

    model = AGTv3(
        dim=dim,
        num_memory_slots=16,
        vocab_size=2,
    ).to(device)

    for p in model.controller.parameters():
        p.requires_grad_(False)

    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.0,
    )

    with torch.no_grad():
        initial_logits = run_sequence(model, subset.x)
        initial_ce = F.cross_entropy(
            initial_logits,
            subset.target,
        ).item()

    print("=== STAGE 1 TASK A TINY-SUBSET OVERFIT ===")
    print(f"device={device}")
    print(f"dim={dim}")
    print(f"length={length}")
    print(f"subset_size={subset_size}")
    print(f"steps={steps}")
    print(f"lr={lr}")
    print("controller=DISABLED")
    print("route=FIXED ONE SWEEP")
    print("loss=FINAL-TOKEN CE ONLY")
    print(f"initial_ce={initial_ce:.6f}")

    last_ce = None
    last_acc = None

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)

        logits = run_sequence(model, subset.x)

        loss = F.cross_entropy(
            logits,
            subset.target,
        )

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            accuracy = (
                predictions == subset.target
            ).float().mean().item()

        last_ce = loss.item()
        last_acc = accuracy

        if step == 1 or step % 50 == 0:
            print(
                f"step={step:03d} "
                f"ce={last_ce:.6f} "
                f"accuracy={last_acc:.4f}"
            )

    assert last_ce is not None
    assert last_acc is not None

    # Re-evaluate AFTER the optimizer update.
    with torch.no_grad():
        final_logits = run_sequence(model, subset.x)
        final_ce = F.cross_entropy(
            final_logits,
            subset.target,
        ).item()
        final_predictions = final_logits.argmax(dim=-1)
        final_accuracy = (
            final_predictions == subset.target
        ).float().mean().item()

    assert final_ce < initial_ce, (
        f"Tiny-subset CE did not decrease: "
        f"{initial_ce:.6f} -> {final_ce:.6f}"
    )

    assert final_accuracy == 1.0, (
        f"Tiny-subset did not reach 100% accuracy: "
        f"{final_accuracy:.4f}"
    )

    print("\n=== RESULT ===")
    print(f"initial_ce={initial_ce:.6f}")
    print(f"final_ce={final_ce:.6f}")
    print(f"final_accuracy={final_accuracy:.4f}")
    print("tiny_subset_overfit=True")


if __name__ == "__main__":
    main()
