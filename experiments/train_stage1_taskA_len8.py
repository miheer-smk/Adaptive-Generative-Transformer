from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from data.task_a import TaskAParity
from src.agt import AGTv3


BANK = ["SWA", "SWA", "GQA", "GQA", "GDN", "GDN", "FFN", "FFN"]
REQUIRED_GROUPS = [
    "memory",
    "prelude",
    "blocks",
    "commit",
    "memory_write",
    "output_head",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_controller(model: AGTv3) -> None:
    for p in model.controller.parameters():
        p.requires_grad_(False)


def assert_fixed_route_model(model: AGTv3) -> None:
    bank = [b.species for b in model.blocks.blocks]
    assert bank == BANK, f"Unexpected bank: {bank}"


def gradient_coverage(model: AGTv3) -> dict[str, int]:
    counts = {name: 0 for name in REQUIRED_GROUPS}

    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue

        if param.grad.abs().sum().item() == 0:
            continue

        top = name.split(".", 1)[0]
        if top in counts:
            counts[top] += 1

    return counts


def parameter_deltas(
    model: AGTv3,
    before: dict[str, torch.Tensor],
) -> dict[str, float]:
    totals = {name: 0.0 for name in REQUIRED_GROUPS}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        top = name.split(".", 1)[0]
        if top not in totals:
            continue

        delta = (param.detach() - before[name]).abs().sum().item()
        totals[top] += delta

    return totals


def snapshot_trainable(model: AGTv3) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def run_sequence(
    model: AGTv3,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Fixed-route token-axis execution.

    x: [B, L, D]
    returns final-token logits: [B, 2]
    """
    batch_size, seq_len, dim = x.shape
    assert dim == model.dim

    state = model.initial_persistent_state(
        batch_size=batch_size,
        device=x.device,
        dtype=x.dtype,
    )

    # Current frozen step() API retains C_prev as an explicit compatibility
    # input. The persistent execution context itself is carried by `state`.
    C_prev = torch.zeros(
        batch_size,
        model.dim,
        device=x.device,
        dtype=x.dtype,
    )

    final_logits = None

    for t in range(seq_len):
        out = model.step(
            x[:, t, :],
            state,
            C_prev,
        )

        state = out["state"]

        assert state.tokens_committed == t + 1
        assert torch.isfinite(out["logits"]).all()

        final_logits = out["logits"]

    assert final_logits is not None
    return final_logits


@torch.no_grad()
def evaluate(
    model: AGTv3,
    task: TaskAParity,
    split: str,
    min_len: int,
    max_len: int,
    per_length: int,
) -> dict:
    model.eval()

    total = 0
    correct = 0
    losses = []

    for length in range(min_len, max_len + 1):
        batch = task.make_fixed_set(
            split=split,
            length=length,
            size=per_length,
        )

        logits = run_sequence(model, batch.x)
        loss = F.cross_entropy(logits, batch.target)

        pred = logits.argmax(dim=-1)

        total += batch.target.numel()
        correct += (pred == batch.target).sum().item()
        losses.append(loss.item())

    return {
        "split": split,
        "accuracy": correct / total,
        "mean_ce": sum(losses) / len(losses),
        "examples": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3001)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="run validation and OOD test after training",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(4)

    device = torch.device(args.device)

    task = TaskAParity(
        dim=args.dim,
        device=device,
    )

    model = AGTv3(
        dim=args.dim,
        num_memory_slots=16,
        vocab_size=2,
    ).to(device)

    freeze_controller(model)
    assert_fixed_route_model(model)

    trainable_params = [
        p for p in model.parameters() if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("=== AGT-v3 STAGE 1 TASK A ===")
    print(f"device={device}")
    print(f"dim={args.dim}")
    print(f"batch_size={args.batch_size}")
    print(f"steps={args.steps}")
    print(f"lr={args.lr}")
    print(f"bank={BANK}")
    print("controller=DISABLED")
    print("route=FIXED ONE SWEEP")
    print("loss=FINAL-TOKEN CE ONLY")
    print(
        "trainable_params=",
        sum(p.numel() for p in trainable_params),
    )
    print(
        "controller_params=",
        sum(p.numel() for p in model.controller.parameters()),
    )

    # Fixed probe set: identical examples are evaluated before and after
    # training. Individual batch losses are not directly comparable because
    # training difficulty (sequence length) changes between batches.
    probe_lengths = list(range(
        task.TRAIN_MIN_LEN,
        task.TRAIN_MAX_LEN + 1,
    ))

    probe_batches = [
        task.make_fixed_set(
            split="train",
            length=length,
            size=16,
        )
        for length in probe_lengths
    ]

    @torch.no_grad()
    def fixed_probe_ce() -> float:
        model.eval()
        losses = []

        for probe in probe_batches:
            logits = run_sequence(model, probe.x)
            losses.append(
                F.cross_entropy(logits, probe.target).item()
            )

        return sum(losses) / len(losses)

    probe_ce_before = fixed_probe_ce()

    first_loss = None
    last_loss = None
    first_grad = None
    update_report = None

    for step in range(1, args.steps + 1):
        # One length per batch keeps the sequential API exact while sampling
        # training difficulty uniformly from {4,...,16}.
        # Controlled Stage-1 probe: fixed sequence length.
        length = 8

        batch = task.make_batch(
            batch_size=args.batch_size,
            length=length,
            split="train",
        )

        model.train()

        logits = run_sequence(model, batch.x)

        loss = F.cross_entropy(
            logits,
            batch.target,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if step == 1:
            first_loss = float(loss.item())
            first_grad = gradient_coverage(model)

            missing = [
                group
                for group, count in first_grad.items()
                if count == 0
            ]

            print("\n=== GRADIENT COVERAGE / STEP 1 ===")
            for group, count in first_grad.items():
                print(f"{group:15s}: {count:4d}")

            assert not missing, (
                f"Missing gradient coverage: {missing}"
            )

            before = snapshot_trainable(model)

        optimizer.step()

        if step == 1:
            update_report = parameter_deltas(model, before)

            print("\n=== PARAMETER UPDATES / STEP 1 ===")
            for group, delta in update_report.items():
                print(f"{group:15s}: {delta:.6e}")

            zero_updates = [
                group
                for group, delta in update_report.items()
                if delta == 0.0
            ]

            assert not zero_updates, (
                f"No parameter update in groups: {zero_updates}"
            )

        last_loss = float(loss.item())

        print(
            f"step={step:03d} "
            f"length={length:02d} "
            f"loss={last_loss:.6f}"
        )

    assert first_loss is not None
    assert last_loss is not None

    probe_ce_after = fixed_probe_ce()

    assert probe_ce_after < probe_ce_before, (
        f"Fixed-probe CE did not decrease: "
        f"{probe_ce_before:.6f} -> {probe_ce_after:.6f}"
    )

    print("\n=== STAGE 1 TINY TRAINING RESULT ===")
    print(f"first_batch_ce={first_loss:.6f}")
    print(f"last_batch_ce={last_loss:.6f}")
    print(f"fixed_probe_ce_before={probe_ce_before:.6f}")
    print(f"fixed_probe_ce_after={probe_ce_after:.6f}")
    print("loss_decreased=True")
    print("gradient_coverage=True")
    print("parameter_updates=True")

    results = {
        "task": "Task A parity",
        "device": str(device),
        "dim": args.dim,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "bank": BANK,
        "controller": "disabled",
        "loss": "final-token cross-entropy",
        "first_batch_ce": first_loss,
        "last_batch_ce": last_loss,
        "fixed_probe_ce_before": probe_ce_before,
        "fixed_probe_ce_after": probe_ce_after,
        "gradient_coverage": first_grad,
        "parameter_update_l1": update_report,
    }

    if args.eval:
        val = evaluate(
            model,
            task,
            split="val",
            min_len=task.VAL_MIN_LEN,
            max_len=task.VAL_MAX_LEN,
            per_length=64,
        )

        test_easy = evaluate(
            model,
            task,
            split="test",
            min_len=17,
            max_len=24,
            per_length=64,
        )

        test_hard = evaluate(
            model,
            task,
            split="test",
            min_len=25,
            max_len=32,
            per_length=64,
        )

        results["validation"] = val
        results["test_easy_ood"] = test_easy
        results["test_hard_ood"] = test_hard

        print("\n=== EVALUATION ===")
        print(val)
        print(test_easy)
        print(test_hard)

    out_dir = Path("experiments/runs")
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "stage1_taskA_tiny.json").open("w") as f:
        json.dump(results, f, indent=2)

    print("\nWROTE experiments/runs/stage1_taskA_tiny.json")


if __name__ == "__main__":
    main()
