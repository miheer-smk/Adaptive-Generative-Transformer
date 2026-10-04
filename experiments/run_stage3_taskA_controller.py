from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from data.task_a import TaskAParity
from src.agt import AGTv3


BANK = ["SWA", "SWA", "GQA", "GQA", "GDN", "GDN", "FFN", "FFN"]
CHECKPOINT = Path(
    "experiments/runs/stage1_parity_seed3305_best.pt"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_fixed_sequence(model: AGTv3, x: torch.Tensor) -> torch.Tensor:
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
        out = model.step(
            x[:, t, :],
            state,
            C_prev,
        )
        state = out["state"]
        logits = out["logits"]

    assert logits is not None
    return logits


def run_diff_sequence(
    model: AGTv3,
    x: torch.Tensor,
    tau: float,
    training: bool,
):
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
    action_history = []

    for t in range(x.shape[1]):
        out = model.adaptive_differentiable_step(
            x_t=x[:, t, :],
            state=state,
            C_prev=C_prev,
            # Gate-B marks budget-relative conditioning as optional.
            # Disable it for the first controlled Gate-C experiment.
            budget=torch.zeros(
                x.shape[0],
                2,
                device=x.device,
                dtype=x.dtype,
            ),
            tau=tau,
            training=training,
        )

        state = out["state"]
        logits = out["logits"]

        if out["execution_count"] != 8:
            raise RuntimeError(
                "Stage-3 pilot expected exactly 8 routing decisions, got "
                f"{out['execution_count']}."
            )

        action_history.append(out["action_history"])

    assert logits is not None

    return logits, action_history


@torch.no_grad()
def evaluate_learned_route(
    model: AGTv3,
    task: TaskAParity,
    split: str,
    min_len: int,
    max_len: int,
    per_length: int,
    tau: float,
):
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

        # The current differentiable executor pilot intentionally requires
        # batch size 1. Evaluate each fixed example independently so the
        # learned-route metric remains deterministic and comparable.
        for i in range(batch.x.shape[0]):
            x_i = batch.x[i:i + 1]
            target_i = batch.target[i:i + 1]

            logits, _history = run_diff_sequence(
                model,
                x_i,
                tau=tau,
                training=False,
            )

            loss = F.cross_entropy(
                logits,
                target_i,
            )

            total += 1
            correct += (
                logits.argmax(dim=-1) == target_i
            ).sum().item()
            losses.append(loss.item())

    return {
        "accuracy": correct / total,
        "mean_ce": sum(losses) / len(losses),
        "examples": total,
    }


@torch.no_grad()
def evaluate(
    model: AGTv3,
    task: TaskAParity,
    split: str,
    min_len: int,
    max_len: int,
    per_length: int,
):
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

        logits = run_fixed_sequence(model, batch.x)
        loss = F.cross_entropy(logits, batch.target)

        total += batch.target.numel()
        correct += (
            logits.argmax(dim=-1) == batch.target
        ).sum().item()
        losses.append(loss.item())

    return {
        "accuracy": correct / total,
        "mean_ce": sum(losses) / len(losses),
        "examples": total,
    }


def controller_gradient_coverage(model: AGTv3):
    total = 0
    nonzero = 0
    finite = 0

    for parameter in model.controller.parameters():
        if parameter.grad is None:
            continue

        total += 1

        if parameter.grad.abs().sum().item() > 0:
            nonzero += 1

        if torch.isfinite(parameter.grad).all():
            finite += 1

    return {
        "parameters_with_grad": total,
        "nonzero_grad_parameters": nonzero,
        "finite_grad_parameters": finite,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3401)
    parser.add_argument(
        "--controller-checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(4)

    device = torch.device(args.device)

    if not CHECKPOINT.exists():
        raise SystemExit(f"Missing checkpoint: {CHECKPOINT}")

    ckpt = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

    dim = int(ckpt["dim"])

    model = AGTv3(
        dim=dim,
        num_memory_slots=16,
        vocab_size=2,
    )

    checkpoint_state = ckpt["state_dict"]

    # Stage-1 used an older controller architecture. The Stage-3 experiment
    # intentionally reuses only the validated backbone and initializes the
    # new unified controller from scratch.
    backbone_state = {
        key: value
        for key, value in checkpoint_state.items()
        if not key.startswith("controller.")
    }

    missing, unexpected = model.load_state_dict(
        backbone_state,
        strict=False,
    )

    expected_controller_keys = {
        key
        for key in model.state_dict().keys()
        if key.startswith("controller.")
    }

    assert set(missing) == expected_controller_keys, (
        "Unexpected missing non-controller parameters: "
        f"{sorted(set(missing) - expected_controller_keys)}"
    )

    assert unexpected == [], (
        "Unexpected checkpoint parameters: "
        f"{unexpected}"
    )

    model.to(device)

    if args.controller_checkpoint is not None:
        controller_checkpoint = torch.load(
            args.controller_checkpoint,
            map_location="cpu",
        )

        controller_state = controller_checkpoint["controller_state_dict"]

        missing, unexpected = model.controller.load_state_dict(
            controller_state,
            strict=True,
        )

        if missing or unexpected:
            raise RuntimeError(
                "Controller checkpoint mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )

        print(
            f"controller_init=WARM_START:{args.controller_checkpoint}"
        )
    else:
        print("controller_init=FRESH")

    # Stage-3 controlled experiment: exactly one 8-decision routing budget,
    # matching the validated Stage-1 8-block reference sweep.
    model.executor.max_execution_depth = 8

    actual_bank = [
        block.species
        for block in model.blocks.blocks
    ]

    if actual_bank != BANK:
        raise RuntimeError(
            f"Unexpected bank: {actual_bank}"
        )

    # Freeze everything except the routing controller.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for parameter in model.controller.parameters():
        parameter.requires_grad_(True)

    trainable = list(model.controller.parameters())

    task = TaskAParity(
        dim=dim,
        device=device,
    )

    print("=== AGT-v3 GATE-C TASK-A CONTROLLER PILOT ===")
    print(f"device={device}")
    print(f"dim={dim}")
    print(f"steps={args.steps}")
    print(f"batch_size={args.batch_size}")
    print(f"lr={args.lr}")
    print(f"tau={args.tau}")
    print("backbone=FROZEN_STAGE1_SEED3305")
    print("trainable=CONTROLLER_ONLY")
    print("routing_steps=8")
    print("halt_training=DISABLED")

    baseline_val = evaluate(
        model,
        task,
        "val",
        task.VAL_MIN_LEN,
        task.VAL_MAX_LEN,
        32,
    )

    baseline_test = evaluate(
        model,
        task,
        "test",
        task.TEST_MIN_LEN,
        task.TEST_MAX_LEN,
        32,
    )

    print("\n=== FIXED-ROUTE BASELINE ===")
    print(json.dumps({
        "val": baseline_val,
        "test": baseline_test,
    }, indent=2))

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=0.0,
    )

    for step in range(1, args.steps + 1):
        length = int(
            torch.randint(
                task.TRAIN_MIN_LEN,
                task.TRAIN_MAX_LEN + 1,
                (1,),
            ).item()
        )

        batch = task.make_batch(
            batch_size=args.batch_size,
            length=length,
            split="train",
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits, history = run_diff_sequence(
            model,
            batch.x,
            tau=args.tau,
            training=True,
        )

        loss = F.cross_entropy(
            logits,
            batch.target,
        )

        loss.backward()

        coverage = controller_gradient_coverage(model)

        if coverage["nonzero_grad_parameters"] == 0:
            raise RuntimeError(
                "Controller received no nonzero gradients."
            )

        if coverage["finite_grad_parameters"] != coverage[
            "parameters_with_grad"
        ]:
            raise RuntimeError(
                "Non-finite controller gradient detected."
            )

        optimizer.step()

        print(
            f"step={step:03d} "
            f"length={length} "
            f"ce={loss.item():.6f} "
            f"grad_params={coverage['nonzero_grad_parameters']}"
        )

    model.eval()

    # Deterministic route inspection after training.
    route_batch = task.make_fixed_set(
        split="train",
        length=8,
        size=1,
    )

    _, route_history = run_diff_sequence(
        model,
        route_batch.x,
        tau=args.tau,
        training=False,
    )

    route_counts = {}
    for token_history in route_history:
        for action, index in token_history:
            key = action if index is None else f"{action}({index})"
            route_counts[key] = route_counts.get(key, 0) + 1

    total_route_decisions = sum(route_counts.values())

    print("\n=== DETERMINISTIC ROUTE INSPECTION ===")
    print(f"routing_decisions_per_token={8}")
    print(f"tokens_inspected={len(route_history)}")
    print(f"total_route_decisions={total_route_decisions}")
    print(json.dumps(route_counts, indent=2, sort_keys=True))

    post_val = evaluate_learned_route(
        model,
        task,
        "val",
        task.VAL_MIN_LEN,
        task.VAL_MAX_LEN,
        32,
        tau=args.tau,
    )

    post_test = evaluate_learned_route(
        model,
        task,
        "test",
        task.TEST_MIN_LEN,
        task.TEST_MAX_LEN,
        32,
        tau=args.tau,
    )

    print("\n=== POST-PILOT EVALUATION ===")
    print(json.dumps({
        "fixed_baseline": {
            "val": baseline_val,
            "test": baseline_test,
        },
        "after_controller_training": {
            "val": post_val,
            "test": post_test,
        },
    }, indent=2))


if __name__ == "__main__":
    main()
