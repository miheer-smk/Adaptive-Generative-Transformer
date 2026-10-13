from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.task_a import TaskAParity
from src.agt import AGTv3


BACKBONE = Path(
    "experiments/runs/stage1_parity_seed3305_best.pt"
)

CONTROLLER = Path(
    "experiments/runs/stage3_controller_route_imitation_v2_seed3401.pt"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_backbone(model, checkpoint):
    state = checkpoint["state_dict"]

    backbone_state = {
        key: value
        for key, value in state.items()
        if not key.startswith("controller.")
    }

    missing, unexpected = model.load_state_dict(
        backbone_state,
        strict=False,
    )

    expected_controller = {
        key
        for key in model.state_dict()
        if key.startswith("controller.")
    }

    assert set(missing) == expected_controller
    assert unexpected == []


def evaluate(
    model,
    task,
    split,
    min_len,
    max_len,
    per_length,
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

        for i in range(batch.x.shape[0]):
            x = batch.x[i:i + 1]
            target = batch.target[i:i + 1]

            state = model.initial_persistent_state(
                batch_size=1,
                device=x.device,
                dtype=x.dtype,
            )

            C_prev = torch.zeros(
                1,
                model.dim,
                device=x.device,
                dtype=x.dtype,
            )

            logits = None

            for t in range(length):
                out = model.adaptive_differentiable_step(
                    x_t=x[:, t, :],
                    state=state,
                    C_prev=C_prev,
                    budget=torch.tensor(
                        [[float(t), float(length)]],
                        device=x.device,
                        dtype=x.dtype,
                    ),
                    tau=1.0,
                    training=False,
                )

                state = out["state"]
                logits = out["logits"]

            assert logits is not None

            loss = F.cross_entropy(
                logits,
                target,
            )

            losses.append(loss.item())
            total += 1
            correct += (
                logits.argmax(dim=-1) == target
            ).sum().item()

    return {
        "accuracy": correct / total,
        "mean_ce": sum(losses) / len(losses),
        "examples": total,
    }


def run_train_sequence(
    model,
    x,
    tau,
):
    state = model.initial_persistent_state(
        batch_size=1,
        device=x.device,
        dtype=x.dtype,
    )

    C_prev = torch.zeros(
        1,
        model.dim,
        device=x.device,
        dtype=x.dtype,
    )

    logits = None

    for t in range(x.shape[1]):
        out = model.adaptive_differentiable_step(
            x_t=x[:, t, :],
            state=state,
            C_prev=C_prev,
            budget=torch.tensor(
                [[float(t), float(x.shape[1])]],
                device=x.device,
                dtype=x.dtype,
            ),
            tau=tau,
            training=True,
        )

        state = out["state"]
        logits = out["logits"]

        if out["execution_count"] != 8:
            raise RuntimeError(
                f"Expected 8 routing decisions, got "
                f"{out['execution_count']}."
            )

    assert logits is not None

    return logits


def main():
    seed = 3501
    steps = 200
    lr = 1e-5
    tau = 2.0

    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    backbone_ckpt = torch.load(
        BACKBONE,
        map_location="cpu",
    )

    controller_ckpt = torch.load(
        CONTROLLER,
        map_location="cpu",
    )

    dim = int(backbone_ckpt["dim"])

    model = AGTv3(
        dim=dim,
        num_memory_slots=16,
        vocab_size=2,
    )

    load_backbone(
        model,
        backbone_ckpt,
    )

    model.controller.load_state_dict(
        controller_ckpt["controller_state_dict"],
        strict=True,
    )

    model.to(device)

    model.executor.max_execution_depth = 8

    # Freeze the proven Stage-1 representation.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # Only routing controller is trainable.
    for parameter in model.controller.parameters():
        parameter.requires_grad_(True)

    task = TaskAParity(
        dim=dim,
        device=device,
    )

    print("=== STAGE-3 WARMSTART TASK-LOSS TRAINING ===")
    print(f"device={device}")
    print(f"dim={dim}")
    print(f"steps={steps}")
    print(f"lr={lr}")
    print(f"tau={tau}")
    print("backbone=FROZEN_STAGE1_SEED3305")
    print("controller=WARMSTART_ROUTE_IMITATION_V2")
    print("routing_steps=8")
    print("trainable=CONTROLLER_ONLY")

    baseline_val = evaluate(
        model,
        task,
        "val",
        task.VAL_MIN_LEN,
        task.VAL_MAX_LEN,
        16,
    )

    baseline_test = evaluate(
        model,
        task,
        "test",
        task.TEST_MIN_LEN,
        task.TEST_MAX_LEN,
        16,
    )

    print("\n=== WARMSTART BASELINE ===")
    print(json.dumps({
        "val": baseline_val,
        "test": baseline_test,
    }, indent=2))

    optimizer = torch.optim.AdamW(
        model.controller.parameters(),
        lr=lr,
        weight_decay=0.0,
    )

    best_val = float("inf")

    for step in range(1, steps + 1):
        length = int(
            torch.randint(
                task.TRAIN_MIN_LEN,
                task.TRAIN_MAX_LEN + 1,
                (1,),
            ).item()
        )

        batch = task.make_batch(
            batch_size=1,
            length=length,
            split="train",
        )

        model.train()

        logits = run_train_sequence(
            model,
            batch.x,
            tau=tau,
        )

        loss = F.cross_entropy(
            logits,
            batch.target,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        for parameter in model.controller.parameters():
            if parameter.grad is None:
                raise RuntimeError(
                    "Missing controller gradient."
                )

            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(
                    "Non-finite controller gradient."
                )

        optimizer.step()

        if step == 1 or step % 25 == 0:
            val = evaluate(
                model,
                task,
                "val",
                task.VAL_MIN_LEN,
                task.VAL_MAX_LEN,
                8,
            )

            print(
                f"step={step:03d} "
                f"train_ce={loss.item():.6f} "
                f"val_ce={val['mean_ce']:.6f} "
                f"val_acc={val['accuracy']:.4f}"
            )

            if val["mean_ce"] < best_val:
                best_val = val["mean_ce"]

    final_val = evaluate(
        model,
        task,
        "val",
        task.VAL_MIN_LEN,
        task.VAL_MAX_LEN,
        16,
    )

    final_test = evaluate(
        model,
        task,
        "test",
        task.TEST_MIN_LEN,
        task.TEST_MAX_LEN,
        16,
    )

    print("\n=== FINAL RESULT ===")
    print(json.dumps({
        "warmstart_baseline": {
            "val": baseline_val,
            "test": baseline_test,
        },
        "after_task_loss": {
            "val": final_val,
            "test": final_test,
        },
    }, indent=2))


if __name__ == "__main__":
    main()
