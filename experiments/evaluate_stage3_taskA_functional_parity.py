from __future__ import annotations

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
ROUTE_CKPT = Path(
    "experiments/runs/stage3_controller_route_imitation_v2_seed3401.pt"
)


def load_model(device):
    backbone_ckpt = torch.load(
        BACKBONE,
        map_location="cpu",
    )
    route_ckpt = torch.load(
        ROUTE_CKPT,
        map_location="cpu",
    )

    dim = int(backbone_ckpt["dim"])

    model = AGTv3(
        dim=dim,
        num_memory_slots=16,
        vocab_size=2,
    )

    backbone_state = {
        key: value
        for key, value in backbone_ckpt["state_dict"].items()
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

    model.controller.load_state_dict(
        route_ckpt["controller_state_dict"],
        strict=True,
    )

    model.to(device)
    model.executor.max_execution_depth = 8

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, dim


@torch.no_grad()
def fixed_run(model, x):
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

    out_logits = None

    for t in range(x.shape[1]):
        out = model.step(
            x[:, t, :],
            state,
            C_prev,
        )
        state = out["state"]
        out_logits = out["logits"]

    assert out_logits is not None
    return out_logits


@torch.no_grad()
def learned_run(model, x):
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

    out_logits = None
    routes = []

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
            tau=1.0,
            training=False,
        )

        state = out["state"]
        out_logits = out["logits"]
        routes.append(out["action_history"])

    assert out_logits is not None

    return out_logits, routes


@torch.no_grad()
def evaluate(model, task, split, min_len, max_len, per_length):
    total = 0
    fixed_correct = 0
    learned_correct = 0
    fixed_losses = []
    learned_losses = []
    max_logit_delta = 0.0
    exact_route_tokens = 0
    route_tokens = 0

    expected_route = [
        ("ADVANCE", 0),
        ("ADVANCE", 1),
        ("ADVANCE", 2),
        ("ADVANCE", 3),
        ("ADVANCE", 4),
        ("ADVANCE", 5),
        ("ADVANCE", 6),
        ("ADVANCE", 7),
    ]

    for length in range(min_len, max_len + 1):
        batch = task.make_fixed_set(
            split=split,
            length=length,
            size=per_length,
        )

        for i in range(batch.x.shape[0]):
            x = batch.x[i:i + 1]
            target = batch.target[i:i + 1]

            fixed_logits = fixed_run(model, x)
            learned_logits, routes = learned_run(model, x)

            fixed_loss = F.cross_entropy(
                fixed_logits,
                target,
            )
            learned_loss = F.cross_entropy(
                learned_logits,
                target,
            )

            fixed_losses.append(
                fixed_loss.item()
            )
            learned_losses.append(
                learned_loss.item()
            )

            fixed_correct += (
                fixed_logits.argmax(dim=-1) == target
            ).sum().item()

            learned_correct += (
                learned_logits.argmax(dim=-1) == target
            ).sum().item()

            total += 1

            max_logit_delta = max(
                max_logit_delta,
                (
                    fixed_logits - learned_logits
                ).abs().max().item(),
            )

            for route in routes:
                route_tokens += 1
                if route == expected_route:
                    exact_route_tokens += 1

    return {
        "examples": total,
        "fixed_accuracy": fixed_correct / total,
        "learned_accuracy": learned_correct / total,
        "fixed_mean_ce": sum(fixed_losses) / len(fixed_losses),
        "learned_mean_ce": sum(learned_losses) / len(learned_losses),
        "max_absolute_logit_delta": max_logit_delta,
        "exact_route_tokens": exact_route_tokens,
        "route_tokens": route_tokens,
    }


def main():
    if not BACKBONE.exists():
        raise SystemExit(f"Missing backbone checkpoint: {BACKBONE}")

    if not ROUTE_CKPT.exists():
        raise SystemExit(f"Missing route checkpoint: {ROUTE_CKPT}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, dim = load_model(device)

    task = TaskAParity(
        dim=dim,
        device=device,
    )

    print("=== STAGE-3 TASK-A FUNCTIONAL PARITY ===")
    print(f"device={device}")
    print(f"dim={dim}")
    print("backbone=FROZEN_STAGE1_SEED3305")
    print("controller=ROUTE_IMITATION_V2")
    print("routing_depth=8")

    val = evaluate(
        model,
        task,
        "val",
        task.VAL_MIN_LEN,
        task.VAL_MAX_LEN,
        8,
    )

    test = evaluate(
        model,
        task,
        "test",
        task.TEST_MIN_LEN,
        task.TEST_MAX_LEN,
        8,
    )

    print("\n=== VAL ===")
    for key, value in val.items():
        print(f"{key}={value}")

    print("\n=== TEST ===")
    for key, value in test.items():
        print(f"{key}={value}")

    print("\n=== RESULT ===")

    all_exact = (
        val["exact_route_tokens"] == val["route_tokens"]
        and test["exact_route_tokens"] == test["route_tokens"]
    )

    accuracy_match = (
        val["learned_accuracy"] == val["fixed_accuracy"]
        and test["learned_accuracy"] == test["fixed_accuracy"]
    )

    ce_match = (
        abs(
            val["learned_mean_ce"]
            - val["fixed_mean_ce"]
        ) < 1e-6
        and abs(
            test["learned_mean_ce"]
            - test["fixed_mean_ce"]
        ) < 1e-6
    )

    print(f"exact_route={all_exact}")
    print(f"accuracy_match={accuracy_match}")
    print(f"ce_match={ce_match}")
    print(
        f"max_logit_delta="
        f"{max(val['max_absolute_logit_delta'], test['max_absolute_logit_delta']):.10e}"
    )

    if not all_exact:
        raise RuntimeError(
            "Functional parity failed: learned controller did not "
            "reproduce the canonical route."
        )

    if not accuracy_match:
        raise RuntimeError(
            "Functional parity failed: accuracy differs."
        )

    if not ce_match:
        raise RuntimeError(
            "Functional parity failed: CE differs by >= 1e-6."
        )

    print("functional_parity=True")


if __name__ == "__main__":
    main()
