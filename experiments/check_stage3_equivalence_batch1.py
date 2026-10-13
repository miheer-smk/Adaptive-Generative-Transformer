from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from data.task_a import TaskAParity
from src.agt import AGTv3


BACKBONE = "experiments/runs/stage1_parity_seed3305_best.pt"
ROUTE_CKPT = "experiments/runs/stage3_controller_route_imitation_v2_seed3401.pt"


@torch.no_grad()
def run_fixed(model, x):
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
        out = model.step(
            x[:, t, :],
            state,
            C_prev,
        )
        state = out["state"]
        logits = out["logits"]

    return logits


@torch.no_grad()
def run_closed_loop(model, x):
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
    route = []

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
        logits = out["logits"]
        route.append(out["action_history"])

    return logits, route


def main():
    torch.manual_seed(123)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    backbone_ckpt = torch.load(
        BACKBONE,
        map_location="cpu",
    )

    route_ckpt = torch.load(
        ROUTE_CKPT,
        map_location="cpu",
    )

    print("=== STAGE-3 BATCH-1 CLOSED-LOOP CHECK ===")
    print(
        "offline_route_accuracy=",
        route_ckpt.get("offline_route_accuracy"),
    )
    print(
        "executor_route_recovered=",
        route_ckpt.get("executor_route_recovered"),
    )

    dim = int(backbone_ckpt["dim"])

    model = AGTv3(
        dim=dim,
        num_memory_slots=16,
        vocab_size=2,
    )

    backbone_state = {
        k: v
        for k, v in backbone_ckpt["state_dict"].items()
        if not k.startswith("controller.")
    }

    missing, unexpected = model.load_state_dict(
        backbone_state,
        strict=False,
    )

    expected_controller = {
        k
        for k in model.state_dict()
        if k.startswith("controller.")
    }

    assert set(missing) == expected_controller
    assert unexpected == []

    model.controller.load_state_dict(
        route_ckpt["controller_state_dict"],
        strict=True,
    )

    model.to(device)

    # Critical: closed-loop diagnostic is exactly one 8-decision budget.
    model.executor.max_execution_depth = 8

    task = TaskAParity(
        dim=dim,
        device=device,
    )

    batch = task.make_fixed_set(
        split="train",
        length=8,
        size=4,
    )

    fixed_correct = 0
    closed_correct = 0
    logit_diffs = []
    route_exact = 0

    for i in range(4):
        x_i = batch.x[i:i + 1]
        y_i = batch.target[i:i + 1]

        fixed_logits = run_fixed(
            model,
            x_i,
        )

        closed_logits, route = run_closed_loop(
            model,
            x_i,
        )

        fixed_pred = fixed_logits.argmax(dim=-1)
        closed_pred = closed_logits.argmax(dim=-1)

        fixed_correct += int(
            (fixed_pred == y_i).item()
        )

        closed_correct += int(
            (closed_pred == y_i).item()
        )

        logit_diffs.append(
            (fixed_logits - closed_logits)
            .abs()
            .max()
            .item()
        )

        exact = route[0] == [
            ("ADVANCE", 0),
            ("ADVANCE", 1),
            ("ADVANCE", 2),
            ("ADVANCE", 3),
            ("ADVANCE", 4),
            ("ADVANCE", 5),
            ("ADVANCE", 6),
            ("ADVANCE", 7),
        ]

        route_exact += int(exact)

        print(
            f"example={i} "
            f"target={int(y_i.item())} "
            f"fixed_pred={int(fixed_pred.item())} "
            f"closed_pred={int(closed_pred.item())} "
            f"route_exact={exact} "
            f"max_logit_diff={logit_diffs[-1]:.6e}"
        )

        print(
            f"  closed_route={route[0]}"
        )

    print("\n=== SUMMARY ===")
    print(f"fixed_correct={fixed_correct}/4")
    print(f"closed_correct={closed_correct}/4")
    print(f"exact_routes={route_exact}/4")
    print(
        "mean_max_logit_diff="
        f"{sum(logit_diffs) / len(logit_diffs):.6e}"
    )


if __name__ == "__main__":
    main()
