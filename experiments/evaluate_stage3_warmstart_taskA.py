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

CONTROLLER = Path(
    "experiments/runs/stage3_controller_route_imitation_v2_seed3401.pt"
)


@torch.no_grad()
def evaluate_split(
    model: AGTv3,
    task: TaskAParity,
    split: str,
    lo: int,
    hi: int,
    per_length: int,
):
    model.eval()

    total = 0
    correct = 0
    losses = []
    route_exact = 0

    for length in range(lo, hi + 1):
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

            final_logits = None
            exact = True

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
                final_logits = out["logits"]

                expected = [
                    ("ADVANCE", 0),
                    ("ADVANCE", 1),
                    ("ADVANCE", 2),
                    ("ADVANCE", 3),
                    ("ADVANCE", 4),
                    ("ADVANCE", 5),
                    ("ADVANCE", 6),
                    ("ADVANCE", 7),
                ]

                if out["action_history"] != expected:
                    exact = False

            assert final_logits is not None

            loss = F.cross_entropy(
                final_logits,
                target,
            )

            losses.append(loss.item())
            total += 1
            correct += (
                final_logits.argmax(dim=-1) == target
            ).sum().item()

            if exact:
                route_exact += 1

    return {
        "accuracy": correct / total,
        "mean_ce": sum(losses) / len(losses),
        "route_exact_fraction": route_exact / total,
        "examples": total,
    }


def main():
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
        controller_ckpt["controller_state_dict"],
        strict=True,
    )

    model.executor.max_execution_depth = 8
    model.to(device)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    task = TaskAParity(
        dim=dim,
        device=device,
    )

    print("=== STAGE-3 WARMSTART TASK-A EVALUATION ===")
    print(f"device={device}")
    print("backbone=Stage1_seed3305")
    print("controller=route_imitation_v2_seed3401")
    print("routing_steps=8")
    print("training=False")

    val = evaluate_split(
        model,
        task,
        "val",
        task.VAL_MIN_LEN,
        task.VAL_MAX_LEN,
        16,
    )

    test = evaluate_split(
        model,
        task,
        "test",
        task.TEST_MIN_LEN,
        task.TEST_MAX_LEN,
        16,
    )

    print("\n=== RESULT ===")
    print("VAL:")
    print(val)
    print("TEST:")
    print(test)


if __name__ == "__main__":
    main()
