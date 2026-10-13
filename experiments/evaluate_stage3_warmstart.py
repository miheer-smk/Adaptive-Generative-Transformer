from __future__ import annotations

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


BACKBONE_CKPT = Path(
    "experiments/runs/stage1_parity_seed3305_best.pt"
)

CONTROLLER_CKPT = Path(
    "experiments/runs/stage3_controller_route_imitation_v2_seed3401.pt"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(device: torch.device) -> AGTv3:
    base = torch.load(
        BACKBONE_CKPT,
        map_location="cpu",
    )

    route_ckpt = torch.load(
        CONTROLLER_CKPT,
        map_location="cpu",
    )

    dim = int(base["dim"])

    model = AGTv3(
        dim=dim,
        num_memory_slots=16,
        vocab_size=2,
    )

    backbone_state = {
        key: value
        for key, value in base["state_dict"].items()
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

    assert set(missing) == expected_controller_keys
    assert unexpected == []

    model.controller.load_state_dict(
        route_ckpt["controller_state_dict"],
        strict=True,
    )

    model.executor.max_execution_depth = 8

    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def evaluate_split(
    model: AGTv3,
    task: TaskAParity,
    split: str,
    min_len: int,
    max_len: int,
    per_length: int,
):
    total = 0
    fixed_correct = 0
    learned_correct = 0

    fixed_losses = []
    learned_losses = []

    route_exact = 0
    route_total = 0

    max_logit_diff = 0.0

    for length in range(min_len, max_len + 1):
        batch = task.make_fixed_set(
            split=split,
            length=length,
            size=per_length,
        )

        for i in range(batch.x.shape[0]):
            x = batch.x[i:i + 1]
            target = batch.target[i:i + 1]

            # -------------------------------------------------
            # Fixed reference route.
            # -------------------------------------------------
            fixed_state = model.initial_persistent_state(
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

            fixed_logits = None

            for t in range(length):
                fixed_out = model.step(
                    x[:, t, :],
                    fixed_state,
                    C_prev,
                )
                fixed_state = fixed_out["state"]
                fixed_logits = fixed_out["logits"]

            assert fixed_logits is not None

            # -------------------------------------------------
            # Learned deterministic route.
            # -------------------------------------------------
            learned_state = model.initial_persistent_state(
                batch_size=1,
                device=x.device,
                dtype=x.dtype,
            )

            learned_logits = None
            routes = []

            for t in range(length):
                learned_out = model.adaptive_differentiable_step(
                    x_t=x[:, t, :],
                    state=learned_state,
                    C_prev=C_prev,
                    budget=torch.tensor(
                        [[float(t), float(length)]],
                        device=x.device,
                        dtype=x.dtype,
                    ),
                    tau=1.0,
                    training=False,
                )

                learned_state = learned_out["state"]
                learned_logits = learned_out["logits"]

                route = learned_out["action_history"]

                routes.append(route)

                route_exact += int(
                    route == [
                        ("ADVANCE", 0),
                        ("ADVANCE", 1),
                        ("ADVANCE", 2),
                        ("ADVANCE", 3),
                        ("ADVANCE", 4),
                        ("ADVANCE", 5),
                        ("ADVANCE", 6),
                        ("ADVANCE", 7),
                    ]
                )
                route_total += 1

            assert learned_logits is not None

            fixed_loss = F.cross_entropy(
                fixed_logits,
                target,
            )

            learned_loss = F.cross_entropy(
                learned_logits,
                target,
            )

            fixed_losses.append(fixed_loss.item())
            learned_losses.append(learned_loss.item())

            fixed_correct += int(
                fixed_logits.argmax(dim=-1).item()
                == target.item()
            )

            learned_correct += int(
                learned_logits.argmax(dim=-1).item()
                == target.item()
            )

            total += 1

            max_logit_diff = max(
                max_logit_diff,
                (
                    fixed_logits
                    - learned_logits
                ).abs().max().item(),
            )

    return {
        "fixed_accuracy": fixed_correct / total,
        "learned_accuracy": learned_correct / total,
        "fixed_mean_ce": sum(fixed_losses) / len(fixed_losses),
        "learned_mean_ce": sum(learned_losses) / len(learned_losses),
        "route_exact_tokens": route_exact,
        "route_total_tokens": route_total,
        "route_exact_accuracy": route_exact / route_total,
        "max_abs_logit_difference": max_logit_diff,
        "examples": total,
    }


def main() -> None:
    set_seed(3501)

    if not BACKBONE_CKPT.exists():
        raise SystemExit(
            f"Missing backbone checkpoint: {BACKBONE_CKPT}"
        )

    if not CONTROLLER_CKPT.exists():
        raise SystemExit(
            f"Missing controller checkpoint: {CONTROLLER_CKPT}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(device)

    task = TaskAParity(
        dim=model.dim,
        device=device,
    )

    print("=== STAGE-3 WARM-START EQUIVALENCE ===")
    print(f"device={device}")
    print(f"dim={model.dim}")
    print("backbone=FROZEN_STAGE1_SEED3305")
    print("controller=ROUTE_IMITATION_V2_SEED3401")
    print("routing_steps=8")
    print("evaluation=DETERMINISTIC")

    val = evaluate_split(
        model,
        task,
        "val",
        task.VAL_MIN_LEN,
        task.VAL_MAX_LEN,
        per_length=8,
    )

    test = evaluate_split(
        model,
        task,
        "test",
        task.TEST_MIN_LEN,
        task.TEST_MAX_LEN,
        per_length=8,
    )

    print("\n=== VAL ===")
    print(val)

    print("\n=== TEST ===")
    print(test)

    # These are diagnostic assertions, not a scientific claim.
    if val["route_exact_accuracy"] != 1.0:
        raise RuntimeError(
            "Warm-start controller failed exact VAL route recovery."
        )

    if test["route_exact_accuracy"] != 1.0:
        raise RuntimeError(
            "Warm-start controller failed exact TEST route recovery."
        )

    if val["learned_accuracy"] != val["fixed_accuracy"]:
        raise RuntimeError(
            "Warm-start learned-route VAL accuracy differs from fixed route."
        )

    if test["learned_accuracy"] != test["fixed_accuracy"]:
        raise RuntimeError(
            "Warm-start learned-route TEST accuracy differs from fixed route."
        )

    print("\n=== RESULT ===")
    print("warm_start_route_recovery=True")
    print("warm_start_task_equivalence=True")


if __name__ == "__main__":
    main()
