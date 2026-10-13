from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.task_a import TaskAParity
from src.agt import AGTv3


CKPT = Path(
    "experiments/runs/stage3_controller_route_imitation_seed3401.pt"
)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ckpt = torch.load(
        CKPT,
        map_location="cpu",
    )

    dim = int(ckpt["dim"])

    model = AGTv3(
        dim=dim,
        num_memory_slots=16,
        vocab_size=2,
    )

    # Closed-loop Stage-3 evaluation uses exactly the controlled
    # eight-decision routing budget.
    model.executor.max_execution_depth = 8

    checkpoint_state = ckpt["backbone_checkpoint"]
    backbone_ckpt = torch.load(
        checkpoint_state,
        map_location="cpu",
    )

    backbone = {
        key: value
        for key, value in backbone_ckpt["state_dict"].items()
        if not key.startswith("controller.")
    }

    missing, unexpected = model.load_state_dict(
        backbone,
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
        ckpt["controller_state_dict"],
        strict=True,
    )

    model.to(device)

    # Exactly the intended Stage-3 fixed decision budget.
    model.executor.max_execution_depth = 8
    model.eval()

    task = TaskAParity(
        dim=dim,
        device=device,
    )

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

    total_tokens = 0
    exact_route_tokens = 0
    correct_task_tokens = 0
    first_error_counts = {}
    route_length_counts = {}
    action_counts = {}

    for length in range(
        task.TEST_MIN_LEN,
        min(task.TEST_MAX_LEN, task.TEST_MIN_LEN + 7) + 1,
    ):
        batch = task.make_fixed_set(
            split="test",
            length=length,
            size=8,
        )

        for b in range(batch.x.shape[0]):
            state = model.initial_persistent_state(
                batch_size=1,
                device=device,
                dtype=batch.x.dtype,
            )

            C_prev = torch.zeros(
                1,
                model.dim,
                device=device,
                dtype=batch.x.dtype,
            )

            token_routes = []
            final_logits = None

            # -------------------------------------------------
            # TRUE CLOSED LOOP:
            #
            # Every token consumes the persistent state produced
            # by the learned-route execution of the previous token.
            # -------------------------------------------------
            for t in range(length):
                out = model.adaptive_differentiable_step(
                    x_t=batch.x[b:b + 1, t, :],
                    state=state,
                    C_prev=C_prev,
                    budget=torch.tensor(
                        [[float(t), float(length)]],
                        device=device,
                        dtype=batch.x.dtype,
                    ),
                    tau=1.0,
                    training=False,
                )

                state = out["state"]
                final_logits = out["logits"]

                history = out["action_history"]
                token_routes.append(history)

                total_tokens += 1

                if history == expected:
                    exact_route_tokens += 1
                elif history:
                    key = str(history[0])
                    first_error_counts[key] = (
                        first_error_counts.get(key, 0) + 1
                    )
                else:
                    first_error_counts["EMPTY"] = (
                        first_error_counts.get("EMPTY", 0) + 1
                    )

                route_length = len(history)
                route_length_counts[route_length] = (
                    route_length_counts.get(route_length, 0) + 1
                )

                for action in history:
                    action_counts[action[0]] = (
                        action_counts.get(action[0], 0) + 1
                    )

            assert final_logits is not None

            pred = final_logits.argmax(dim=-1)

            if pred.item() == batch.target[b].item():
                correct_task_tokens += 1

    exact_route_accuracy = (
        exact_route_tokens / total_tokens
    )

    task_accuracy = (
        correct_task_tokens / total_tokens
    )

    print("=== TRUE CLOSED-LOOP STAGE-3 ROUTE EVALUATION ===")
    print(f"checkpoint={CKPT}")
    print(f"device={device}")
    print(
        f"offline_route_accuracy="
        f"{ckpt['offline_route_accuracy']:.6f}"
    )
    print(f"tokens_evaluated={total_tokens}")
    print(f"exact_route_tokens={exact_route_tokens}")
    print(
        f"exact_route_accuracy="
        f"{exact_route_accuracy:.6f}"
    )
    print(f"task_accuracy={task_accuracy:.6f}")

    print("first_error_distribution=")
    for key, value in sorted(
        first_error_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {key}: {value}")

    print("route_length_distribution=")
    for key, value in sorted(route_length_counts.items()):
        print(f"  {key}: {value}")

    print("action_distribution=")
    for key, value in sorted(action_counts.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
