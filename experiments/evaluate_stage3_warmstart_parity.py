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

    assert int(controller_ckpt["dim"]) == dim
    assert controller_ckpt["offline_route_accuracy"] == 1.0

    model = AGTv3(
        dim=dim,
        num_memory_slots=16,
        vocab_size=2,
    )

    # Load validated Stage-1 backbone only.
    backbone_state = {
        k: v
        for k, v in backbone_ckpt["state_dict"].items()
        if not k.startswith("controller.")
    }

    missing, unexpected = model.load_state_dict(
        backbone_state,
        strict=False,
    )

    expected_controller_keys = {
        k
        for k in model.state_dict()
        if k.startswith("controller.")
    }

    assert set(missing) == expected_controller_keys
    assert unexpected == []

    # Load the perfectly recovered Stage-3 controller.
    model.controller.load_state_dict(
        controller_ckpt["controller_state_dict"],
        strict=True,
    )

    model.to(device)

    # Exactly the validated 8-decision route budget.
    model.executor.max_execution_depth = 8

    model.eval()

    task = TaskAParity(
        dim=dim,
        device=device,
    )

    total = 0
    correct = 0
    losses = []

    route_exact = 0
    route_total = 0

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

    for length in range(
        task.VAL_MIN_LEN,
        task.TEST_MAX_LEN + 1,
    ):
        split = (
            "val"
            if length <= task.VAL_MAX_LEN
            else "test"
        )

        batch = task.make_fixed_set(
            split=split,
            length=length,
            size=32,
        )

        for i in range(batch.x.shape[0]):
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

            final_logits = None
            exact = True

            for t in range(length):
                out = model.adaptive_differentiable_step(
                    x_t=batch.x[i:i + 1, t, :],
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

                if out["action_history"] != expected_route:
                    exact = False

            assert final_logits is not None

            target = batch.target[i:i + 1]
            loss = F.cross_entropy(
                final_logits,
                target,
            )

            losses.append(loss.item())
            total += 1

            correct += (
                final_logits.argmax(dim=-1) == target
            ).sum().item()

            route_total += 1
            route_exact += int(exact)

    print("=== STAGE-3 WARMSTART PARITY ===")
    print(f"device={device}")
    print("backbone=STAGE1_SEED3305")
    print("controller=ROUTE_IMITATION_V2_SEED3401")
    print("routing_decisions=8")
    print(f"examples={total}")
    print(f"parity_accuracy={correct / total:.6f}")
    print(f"mean_ce={sum(losses) / len(losses):.6f}")
    print(f"exact_route_accuracy={route_exact / route_total:.6f}")

    assert correct / total == 1.0
    assert route_exact / route_total == 1.0

    print("warmstart_parity=True")


if __name__ == "__main__":
    main()
