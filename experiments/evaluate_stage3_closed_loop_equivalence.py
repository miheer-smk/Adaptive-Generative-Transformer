from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from data.task_a import TaskAParity
from src.agt import AGTv3


BACKBONE = "experiments/runs/stage1_parity_seed3305_best.pt"
ROUTE_CKPT = "experiments/runs/stage3_controller_route_imitation_v2_seed3401.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(device, backbone_ckpt, route_ckpt):
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

    model.executor.max_execution_depth = 8
    model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


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

    final = None

    for t in range(x.shape[1]):
        out = model.step(
            x[:, t, :],
            state,
            C_prev,
        )
        state = out["state"]
        final = out

    return final


@torch.no_grad()
def run_adaptive(model, x):
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

    final = None
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
        final = out
        routes.append(out["action_history"])

    return final, routes


def main():
    set_seed(999)

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

    model = make_model(
        device,
        backbone_ckpt,
        route_ckpt,
    )

    task = TaskAParity(
        dim=model.dim,
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

    max_h = 0.0
    max_H = 0.0
    max_logits = 0.0

    failures = []

    print("=== MULTI-LENGTH CLOSED-LOOP EQUIVALENCE ===")
    print(f"device={device}")
    print("backbone=STAGE1_SEED3305")
    print("controller=ROUTE_IMITATION_V2_SEED3401")
    print("routing_depth=8")

    for length in range(4, 33):
        batch = task.make_fixed_set(
            split="train" if length <= 16 else "test",
            length=length,
            size=1,
        )

        fixed = run_fixed(model, batch.x)
        adaptive, routes = run_adaptive(model, batch.x)

        h_diff = (
            fixed["h_terminal"] - adaptive["h_terminal"]
        ).abs().max().item()

        H_diff = (
            fixed["H_t"] - adaptive["H_t"]
        ).abs().max().item()

        logit_diff = (
            fixed["logits"] - adaptive["logits"]
        ).abs().max().item()

        route_ok = all(
            route == expected
            for route in routes
        )

        max_h = max(max_h, h_diff)
        max_H = max(max_H, H_diff)
        max_logits = max(max_logits, logit_diff)

        print(
            f"L={length:02d} "
            f"h_diff={h_diff:.3e} "
            f"H_diff={H_diff:.3e} "
            f"logit_diff={logit_diff:.3e} "
            f"route_ok={route_ok}"
        )

        if (
            h_diff != 0.0
            or H_diff != 0.0
            or logit_diff != 0.0
            or not route_ok
        ):
            failures.append(
                (
                    length,
                    h_diff,
                    H_diff,
                    logit_diff,
                    route_ok,
                )
            )

    print("\n=== MAX DIFFERENCES ===")
    print(f"max_h_diff={max_h:.3e}")
    print(f"max_H_diff={max_H:.3e}")
    print(f"max_logit_diff={max_logits:.3e}")

    print("\n=== RESULT ===")
    print(f"lengths_tested={len(range(4, 33))}")
    print(f"failures={len(failures)}")

    if failures:
        for item in failures:
            print("FAIL:", item)
        raise RuntimeError(
            "Multi-length closed-loop equivalence failed."
        )

    print("closed_loop_equivalence=True")


if __name__ == "__main__":
    main()
