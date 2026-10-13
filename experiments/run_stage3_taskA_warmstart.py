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


BACKBONE_CKPT = Path(
    "experiments/runs/stage1_parity_seed3305_best.pt"
)
CONTROLLER_CKPT = Path(
    "experiments/runs/stage3_controller_route_imitation_v2_seed3401.pt"
)
OUT = Path(
    "experiments/runs/stage3_taskA_warmstart_seed3401_steps200.pt"
)

BANK = [
    "SWA", "SWA", "GQA", "GQA",
    "GDN", "GDN", "FFN", "FFN",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_frozen_backbone(
    model: AGTv3,
    checkpoint: dict,
) -> None:
    state = checkpoint["state_dict"]

    backbone = {
        key: value
        for key, value in state.items()
        if not key.startswith("controller.")
    }

    missing, unexpected = model.load_state_dict(
        backbone,
        strict=False,
    )

    expected_controller = {
        key
        for key in model.state_dict().keys()
        if key.startswith("controller.")
    }

    assert set(missing) == expected_controller, (
        "Unexpected missing keys: "
        f"{sorted(set(missing) - expected_controller)}"
    )

    assert unexpected == [], (
        f"Unexpected checkpoint keys: {unexpected}"
    )


def load_controller(
    model: AGTv3,
    checkpoint: dict,
) -> None:
    if "controller_state_dict" in checkpoint:
        controller_state = checkpoint["controller_state_dict"]

    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        controller_state = {
            key.removeprefix("controller."): value
            for key, value in state.items()
            if key.startswith("controller.")
        }

    else:
        raise RuntimeError(
            "Warm-start checkpoint contains neither "
            "'controller_state_dict' nor 'state_dict'."
        )

    missing, unexpected = model.controller.load_state_dict(
        controller_state,
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            "Controller checkpoint mismatch:\n"
            f"missing={missing}\n"
            f"unexpected={unexpected}"
        )


def run_learned_sequence(
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
    histories = []

    for t in range(x.shape[1]):
        out = model.adaptive_differentiable_step(
            x_t=x[:, t, :],
            state=state,
            C_prev=C_prev,
            budget=torch.tensor(
                [[float(t), float(x.shape[1])]],
                device=x.device,
                dtype=x.dtype,
            ).expand(x.shape[0], -1),
            tau=tau,
            training=training,
        )

        if out["execution_count"] != 8:
            raise RuntimeError(
                "Expected exactly 8 routing decisions, got "
                f"{out['execution_count']}."
            )

        state = out["state"]
        logits = out["logits"]
        histories.append(out["action_history"])

    assert logits is not None

    return logits, histories


@torch.no_grad()
def evaluate_learned(
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

        for i in range(batch.x.shape[0]):
            logits, _ = run_learned_sequence(
                model,
                batch.x[i:i + 1],
                tau=tau,
                training=False,
            )

            target = batch.target[i:i + 1]

            losses.append(
                F.cross_entropy(
                    logits,
                    target,
                ).item()
            )

            correct += (
                logits.argmax(dim=-1) == target
            ).sum().item()

            total += 1

    return {
        "accuracy": correct / total,
        "mean_ce": sum(losses) / len(losses),
        "examples": total,
    }


def route_summary(
    model: AGTv3,
    task: TaskAParity,
    tau: float,
):
    probe = task.make_fixed_set(
        split="train",
        length=8,
        size=8,
    )

    _, histories = run_learned_sequence(
        model,
        probe.x,
        tau=tau,
        training=False,
    )

    counts = {}

    for token_history in histories:
        for action, index in token_history:
            key = (
                action
                if index is None
                else f"{action}({index})"
            )
            counts[key] = counts.get(key, 0) + 1

    total = sum(counts.values())

    return {
        "total_decisions": total,
        "counts": counts,
    }


def main() -> None:
    seed = 3401
    steps = 200
    batch_size = 1
    lr = 1e-4
    tau = 2.0

    set_seed(seed)
    torch.set_num_threads(4)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if not BACKBONE_CKPT.exists():
        raise SystemExit(
            f"Missing backbone checkpoint: {BACKBONE_CKPT}"
        )

    if not CONTROLLER_CKPT.exists():
        raise SystemExit(
            f"Missing controller checkpoint: {CONTROLLER_CKPT}"
        )

    backbone_ckpt = torch.load(
        BACKBONE_CKPT,
        map_location="cpu",
    )

    controller_ckpt = torch.load(
        CONTROLLER_CKPT,
        map_location="cpu",
    )

    dim = int(backbone_ckpt["dim"])

    model = AGTv3(
        dim=dim,
        num_memory_slots=16,
        vocab_size=2,
    )

    load_frozen_backbone(
        model,
        backbone_ckpt,
    )

    load_controller(
        model,
        controller_ckpt,
    )

    model.to(device)

    actual_bank = [
        block.species
        for block in model.blocks.blocks
    ]

    assert actual_bank == BANK, (
        f"Unexpected block bank: {actual_bank}"
    )

    # Exact controlled Stage-3 budget.
    model.executor.max_execution_depth = 8

    # Freeze the validated backbone.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # Train ONLY the unified controller.
    for parameter in model.controller.parameters():
        parameter.requires_grad_(True)

    trainable = list(
        model.controller.parameters()
    )

    task = TaskAParity(
        dim=dim,
        device=device,
    )

    baseline = evaluate_learned(
        model,
        task,
        "val",
        task.VAL_MIN_LEN,
        task.VAL_MAX_LEN,
        16,
        tau,
    )

    route_before = route_summary(
        model,
        task,
        tau,
    )

    print("=== AGT-v3 GATE-C TASK-A WARM-START ===")
    print(f"device={device}")
    print(f"dim={dim}")
    print(f"steps={steps}")
    print(f"batch_size={batch_size}")
    print(f"lr={lr}")
    print(f"tau={tau}")
    print("backbone=FROZEN_STAGE1_SEED3305")
    print("controller=WARMSTART_ROUTE_IMITATION_V2")
    print("routing_steps=8")
    print("trainable=CONTROLLER_ONLY")

    print("\n=== PRE-TRAINING ===")
    print(json.dumps({
        "val": baseline,
        "route": route_before,
    }, indent=2))

    optimizer = torch.optim.AdamW(
        trainable,
        lr=lr,
        weight_decay=0.0,
    )

    for step in range(1, steps + 1):
        length = int(
            torch.randint(
                task.TRAIN_MIN_LEN,
                task.TRAIN_MAX_LEN + 1,
                (1,),
            ).item()
        )

        batch = task.make_batch(
            batch_size=batch_size,
            length=length,
            split="train",
        )

        model.train()

        logits, _ = run_learned_sequence(
            model,
            batch.x,
            tau=tau,
            training=True,
        )

        loss = F.cross_entropy(
            logits,
            batch.target,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        for parameter in trainable:
            if parameter.grad is None:
                raise RuntimeError(
                    "Controller parameter has no gradient."
                )

            if not torch.isfinite(
                parameter.grad
            ).all():
                raise RuntimeError(
                    "Non-finite controller gradient."
                )

        optimizer.step()

        if step == 1 or step % 25 == 0:
            print(
                f"step={step:03d} "
                f"length={length} "
                f"ce={loss.item():.6f}"
            )

    model.eval()

    post_val = evaluate_learned(
        model,
        task,
        "val",
        task.VAL_MIN_LEN,
        task.VAL_MAX_LEN,
        16,
        tau,
    )

    post_test = evaluate_learned(
        model,
        task,
        "test",
        task.TEST_MIN_LEN,
        task.TEST_MAX_LEN,
        16,
        tau,
    )

    route_after = route_summary(
        model,
        task,
        tau,
    )

    print("\n=== POST-TRAINING ===")
    print(json.dumps({
        "val": post_val,
        "test": post_test,
        "route": route_after,
    }, indent=2))

    torch.save(
        {
            "seed": seed,
            "dim": dim,
            "steps": steps,
            "lr": lr,
            "tau": tau,
            "routing_steps": 8,
            "backbone_checkpoint": str(
                BACKBONE_CKPT
            ),
            "controller_checkpoint": str(
                CONTROLLER_CKPT
            ),
            "controller_state_dict":
                model.controller.state_dict(),
            "pre_val": baseline,
            "post_val": post_val,
            "post_test": post_test,
            "route_before": route_before,
            "route_after": route_after,
        },
        OUT,
    )

    print("\n=== RESULT ===")
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
