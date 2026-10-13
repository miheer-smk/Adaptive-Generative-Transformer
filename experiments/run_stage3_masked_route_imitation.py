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
from src.masks import apply_action_mask, build_execution_mask


CKPT = Path(
    "experiments/runs/stage1_parity_seed3305_best.pt"
)

OUT = Path(
    "experiments/runs/stage3_controller_masked_route_imitation_seed3401.pt"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_seed(3401)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CKPT,
        map_location="cpu",
    )

    dim = int(checkpoint["dim"])

    model = AGTv3(
        dim=dim,
        num_memory_slots=16,
        vocab_size=2,
    )

    checkpoint_state = checkpoint["state_dict"]

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

    assert set(missing) == expected_controller_keys
    assert unexpected == []

    model.to(device)

    # Exact Stage-3 pilot budget.
    model.executor.max_execution_depth = 8

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for parameter in model.controller.parameters():
        parameter.requires_grad_(True)

    model.eval()

    task = TaskAParity(
        dim=dim,
        device=device,
    )

    # ---------------------------------------------------------
    # Collect teacher-forced fixed-route observations.
    # ---------------------------------------------------------

    captured_h = [[] for _ in range(8)]
    token_memory = []
    token_memory_mean = []
    token_position = []
    token_length = []

    hooks = []

    def make_hook(index):
        def hook(module, args):
            x = args[0]

            if x.ndim != 3:
                raise RuntimeError(
                    f"Unexpected block input shape: {tuple(x.shape)}"
                )

            captured_h[index].append(
                x[:, 0, :].detach().clone()
            )

        return hook

    for i, block in enumerate(model.blocks.blocks):
        hooks.append(
            block.register_forward_pre_hook(
                make_hook(i)
            )
        )

    try:
        for length in range(
            task.TRAIN_MIN_LEN,
            task.TRAIN_MAX_LEN + 1,
        ):
            batch = task.make_fixed_set(
                split="train",
                length=length,
                size=8,
            )

            state = model.initial_persistent_state(
                batch_size=batch.x.shape[0],
                device=device,
                dtype=batch.x.dtype,
            )

            C_prev = torch.zeros(
                batch.x.shape[0],
                model.dim,
                device=device,
                dtype=batch.x.dtype,
            )

            for t in range(length):
                with torch.no_grad():
                    memory_read, _ = model.memory.read(
                        state.s,
                        state.memory,
                    )

                    memory_mean = state.memory.mean(dim=1)

                    out = model.step(
                        batch.x[:, t, :],
                        state,
                        C_prev,
                    )

                token_memory.append(
                    memory_read.detach().clone()
                )

                token_memory_mean.append(
                    memory_mean.detach().clone()
                )

                token_position.append(t)
                token_length.append(length)

                state = out["state"]

    finally:
        for hook in hooks:
            hook.remove()

    num_tokens = len(token_memory)

    for i in range(8):
        assert len(captured_h[i]) == num_tokens

    # ---------------------------------------------------------
    # Build TOKEN -> BLOCK aligned examples.
    # ---------------------------------------------------------

    H_rows = []
    M_rows = []
    MM_rows = []
    P_rows = []
    B_rows = []
    MASK_rows = []
    Y_rows = []

    for t in range(num_tokens):
        for block_index in range(8):
            batch_dim = token_memory[t].shape[0]

            H_rows.append(
                captured_h[block_index][t]
            )

            M_rows.append(
                token_memory[t]
            )

            MM_rows.append(
                token_memory_mean[t]
            )

            pointer = torch.zeros(
                batch_dim,
                9,
                device=device,
                dtype=token_memory[t].dtype,
            )

            # Before block i:
            #   i=0 -> initial pointer 0
            #   i>0 -> pointer remains at i-1
            current_pointer = (
                0 if block_index == 0
                else block_index - 1
            )

            pointer[:, current_pointer] = 1.0
            P_rows.append(pointer)

            budget = torch.tensor(
                [
                    [
                        float(token_position[t]),
                        float(token_length[t]),
                    ]
                ],
                device=device,
                dtype=token_memory[t].dtype,
            ).expand(
                batch_dim,
                -1,
            )

            B_rows.append(budget)

            # EXACT executor legality mask.
            mask = build_execution_mask(
                pointer=current_pointer,
                execution_count=(
                    0 if block_index == 0
                    else 1
                ),
                just_restarted=False,
                num_blocks=8,
            ).to(device=device)

            MASK_rows.append(
                mask.unsqueeze(0).expand(
                    batch_dim,
                    -1,
                )
            )

            Y_rows.append(
                torch.full(
                    (batch_dim,),
                    block_index,
                    device=device,
                    dtype=torch.long,
                )
            )

    H = torch.cat(H_rows, dim=0)
    M = torch.cat(M_rows, dim=0)
    MM = torch.cat(MM_rows, dim=0)
    P = torch.cat(P_rows, dim=0)
    B = torch.cat(B_rows, dim=0)
    MASK = torch.cat(MASK_rows, dim=0)
    Y = torch.cat(Y_rows, dim=0)

    n = Y.numel()

    assert n == 8320

    print("=== STAGE-3 MASKED ROUTE IMITATION ===")
    print(f"device={device}")
    print(f"route_examples={n}")
    print("backbone=FROZEN_STAGE1_SEED3305")
    print("controller=TRAINABLE")
    print("route=ADVANCE(0..7)")
    print("mask=EXECUTOR_LEGALITY_MASK")
    print("alignment=TOKEN_BLOCK_EXPLICIT")

    optimizer = torch.optim.AdamW(
        model.controller.parameters(),
        lr=1e-3,
        weight_decay=0.0,
    )

    batch_size = 1024

    for step in range(1, 3001):
        indices = torch.randint(
            low=0,
            high=n,
            size=(batch_size,),
            device=device,
        )

        logits = model.controller(
            h=H[indices],
            memory_read=M[indices],
            memory_mean=MM[indices],
            pointer_one_hot=P[indices],
            budget=B[indices],
        )["logits"]

        masked_logits = apply_action_mask(
            logits,
            MASK[indices],
        )

        loss = F.cross_entropy(
            masked_logits,
            Y[indices],
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        for parameter in model.controller.parameters():
            if parameter.grad is None:
                raise RuntimeError(
                    "Controller parameter has no gradient."
                )
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(
                    "Non-finite controller gradient."
                )

        optimizer.step()

        if step == 1 or step % 300 == 0:
            with torch.no_grad():
                predictions = masked_logits.argmax(dim=-1)
                accuracy = (
                    predictions == Y[indices]
                ).float().mean().item()

            print(
                f"step={step:04d} "
                f"masked_ce={loss.item():.6f} "
                f"masked_route_acc={accuracy:.4f}"
            )

    # ---------------------------------------------------------
    # Full masked offline recovery.
    # ---------------------------------------------------------

    model.controller.eval()

    with torch.no_grad():
        logits = model.controller(
            h=H,
            memory_read=M,
            memory_mean=MM,
            pointer_one_hot=P,
            budget=B,
        )["logits"]

        masked_logits = apply_action_mask(
            logits,
            MASK,
        )

        predictions = masked_logits.argmax(dim=-1)

        offline_accuracy = (
            predictions == Y
        ).float().mean().item()

    print("\n=== MASKED OFFLINE ROUTE RECOVERY ===")
    print(f"accuracy={offline_accuracy:.6f}")

    for decision in range(8):
        rows = Y == decision

        decision_accuracy = (
            predictions[rows] == Y[rows]
        ).float().mean().item()

        print(
            f"decision={decision} "
            f"accuracy={decision_accuracy:.6f}"
        )

    # ---------------------------------------------------------
    # Actual deterministic executor route recovery.
    # Test several independent tokens rather than one token.
    # ---------------------------------------------------------

    route_set = task.make_fixed_set(
        split="train",
        length=8,
        size=16,
    )

    route_successes = 0

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

    for example in range(route_set.x.shape[0]):
        state = model.initial_persistent_state(
            batch_size=1,
            device=device,
            dtype=torch.float32,
        )

        C_prev = torch.zeros(
            1,
            model.dim,
            device=device,
            dtype=torch.float32,
        )

        token_route = []

        for t in range(8):
            out = model.adaptive_differentiable_step(
                x_t=route_set.x[
                    example:example + 1,
                    t,
                ],
                state=state,
                C_prev=C_prev,
                budget=torch.tensor(
                    [[float(t), 8.0]],
                    device=device,
                    dtype=torch.float32,
                ),
                tau=1.0,
                training=False,
            )

            state = out["state"]
            token_route.append(
                out["action_history"]
            )

        # All eight token histories must be exactly the teacher route.
        if all(
            history == expected
            for history in token_route
        ):
            route_successes += 1

    route_accuracy = (
        route_successes / route_set.x.shape[0]
    )

    print("\n=== ACTUAL EXECUTOR ROUTE RECOVERY ===")
    print(
        f"examples={route_set.x.shape[0]}"
    )
    print(
        f"accuracy={route_accuracy:.6f}"
    )

    torch.save(
        {
            "seed": 3401,
            "dim": dim,
            "backbone_checkpoint": str(CKPT),
            "masked_offline_route_accuracy": offline_accuracy,
            "executor_route_accuracy": route_accuracy,
            "controller_state_dict": model.controller.state_dict(),
            "route": expected,
        },
        OUT,
    )

    print("\n=== RESULT ===")
    print(
        f"masked_offline_route_accuracy={offline_accuracy:.6f}"
    )
    print(
        f"executor_route_accuracy={route_accuracy:.6f}"
    )
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
