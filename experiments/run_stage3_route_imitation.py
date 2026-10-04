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


CKPT = Path("experiments/runs/stage1_parity_seed3305_best.pt")
OUT = Path("experiments/runs/stage3_controller_route_imitation_v2_seed3401.pt")


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

    ckpt = torch.load(CKPT, map_location="cpu")
    dim = int(ckpt["dim"])

    model = AGTv3(
        dim=dim,
        num_memory_slots=16,
        vocab_size=2,
    )

    checkpoint_state = ckpt["state_dict"]

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
        for key in model.state_dict()
        if key.startswith("controller.")
    }

    assert set(missing) == expected_controller_keys
    assert unexpected == []

    model.to(device)

    # Stage-3 route diagnostic uses exactly one 8-decision routing budget.
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
    # Collect correctly aligned teacher-forced route examples.
    #
    # One row is:
    #
    #   (h_before_decision,
    #    memory_read,
    #    memory_mean,
    #    pointer_one_hot,
    #    budget,
    #    target_action)
    #
    # Target sequence is:
    #
    #   ADVANCE(0), ADVANCE(1), ..., ADVANCE(7)
    #
    # under the validated fixed-route executor.
    # ---------------------------------------------------------

    block_inputs = [[] for _ in range(8)]
    token_memory = []
    token_memory_mean = []
    token_lengths = []
    token_positions = []

    hooks = []

    def make_hook(index):
        def hook(module, args):
            x = args[0]

            if x.ndim != 3 or x.shape[1:] != (1, dim):
                raise RuntimeError(
                    f"Unexpected block {index} input shape: "
                    f"{tuple(x.shape)}"
                )

            block_inputs[index].append(
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

                token_lengths.append(length)
                token_positions.append(t)

                state = out["state"]

    finally:
        for hook in hooks:
            hook.remove()

    # Each fixed-route token produced one call to every block.
    num_tokens = len(token_memory)

    for i in range(8):
        assert len(block_inputs[i]) == num_tokens, (
            f"Block {i} captures {len(block_inputs[i])}, "
            f"expected {num_tokens}."
        )

    # ---------------------------------------------------------
    # Build aligned rows in TOKEN -> BLOCK order.
    # ---------------------------------------------------------

    H_rows = []
    M_rows = []
    MM_rows = []
    P_rows = []
    B_rows = []
    Y_rows = []

    for t in range(num_tokens):
        for i in range(8):
            H_rows.append(block_inputs[i][t])

            M_rows.append(token_memory[t])
            MM_rows.append(token_memory_mean[t])

            pointer = torch.zeros(
                token_memory[t].shape[0],
                9,
                device=device,
                dtype=token_memory[t].dtype,
            )

            current_pointer = 0 if i == 0 else i - 1
            pointer[:, current_pointer] = 1.0
            P_rows.append(pointer)

            # Budget-relative conditioning is optional in Gate-B.
            # Disable it in the first controlled route-recovery diagnostic.
            budget = torch.zeros(
                token_memory[t].shape[0],
                2,
                device=device,
                dtype=token_memory[t].dtype,
            )

            B_rows.append(budget)

            Y_rows.append(
                torch.full(
                    (token_memory[t].shape[0],),
                    i,
                    device=device,
                    dtype=torch.long,
                )
            )

    H = torch.cat(H_rows, dim=0)
    M = torch.cat(M_rows, dim=0)
    MM = torch.cat(MM_rows, dim=0)
    P = torch.cat(P_rows, dim=0)
    B = torch.cat(B_rows, dim=0)
    Y = torch.cat(Y_rows, dim=0)

    n = Y.numel()

    assert n == 8320, f"Expected 8320 route examples, got {n}."

    # Exact legality masks used by the real executor for the canonical
    # teacher-forced sweep. The mask depends on the decision index,
    # current pointer, and execution count; it is not a learned feature.
    decision_masks = torch.stack(
        [
            build_execution_mask(
                pointer=0 if decision == 0 else decision - 1,
                execution_count=decision,
                just_restarted=False,
                num_blocks=8,
            )
            for decision in range(8)
        ],
        dim=0,
    ).to(device=device)

    print("=== STAGE-3 ROUTE IMITATION DIAGNOSTIC ===")
    print(f"device={device}")
    print(f"route_examples={n}")
    print("backbone=FROZEN_STAGE1_SEED3305")
    print("controller=TRAINABLE")
    print("target_route=ADVANCE(0..7)")
    print("alignment=TOKEN_BLOCK_EXPLICIT")

    optimizer = torch.optim.AdamW(
        model.controller.parameters(),
        lr=1e-3,
        weight_decay=0.0,
    )

    batch_size = 1024

    for step in range(1, 5001):
        idx = torch.randint(
            low=0,
            high=n,
            size=(batch_size,),
            device=device,
        )

        logits = model.controller(
            h=H[idx],
            memory_read=M[idx],
            memory_mean=MM[idx],
            pointer_one_hot=P[idx],
            budget=B[idx],
        )["logits"]

        masked_logits = apply_action_mask(
            logits,
            decision_masks[Y[idx]],
        )

        loss = F.cross_entropy(
            masked_logits,
            Y[idx],
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

        if step == 1 or step % 200 == 0:
            with torch.no_grad():
                pred = masked_logits.argmax(dim=-1)
                acc = (
                    pred == Y[idx]
                ).float().mean().item()

            print(
                f"step={step:04d} "
                f"ce={loss.item():.6f} "
                f"route_acc={acc:.4f}"
            )

    # ---------------------------------------------------------
    # Full offline route recovery.
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
            decision_masks[Y],
        )

        pred = masked_logits.argmax(dim=-1)
        offline_acc = (
            pred == Y
        ).float().mean().item()

    print("\n=== OFFLINE ROUTE RECOVERY ===")
    print(f"accuracy={offline_acc:.6f}")

    print("decision_accuracy:")
    for decision in range(8):
        decision_mask = Y == decision
        decision_accuracy = (
            pred[decision_mask] == Y[decision_mask]
        ).float().mean().item()
        print(
            f"  decision={decision} "
            f"accuracy={decision_accuracy:.6f}"
        )

    # ---------------------------------------------------------
    # Actual deterministic executor check.
    # ---------------------------------------------------------

    route_batch = task.make_fixed_set(
        split="train",
        length=8,
        size=1,
    )

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

    actual_route = []

    for t in range(8):
        out = model.adaptive_differentiable_step(
            x_t=route_batch.x[:, t, :],
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
        actual_route.append(out["action_history"])

    print("\n=== EXECUTOR ROUTE CHECK ===")

    for t, history in enumerate(actual_route):
        print(
            f"token={t} route={history}"
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

    route_ok = all(
        history == expected
        for history in actual_route
    )

    print(f"executor_route_recovered={route_ok}")

    # Offline classification accuracy is diagnostic only. The decisive
    # metric is exact route recovery through the actual executor.
    if not route_ok:
        raise RuntimeError(
            "Deterministic executor did not recover the canonical route "
            "on the initial probe."
        )

    # ---------------------------------------------------------
    # Held-out executor route recovery.
    # ---------------------------------------------------------
    #
    # A token is counted as recovered only when all 8 decisions equal:
    #
    #   ADVANCE(0), ..., ADVANCE(7)
    #
    # This tests the trained controller through the real AGTv3
    # executor rather than through an offline classifier.
    # ---------------------------------------------------------

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

    def exact_route_for_token(x_token, token_index, sequence_length, split):
        state = model.initial_persistent_state(
            batch_size=1,
            device=device,
            dtype=x_token.dtype,
        )

        C_prev = torch.zeros(
            1,
            model.dim,
            device=device,
            dtype=x_token.dtype,
        )

        # Reconstruct the persistent history up to the selected token.
        # This keeps the held-out route test faithful to autoregressive state.
        task_batch = task.make_fixed_set(
            split=split,
            length=sequence_length,
            size=1,
        )

        for j in range(token_index + 1):
            out = model.adaptive_differentiable_step(
                x_t=task_batch.x[:, j, :],
                state=state,
                C_prev=C_prev,
                budget=torch.tensor(
                    [[float(j), float(sequence_length)]],
                    device=device,
                    dtype=x_token.dtype,
                ),
                tau=1.0,
                training=False,
            )
            state = out["state"]

        return out["action_history"]

    heldout = {}

    for split, lengths in (
        ("val", range(task.VAL_MIN_LEN, task.VAL_MAX_LEN + 1)),
        ("test", range(task.TEST_MIN_LEN, task.TEST_MAX_LEN + 1)),
    ):
        total_tokens = 0
        recovered_tokens = 0

        for length in lengths:
            probe = task.make_fixed_set(
                split=split,
                length=length,
                size=1,
            )

            # Probe the final token so the persistent state includes the
            # preceding token history.
            history = exact_route_for_token(
                probe.x[:, -1, :],
                token_index=length - 1,
                sequence_length=length,
                split=split,
            )

            total_tokens += 1
            recovered_tokens += int(history == expected_route)

        heldout[split] = {
            "recovered_tokens": recovered_tokens,
            "total_tokens": total_tokens,
            "exact_route_accuracy": (
                recovered_tokens / total_tokens
            ),
        }

    print("\n=== HELD-OUT EXECUTOR ROUTE RECOVERY ===")
    print(heldout)

    torch.save(
        {
            "seed": 3401,
            "dim": dim,
            "backbone_checkpoint": str(CKPT),
            "offline_route_accuracy": offline_acc,
            "controller_state_dict": model.controller.state_dict(),
            "route": expected,
        },
        OUT,
    )

    print("\n=== RESULT ===")
    print(f"offline_route_accuracy={offline_acc:.6f}")
    print(f"saved={OUT}")
    print("route_recovery=True")


if __name__ == "__main__":
    main()
