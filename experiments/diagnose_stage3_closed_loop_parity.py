from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from data.task_a import TaskAParity
from src.agt import AGTv3


CKPT = Path(
    "experiments/runs/stage1_parity_seed3305_best.pt"
)


class ForcedCanonicalController(torch.nn.Module):
    """
    Controller stub that forces exactly:

        ADVANCE(0), ADVANCE(1), ..., ADVANCE(7)

    independent of controller inputs.

    This isolates executor semantics from controller learning.
    """

    def __init__(self, num_blocks: int):
        super().__init__()
        self.num_blocks = num_blocks
        self.decision = 0

    def reset(self):
        self.decision = 0

    def forward(
        self,
        h=None,
        memory_read=None,
        memory_mean=None,
        pointer_one_hot=None,
        budget=None,
        **kwargs,
    ):
        logits = torch.full(
            (
                h.shape[0],
                self.num_blocks + 3,
            ),
            -100.0,
            device=h.device,
            dtype=h.dtype,
        )

        if self.decision >= self.num_blocks:
            raise RuntimeError(
                f"Forced controller exhausted at decision={self.decision}"
            )

        logits[:, self.decision] = 100.0

        self.decision += 1

        return {
            "logits": logits,
            "advance": logits[:, :self.num_blocks],
            "repeat": logits[:, self.num_blocks],
            "restart": logits[:, self.num_blocks + 1],
            "halt": logits[:, self.num_blocks + 2],
            "legacy_mode": False,
        }


@torch.no_grad()
def main():
    torch.manual_seed(4242)

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

    backbone_state = {
        key: value
        for key, value in ckpt["state_dict"].items()
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

    model.to(device)
    model.eval()

    model.executor.max_execution_depth = 8

    task = TaskAParity(
        dim=dim,
        device=device,
    )

    batch = task.make_fixed_set(
        split="test",
        length=8,
        size=1,
    )

    x = batch.x
    C_prev = torch.zeros(
        1,
        dim,
        device=device,
        dtype=x.dtype,
    )

    # =========================================================
    # Reference: existing validated fixed-route step.
    # =========================================================

    state_fixed = model.initial_persistent_state(
        batch_size=1,
        device=device,
        dtype=x.dtype,
    )

    fixed = model.step(
        x_t=x[:, 0, :],
        state=state_fixed,
        C_prev=C_prev,
    )

    # =========================================================
    # Forced canonical differentiable executor.
    # =========================================================

    forced_controller = ForcedCanonicalController(
        num_blocks=8,
    ).to(device)

    model.controller = forced_controller

    forced_controller.reset()

    state_diff = model.initial_persistent_state(
        batch_size=1,
        device=device,
        dtype=x.dtype,
    )

    diff = model.adaptive_differentiable_step(
        x_t=x[:, 0, :],
        state=state_diff,
        C_prev=C_prev,
        budget=torch.tensor(
            [[0.0, 8.0]],
            device=device,
            dtype=x.dtype,
        ),
        tau=1.0,
        training=False,
    )

    print("=== STAGE-3 CLOSED-LOOP PARITY DIAGNOSTIC ===")
    print(f"device={device}")
    print("route=ADVANCE(0..7)")
    print("routing_depth=8")

    print("\n=== ROUTE ===")
    print(diff["action_history"])

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

    assert diff["action_history"] == expected_route

    print("\n=== COMPARISON ===")

    h0_diff = (
        fixed["h0"] - diff["h0"]
    ).abs().max().item()

    h_terminal_diff = (
        fixed["h_terminal"] - diff["h_terminal"]
    ).abs().max().item()

    H_diff = (
        fixed["H_t"] - diff["H_t"]
    ).abs().max().item()

    logits_diff = (
        fixed["logits"] - diff["logits"]
    ).abs().max().item()

    print(f"h0_max_abs_diff={h0_diff:.10e}")
    print(f"h_terminal_max_abs_diff={h_terminal_diff:.10e}")
    print(f"H_t_max_abs_diff={H_diff:.10e}")
    print(f"logits_max_abs_diff={logits_diff:.10e}")

    print("\n=== EXECUTION COUNTS ===")
    print(f"fixed_route_expected=8")
    print(f"differentiable_execution={diff['execution_count']}")

    print("\n=== RESULTS ===")

    parity = (
        h0_diff < 1e-6
        and h_terminal_diff < 1e-6
        and H_diff < 1e-6
        and logits_diff < 1e-6
    )

    print(f"exact_forward_parity={parity}")

    if not parity:
        raise RuntimeError(
            "Fixed-route and differentiable executor paths are not "
            "forward-equivalent."
        )

    print("closed_loop_executor_parity=True")


if __name__ == "__main__":
    main()
