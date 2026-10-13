from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.task_a import TaskAParity
from src.agt import AGTv3
from src.terminal_commit import block_context


BACKBONE_CKPT = Path(
    "experiments/runs/stage1_parity_seed3305_best.pt"
)

ROUTE_CKPT = Path(
    "experiments/runs/stage3_controller_route_imitation_v2_seed3401.pt"
)


def max_diff(a, b):
    return (a.detach() - b.detach()).abs().max().item()


def main():
    torch.manual_seed(1234)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=== STAGE-3 FIXED VS ADAPTIVE EQUIVALENCE ===")
    print(f"device={device}")

    backbone = torch.load(
        BACKBONE_CKPT,
        map_location="cpu",
    )

    route = torch.load(
        ROUTE_CKPT,
        map_location="cpu",
    )

    model = AGTv3(
        dim=int(backbone["dim"]),
        num_memory_slots=16,
        vocab_size=2,
    )

    backbone_state = {
        k: v
        for k, v in backbone["state_dict"].items()
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

    controller_state = route.get("controller_state_dict")

    if controller_state is None:
        raise RuntimeError(
            "Route checkpoint missing controller_state_dict. "
            f"keys={sorted(route.keys())}"
        )

    model.controller.load_state_dict(
        controller_state,
        strict=True,
    )

    model.to(device)
    model.eval()
    model.executor.max_execution_depth = 8

    task = TaskAParity(
        dim=model.dim,
        device=device,
    )

    batch = task.make_fixed_set(
        split="train",
        length=8,
        size=1,
    )

    state_fixed = model.initial_persistent_state(
        batch_size=1,
        device=device,
        dtype=torch.float32,
    )

    state_adaptive = state_fixed.clone()

    C_prev_fixed = torch.zeros(
        1,
        model.dim,
        device=device,
        dtype=torch.float32,
    )

    C_prev_adaptive = C_prev_fixed.clone()

    print("\n=== PER-TOKEN COMPARISON ===")

    for t in range(8):
        x_t = batch.x[:, t, :]

        # -----------------------------------------------------
        # Fixed canonical path.
        # -----------------------------------------------------
        fixed = model.step(
            x_t=x_t,
            state=state_fixed,
            C_prev=C_prev_fixed,
        )

        # -----------------------------------------------------
        # Adaptive deterministic path.
        # -----------------------------------------------------
        adaptive = model.adaptive_differentiable_step(
            x_t=x_t,
            state=state_adaptive,
            C_prev=C_prev_adaptive,
            budget=torch.tensor(
                [[float(t), 8.0]],
                device=device,
                dtype=torch.float32,
            ),
            tau=1.0,
            training=False,
        )

        print(
            f"token={t} "
            f"route={adaptive['action_history']}"
        )

        print(
            f"  h0_diff={max_diff(fixed['h0'], adaptive['h0']):.6e}"
        )

        print(
            f"  h_terminal_diff="
            f"{max_diff(fixed['h_terminal'], adaptive['h_terminal']):.6e}"
        )

        print(
            f"  H_t_diff="
            f"{max_diff(fixed['H_t'], adaptive['H_t']):.6e}"
        )

        print(
            f"  logits_diff="
            f"{max_diff(fixed['logits'], adaptive['logits']):.6e}"
        )

        print(
            f"  memory_read_diff="
            f"{max_diff(fixed['memory_read'], adaptive['memory_read']):.6e}"
        )

        state_s_diff = max_diff(
            fixed["state"].s,
            adaptive["state"].s,
        )

        state_m_diff = max_diff(
            fixed["state"].memory,
            adaptive["state"].memory,
        )

        print(
            f"  state.s_diff={state_s_diff:.6e}"
        )

        print(
            f"  state.memory_diff={state_m_diff:.6e}"
        )

        # -----------------------------------------------------
        # Direct executor-vs-manual block-sweep comparison.
        # -----------------------------------------------------
        if t == 0:
            work_state = state_fixed.clone()

            memory_read, _ = model.memory.read(
                work_state.s,
                work_state.memory,
            )

            h0 = model.prelude(
                x_t=x_t,
                H_prev=work_state.s,
                memory_read=memory_read,
                C_prev=C_prev_fixed,
            )

            h_manual = h0.unsqueeze(1)
            h0_3 = h0.unsqueeze(1)
            R_c = memory_read.unsqueeze(1)

            for i, blk in enumerate(model.blocks.blocks):
                cache = block_context(
                    work_state.ctx,
                    i,
                    blk.species,
                )

                if blk.species == "FFN":
                    out = blk(
                        h_manual,
                        h0=h0_3,
                        R=R_c,
                        C_prev=None,
                    )
                else:
                    out = blk(
                        h_manual,
                        h0=h0_3,
                        R=R_c,
                        C_prev=cache,
                    )

                if blk.species == "GDN":
                    h_manual, _ = out
                else:
                    h_manual = out

            executor_result = model.executor.execute_differentiable(
                h0=h0,
                memory=memory_read,
                C_prev=C_prev_fixed,
                context=work_state.ctx,
                memory_mean=work_state.memory.mean(dim=1),
                budget=torch.tensor(
                    [[0.0, 8.0]],
                    device=device,
                    dtype=torch.float32,
                ),
                tau=1.0,
                training=False,
            )

            print("\n=== TOKEN-0 EXECUTOR ISOLATION ===")
            print(
                "manual_vs_executor_h_terminal_diff="
                f"{max_diff(h_manual.squeeze(1), executor_result.h_terminal):.6e}"
            )

            print(
                "executor_route="
                f"{executor_result.action_history}"
            )

        state_fixed = fixed["state"]
        state_adaptive = adaptive["state"]

    print("\n=== FINAL ===")
    print("comparison_complete=True")


if __name__ == "__main__":
    main()
