import torch

from src.executor import AdaptiveExecutor


DIM = 64
NUM_BLOCKS = 8


class ToyBank:
    def __init__(self, num_blocks):
        self.blocks = [object()] * num_blocks

    def __call__(self, block_index, h):
        # Make ADVANCE visibly change h so RESTART preservation is testable.
        if block_index == 0:
            return h + 1.0
        return h


class SequenceController:
    """
    Force:
        call 1 -> ADVANCE(0)
        call 2 -> RESTART
        call 3 -> HALT

    Also records the controller-visible h at every decision.
    """

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.calls = 0
        self.seen_h = []

    def __call__(self, h):
        self.seen_h.append(h.clone())

        batch = h.shape[0]
        device = h.device

        advance = torch.full(
            (batch, self.num_blocks),
            -100.0,
            device=device,
        )
        repeat = torch.full(
            (batch,),
            -100.0,
            device=device,
        )
        restart = torch.full(
            (batch,),
            -100.0,
            device=device,
        )
        halt = torch.full(
            (batch,),
            -100.0,
            device=device,
        )

        if self.calls == 0:
            advance[:, 0] = 100.0
        elif self.calls == 1:
            restart[:] = 100.0
        else:
            halt[:] = 100.0

        self.calls += 1

        return {
            "advance": advance,
            "repeat": repeat,
            "restart": restart,
            "halt": halt,
        }


@torch.no_grad()
def test_t11_restart_preserves_h():
    """
    T11 — RESTART preservation.

    The live h must be bitwise identical immediately before
    and immediately after the RESTART decision.
    """
    torch.manual_seed(2110)

    bank = ToyBank(NUM_BLOCKS)
    controller = SequenceController(NUM_BLOCKS)

    executor = AdaptiveExecutor(
        block_bank=bank,
        controller=controller,
        convergence=None,
        max_execution_depth=16,
    )

    h0 = torch.randn(1, DIM)
    memory = torch.zeros(1, DIM)
    C_prev = torch.zeros(1, DIM)

    result = executor.execute(
        h0=h0,
        memory=memory,
        C_prev=C_prev,
    )

    assert result.action_history == [
        ("ADVANCE", 0),
        ("RESTART", None),
        ("HALT", None),
    ]

    # seen_h[1] = immediately before RESTART decision
    # seen_h[2] = immediately after RESTART action
    h_before_restart = controller.seen_h[1]
    h_after_restart = controller.seen_h[2]

    assert torch.equal(
        h_before_restart,
        h_after_restart,
    ), (
        "T11 failed: h changed across RESTART decision.\n"
        f"before norm={h_before_restart.norm().item():.10e}\n"
        f"after  norm={h_after_restart.norm().item():.10e}"
    )
