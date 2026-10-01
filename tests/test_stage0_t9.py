import torch
import torch.nn as nn

from src.agt import AGTv3


DIM = 64
VOCAB = 97
SLOTS = 16


class ForcedController(nn.Module):
    """
    Force:
        step 1 -> RESTART
        step 2 -> RESTART
        step 3 -> HALT

    This directly probes the T9 rule that RESTART cannot immediately
    follow another RESTART.
    """

    def __init__(self, num_blocks):
        super().__init__()
        self.num_blocks = num_blocks
        self.calls = 0

    def forward(self, state):
        device = state.device
        batch = state.shape[0]

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

        if self.calls < 2:
            restart.fill_(100.0)
        else:
            halt.fill_(100.0)

        self.calls += 1

        return {
            "advance": advance,
            "repeat": repeat,
            "restart": restart,
            "halt": halt,
        }


@torch.no_grad()
def test_t9_restart_cannot_immediately_follow_restart():
    """
    T9 — explicit RESTART legality rule.

    A RESTART must not be followed immediately by another RESTART.
    """
    torch.manual_seed(2090)

    model = AGTv3(
        dim=DIM,
        num_memory_slots=SLOTS,
        vocab_size=VOCAB,
    ).eval()

    controller = ForcedController(
        num_blocks=len(model.blocks.blocks),
    )

    # T9 is a routing-legality test; isolate it from the currently
    # incompatible convergence-gate call signature.
    model.executor.controller = controller
    model.executor.convergence = None

    h0 = torch.randn(1, DIM)
    memory = model.memory.initial_state(batch_size=1)
    C_prev = torch.zeros(1, DIM)

    result = model.executor.execute(
        h0=h0,
        memory=memory,
        C_prev=C_prev,
    )

    history = result.action_history

    for prev_action, current_action in zip(history, history[1:]):
        assert not (
            prev_action[0] == "RESTART"
            and current_action[0] == "RESTART"
        ), f"Illegal consecutive RESTART actions: {history}"
