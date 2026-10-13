import torch

from src.executor import AdaptiveExecutor


class AddBlock:
    def __init__(self, amount=1.0):
        self.amount = float(amount)

    def __call__(self, x, **kwargs):
        return x + self.amount


class OneBlockBank:
    def __init__(self):
        self.blocks = [AddBlock(1.0)]

    def __call__(self, index, x, **kwargs):
        return self.blocks[index](x, **kwargs)


class AlwaysHaltController:
    def __call__(self, h):
        batch = h.shape[0]
        device = h.device
        dtype = h.dtype

        return {
            "advance": torch.full(
                (batch, 1),
                -10.0,
                device=device,
                dtype=dtype,
            ),
            "repeat": torch.full(
                (batch,),
                -10.0,
                device=device,
                dtype=dtype,
            ),
            "restart": torch.full(
                (batch,),
                -10.0,
                device=device,
                dtype=dtype,
            ),
            "halt": torch.full(
                (batch,),
                10.0,
                device=device,
                dtype=dtype,
            ),
        }


class HaltAfterController:
    """
    Execute exactly `num_steps` block decisions, then HALT.

    With one block:
        decision 1 -> ADVANCE(0)
        decisions 2..N -> REPEAT
        decision N+1 -> HALT
    """

    def __init__(self, num_steps):
        self.num_steps = int(num_steps)
        self.calls = 0

    def __call__(self, h):
        self.calls += 1

        batch = h.shape[0]
        device = h.device
        dtype = h.dtype

        advance = torch.full(
            (batch, 1),
            -10.0,
            device=device,
            dtype=dtype,
        )
        repeat = torch.full(
            (batch,),
            -10.0,
            device=device,
            dtype=dtype,
        )
        restart = torch.full(
            (batch,),
            -10.0,
            device=device,
            dtype=dtype,
        )
        halt = torch.full(
            (batch,),
            -10.0,
            device=device,
            dtype=dtype,
        )

        if self.calls == 1:
            advance[:, 0] = 10.0
        elif self.calls <= self.num_steps:
            repeat[:] = 10.0
        else:
            halt[:] = 10.0

        return {
            "advance": advance,
            "repeat": repeat,
            "restart": restart,
            "halt": halt,
        }


def run(controller, max_depth=8):
    executor = AdaptiveExecutor(
        block_bank=OneBlockBank(),
        controller=controller,
        convergence=None,
        max_execution_depth=max_depth,
    )

    h0 = torch.zeros(1, 4)
    memory = torch.zeros_like(h0)
    C_prev = torch.zeros_like(h0)

    return executor.execute(
        h0=h0,
        memory=memory,
        C_prev=C_prev,
    )


def test_stage4_always_halt_zero_compute():
    result = run(AlwaysHaltController())

    assert result.halted
    assert not result.safety_cap_hit
    assert result.execution_count == 0
    assert result.action_history == [
        ("HALT", None),
    ]

    assert torch.equal(
        result.h_terminal,
        torch.zeros(1, 4),
    )


def test_stage4_halt_after_three_execution_decisions():
    result = run(
        HaltAfterController(num_steps=3),
        max_depth=8,
    )

    assert result.halted
    assert not result.safety_cap_hit
    assert result.execution_count == 3

    assert result.action_history == [
        ("ADVANCE", 0),
        ("REPEAT", 0),
        ("REPEAT", 0),
        ("HALT", None),
    ]

    assert torch.equal(
        result.h_terminal,
        torch.full((1, 4), 3.0),
    )


def test_stage4_no_halt_hits_fixed_safety_cap():
    result = run(
        HaltAfterController(num_steps=100),
        max_depth=3,
    )

    assert not result.halted
    assert result.safety_cap_hit
    assert result.execution_count == 3

    assert result.action_history == [
        ("ADVANCE", 0),
        ("REPEAT", 0),
        ("REPEAT", 0),
    ]

    assert torch.equal(
        result.h_terminal,
        torch.full((1, 4), 3.0),
    )
