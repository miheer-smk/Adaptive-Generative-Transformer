import torch

from src.executor import AdaptiveExecutor


class CountingAddBlock:
    def __init__(self, amount):
        self.amount = float(amount)
        self.calls = 0

    def __call__(self, x, **kwargs):
        self.calls += 1
        return x + self.amount


class CountingBank:
    def __init__(self):
        self.blocks = [
            CountingAddBlock(1.0),
            CountingAddBlock(2.0),
            CountingAddBlock(3.0),
            CountingAddBlock(4.0),
        ]

    def __call__(self, index, x, **kwargs):
        return self.blocks[index](x, **kwargs)


class HaltController:
    def __call__(
        self,
        *,
        h,
        memory_read,
        memory_mean,
        pointer_one_hot,
        budget,
    ):
        # N=4 => actions:
        # 0..3 ADVANCE, 4 REPEAT, 5 RESTART, 6 HALT
        logits = torch.zeros(
            h.shape[0],
            7,
            device=h.device,
            dtype=h.dtype,
        )
        logits[:, 6] = 100.0
        return {"logits": logits}


class AdvanceThenHaltController:
    def __init__(self):
        self.calls = 0

    def __call__(
        self,
        *,
        h,
        memory_read,
        memory_mean,
        pointer_one_hot,
        budget,
    ):
        logits = torch.zeros(
            h.shape[0],
            7,
            device=h.device,
            dtype=h.dtype,
        )

        if self.calls == 0:
            logits[:, 0] = 100.0       # ADVANCE(0)
        else:
            logits[:, 6] = 100.0      # HALT

        self.calls += 1
        return {"logits": logits}


def test_stage4_halt_terminates_before_any_block_execution():
    bank = CountingBank()
    controller = HaltController()

    executor = AdaptiveExecutor(
        block_bank=bank,
        controller=controller,
        convergence=None,
        max_execution_depth=8,
    )

    h0 = torch.randn(1, 4)
    memory = torch.zeros(1, 4)

    result = executor.execute_adaptive(
        h0=h0,
        memory=memory,
        context=None,
        memory_mean=torch.zeros(1, 256),
        budget=torch.tensor([[0.0, 8.0]]),
        training=False,
    )

    assert result.halted is True
    assert result.safety_cap_hit is False
    assert result.execution_count == 0
    assert result.action_history == [
        ("HALT", None)
    ]
    assert torch.equal(result.h_terminal, h0)

    assert all(
        block.calls == 0
        for block in bank.blocks
    )


def test_stage4_halt_after_execution_preserves_current_live_state():
    bank = CountingBank()
    controller = AdvanceThenHaltController()

    executor = AdaptiveExecutor(
        block_bank=bank,
        controller=controller,
        convergence=None,
        max_execution_depth=8,
    )

    h0 = torch.zeros(1, 4)
    memory = torch.zeros(1, 4)

    result = executor.execute_adaptive(
        h0=h0,
        memory=memory,
        context=None,
        memory_mean=torch.zeros(1, 256),
        budget=torch.tensor([[0.0, 8.0]]),
        training=False,
    )

    assert result.halted is True
    assert result.safety_cap_hit is False
    assert result.execution_count == 1

    assert result.action_history == [
        ("ADVANCE", 0),
        ("HALT", None),
    ]

    # Block 0 was executed once; HALT does not execute a successor.
    assert bank.blocks[0].calls == 1
    assert all(
        block.calls == 0
        for block in bank.blocks[1:]
    )

    assert torch.equal(
        result.h_terminal,
        torch.ones(1, 4),
    )


