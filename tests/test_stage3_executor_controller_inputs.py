import torch

from src.caches import ExecutionCache
from src.executor import AdaptiveExecutor


class OneBlock:
    species = "FFN"

    def __call__(self, x, h0=None, R=None, C_prev=None, **kwargs):
        assert C_prev is None
        return x + 1.0


class OneBlockBank:
    def __init__(self):
        self.blocks = [OneBlock()]

    def __call__(self, index, x, **kwargs):
        return self.blocks[index](x, **kwargs)


class RecordingController:
    """
    First decision: ADVANCE(0)
    Second decision: HALT

    The controller records the complete Stage-3 input signature.
    """

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        *,
        h,
        memory_read,
        memory_mean,
        pointer_one_hot,
        budget,
    ):
        self.calls.append(
            {
                "h": h.detach().clone(),
                "memory_read": memory_read.detach().clone(),
                "memory_mean": memory_mean.detach().clone(),
                "pointer_one_hot": pointer_one_hot.detach().clone(),
                "budget": budget.detach().clone(),
            }
        )

        device = h.device
        dtype = h.dtype

        # Actions for N=1:
        # 0 = ADVANCE(0)
        # 1 = REPEAT
        # 2 = RESTART
        # 3 = HALT
        logits = torch.full(
            (h.shape[0], 4),
            -100.0,
            device=device,
            dtype=dtype,
        )

        if len(self.calls) == 1:
            logits[:, 0] = 10.0
        else:
            logits[:, 3] = 10.0

        return {
            "logits": logits,
            "advance": logits[:, :1],
            "repeat": logits[:, 1],
            "restart": logits[:, 2],
            "halt": logits[:, 3],
        }


def test_executor_supplies_full_stage3_controller_state():
    controller = RecordingController()

    executor = AdaptiveExecutor(
        block_bank=OneBlockBank(),
        controller=controller,
        convergence=None,
        max_execution_depth=4,
    )

    h0 = torch.zeros(1, 64)
    memory_read = torch.ones(1, 64)
    memory_mean = torch.full((1, 256), 3.0)
    budget = torch.tensor([[0.25, 1.0]])

    result = executor.execute(
        h0=h0,
        memory=memory_read,
        C_prev=torch.zeros(1, 64),
        context=ExecutionCache(),
        memory_mean=memory_mean,
        budget=budget,
    )

    assert result.halted
    assert result.execution_count == 1

    assert len(controller.calls) == 2

    first = controller.calls[0]
    second = controller.calls[1]

    assert first["h"].shape == (1, 64)
    assert first["memory_read"].shape == (1, 64)
    assert first["memory_mean"].shape == (1, 256)
    assert first["pointer_one_hot"].shape == (1, 2)
    assert first["budget"].shape == (1, 2)

    assert torch.equal(
        first["memory_read"],
        memory_read,
    )

    assert torch.equal(
        first["memory_mean"],
        memory_mean,
    )

    assert torch.equal(
        first["pointer_one_hot"],
        torch.tensor([[1.0, 0.0]]),
    )

    assert torch.equal(
        second["pointer_one_hot"],
        torch.tensor([[1.0, 0.0]]),
    )

    # The one-block case keeps the pointer at block 0 after ADVANCE(0).
    # The important contract is that a valid one-hot pointer is supplied
    # on every controller decision.
    assert torch.equal(
        first["budget"],
        budget,
    )

    assert torch.equal(
        second["budget"],
        budget,
    )
