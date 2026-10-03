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


def test_stage3_differentiable_executor_loop_backpropagates_to_controller():
    import torch

    from src.controller import Controller
    from src.executor import AdaptiveExecutor

    torch.manual_seed(7)

    num_blocks = 4
    dim = 64
    batch_size = 1

    class AddBlock:
        def __init__(self, amount):
            self.amount = float(amount)

        def __call__(self, x, **kwargs):
            return x + self.amount

    class FourBlockBank:
        def __init__(self):
            self.blocks = [
                AddBlock(1.0),
                AddBlock(2.0),
                AddBlock(3.0),
                AddBlock(4.0),
            ]

        def __call__(self, index, x, **kwargs):
            return self.blocks[index](x, **kwargs)

    controller = Controller(
        dim=dim,
        num_blocks=num_blocks,
        memory_dim=256,
        hidden_dim=256,
    )

    executor = AdaptiveExecutor(
        block_bank=FourBlockBank(),
        controller=controller,
        max_execution_depth=3,
        convergence=None,
    )

    h0 = torch.randn(batch_size, dim)
    memory_read = torch.zeros_like(h0)
    memory_mean = torch.zeros(batch_size, 256)
    budget = torch.tensor([[0.0, 1.0]])

    result = executor.execute_differentiable(
        h0=h0,
        memory=memory_read,
        C_prev=None,
        context=None,
        memory_mean=memory_mean,
        budget=budget,
        tau=1.0,
        training=True,
    )

    loss = result.h_terminal.sum()
    loss.backward()

    assert result.h_terminal.shape == (batch_size, dim)
    assert result.execution_count == 3

    controller_grads = [
        parameter.grad
        for parameter in controller.parameters()
        if parameter.requires_grad
    ]

    assert any(
        grad is not None
        and torch.isfinite(grad).all()
        and grad.abs().sum() > 0
        for grad in controller_grads
    )

def test_stage3_real_agt_differentiable_step_backpropagates():
    import torch

    from src.agt import AGTv3

    torch.manual_seed(17)

    model = AGTv3(
        dim=64,
        num_memory_slots=16,
        vocab_size=2,
    )

    state = model.initial_persistent_state(
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
    )

    x_t = torch.randn(1, 64)
    C_prev = torch.zeros(1, 64)

    result = model.adaptive_differentiable_step(
        x_t=x_t,
        state=state,
        C_prev=C_prev,
        budget=torch.tensor([[0.0, 1.0]]),
        tau=1.0,
        training=True,
    )

    loss = result["logits"].sum()
    loss.backward()

    assert result["logits"].shape == (1, 2)
    assert result["execution_count"] == model.executor.max_execution_depth

    controller_grads = [
        parameter.grad
        for parameter in model.controller.parameters()
        if parameter.requires_grad
    ]

    assert any(
        grad is not None
        and torch.isfinite(grad).all()
        and grad.abs().sum() > 0
        for grad in controller_grads
    )
