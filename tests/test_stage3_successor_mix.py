import torch

from src.caches import ExecutionCache
from src.executor import AdaptiveExecutor

from src.routing import st_gumbel_successor_mix


def test_successor_mix_is_hard_on_forward():
    torch.manual_seed(1234)

    candidates = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [10.0, 20.0, 30.0],
                [100.0, 200.0, 300.0],
                [-5.0, -6.0, -7.0],
            ]
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    logits = torch.tensor(
        [[0.1, 2.0, -0.5, 0.7]],
        dtype=torch.float32,
        requires_grad=True,
    )

    mask = torch.tensor(
        [[True, True, True, True]],
    )

    mixed, action, probs = st_gumbel_successor_mix(
        candidates,
        logits,
        mask,
        tau=1.0,
        training=True,
    )

    assert mixed.shape == (1, 3)
    assert action.shape == (1, 4)
    assert probs.shape == (1, 4)

    chosen = torch.argmax(action, dim=-1).item()

    assert torch.equal(
        mixed,
        candidates.detach()[0, chosen].unsqueeze(0),
    )


def test_successor_mix_backward_reaches_logits_and_candidates():
    torch.manual_seed(42)

    candidates = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [3.0, 0.0],
            ]
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    logits = torch.tensor(
        [[0.5, 1.0, -0.25]],
        dtype=torch.float32,
        requires_grad=True,
    )

    mask = torch.tensor(
        [[True, True, True]],
    )

    mixed, _, _ = st_gumbel_successor_mix(
        candidates,
        logits,
        mask,
        tau=0.7,
        training=True,
    )

    loss = mixed.sum()
    loss.backward()

    assert candidates.grad is not None
    assert logits.grad is not None

    assert torch.isfinite(candidates.grad).all()
    assert torch.isfinite(logits.grad).all()

    assert candidates.grad.abs().sum().item() > 0.0
    assert logits.grad.abs().sum().item() > 0.0


def test_invalid_successor_has_zero_forward_probability_and_gradient():
    torch.manual_seed(99)

    candidates = torch.randn(
        1,
        4,
        5,
        requires_grad=True,
    )

    logits = torch.randn(
        1,
        4,
        requires_grad=True,
    )

    mask = torch.tensor(
        [[True, False, True, False]],
    )

    mixed, action, probs = st_gumbel_successor_mix(
        candidates,
        logits,
        mask,
        tau=1.0,
        training=True,
    )

    assert torch.equal(
        action[:, 1],
        torch.zeros(1),
    )
    assert torch.equal(
        action[:, 3],
        torch.zeros(1),
    )

    assert probs[0, 1].item() == 0.0
    assert probs[0, 3].item() == 0.0

    mixed.sum().backward()

    assert candidates.grad is not None
    assert logits.grad is not None

    assert torch.equal(
        candidates.grad[:, 1],
        torch.zeros_like(candidates.grad[:, 1]),
    )
    assert torch.equal(
        candidates.grad[:, 3],
        torch.zeros_like(candidates.grad[:, 3]),
    )


def test_successor_mix_inference_is_deterministic():
    candidates = torch.tensor(
        [
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ]
        ],
        dtype=torch.float32,
    )

    logits = torch.tensor(
        [[0.0, 4.0, 1.0]],
        dtype=torch.float32,
    )

    mask = torch.tensor(
        [[True, True, True]],
    )

    mixed, action, probs = st_gumbel_successor_mix(
        candidates,
        logits,
        mask,
        tau=0.5,
        training=False,
    )

    assert torch.equal(
        mixed,
        candidates[0, 1].unsqueeze(0),
    )

    assert torch.equal(
        action,
        torch.tensor([[0.0, 1.0, 0.0]]),
    )

    assert torch.argmax(probs, dim=-1).item() == 1


def test_executor_generates_independent_action_successors():
    class AddBlock:
        species = "FFN"

        def __init__(self, value):
            self.value = value

        def __call__(self, x, h0=None, R=None, C_prev=None, **kwargs):
            assert C_prev is None
            return x + self.value

    class Bank:
        def __init__(self):
            self.blocks = [
                AddBlock(1.0),
                AddBlock(2.0),
                AddBlock(4.0),
            ]

        def __call__(self, index, x, **kwargs):
            return self.blocks[index](x, **kwargs)

    executor = AdaptiveExecutor(
        controller=None,
        block_bank=Bank(),
    )

    h = torch.zeros(1, 4)
    h0 = h.clone()
    memory = h.clone()
    context = ExecutionCache()

    candidates = executor.candidate_successors(
        h=h,
        h0=h0,
        memory=memory,
        context=context,
        pointer=1,
    )

    assert candidates["h"].shape == (1, 5, 4)

    # ADVANCE(0..2)
    assert torch.equal(
        candidates["h"][0, 0],
        torch.full((4,), 1.0),
    )
    assert torch.equal(
        candidates["h"][0, 1],
        torch.full((4,), 2.0),
    )
    assert torch.equal(
        candidates["h"][0, 2],
        torch.full((4,), 4.0),
    )

    # REPEAT(pointer=1)
    assert torch.equal(
        candidates["h"][0, 3],
        torch.full((4,), 2.0),
    )

    # RESTART: zero compute, hidden state preserved.
    assert torch.equal(
        candidates["h"][0, 4],
        h[0],
    )

    assert candidates["next_pointer"] == [0, 1, 2, 1, 0]

    # HALT has no successor state and is not represented here.
    assert context.gdn == {}
    assert context.kv == {}
    assert context.swa == {}



def test_executor_differentiable_route_step_hard_forward_and_gradients():
    class AddBlock:
        species = "FFN"

        def __init__(self, value):
            self.value = value

        def __call__(self, x, h0=None, R=None, C_prev=None, **kwargs):
            return x + self.value

    class Bank:
        def __init__(self):
            self.blocks = [
                AddBlock(1.0),
                AddBlock(2.0),
                AddBlock(4.0),
            ]

        def __call__(self, index, x, **kwargs):
            return self.blocks[index](x, **kwargs)

    executor = AdaptiveExecutor(
        controller=None,
        block_bank=Bank(),
    )

    torch.manual_seed(4321)

    h = torch.randn(2, 4, requires_grad=True)
    h0 = torch.zeros_like(h)
    memory = torch.zeros_like(h)
    context = ExecutionCache()

    action_logits = torch.randn(
        2, 6, requires_grad=True
    )

    result = executor.differentiable_route_step(
        h=h,
        h0=h0,
        memory=memory,
        context=context,
        pointer=1,
        action_logits=action_logits,
        tau=1.0,
        training=True,
    )

    # N=3 => N+2 = 5 successor states.
    assert result["candidates"].shape == (2, 5, 4)

    assert result["h_next"].shape == (2, 4)
    assert result["action"].shape == (2, 5)
    assert result["probs"].shape == (2, 5)

    # ST hard-forward action.
    assert torch.all(result["action"].sum(dim=-1) == 1)
    assert torch.all(
        (result["action"] == 0) |
        (result["action"] == 1)
    )

    # The forward output must equal the selected hard candidate.
    selected = result["selected_action"]

    for b in range(h.shape[0]):
        expected = result["candidates"][
            b,
            selected[b],
        ]
        assert torch.equal(
            result["h_next"][b],
            expected,
        )

    # RESTART is candidate index 4.
    assert result["next_pointer"].shape == (2,)
    assert torch.all(
        (result["next_pointer"] >= 0) &
        (result["next_pointer"] < 3)
    )

    loss = result["h_next"].square().sum()
    loss.backward()

    assert h.grad is not None
    assert action_logits.grad is not None

    assert torch.isfinite(h.grad).all()
    assert torch.isfinite(action_logits.grad).all()

    assert h.grad.abs().sum() > 0
    assert action_logits.grad.abs().sum() > 0
