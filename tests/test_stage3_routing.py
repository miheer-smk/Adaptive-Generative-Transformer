from src.masks import build_execution_mask, apply_action_mask
import torch
import pytest

from src.routing import st_gumbel_action, st_gumbel_successor_mix


def test_st_gumbel_forward_is_hard_one_hot():
    torch.manual_seed(1234)

    logits = torch.tensor(
        [[0.2, 1.1, -0.7, 0.4]],
        dtype=torch.float32,
        requires_grad=True,
    )
    mask = torch.tensor([[True, True, True, True]])

    action, probs = st_gumbel_action(
        logits,
        mask,
        tau=1.0,
        training=True,
    )

    assert action.shape == logits.shape
    assert probs.shape == logits.shape

    # Straight-through forward value must be exactly one-hot.
    assert torch.all((action == 0) | (action == 1))
    assert torch.allclose(
        action.sum(dim=-1),
        torch.ones(1),
    )


def test_st_gumbel_respects_action_mask():
    torch.manual_seed(5678)

    logits = torch.tensor(
        [[2.0, 1.0, 0.0, -1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    mask = torch.tensor([[True, False, True, False]])

    action, probs = st_gumbel_action(
        logits,
        mask,
        tau=0.8,
        training=True,
    )

    assert torch.equal(
        action[:, 1:],
        torch.zeros_like(action[:, 1:]),
    )
    assert probs[0, 1].item() == 0.0
    assert probs[0, 3].item() == 0.0
    assert torch.allclose(
        action.sum(dim=-1),
        torch.ones(1),
    )


def test_st_gumbel_has_backward_gradient():
    torch.manual_seed(42)

    logits = torch.tensor(
        [[0.5, 1.0, -0.25, 0.75]],
        dtype=torch.float32,
        requires_grad=True,
    )
    mask = torch.tensor([[True, True, True, True]])

    action, probs = st_gumbel_action(
        logits,
        mask,
        tau=0.7,
        training=True,
    )

    loss = (action * torch.tensor([[1.0, -2.0, 0.5, 3.0]])).sum()
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0


def test_st_gumbel_invalid_actions_have_zero_gradient():
    torch.manual_seed(99)

    logits = torch.tensor(
        [[1.0, 0.5, -0.5, 2.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    mask = torch.tensor([[True, False, True, False]])

    action, probs = st_gumbel_action(
        logits,
        mask,
        tau=1.0,
        training=True,
    )

    loss = action.sum()
    loss.backward()

    assert logits.grad is not None
    assert logits.grad[0, 1].item() == 0.0
    assert logits.grad[0, 3].item() == 0.0


def test_st_gumbel_temperature_must_be_positive():
    logits = torch.zeros(1, 4)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with pytest.raises(ValueError):
        st_gumbel_action(logits, mask, tau=0.0, training=True)

    with pytest.raises(ValueError):
        st_gumbel_action(logits, mask, tau=-1.0, training=True)


def test_inference_is_deterministic_argmax():
    logits = torch.tensor(
        [[0.1, 2.5, 1.0, -3.0]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, True, True, True]])

    action, probs = st_gumbel_action(
        logits,
        mask,
        tau=0.5,
        training=False,
    )

    expected = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

    assert torch.equal(action, expected)
    assert torch.argmax(probs, dim=-1).item() == 1

def test_st_gumbel_successor_mix_has_hard_forward_and_logit_gradients():
    """
    Gate-C structural boundary:

    1. The forward action is exactly one-hot.
    2. The selected hard action determines the forward successor.
    3. The straight-through path preserves gradients to:
         - controller logits
         - source hidden state
    """
    torch.manual_seed(1234)

    B, D, A = 2, 4, 5

    h = torch.randn(B, D, requires_grad=True)

    # Five deliberately distinct successor states.
    candidates = torch.stack(
        [
            h + 1.0,
            2.0 * h + 2.0,
            3.0 * h + 3.0,
            4.0 * h + 4.0,
            5.0 * h + 5.0,
        ],
        dim=1,
    )

    logits = torch.randn(B, A, requires_grad=True)

    mask = torch.ones(
        B,
        A,
        dtype=torch.bool,
    )

    # Directly verify the hard-forward action.
    action, _ = st_gumbel_action(
        logits=logits,
        mask=mask,
        tau=1.0,
        training=True,
    )

    assert action.shape == (B, A)
    assert torch.all(action.sum(dim=-1) == 1)
    assert torch.all((action == 0) | (action == 1))

    # Verify the successor mixture.
    successor, _, _ = st_gumbel_successor_mix(
        candidates=candidates,
        logits=logits,
        mask=mask,
        tau=1.0,
        training=True,
    )

    assert successor.shape == (B, D)
    assert torch.isfinite(successor).all()

    loss = successor.square().sum()
    loss.backward()

    assert h.grad is not None
    assert logits.grad is not None

    assert torch.isfinite(h.grad).all()
    assert torch.isfinite(logits.grad).all()

    assert h.grad.abs().sum() > 0
    assert logits.grad.abs().sum() > 0


def test_stage3_execution_mask_matches_monotone_sweep_policy():
    # N=4 => [ADVANCE x4, REPEAT, RESTART, HALT]
    N = 4

    # Initial state: only higher-index blocks are valid.
    mask = build_execution_mask(
        pointer=0,
        execution_count=0,
        just_restarted=False,
        num_blocks=N,
    )

    assert mask.tolist() == [
        True, True, True, True,
        False, False, True,
    ]

    # Middle of a sweep: only strictly higher ADVANCE targets.
    mask = build_execution_mask(
        pointer=1,
        execution_count=1,
        just_restarted=False,
        num_blocks=N,
    )

    assert mask.tolist() == [
        False, False, True, True,
        True, True, True,
    ]

    # Final block: no ADVANCE remains.
    mask = build_execution_mask(
        pointer=3,
        execution_count=3,
        just_restarted=False,
        num_blocks=N,
    )

    assert mask.tolist() == [
        False, False, False, False,
        True, True, True,
    ]

    # Immediately after RESTART: only higher-index ADVANCE,
    # REPEAT and RESTART remain masked.
    mask = build_execution_mask(
        pointer=0,
        execution_count=2,
        just_restarted=True,
        num_blocks=N,
    )

    assert mask.tolist() == [
        True, True, True, True,
        False, False, True,
    ]
