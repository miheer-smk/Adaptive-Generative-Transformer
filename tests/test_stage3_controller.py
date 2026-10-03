import torch

from src.controller import Controller


def test_d64_controller_contract():
    controller = Controller(
        dim=64,
        num_blocks=8,
    )

    h = torch.randn(1, 64)
    memory_read = torch.randn(1, 64)
    memory_mean = torch.randn(1, 256)

    pointer_one_hot = torch.zeros(1, 9)
    pointer_one_hot[0, 0] = 1.0

    budget = torch.tensor([[0.25, 1.0]])

    out = controller(
        h=h,
        memory_read=memory_read,
        memory_mean=memory_mean,
        pointer_one_hot=pointer_one_hot,
        budget=budget,
    )

    assert isinstance(out, dict)

    assert out["logits"].shape == (1, 11)
    assert out["advance"].shape == (1, 8)
    assert out["repeat"].shape == (1,)
    assert out["restart"].shape == (1,)
    assert out["halt"].shape == (1,)


def test_controller_uses_all_state_inputs():
    torch.manual_seed(123)

    controller = Controller(
        dim=64,
        num_blocks=8,
    ).eval()

    h = torch.randn(1, 64)
    memory_read = torch.randn(1, 64)
    memory_mean = torch.randn(1, 256)
    pointer = torch.zeros(1, 9)
    pointer[0, 3] = 1.0
    budget = torch.tensor([[0.5, 1.0]])

    base = controller(
        h=h,
        memory_read=memory_read,
        memory_mean=memory_mean,
        pointer_one_hot=pointer,
        budget=budget,
    )["logits"]

    changed_memory = controller(
        h=h,
        memory_read=memory_read + 1.0,
        memory_mean=memory_mean,
        pointer_one_hot=pointer,
        budget=budget,
    )["logits"]

    changed_pointer = controller(
        h=h,
        memory_read=memory_read,
        memory_mean=memory_mean,
        pointer_one_hot=torch.roll(pointer, shifts=1, dims=-1),
        budget=budget,
    )["logits"]

    changed_budget = controller(
        h=h,
        memory_read=memory_read,
        memory_mean=memory_mean,
        pointer_one_hot=pointer,
        budget=torch.tensor([[0.9, 1.0]]),
    )["logits"]

    assert not torch.allclose(base, changed_memory)
    assert not torch.allclose(base, changed_pointer)
    assert not torch.allclose(base, changed_budget)


def test_controller_has_395_input_projection_for_d64():
    controller = Controller(
        dim=64,
        num_blocks=8,
    )

    # The D64 pilot contract is 395 -> 256 -> 11.
    linear_layers = [
        module
        for module in controller.modules()
        if isinstance(module, torch.nn.Linear)
    ]

    assert linear_layers[0].in_features == 395
    assert linear_layers[0].out_features == 256
    assert linear_layers[-1].out_features == 11
