import torch

from data.task_a import TaskAParity


def test_task_a_shapes_and_targets():
    task = TaskAParity(dim=32)

    for split, length in [
        ("train", 4),
        ("train", 16),
        ("val", 8),
        ("val", 16),
        ("test", 17),
        ("test", 32),
    ]:
        batch = task.make_fixed_set(split, length, size=64)

        assert batch.x.shape == (64, length, 32)
        assert batch.target.shape == (64,)
        assert batch.length == length
        assert torch.all((batch.target == 0) | (batch.target == 1))


def test_task_a_difficulty_is_length():
    task = TaskAParity(dim=8)

    for length in range(4, 33):
        assert task.difficulty(length) == length


def test_task_a_deterministic_feature_table():
    a = TaskAParity(dim=16)
    b = TaskAParity(dim=16)

    assert torch.equal(a.token_features, b.token_features)


def test_task_a_invalid_split_length_rejected():
    task = TaskAParity(dim=8)

    for split, bad_length in [
        ("train", 3),
        ("train", 17),
        ("val", 3),
        ("val", 17),
        ("test", 16),
        ("test", 33),
    ]:
        try:
            task.make_batch(4, bad_length, split=split)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"{split} accepted invalid length {bad_length}"
            )
