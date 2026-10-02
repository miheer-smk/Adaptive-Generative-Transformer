from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TaskABatch:
    x: torch.Tensor
    target: torch.Tensor
    length: int


class TaskAParity:
    """
    AGT-v3 Stage-1 Task A: binary sequence parity.

    Difficulty:
        D(x) = sequence length L.

    The two token representations are frozen and deterministic.
    """

    TASK_SEED = 3001
    FEATURE_SEED = 3002

    TRAIN_MIN_LEN = 4
    TRAIN_MAX_LEN = 16
    VAL_MIN_LEN = 4
    VAL_MAX_LEN = 16
    TEST_MIN_LEN = 17
    TEST_MAX_LEN = 32

    def __init__(self, dim: int, device: str | torch.device = "cpu") -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")

        self.dim = dim
        self.device = torch.device(device)

        g = torch.Generator(device="cpu").manual_seed(self.FEATURE_SEED)
        self.token_features = torch.randn(
            2,
            dim,
            generator=g,
            dtype=torch.float32,
        ).to(self.device)

        self._train_generator = torch.Generator(device="cpu").manual_seed(
            self.TASK_SEED
        )
        self._eval_generators = {
            "val": torch.Generator(device="cpu").manual_seed(self.TASK_SEED + 1),
            "test": torch.Generator(device="cpu").manual_seed(self.TASK_SEED + 2),
        }

    @staticmethod
    def difficulty(length: int) -> int:
        if length < 1:
            raise ValueError("length must be positive")
        return length

    @staticmethod
    def target_from_bits(bits: torch.Tensor) -> torch.Tensor:
        if bits.ndim != 2:
            raise ValueError("bits must have shape [B, L]")
        if not torch.all((bits == 0) | (bits == 1)):
            raise ValueError("bits must contain only 0/1")
        return bits.sum(dim=1).remainder(2).long()

    def make_batch(
        self,
        batch_size: int,
        length: int,
        split: str = "train",
    ) -> TaskABatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if length < 1:
            raise ValueError("length must be positive")

        ranges = {
            "train": (self.TRAIN_MIN_LEN, self.TRAIN_MAX_LEN),
            "val": (self.VAL_MIN_LEN, self.VAL_MAX_LEN),
            "test": (self.TEST_MIN_LEN, self.TEST_MAX_LEN),
        }
        lo, hi = ranges[split]

        if not (lo <= length <= hi):
            raise ValueError(
                f"length={length} is outside the {split} range [{lo}, {hi}]"
            )

        if split == "train":
            g = self._train_generator
        else:
            g = self._eval_generators[split]

        bits = torch.randint(
            low=0,
            high=2,
            size=(batch_size, length),
            generator=g,
            device="cpu",
            dtype=torch.long,
        ).to(self.device)

        x = self.token_features[bits]
        target = self.target_from_bits(bits)

        return TaskABatch(
            x=x,
            target=target,
            length=length,
        )

    def make_fixed_set(
        self,
        split: str,
        length: int,
        size: int,
    ) -> TaskABatch:
        if size <= 0:
            raise ValueError("size must be positive")

        return self.make_batch(
            batch_size=size,
            length=length,
            split=split,
        )


def validate_task_a() -> None:
    task = TaskAParity(dim=16)

    train = task.make_fixed_set("train", length=8, size=128)
    val = task.make_fixed_set("val", length=12, size=128)
    test = task.make_fixed_set("test", length=20, size=128)

    assert train.x.shape == (128, 8, 16)
    assert val.x.shape == (128, 12, 16)
    assert test.x.shape == (128, 20, 16)

    assert train.target.shape == (128,)
    assert val.target.shape == (128,)
    assert test.target.shape == (128,)

    assert set(train.target.tolist()) <= {0, 1}
    assert set(val.target.tolist()) <= {0, 1}
    assert set(test.target.tolist()) <= {0, 1}

    # Reconstruct bits from the frozen 2-row representation.
    for batch in (train, val, test):
        distances = torch.cdist(
            batch.x.reshape(-1, task.dim),
            task.token_features,
        )
        recovered = distances.argmin(dim=1).reshape(batch.x.shape[:2])
        expected = task.target_from_bits(recovered)
        assert torch.equal(expected, batch.target)


if __name__ == "__main__":
    validate_task_a()
    print("TASK A DATA SANITY: PASS")
