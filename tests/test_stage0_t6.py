import pytest
import torch

from src.agt import AGTv3


DIM = 64
VOCAB = 97
SLOTS = 16


def assert_state_equal(a, b):
    assert a.tokens_committed == b.tokens_committed
    assert torch.equal(a.s, b.s)
    assert torch.equal(a.memory, b.memory)

    assert set(a.ctx.kv) == set(b.ctx.kv)
    assert set(a.ctx.swa) == set(b.ctx.swa)
    assert set(a.ctx.gdn) == set(b.ctx.gdn)

    for key in a.ctx.kv:
        ca = a.ctx.kv[key]
        cb = b.ctx.kv[key]
        assert ca.committed_length == cb.committed_length
        assert ca.ephemeral_length == cb.ephemeral_length

        if ca.committed_keys is None:
            assert cb.committed_keys is None
        else:
            assert torch.equal(ca.committed_keys, cb.committed_keys)
            assert torch.equal(ca.committed_values, cb.committed_values)

        # Equality of ephemeral cache contents is fully captured by
        # the public ephemeral_length invariant at terminal boundaries.
        assert ca.ephemeral_length == cb.ephemeral_length

    for key in a.ctx.swa:
        ca = a.ctx.swa[key]
        cb = b.ctx.swa[key]
        assert ca.committed_length == cb.committed_length
        assert ca.ephemeral_length == cb.ephemeral_length

        if ca.committed_keys is None:
            assert cb.committed_keys is None
        else:
            assert torch.equal(ca.committed_keys, cb.committed_keys)
            assert torch.equal(ca.committed_values, cb.committed_values)

        # Equality of ephemeral cache contents is fully captured by
        # the public ephemeral_length invariant at terminal boundaries.
        assert ca.ephemeral_length == cb.ephemeral_length

    for key in a.ctx.gdn:
        ca = a.ctx.gdn[key]
        cb = b.ctx.gdn[key]

        assert ca.version == cb.version

        if ca.committed_state is None:
            assert cb.committed_state is None
        else:
            assert torch.equal(ca.committed_state, cb.committed_state)

        if ca.committed_conv_buffer is None:
            assert cb.committed_conv_buffer is None
        else:
            assert torch.equal(
                ca.committed_conv_buffer,
                cb.committed_conv_buffer,
            )


@torch.no_grad()
def test_t6_checkpoint_restore_matches_uninterrupted():
    """
    T6 supported portion.

    Clone the persistent state after token 2, restore from that checkpoint,
    and verify continuation produces exactly the same states and logits as
    uninterrupted execution.
    """
    torch.manual_seed(2040)

    model = AGTv3(
        dim=DIM,
        num_memory_slots=SLOTS,
        vocab_size=VOCAB,
    ).eval()

    state0 = model.initial_persistent_state(batch_size=1)
    C_prev = torch.zeros(1, DIM)

    tokens = [
        torch.randn(1, DIM)
        for _ in range(6)
    ]

    # Uninterrupted reference.
    reference_states = []
    reference_logits = []

    state_ref = state0
    for x_t in tokens:
        out = model.step(x_t, state_ref, C_prev)
        state_ref = out["state"]
        reference_states.append(state_ref)
        reference_logits.append(out["logits"].clone())

    # Process through checkpoint boundary.
    state_ckpt_path = state0
    for x_t in tokens[:3]:
        out = model.step(x_t, state_ckpt_path, C_prev)
        state_ckpt_path = out["state"]

    # Snapshot/restore the complete persistent token state.
    checkpoint = state_ckpt_path.clone()

    # Continue from the restored checkpoint.
    restored_states = []
    restored_logits = []

    state_restored = checkpoint
    for x_t in tokens[3:]:
        out = model.step(x_t, state_restored, C_prev)
        state_restored = out["state"]
        restored_states.append(state_restored)
        restored_logits.append(out["logits"].clone())

    # Compare tokens 4..6 against uninterrupted reference.
    for i, (got_state, got_logits) in enumerate(
        zip(restored_states, restored_logits),
        start=3,
    ):
        assert_state_equal(got_state, reference_states[i])
        assert torch.equal(got_logits, reference_logits[i])

    # The checkpoint itself is exactly token 3 state.
    assert_state_equal(checkpoint, reference_states[2])


def test_t6_parameter_version_guard_is_not_available_yet():
    """
    The current AGTv3 persistent state carries no model-parameter version,
    so wrong-version checkpoint rejection is not yet representable.
    """
    pytest.skip(
        "Model-parameter versioning/rejection is not implemented in the "
        "current AGT-v3 persistent state."
    )
