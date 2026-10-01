import torch

from src.agt import AGTv3


DIM = 64
VOCAB = 97
SLOTS = 16


def assert_tensor_close(actual, expected):
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

def assert_cache_row_equal(batch_cache, single_cache, row):
    assert batch_cache.version == single_cache.version

    if batch_cache.committed_keys is None:
        assert single_cache.committed_keys is None
    else:
        assert_tensor_close(
            batch_cache.committed_keys[row:row + 1],
            single_cache.committed_keys,
        )
        assert_tensor_close(
            batch_cache.committed_values[row:row + 1],
            single_cache.committed_values,
        )

    assert batch_cache.ephemeral_length == single_cache.ephemeral_length == 0


def assert_state_row_equal(batch_state, single_state, row):
    assert batch_state.tokens_committed == single_state.tokens_committed

    assert_tensor_close(
        batch_state.s[row:row + 1],
        single_state.s,
    )

    assert_tensor_close(
        batch_state.memory[row:row + 1],
        single_state.memory,
    )

    assert set(batch_state.ctx.kv) == set(single_state.ctx.kv)
    assert set(batch_state.ctx.swa) == set(single_state.ctx.swa)
    assert set(batch_state.ctx.gdn) == set(single_state.ctx.gdn)

    for key in batch_state.ctx.kv:
        assert_cache_row_equal(
            batch_state.ctx.kv[key],
            single_state.ctx.kv[key],
            row,
        )

    for key in batch_state.ctx.swa:
        assert_cache_row_equal(
            batch_state.ctx.swa[key],
            single_state.ctx.swa[key],
            row,
        )

    for key in batch_state.ctx.gdn:
        batch_cache = batch_state.ctx.gdn[key]
        single_cache = single_state.ctx.gdn[key]

        assert batch_cache.version == single_cache.version

        if batch_cache.committed_state is None:
            assert single_cache.committed_state is None
        else:
            assert_tensor_close(
                batch_cache.committed_state[row:row + 1],
                single_cache.committed_state,
            )

        if batch_cache.committed_conv_buffer is None:
            assert single_cache.committed_conv_buffer is None
        else:
            assert_tensor_close(
                batch_cache.committed_conv_buffer[row:row + 1],
                single_cache.committed_conv_buffer,
            )


@torch.no_grad()
def test_t10_batch_composition_invariance():
    """
    T10 — Batch invariance.

    Processing samples A and B independently must produce the same
    persistent states and logits as processing A and B together in one
    batch.
    """
    torch.manual_seed(2100)

    model = AGTv3(
        dim=DIM,
        num_memory_slots=SLOTS,
        vocab_size=VOCAB,
    ).eval()

    C_single = torch.zeros(1, DIM)
    C_batch = torch.zeros(2, DIM)

    x_a = torch.randn(1, DIM)
    x_b = torch.randn(1, DIM)

    # ---------------------------------------------------------
    # Independent executions.
    # ---------------------------------------------------------
    state_a = model.initial_persistent_state(batch_size=1)
    out_a = model.step(x_a, state_a, C_single)

    state_b = model.initial_persistent_state(batch_size=1)
    out_b = model.step(x_b, state_b, C_single)

    # ---------------------------------------------------------
    # Same samples together in one batch.
    # ---------------------------------------------------------
    state_batch = model.initial_persistent_state(batch_size=2)
    x_batch = torch.cat([x_a, x_b], dim=0)

    out_batch = model.step(x_batch, state_batch, C_batch)

    # ---------------------------------------------------------
    # Persistent state invariance.
    # ---------------------------------------------------------
    assert_state_row_equal(
        out_batch["state"],
        out_a["state"],
        row=0,
    )

    assert_state_row_equal(
        out_batch["state"],
        out_b["state"],
        row=1,
    )

    # ---------------------------------------------------------
    # Output invariance.
    # ---------------------------------------------------------
    assert_tensor_close(
        out_batch["H_t"][0:1],
        out_a["H_t"],
    )

    assert_tensor_close(
        out_batch["H_t"][1:2],
        out_b["H_t"],
    )

    assert_tensor_close(
        out_batch["logits"][0:1],
        out_a["logits"],
    )

    assert_tensor_close(
        out_batch["logits"][1:2],
        out_b["logits"],
    )
