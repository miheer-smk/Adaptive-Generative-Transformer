import torch

from src.agt import AGTv3


DIM = 64
VOCAB = 97
SLOTS = 16
N_TOKENS = 256


@torch.no_grad()
def test_t2_cache_cardinality_independent_of_execution_depth():
    """
    T2 — Cache cardinality.

    After token t:
      GQA cache = exactly t committed entries.
      SWA cache = min(t, W-1) committed entries.

    Cardinality is determined by token position, not by how many
    within-token executions occurred.
    """
    torch.manual_seed(2027)

    model = AGTv3(
        dim=DIM,
        num_memory_slots=SLOTS,
        vocab_size=VOCAB,
    ).eval()

    state = model.initial_persistent_state(batch_size=1)
    C_prev = torch.zeros(1, DIM)

    # Verify the bank itself has the expected block species.
    species = [block.species for block in model.blocks.blocks]
    assert species == [
        "SWA",
        "SWA",
        "GQA",
        "GQA",
        "GDN",
        "GDN",
        "FFN",
        "FFN",
    ]

    swa_indices = [i for i, s in enumerate(species) if s == "SWA"]
    gqa_indices = [i for i, s in enumerate(species) if s == "GQA"]

    assert len(swa_indices) == 2
    assert len(gqa_indices) == 2

    for t in range(1, N_TOKENS + 1):
        x_t = torch.randn(1, DIM)

        out = model.step(x_t, state, C_prev)
        state = out["state"]

        # Global GQA: exactly one committed token per processed token.
        for block_id in gqa_indices:
            cache = state.ctx.kv[block_id]
            assert cache.ephemeral_length == 0
            assert cache.committed_length == t, (
                f"GQA block {block_id}: expected {t}, "
                f"got {cache.committed_length} at token {t}"
            )

        # SWA: capacity must be min(t, W-1), per Stage-0 specification.
        for block_id in swa_indices:
            cache = state.ctx.swa[block_id]
            assert cache.ephemeral_length == 0

            expected = min(t, cache.window_size - 1)

            assert cache.committed_length == expected, (
                f"SWA block {block_id}: expected {expected}, "
                f"got {cache.committed_length} at token {t}; "
                f"window_size={cache.window_size}"
            )
