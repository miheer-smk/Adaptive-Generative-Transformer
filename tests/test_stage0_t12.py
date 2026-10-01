import torch

from src.agt import AGTv3
from src.terminal_commit import block_context, PRELUDE_KEY


DIM = 64
VOCAB = 97
SLOTS = 16


def assert_payload_close(actual, expected, label):
    assert len(actual) == len(expected), f"{label}: payload length mismatch"
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert torch.allclose(a, e, atol=1e-6, rtol=1e-5), (
            f"{label}[{i}] does not match terminal-state projection"
        )


@torch.no_grad()
def test_t12_semantic_coherence():
    """
    T12 — semantic coherence.

    Every committed entry must equal the corresponding block's
    projection of the terminal-state input, using the committed
    pre-token context for its position/state semantics.
    """
    torch.manual_seed(2120)

    model = AGTv3(
        dim=DIM,
        num_memory_slots=SLOTS,
        vocab_size=VOCAB,
    ).eval()

    state = model.initial_persistent_state(batch_size=1)
    pre_state = state.clone()

    x_t = torch.randn(1, DIM)
    C_prev = torch.zeros(1, DIM)

    out = model.step(x_t, state, C_prev)
    plan = out["commit_plan"]

    h_terminal = out["h_terminal"].unsqueeze(1)
    h0 = out["h0"].unsqueeze(1)
    R_c = out["memory_read"].unsqueeze(1)

    entry_index = 0

    for i, blk in enumerate(model.blocks.blocks):
        cache = block_context(
            pre_state.ctx,
            i,
            blk.species,
        )

        expected = blk.commit_entry(
            h_terminal,
            h0,
            R_c,
            cache,
        )

        if expected is None:
            continue

        _, _, actual_payload = plan.entries[entry_index]

        assert_payload_close(
            actual_payload,
            tuple(expected),
            f"block {i} ({blk.species})",
        )

        entry_index += 1

    pre_cache = pre_state.ctx.swa.get(PRELUDE_KEY)
    assert pre_cache is not None

    expected_prelude = model.prelude_block.attn_commit_entry(
        x_t.unsqueeze(1),
        None,
        None,
        pre_cache,
    )

    _, _, actual_prelude = plan.entries[entry_index]

    assert_payload_close(
        actual_prelude,
        tuple(expected_prelude),
        "prelude",
    )

    entry_index += 1

    assert entry_index == len(plan.entries)
