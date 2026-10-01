import torch

from src.agt import AGTv3
from tests.test_stage0_t6 import assert_state_equal


DIM = 64
VOCAB = 97
SLOTS = 16


@torch.no_grad()
def test_t7_reset_isolation():
    """
    T7 — Reset isolation.

    Process Doc A, reset by creating a fresh persistent H_0, then process
    Doc B. The resulting Doc B states/logits must equal processing Doc B
    alone from a fresh H_0.
    """
    torch.manual_seed(2050)

    model = AGTv3(
        dim=DIM,
        num_memory_slots=SLOTS,
        vocab_size=VOCAB,
    ).eval()

    C_zero = torch.zeros(1, DIM)

    doc_a = [torch.randn(1, DIM) for _ in range(4)]
    doc_b = [torch.randn(1, DIM) for _ in range(3)]

    # ---------------------------------------------------------
    # Run Doc A.
    # ---------------------------------------------------------
    state_a = model.initial_persistent_state(batch_size=1)

    for x_t in doc_a:
        out = model.step(x_t, state_a, C_zero)
        state_a = out["state"]

    assert state_a.tokens_committed == len(doc_a)

    # ---------------------------------------------------------
    # Reset: fresh persistent state.
    # ---------------------------------------------------------
    state_after_reset = model.initial_persistent_state(batch_size=1)
    C_after_reset = torch.zeros(1, DIM)

    reset_states = []
    reset_logits = []

    for x_t in doc_b:
        out = model.step(x_t, state_after_reset, C_after_reset)
        state_after_reset = out["state"]

        reset_states.append(state_after_reset)
        reset_logits.append(out["logits"].clone())

    # ---------------------------------------------------------
    # Doc B alone from a fresh initial state.
    # ---------------------------------------------------------
    state_b_alone = model.initial_persistent_state(batch_size=1)
    C_b_alone = torch.zeros(1, DIM)

    alone_states = []
    alone_logits = []

    for x_t in doc_b:
        out = model.step(x_t, state_b_alone, C_b_alone)
        state_b_alone = out["state"]

        alone_states.append(state_b_alone)
        alone_logits.append(out["logits"].clone())

    # ---------------------------------------------------------
    # Reset isolation: A must have no effect on B.
    # ---------------------------------------------------------
    for got_state, ref_state, got_logits, ref_logits in zip(
        reset_states,
        alone_states,
        reset_logits,
        alone_logits,
    ):
        assert_state_equal(got_state, ref_state)
        assert torch.equal(got_logits, ref_logits)

    assert state_after_reset.tokens_committed == len(doc_b)
    assert state_b_alone.tokens_committed == len(doc_b)
