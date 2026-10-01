import torch

from src.agt import AGTv3


DIM = 64
VOCAB = 97
SLOTS = 16
PREFIX_LEN = 3
TOTAL_LEN = 5


def run_sequence(model, tokens):
    """
    Execute a token sequence with the current public sequential API.

    Record the observable state at every token:
      - H_t
      - P_t (= logits exposed by current model)
      - controller action logits derived from H_t
      - persistent s
      - persistent memory

    No future token is supplied to step().
    """
    state = model.initial_persistent_state(batch_size=1)
    C_prev = torch.zeros(1, DIM)

    records = []

    with torch.no_grad():
        for x_t in tokens:
            out = model.step(x_t, state, C_prev)
            state = out["state"]

            records.append(
                {
                    "H_t": out["H_t"].detach().clone(),
                    "P_t": out["logits"].detach().clone(),
                    "action_logits": model.controller.action_logits(
                        out["H_t"]
                    ).detach().clone(),
                    "s": state.s.detach().clone(),
                    "memory": state.memory.detach().clone(),
                }
            )

    return records


def assert_record_equal(a, b):
    assert torch.equal(a["H_t"], b["H_t"])
    assert torch.equal(a["P_t"], b["P_t"])
    assert torch.equal(a["action_logits"], b["action_logits"])
    assert torch.equal(a["s"], b["s"])
    assert torch.equal(a["memory"], b["memory"])


@torch.no_grad()
def test_t1_future_token_invariance():
    """
    T1 — Future-token invariance.

    Two sequences share the same prefix through token t and differ only
    in future tokens. Everything observable at and before t must match.
    """
    torch.manual_seed(2026)

    model = AGTv3(
        dim=DIM,
        num_memory_slots=SLOTS,
        vocab_size=VOCAB,
    ).eval()

    prefix = [torch.randn(1, DIM) for _ in range(PREFIX_LEN)]

    future_a = [
        torch.randn(1, DIM),
        torch.randn(1, DIM),
    ]

    future_b = [
        torch.randn(1, DIM),
        torch.randn(1, DIM),
    ]

    # Make the futures definitely different.
    assert not torch.equal(future_a[0], future_b[0])
    assert not torch.equal(future_a[1], future_b[1])

    seq_a = prefix + future_a
    seq_b = prefix + future_b

    records_a = run_sequence(model, seq_a)
    records_b = run_sequence(model, seq_b)

    # Prefix through t must be identical regardless of the future suffix.
    for t in range(PREFIX_LEN):
        assert_record_equal(records_a[t], records_b[t])
