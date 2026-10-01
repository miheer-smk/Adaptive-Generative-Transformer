import torch

from src.agt import AGTv3


DIM = 64
VOCAB = 97
SLOTS = 16
TOKENS = 4


@torch.no_grad()
def test_stage0_determinism():
    torch.manual_seed(2140)

    model = AGTv3(
        dim=DIM,
        num_memory_slots=SLOTS,
        vocab_size=VOCAB,
    ).eval()

    x = torch.randn(TOKENS, DIM)

    state_a = model.initial_persistent_state(batch_size=1)
    state_b = model.initial_persistent_state(batch_size=1)

    C_a = torch.zeros(1, DIM)
    C_b = torch.zeros(1, DIM)

    for t in range(TOKENS):
        out_a = model.step(
            x[t:t + 1],
            state_a,
            C_a,
        )
        out_b = model.step(
            x[t:t + 1],
            state_b,
            C_b,
        )

        assert torch.equal(out_a["h0"], out_b["h0"])
        assert torch.equal(out_a["h_terminal"], out_b["h_terminal"])
        assert torch.equal(out_a["H_t"], out_b["H_t"])
        assert torch.equal(out_a["logits"], out_b["logits"])

        assert torch.equal(out_a["state"].s, out_b["state"].s)
        assert torch.equal(out_a["state"].memory, out_b["state"].memory)

        state_a = out_a["state"]
        state_b = out_b["state"]

    print("\nStage-0 determinism: exact equality across repeated runs.")
