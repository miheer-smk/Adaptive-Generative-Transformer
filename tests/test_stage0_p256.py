import torch
import torch.nn as nn

from src.agt import AGTv3


DIM = 64
VOCAB = 97
SLOTS = 16
P = 256


class AlwaysAdvanceController(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        self.num_blocks = num_blocks
        self.seen_h = []

    def forward(self, h):
        self.seen_h.append(h.detach().clone())

        batch = h.shape[0]
        device = h.device

        advance = torch.full(
            (batch, self.num_blocks),
            -100.0,
            device=device,
        )
        advance[:, 0] = 100.0

        return {
            "advance": advance,
            "repeat": torch.full((batch,), -100.0, device=device),
            "restart": torch.full((batch,), -100.0, device=device),
            "halt": torch.full((batch,), -100.0, device=device),
        }


@torch.no_grad()
def test_stage0_p256_norm_stability():
    torch.manual_seed(2130)

    model = AGTv3(
        dim=DIM,
        num_memory_slots=SLOTS,
        vocab_size=VOCAB,
    ).eval()

    controller = AlwaysAdvanceController(
        num_blocks=len(model.blocks.blocks),
    )

    model.executor.controller = controller
    model.executor.convergence = None
    model.executor.max_execution_depth = P

    x_t = torch.randn(1, DIM)
    H_prev = torch.zeros(1, DIM)
    memory_read = torch.zeros(1, DIM)
    C_prev = torch.zeros(1, DIM)

    h0 = model.prelude(
        x_t=x_t,
        H_prev=H_prev,
        memory_read=memory_read,
        C_prev=C_prev,
    )

    result = model.executor.execute(
        h0=h0,
        memory=torch.zeros(1, DIM),
        C_prev=C_prev,
    )

    assert result.execution_count == P
    assert result.safety_cap_hit
    assert torch.isfinite(h0).all()
    assert torch.isfinite(result.h_terminal).all()

    norm0 = h0.norm().item()
    normp = result.h_terminal.norm().item()

    assert norm0 > 0.0

    ratio = normp / norm0

    print(f"\nP=256 norm(h_0)       = {norm0:.10e}")
    print(f"P=256 norm(h_256)     = {normp:.10e}")
    print(f"P=256 norm ratio      = {ratio:.10e}")

    # Measurement-only stability check: no NaN/Inf through depth 256.
