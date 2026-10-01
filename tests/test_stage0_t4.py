import torch

from src.agt import AGTv3


DIM = 64
VOCAB = 97
SLOTS = 16


def assert_no_ephemeral_state(state):
    ctx = state.ctx

    # KV/SWA must have no in-flight entries after terminal commit.
    for cache in ctx.kv.values():
        assert cache.ephemeral_length == 0

    for cache in ctx.swa.values():
        assert cache.ephemeral_length == 0


@torch.no_grad()
def test_t4_exactly_one_terminal_commit_per_token():
    """
    T4 — Single commit.

    CommitPlan.token_index is zero-based.
    tokens_committed is a count and therefore becomes old_count + 1.
    """
    torch.manual_seed(2029)

    model = AGTv3(
        dim=DIM,
        num_memory_slots=SLOTS,
        vocab_size=VOCAB,
    ).eval()

    state = model.initial_persistent_state(batch_size=1)
    C_prev = torch.zeros(1, DIM)

    for _ in range(3):
        pre_count = state.tokens_committed
        pre_gdn_versions = {
            block_id: cache.version
            for block_id, cache in state.ctx.gdn.items()
        }
        x_t = torch.randn(1, DIM)

        out = model.step(x_t, state, C_prev)

        plan = out["commit_plan"]
        next_state = out["state"]

        assert plan is not None

        # CommitPlan identifies the token being committed, zero-based.
        assert plan.token_index == pre_count

        # Exactly one token is committed by this step.
        assert next_state.tokens_committed == pre_count + 1

        # Each GDN terminal commit advances its persistent version exactly once.
        for block_id, old_version in pre_gdn_versions.items():
            assert next_state.ctx.gdn[block_id].version == old_version + 1

        # The terminal boundary leaves no ephemeral cache state.
        assert_no_ephemeral_state(next_state)

        # Commit produces a new persistent state.
        assert next_state is not state

        state = next_state


@torch.no_grad()
def test_t4_repeated_step_from_same_state_does_not_double_commit():
    """
    Two independent calls from the same caller-owned state each commit
    exactly one token; the original state remains at its old count.
    """
    torch.manual_seed(2030)

    model = AGTv3(
        dim=DIM,
        num_memory_slots=SLOTS,
        vocab_size=VOCAB,
    ).eval()

    state0 = model.initial_persistent_state(batch_size=1)
    C_prev = torch.zeros(1, DIM)
    x_t = torch.randn(1, DIM)

    pre_gdn_versions = {
        block_id: cache.version
        for block_id, cache in state0.ctx.gdn.items()
    }

    out_a = model.step(x_t, state0, C_prev)
    out_b = model.step(x_t, state0, C_prev)

    assert state0.tokens_committed == 0

    assert out_a["commit_plan"] is not None
    assert out_b["commit_plan"] is not None

    # Both calls start from token count 0, so both plans identify token 0.
    assert out_a["commit_plan"].token_index == 0
    assert out_b["commit_plan"].token_index == 0

    # Each independent call commits exactly one token.
    assert out_a["state"].tokens_committed == 1
    assert out_b["state"].tokens_committed == 1

    # Each independent execution from state0 performs exactly one GDN commit.
    for block_id, old_version in pre_gdn_versions.items():
        assert out_a["state"].ctx.gdn[block_id].version == old_version + 1
        assert out_b["state"].ctx.gdn[block_id].version == old_version + 1

    assert_no_ephemeral_state(out_a["state"])
    assert_no_ephemeral_state(out_b["state"])
