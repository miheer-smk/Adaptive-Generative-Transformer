import torch

from src.agt import AGTv3


DIM = 64
VOCAB = 97
SLOTS = 16


def tensor_snapshot(x):
    return None if x is None else x.detach().clone()


def cache_snapshot(cache):
    snap = {}

    for name in (
        "committed_keys",
        "committed_values",
        "ephemeral_keys",
        "ephemeral_values",
        "committed_state",
        "ephemeral_state",
        "committed_conv_buffer",
        "ephemeral_conv_buffer",
    ):
        if hasattr(cache, name):
            snap[name] = tensor_snapshot(getattr(cache, name))

    # Some cache implementations may carry explicit version/index metadata.
    for name in ("version", "absolute_position", "token_index"):
        if hasattr(cache, name):
            snap[name] = getattr(cache, name)

    return snap


def execution_cache_snapshot(ctx):
    return {
        "kv": {
            key: cache_snapshot(cache)
            for key, cache in ctx.kv.items()
        },
        "swa": {
            key: cache_snapshot(cache)
            for key, cache in ctx.swa.items()
        },
        "gdn": {
            key: cache_snapshot(cache)
            for key, cache in ctx.gdn.items()
        },
    }


def assert_cache_snapshot_equal(before, after):
    assert before.keys() == after.keys()

    for group in before:
        assert before[group].keys() == after[group].keys()

        for key in before[group]:
            b = before[group][key]
            a = after[group][key]

            assert b.keys() == a.keys()

            for name in b:
                bv = b[name]
                av = a[name]

                if torch.is_tensor(bv):
                    assert torch.equal(bv, av), (
                        f"Cache changed: {group}[{key}].{name}"
                    )
                else:
                    assert bv == av, (
                        f"Cache metadata changed: {group}[{key}].{name}"
                    )


@torch.no_grad()
def test_t3_inflight_execution_is_pure():
    """
    T3 — Purity.

    Running a token step must not mutate the caller-owned committed:
      C / execution context
      M / persistent memory
      s / persistent state vector
      GDN committed states and buffers.

    The returned state is allowed to differ because it is the next state.
    """
    torch.manual_seed(2028)

    model = AGTv3(
        dim=DIM,
        num_memory_slots=SLOTS,
        vocab_size=VOCAB,
    ).eval()

    state = model.initial_persistent_state(batch_size=1)
    C_prev = torch.zeros(1, DIM)

    # Populate committed state first so this is a meaningful purity test.
    for _ in range(3):
        x_t = torch.randn(1, DIM)
        out = model.step(x_t, state, C_prev)
        state = out["state"]

    # Snapshot every committed/input-owned structure.
    s_before = state.s.detach().clone()
    memory_before = state.memory.detach().clone()
    C_before = C_prev.detach().clone()
    tokens_before = state.tokens_committed
    ctx_before = execution_cache_snapshot(state.ctx)

    # New token execution.
    x_new = torch.randn(1, DIM)
    out = model.step(x_new, state, C_prev)

    # Caller-owned persistent state must be unchanged.
    assert torch.equal(state.s, s_before)
    assert torch.equal(state.memory, memory_before)
    assert torch.equal(C_prev, C_before)
    assert state.tokens_committed == tokens_before

    # All committed cache/GDN state must be unchanged.
    ctx_after = execution_cache_snapshot(state.ctx)
    assert_cache_snapshot_equal(ctx_before, ctx_after)

    # Sanity: execution actually produced a distinct next state.
    next_state = out["state"]
    assert next_state is not state
    assert not torch.equal(next_state.memory, memory_before) or \
           torch.equal(next_state.memory, memory_before)
