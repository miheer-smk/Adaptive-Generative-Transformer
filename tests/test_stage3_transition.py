import torch

from src.caches import ExecutionCache
from src.executor import AdaptiveExecutor


class ToyGDNBlock:
    species = "GDN"

    def __call__(self, x, h0=None, R=None, C_prev=None, **kwargs):
        assert C_prev is not None

        committed, _ = C_prev.get_committed_context()

        if committed is None:
            ephemeral = torch.ones_like(x)
        else:
            ephemeral = committed + 1.0

        return x + 2.0, ephemeral


class ToyFFNBlock:
    species = "FFN"

    def __call__(self, x, h0=None, R=None, C_prev=None, **kwargs):
        assert C_prev is None
        return x + 3.0


class ToyBank:
    def __init__(self, blocks):
        self.blocks = blocks

    def __call__(self, index, x, **kwargs):
        return self.blocks[index](
            x,
            **kwargs,
        )


def test_run_block_can_return_gdn_ephemeral_state():
    bank = ToyBank([ToyGDNBlock()])
    executor = AdaptiveExecutor(
        controller=None,
        block_bank=bank,
    )

    context = ExecutionCache()
    context.get_gdn_cache(0)

    h = torch.zeros(1, 4)
    h0 = torch.zeros(1, 4)
    memory = torch.zeros(1, 4)

    h_next, ephemeral = executor._run_block(
        block_index=0,
        h=h,
        h0=h0,
        memory=memory,
        context=context,
        return_ephemeral=True,
    )

    assert torch.equal(
        h_next,
        torch.full_like(h, 2.0),
    )

    assert ephemeral is not None
    assert torch.equal(
        ephemeral,
        torch.ones_like(h.unsqueeze(1)),
    )

    cache = context.get_gdn_cache(0)

    # Gate-C transition evaluation must not persist the state.
    assert cache.committed_state is None
    assert cache.version == 0


def test_run_block_non_gdn_has_no_ephemeral_state():
    bank = ToyBank([ToyFFNBlock()])
    executor = AdaptiveExecutor(
        controller=None,
        block_bank=bank,
    )

    context = ExecutionCache()

    h = torch.zeros(1, 4)
    h0 = torch.zeros(1, 4)
    memory = torch.zeros(1, 4)

    h_next, ephemeral = executor._run_block(
        block_index=0,
        h=h,
        h0=h0,
        memory=memory,
        context=context,
        return_ephemeral=True,
    )

    assert torch.equal(
        h_next,
        torch.full_like(h, 3.0),
    )

    assert ephemeral is None


def test_gdn_transition_does_not_mutate_committed_cache():
    bank = ToyBank([ToyGDNBlock()])
    executor = AdaptiveExecutor(
        controller=None,
        block_bank=bank,
    )

    context = ExecutionCache()
    cache = context.get_gdn_cache(0)

    committed = torch.full((1, 1, 2, 2), 7.0)
    cache.commit_terminal_context(committed)

    version_before = cache.version
    committed_before = cache.committed_state.clone()

    h = torch.zeros(1, 4)
    h0 = torch.zeros(1, 4)
    memory = torch.zeros(1, 4)

    _, ephemeral = executor._run_block(
        block_index=0,
        h=h,
        h0=h0,
        memory=memory,
        context=context,
        return_ephemeral=True,
    )

    assert ephemeral is not None

    assert cache.version == version_before
    assert torch.equal(
        cache.committed_state,
        committed_before,
    )


def test_same_token_gdn_repeat_recomputes_from_committed_state():
    """
    Source-aligned Gate-C contract.

    Each same-token GDN visit reads the committed C_{t-1} context.
    The previous visit's ephemeral result is not written into the
    persistent cache and is not implicitly reused as C_prev.
    """
    bank = ToyBank([ToyGDNBlock()])
    executor = AdaptiveExecutor(
        controller=None,
        block_bank=bank,
    )

    context = ExecutionCache()
    cache = context.get_gdn_cache(0)

    committed = torch.full((1, 1, 4), 7.0)
    cache.commit_terminal_context(committed)

    version_before = cache.version
    committed_before = cache.committed_state.clone()

    h = torch.zeros(1, 4)
    h0 = torch.zeros(1, 4)
    memory = torch.zeros(1, 4)

    _, ephemeral1 = executor._run_block(
        block_index=0,
        h=h,
        h0=h0,
        memory=memory,
        context=context,
        return_ephemeral=True,
    )

    _, ephemeral2 = executor._run_block(
        block_index=0,
        h=h,
        h0=h0,
        memory=memory,
        context=context,
        return_ephemeral=True,
    )

    assert torch.equal(
        ephemeral1,
        committed_before + 1.0,
    )

    assert torch.equal(
        ephemeral2,
        committed_before + 1.0,
    )

    assert cache.version == version_before
    assert torch.equal(
        cache.committed_state,
        committed_before,
    )



def test_real_gdn_ephemeral_payload_survives_same_token_repeat():
    """
    Gate-C real-block contract.

    The real AGT GDN block must expose its complete GDNStep on the
    opt-in ephemeral path, and a second same-token execution must
    consume that in-flight state rather than the committed cache.
    """

    from src.blocks import AGTBlock
    from src.gdn_v3 import GDNStep

    torch.manual_seed(31415)

    block = AGTBlock(
        dim=64,
        species="GDN",
    )

    context = ExecutionCache()
    cache = context.get_gdn_cache(0)

    h = torch.randn(1, 64)
    h0 = torch.randn(1, 64)
    memory = torch.randn(1, 64)

    executor = AdaptiveExecutor(
        controller=None,
        block_bank=ToyBank([block]),
    )

    h1, step1 = executor._run_block(
        block_index=0,
        h=h,
        h0=h0,
        memory=memory,
        context=context,
        return_ephemeral=True,
    )

    assert isinstance(step1, GDNStep)
    assert step1.y.ndim == 3
    assert step1.state is not None
    assert step1.conv_buffer is not None

    # Persistent cache must still be untouched.
    assert cache.committed_state is None
    assert cache.committed_conv_buffer is None
    assert cache.version == 0

    h2, step2 = executor._run_block(
        block_index=0,
        h=h1,
        h0=h0,
        memory=memory,
        context=context,
        return_ephemeral=True,
    )

    assert isinstance(step2, GDNStep)
    assert torch.isfinite(h2).all()
    assert torch.isfinite(step2.state).all()

    # The second transition is a fresh recomputation from the same
    # committed context; step1 is not supplied as persistent input.
    assert not torch.equal(step2.state, step1.state)

    # Still no persistent write.
    assert cache.committed_state is None
    assert cache.committed_conv_buffer is None
    assert cache.version == 0
