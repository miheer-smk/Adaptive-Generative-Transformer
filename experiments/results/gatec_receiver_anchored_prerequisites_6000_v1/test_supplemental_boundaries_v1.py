"""Supplementary synthetic boundary checks and eight disposable model tokens."""
import json
from pathlib import Path
import pytest
import torch
import test_receiver_anchored_v3 as original
from test_receiver_anchored_v3 import wrapper, ledger
from receiver_anchored_v2 import P0, P1, fixture_encoding
from statistical_rules_v3 import make_bank, ablation_rule, latency_rule, stratified_means

OUT = Path(__file__).resolve().parent
original.FACTS.update(json.loads((OUT / "test_measurements.json").read_text()))


def test_ablation_accepts_positive_bound_below_five_points():
    bank = make_bank(torch.arange(4).repeat_interleave(128))
    u = torch.ones(512, 4, dtype=torch.int64)
    a = u.clone()
    for group in bank.groups:
        a[group[:8]] = 0
    r = ablation_rule(u, a, bank)
    assert bool((r["point"] == .0625).all())
    assert bool(((r["lower"] > 0) & (r["lower"] < .05)).all())
    assert bool(r["passes"].all())
    with pytest.raises(ValueError):
        ablation_rule(torch.ones(4, 4), torch.zeros(4, 4), make_bank([0,0,1,1]))
    original.fact("synthetic_ablation_boundary", dict(point=r["point"].tolist(), lower=r["lower"].tolist(),
                   passes=r["passes"].tolist(), lower_above_five_points_not_required=True))


def test_latency_rejects_good_point_with_excessive_upper_bound():
    lengths = torch.tensor([16,24]).repeat_interleave(16)
    parities = torch.tensor([0,1]).repeat_interleave(8).repeat(2)
    bank = make_bank(lengths*2 + parities)
    t = torch.ones(32, 2, 2, 5, dtype=torch.float64)
    t[::2, 0] = 1.4
    t[1::2, 0] = .7
    r = latency_rule(t, lengths, parities, bank)
    assert bool((r["point"] < 1.10).all()) and bool((r["upper"] > 1.10).all())
    assert not bool(r["passes"].any())
    original.fact("synthetic_latency_boundary", dict(point=r["point"].tolist(), upper=r["upper"].tolist(),
                   passes=r["passes"].tolist(), point_estimate_alone_insufficient=True))


def test_equal_stratum_weighting_not_case_pooled_weighting():
    strata = torch.tensor([0]*2 + [1]*6 + [2]*4 + [3]*8)
    bank = make_bank(strata)
    values = strata.to(torch.float64) + 1
    point, draws = stratified_means(values, bank)
    assert float(point) == 2.5 and float(values.mean()) == 2.9
    assert bool((draws == 2.5).all())


def test_all_committed_cardinalities_positions_and_read_purity(wrapper):
    device = next(wrapper.parameters()).device
    x, labels = fixture_encoding([1,0], device)
    ex = wrapper.model.executor
    old = ex._run_block
    def observe(block_index, h, h0, memory, context, **kwargs):
        before = memory.detach().clone()
        result = old(block_index, h, h0, memory, context, **kwargs)
        assert torch.equal(before, memory), "In-flight memory-read mutation"
        return result
    ex._run_block = observe
    count = 0
    try:
        for program in (P0,P1):
            state = wrapper.initial(x)
            for j in range(2):
                state = wrapper(x[:,j], state, program, "cache-fixture/" + program.name)["state"]
                n = j + 1
                assert state.tokens_committed == n and not state._consumed
                for c in state.ctx.kv.values():
                    assert c.version == c.committed_length == c.next_position == n
                for c in state.ctx.swa.values():
                    assert c.version == c.num_committed_tokens == c.next_position == n
                    assert c.committed_length == min(n, c.window_size-1)
                for i,c in state.ctx.gdn.items():
                    core = wrapper.model.blocks.blocks[i].core
                    assert c.version == n
                    assert c.committed_state.shape == (1,core.num_heads,core.dv,core.dk)
                    assert c.committed_conv_buffer.shape == (1,core.conv_kernel-1,core.conv_channels)
                count += 1
    finally:
        ex._run_block = old
    original.fact("complete_cache_metadata/" + str(device), dict(tokens=count,
                  every_cache_checked=True,memory_read_pure=True,unselected_GDN4_commit_checked=True))
