import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agt import AGTv3


def snapshot_state(state):
    out = {
        "s": state.s.clone(),
        "memory": state.memory.clone(),
        "tokens_committed": state.tokens_committed,
        "ctx": {},
    }

    for kind in ("kv", "swa", "gdn"):
        bucket = getattr(state.ctx, kind)
        out["ctx"][kind] = {}

        for key, cache in bucket.items():
            entry = {
                "version": cache.version,
            }

            if kind in ("kv", "swa"):
                entry["committed_keys"] = (
                    None if cache.committed_keys is None
                    else cache.committed_keys.clone()
                )
                entry["committed_values"] = (
                    None if cache.committed_values is None
                    else cache.committed_values.clone()
                )
                if kind == "swa":
                    entry["num_committed_tokens"] = cache.num_committed_tokens

            if kind == "gdn":
                entry["committed_state"] = (
                    None if cache.committed_state is None
                    else cache.committed_state.clone()
                )
                entry["committed_conv_buffer"] = (
                    None if cache.committed_conv_buffer is None
                    else cache.committed_conv_buffer.clone()
                )

            out["ctx"][kind][key] = entry

    return out


def assert_state_unchanged(state, snap):
    assert torch.equal(state.s, snap["s"])
    assert torch.equal(state.memory, snap["memory"])
    assert state.tokens_committed == snap["tokens_committed"]

    for kind in ("kv", "swa", "gdn"):
        bucket = getattr(state.ctx, kind)

        for key, entry in snap["ctx"][kind].items():
            cache = bucket[key]
            assert cache.version == entry["version"]

            if kind in ("kv", "swa"):
                if entry["committed_keys"] is None:
                    assert cache.committed_keys is None
                else:
                    assert torch.equal(
                        cache.committed_keys,
                        entry["committed_keys"],
                    )

                if entry["committed_values"] is None:
                    assert cache.committed_values is None
                else:
                    assert torch.equal(
                        cache.committed_values,
                        entry["committed_values"],
                    )

                if kind == "swa":
                    assert (
                        cache.num_committed_tokens
                        == entry["num_committed_tokens"]
                    )

            if kind == "gdn":
                if entry["committed_state"] is None:
                    assert cache.committed_state is None
                else:
                    assert torch.equal(
                        cache.committed_state,
                        entry["committed_state"],
                    )

                if entry["committed_conv_buffer"] is None:
                    assert cache.committed_conv_buffer is None
                else:
                    assert torch.equal(
                        cache.committed_conv_buffer,
                        entry["committed_conv_buffer"],
                    )


@torch.no_grad()
def test_two_consecutive_tokens_and_state_carry():
    torch.manual_seed(123)
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=97).eval()

    state0 = model.initial_persistent_state(batch_size=1)
    C0 = torch.zeros(1, 64)

    x1 = torch.randn(1, 64)
    x2 = torch.randn(1, 64)

    out1 = model.step(x1, state0, C0)
    state1 = out1["state"]

    assert state1.tokens_committed == 1
    assert torch.equal(state1.s, out1["H_t"])

    out2 = model.step(x2, state1, C0)
    state2 = out2["state"]

    assert state2.tokens_committed == 2

    for i in range(8):
        blk = model.blocks.blocks[i]
        if blk.species == "FFN":
            continue

        cache = (
            state2.ctx.kv[i]
            if blk.species == "GQA"
            else state2.ctx.swa[i]
            if blk.species == "SWA"
            else state2.ctx.gdn[i]
        )

        assert cache.version == 2

    assert state2.ctx.swa["pre"].version == 2


@torch.no_grad()
def test_gqa_and_swa_advance_once_per_token():
    torch.manual_seed(456)
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=97).eval()

    state = model.initial_persistent_state(batch_size=1)
    C = torch.zeros(1, 64)

    for t in range(3):
        x = torch.randn(1, 64)
        out = model.step(x, state, C)
        state = out["state"]

        for i in (0, 1):
            assert state.ctx.swa[i].version == t + 1
            assert state.ctx.swa[i].next_position == t + 1

        for i in (2, 3):
            assert state.ctx.kv[i].version == t + 1
            assert state.ctx.kv[i].next_position == t + 1
            assert state.ctx.kv[i].committed_length == t + 1


@torch.no_grad()
def test_gdn_advances_once_per_token():
    torch.manual_seed(789)
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=97).eval()

    state = model.initial_persistent_state(batch_size=1)
    C = torch.zeros(1, 64)

    for t in range(2):
        x = torch.randn(1, 64)
        out = model.step(x, state, C)
        state = out["state"]

        for i in (4, 5):
            cache = state.ctx.gdn[i]
            assert cache.version == t + 1
            assert cache.committed_state is not None
            assert cache.committed_conv_buffer is not None


@torch.no_grad()
def test_original_state_is_not_mutated_and_is_reusable():
    torch.manual_seed(321)
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=97).eval()

    state0 = model.initial_persistent_state(batch_size=1)
    C = torch.zeros(1, 64)
    x = torch.randn(1, 64)

    snap = snapshot_state(state0)

    out_a = model.step(x, state0, C)
    out_b = model.step(x, state0, C)

    assert_state_unchanged(state0, snap)

    assert torch.equal(out_a["H_t"], out_b["H_t"])
    assert torch.equal(out_a["logits"], out_b["logits"])
    assert torch.equal(out_a["state"].s, out_b["state"].s)
    assert torch.equal(out_a["state"].memory, out_b["state"].memory)


@torch.no_grad()
def test_memory_write_exactly_once_per_token():
    torch.manual_seed(654)
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=97).eval()

    calls = {"n": 0}
    original_forward = model.memory_write.forward

    def counted_forward(s, memory, memory_keys):
        calls["n"] += 1
        return original_forward(s, memory, memory_keys)

    model.memory_write.forward = counted_forward

    state = model.initial_persistent_state(batch_size=1)
    C = torch.zeros(1, 64)

    model.step(torch.randn(1, 64), state, C)
    assert calls["n"] == 1


@torch.no_grad()
def test_shapes_and_finite_outputs():
    torch.manual_seed(987)
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=97).eval()

    state = model.initial_persistent_state(batch_size=2)
    C = torch.zeros(2, 64)
    x = torch.randn(2, 64)

    out = model.step(x, state, C)

    assert out["h0"].shape == (2, 64)
    assert out["h_terminal"].shape == (2, 64)
    assert out["H_t"].shape == (2, 64)
    assert out["logits"].shape == (2, 97)
    assert out["memory_read"].shape == (2, 64)

    assert torch.isfinite(out["H_t"]).all()
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["state"].memory).all()


@torch.no_grad()
def test_state_token_count_and_prelude_context():
    torch.manual_seed(135)
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=97).eval()

    state = model.initial_persistent_state(batch_size=1)
    C = torch.zeros(1, 64)

    for t in range(2):
        out = model.step(torch.randn(1, 64), state, C)
        state = out["state"]

    assert state.tokens_committed == 2
    assert state.ctx.swa["pre"].version == 2
    assert state.ctx.swa["pre"].next_position == 2
