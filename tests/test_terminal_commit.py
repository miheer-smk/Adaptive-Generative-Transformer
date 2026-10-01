import os, sys
from types import SimpleNamespace
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.blocks import AGTBlock, HeterogeneousBlockBank
from src.caches import SWARingCache
from src.norms import RMSNorm
from src.terminal_commit import (
    AGTPersistentState, CommitError, MemoryWrite, PRELUDE_KEY,
    apply_commit_plan, block_context, build_commit_plan, terminal_commit,
)

D, W, SLOTS, DM, MAXLEN = 64, 4, 4, 16, 32


def rnd(*shape, seed=1):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def make_world(seed=0):
    torch.manual_seed(seed)
    bank = HeterogeneousBlockBank(dim=D, swa_window=W, ffn_hidden_dim=128, max_seq_len=MAXLEN).eval()
    pre = AGTBlock(dim=D, species="SWA", swa_window=W, max_seq_len=MAXLEN).eval()
    g = torch.Generator().manual_seed(seed + 50)
    with torch.no_grad():
        for blk in list(bank.blocks) + [pre]:
            blk.anchor_gate.copy_(torch.randn(D, generator=g) * 0.5)
            blk.memory_gate.copy_(torch.randn(D, generator=g) * 0.5)
    fn = RMSNorm(D)
    mw = MemoryWrite(D, DM).eval()
    state = AGTPersistentState.initial(rnd(D, seed=1), rnd(SLOTS, DM, seed=2), bank.species_order, W)
    return SimpleNamespace(bank=bank, pre=pre, fn=fn, mw=mw, state=state)


def inputs(t, variant=0):
    off = 100000 * variant if t >= 5 else 0
    return dict(
        h_terminal=rnd(1, 1, D, seed=1000 + t + off),
        h0=rnd(1, 1, D, seed=2000 + t + off),
        R_c=rnd(1, 1, D, seed=3000 + t + off),
        memory_keys=rnd(1, SLOTS, DM, seed=4000 + t + off),
        prelude_input=rnd(1, 1, D, seed=5000 + t + off),
    )


def ckw(w, t, variant=0):
    return dict(bank=w.bank, prelude_block=w.pre, final_norm=w.fn, memory_write=w.mw,
                token_index=t, **inputs(t, variant))


def run_inflight(w, state, t, blocks):
    inp = inputs(t)
    h = rnd(1, 1, D, seed=7000 + t)
    for j in blocks:
        blk = w.bank.blocks[j]
        blk(h, h0=inp["h0"], R=inp["R_c"], C_prev=block_context(state.ctx, j, blk.species))


def snapshot(state):
    snap = {}
    for kind, d in (("kv", state.ctx.kv), ("swa", state.ctx.swa), ("gdn", state.ctx.gdn)):
        for k, c in d.items():
            a, b = c.get_committed_context()
            extra = c.next_position if kind != "gdn" else None
            snap[(kind, k)] = (c.version, None if a is None else a.clone(),
                               None if b is None else b.clone(), extra)
    snap["memory"], snap["s"], snap["t"] = state.memory.clone(), state.s.clone(), state.tokens_committed
    return snap


def deep_equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    return a == b


# ----------------------------------------------------------------- memory write
def test_memory_write_param_count():
    assert sum(p.numel() for p in MemoryWrite(768, 256).parameters()) == 394_241


@torch.no_grad()
def test_memory_write_matches_reference_and_invariants():
    torch.manual_seed(0)
    mw = MemoryWrite(D, DM)
    for p in mw.parameters():
        p.copy_(torch.randn(p.shape) * 0.3)
    s, M, K = rnd(2, D, seed=3), rnd(2, SLOTS, DM, seed=4), rnd(2, SLOTS, DM, seed=5)
    r = mw(s, M, K)
    q = mw.addr_q(s)
    out = torch.empty_like(M)
    for b in range(2):
        sc = torch.stack([(q[b] * K[b, i]).sum() / (DM ** 0.5) for i in range(SLOTS)])
        a = torch.softmax(sc, 0)
        g = torch.sigmoid(mw.gate_w(s[b]) + mw.gate_b)
        nu = mw.val_norm(mw.val_proj(s[b]))
        for i in range(SLOTS):
            wi = g * a[i]
            out[b, i] = (1 - wi) * M[b, i] + wi * nu
        assert torch.allclose(r.alpha[b], a, atol=1e-6)
    assert torch.allclose(r.memory, out, atol=1e-6)
    assert torch.allclose(r.alpha.sum(-1), torch.ones(2), atol=1e-6)
    w_all = r.g * r.alpha
    assert torch.allclose(w_all.sum(-1), r.g.squeeze(-1), atol=1e-6)      # sum_i w_i = g
    assert ((w_all > 0) & (w_all < 1)).all()


@torch.no_grad()
def test_memory_write_gate_init():
    mw = MemoryWrite(D, DM)
    mw.gate_w.weight.zero_()
    r = mw(rnd(1, D), rnd(1, SLOTS, DM, seed=2), rnd(1, SLOTS, DM, seed=3))
    assert torch.allclose(r.g, torch.sigmoid(torch.tensor(-2.0)).expand_as(r.g), atol=1e-7)


@torch.no_grad()
def test_memory_bounded_prop_14_1():
    mw = MemoryWrite(D, DM)
    for p in mw.parameters():
        p.copy_(torch.randn(p.shape) * 0.3)
    M = rnd(1, SLOTS, DM, seed=1)
    vmax = M.norm(dim=-1).max()
    for t in range(30):
        r = mw(rnd(1, D, seed=10 + t), M, rnd(1, SLOTS, DM, seed=300 + t))
        M = r.memory
        vmax = torch.maximum(vmax, r.nu.norm(dim=-1).max())
        assert M.norm(dim=-1).max() <= vmax + 1e-5


def test_memory_write_gradients():
    mw = MemoryWrite(D, DM)
    r = mw(rnd(2, D, seed=3), rnd(2, SLOTS, DM, seed=4), rnd(2, SLOTS, DM, seed=5))
    (r.memory ** 2).sum().backward()
    for n, p in mw.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
        assert p.grad.abs().sum() > 0, f"zero gradient: {n}"


# -------------------------------------------------------------- block entries
@torch.no_grad()
def test_species_entries():
    w = make_world()
    inp = inputs(0)
    for i, sp in enumerate(w.bank.species_order):
        blk = w.bank.blocks[i]
        c = block_context(w.state.ctx, i, sp)
        e = blk.commit_entry(inp["h_terminal"], inp["h0"], inp["R_c"], c)
        if sp == "FFN":
            assert e is None
        elif sp == "GDN":
            S, buf = blk.gdn_commit_entry(inp["h_terminal"], inp["h0"], inp["R_c"], c)
            assert torch.equal(e[0], S) and torch.equal(e[1], buf)
        else:
            assert len(e) == 2 and e[0].shape[2] == 1 and e[1].shape[2] == 1
        assert c is None or c.version == 0                      # entry building is pure


@torch.no_grad()
@pytest.mark.parametrize("j", [0, 2])                           # SWA, GQA
def test_attention_entry_is_i4(j):
    w = make_world()
    state = w.state
    blk = w.bank.blocks[j]
    for t in range(6):
        inp = inputs(t)
        cache = block_context(state.ctx, j, blk.species)
        pos = cache.next_position
        u = blk.norm(inp["h_terminal"] + blk.injection(inp["h_terminal"], inp["h0"], inp["R_c"]))
        _, k_if, v_if = blk.core._project_qkv_rope(u, pos)[0:3]
        state, _ = terminal_commit(state, **ckw(w, t))
        assert torch.equal(cache.committed_keys[:, :, -1:], k_if)
        assert torch.equal(cache.committed_values[:, :, -1:], v_if)


@torch.no_grad()
def test_prelude_entry_from_prelude_input():
    w = make_world()
    state = w.state
    for t in range(5):
        inp = inputs(t)
        pc = state.ctx.swa[PRELUDE_KEY]
        k, v = w.pre.core.project_kv(w.pre.norm(inp["prelude_input"]), pc.next_position)
        state, _ = terminal_commit(state, **ckw(w, t))
        assert torch.equal(pc.committed_keys[:, :, -1:], k)
        assert torch.equal(pc.committed_values[:, :, -1:], v)
        assert pc.next_position == t + 1


# ------------------------------------------------------------- commit contract
@torch.no_grad()
def test_commit_all_species_cardinality_and_unvisited():
    w = make_world()
    state = w.state
    visits = [[], [0], [2, 2, 3], [7, 7, 7, 7], [4, 5, 4], [], [1], [6], [0, 1, 2, 3, 4, 5, 6, 7], [3]]
    prev_mem = state.memory.clone()
    for t, vis in enumerate(visits):
        run_inflight(w, state, t, vis)
        state, plan = terminal_commit(state, **ckw(w, t))
        n = t + 1
        assert state.tokens_committed == n
        for i, sp in enumerate(w.bank.species_order):
            if sp == "GQA":
                c = state.ctx.kv[i]
                assert c.committed_length == n and c.next_position == n and c.version == n
            elif sp == "SWA":
                c = state.ctx.swa[i]
                assert c.committed_length == min(n, W - 1) and c.next_position == n and c.version == n
            elif sp == "GDN":
                assert state.ctx.gdn[i].version == n
            else:
                assert i not in state.ctx.kv and i not in state.ctx.swa and i not in state.ctx.gdn
        pc = state.ctx.swa[PRELUDE_KEY]
        assert pc.committed_length == min(n, W - 1) and pc.next_position == n and pc.version == n
        assert torch.equal(state.s, w.fn(inputs(t)["h_terminal"]).squeeze(1))
        assert not torch.equal(state.memory, prev_mem)
        prev_mem = state.memory.clone()


@torch.no_grad()
def test_inflight_and_plan_build_are_pure():
    w = make_world()
    state = w.state
    for t in range(3):
        state, _ = terminal_commit(state, **ckw(w, t))
    before = snapshot(state)
    for _ in range(4):
        run_inflight(w, state, 3, [0, 1, 2, 3, 4, 5, 6, 7, 2, 2, 5])
    assert deep_equal(before, snapshot(state))
    p1 = build_commit_plan(state, **ckw(w, 3))
    p2 = build_commit_plan(state, **ckw(w, 3))
    assert deep_equal(before, snapshot(state))
    assert torch.equal(p1.memory, p2.memory) and torch.equal(p1.s, p2.s)
    assert apply_commit_plan(state, p1).tokens_committed == 4


@torch.no_grad()
def test_exactly_one_commit_per_token():
    w = make_world()
    state = w.state
    plan = build_commit_plan(state, **ckw(w, 0))
    new = apply_commit_plan(state, plan)
    with pytest.raises(CommitError):
        apply_commit_plan(state, plan)                           # re-apply
    with pytest.raises(CommitError):
        terminal_commit(state, **ckw(w, 0))                      # consumed state
    for d in (new.ctx.kv, new.ctx.swa, new.ctx.gdn):
        for c in d.values():
            assert c.version == 1
    fresh = make_world(1)
    with pytest.raises(CommitError):
        terminal_commit(fresh.state, **ckw(fresh, 1))            # wrong token index
    assert all(c.version == 0 for d in (fresh.state.ctx.kv, fresh.state.ctx.swa, fresh.state.ctx.gdn)
               for c in d.values())


@torch.no_grad()
def test_atomicity_nothing_written_on_failure():
    w = make_world()
    state = w.state
    state.ctx.kv[2] = SWARingCache(window_size=W)               # wrong cache type for a GQA block
    before = snapshot(state)
    with pytest.raises(TypeError):
        terminal_commit(state, **ckw(w, 0))
    assert deep_equal(before, snapshot(state)) and state.tokens_committed == 0

    w = make_world()
    state = w.state
    state.ctx.gdn[4].version += 1                                # inconsistent state
    before = snapshot(state)
    with pytest.raises(CommitError):
        terminal_commit(state, **ckw(w, 0))
    assert deep_equal(before, snapshot(state))

    w = make_world()
    state = w.state
    plan = build_commit_plan(state, **ckw(w, 0))
    state.ctx.gdn[5].version += 1                                # cache changed after the plan
    with pytest.raises(CommitError):
        apply_commit_plan(state, plan)
    for kind, d in (("kv", state.ctx.kv), ("swa", state.ctx.swa), ("gdn", state.ctx.gdn)):
        for k, c in d.items():
            assert c.version == (1 if (kind, k) == ("gdn", 5) else 0)


@torch.no_grad()
def test_reset_isolation():
    w = make_world()
    st = w.state
    for t in range(4):
        st, _ = terminal_commit(st, **ckw(w, t))
    fresh = AGTPersistentState.initial(rnd(D, seed=1), rnd(SLOTS, DM, seed=2), w.bank.species_order, W)
    ref = make_world()
    a, _ = terminal_commit(fresh, **ckw(w, 0))
    b, _ = terminal_commit(ref.state, **ckw(ref, 0))
    assert deep_equal(snapshot(a), snapshot(b))


@torch.no_grad()
def test_later_tokens_do_not_change_earlier_entries():
    wa, wb = make_world(), make_world()
    sa, sb = wa.state, wb.state
    for t in range(8):
        sa, _ = terminal_commit(sa, **ckw(wa, t, variant=0))
        sb, _ = terminal_commit(sb, **ckw(wb, t, variant=1))
    assert torch.equal(sa.ctx.kv[2].committed_keys[:, :, :5], sb.ctx.kv[2].committed_keys[:, :, :5])
    assert torch.equal(sa.ctx.kv[3].committed_values[:, :, :5], sb.ctx.kv[3].committed_values[:, :, :5])
    assert not torch.equal(sa.ctx.kv[2].committed_keys[:, :, 5:], sb.ctx.kv[2].committed_keys[:, :, 5:])


@torch.no_grad()
def test_shape_guards():
    w = make_world()
    kw = ckw(w, 0)
    kw["h_terminal"] = rnd(1, 2, D)
    with pytest.raises(ValueError):
        terminal_commit(w.state, **kw)
    assert w.state.tokens_committed == 0
