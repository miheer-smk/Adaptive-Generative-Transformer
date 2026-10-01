import os, sys
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.blocks import AGTBlock, HeterogeneousBlockBank
from src.caches import KVCache, SWARingCache, GDNStateCache, ExecutionCache

D, W, MAXLEN = 64, 4, 32
NOCACHE = ["SWA", "GQA", "FFN"]


def mk(species, dim=D, seed=0):
    torch.manual_seed(seed)
    return AGTBlock(dim=dim, species=species, swa_window=W,
                    ffn_hidden_dim=128, max_seq_len=MAXLEN).eval()


def rnd(*shape, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def randomize_gates(b, seed=5):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        b.anchor_gate.copy_(torch.randn(b.dim, generator=g) * 0.5)
        b.memory_gate.copy_(torch.randn(b.dim, generator=g) * 0.5)


def ref(b, x, h0=None, R=None, **core_kw):
    """Independent re-statement of the AMB-1 block equations."""
    inj = None
    if h0 is not None:
        inj = b.anchor_gate * h0
    if R is not None:
        inj = b.memory_gate * R if inj is None else inj + b.memory_gate * R
    pre = x if inj is None else x + inj
    if b.species == "FFN":
        c = b.core
        return x + c.mhc(c.ffn(c.norm(pre)))
    return x + b.mhc(b.core(b.norm(pre), **core_kw))


def snap(s):
    if torch.is_tensor(s):
        return s.detach().clone()
    if isinstance(s, (tuple, list)):
        return type(s)(snap(e) for e in s)
    return s


def same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


class Zero(nn.Module):
    def forward(self, x, **kw):
        return torch.zeros_like(x)


# ------------------------------------------------------------ parameters
@pytest.mark.parametrize("sp", NOCACHE + ["GDN"])
def test_gates_exist_init_and_train(sp):
    b = mk(sp, dim=768 if sp == "GDN" else D)
    for g in (b.anchor_gate, b.memory_gate):
        assert g.shape == (b.dim,) and g.requires_grad
        assert torch.allclose(g, torch.full_like(g, 0.1))


# ------------------------------------------------- equations / AMB-1
@pytest.mark.parametrize("sp", NOCACHE)
def test_no_injection_matches_legacy_form(sp):
    b, x = mk(sp), rnd(1, 3, D)
    with torch.no_grad():
        assert torch.allclose(b(x), ref(b, x), atol=1e-6)


@pytest.mark.parametrize("sp", NOCACHE)
def test_injection_form(sp):
    b = mk(sp); randomize_gates(b)
    x, h0, R = rnd(1, 3, D, seed=1), rnd(1, 3, D, seed=2), rnd(1, 3, D, seed=3)
    with torch.no_grad():
        assert torch.allclose(b(x, h0=h0, R=R), ref(b, x, h0, R), atol=1e-6)
        assert torch.allclose(b(x, h0=h0), ref(b, x, h0, None), atol=1e-6)
        assert torch.allclose(b(x, R=R), ref(b, x, None, R), atol=1e-6)


@pytest.mark.parametrize("sp", NOCACHE)
def test_residual_path_is_raw_h(sp):
    # AMB-1: injection must NOT reach the residual path. With a zero sub-layer
    # and identity mHC the block must return h exactly, for any h0, R, gates.
    b = mk(sp); randomize_gates(b)
    if sp == "FFN":
        b.core.ffn = Zero(); b.core.mhc = nn.Identity()
    else:
        b.core = Zero(); b.mhc = nn.Identity()
    x, h0, R = rnd(1, 3, D, seed=1), rnd(1, 3, D, seed=2), rnd(1, 3, D, seed=3)
    with torch.no_grad():
        assert torch.equal(b(x, h0=h0, R=R), x)


@pytest.mark.parametrize("sp", NOCACHE)
def test_zero_gates_equal_no_injection(sp):
    b = mk(sp)
    with torch.no_grad():
        b.anchor_gate.zero_(); b.memory_gate.zero_()
    x, h0, R = rnd(1, 3, D, seed=1), rnd(1, 3, D, seed=2), rnd(1, 3, D, seed=3)
    with torch.no_grad():
        assert torch.allclose(b(x, h0=h0, R=R), b(x), atol=1e-6)


@pytest.mark.parametrize("sp", NOCACHE)
def test_injection_gradients(sp):
    b = mk(sp)
    x = rnd(1, 3, D, seed=1).requires_grad_()
    h0 = rnd(1, 3, D, seed=2).requires_grad_()
    R = rnd(1, 3, D, seed=3).requires_grad_()
    b(x, h0=h0, R=R).pow(2).sum().backward()
    for t in (b.anchor_gate.grad, b.memory_gate.grad, h0.grad, R.grad, x.grad):
        assert t is not None and torch.isfinite(t).all() and t.abs().sum() > 0


def test_shape_guards():
    b = mk("SWA")
    x = rnd(1, 3, D)
    with pytest.raises(ValueError):
        b(x, h0=rnd(1, 2, D))
    with pytest.raises(ValueError):
        b(x, R=rnd(1, 3, D + 1))


# --------------------------------------------------- context (C_prev)
def commit_some(b, cache, n=6):
    for t in range(n):
        xt = rnd(1, 1, D, seed=10 + t)
        k, v = b.core.project_kv(b.norm(xt), cache.next_position)
        cache.commit_terminal_context(k, v)


@pytest.mark.parametrize("sp", ["GQA", "SWA"])
def test_cached_block_reference_purity_and_influence(sp):
    b = mk(sp); randomize_gates(b)
    new = (lambda: KVCache()) if sp == "GQA" else (lambda: SWARingCache(window_size=W))
    cache = new()
    commit_some(b, cache)
    before = (cache.version, cache.committed_length, cache.next_position,
              cache.committed_keys.clone(), cache.committed_values.clone())
    x, h0, R = rnd(1, 1, D, seed=21), rnd(1, 1, D, seed=22), rnd(1, 1, D, seed=23)
    with torch.no_grad():
        y1 = b(x, h0=h0, R=R, C_prev=cache)
        y2 = b(x, h0=h0, R=R, C_prev=cache)          # REPEAT-style revisit
        r = ref(b, x, h0, R, kv_cache=cache)
        y_empty = b(x, h0=h0, R=R, C_prev=new())
    assert torch.equal(y1, y2)
    assert torch.allclose(y1, r, atol=1e-6)
    assert not torch.allclose(y1, y_empty, atol=1e-6), "context has no influence"
    assert (cache.version, cache.committed_length, cache.next_position) == before[:3]
    assert torch.equal(cache.committed_keys, before[3])
    assert torch.equal(cache.committed_values, before[4])


def test_context_type_and_conflict_guards():
    x = rnd(1, 1, D)
    with pytest.raises(TypeError):
        mk("GQA")(x, C_prev=SWARingCache(window_size=W))
    with pytest.raises(TypeError):
        mk("GQA")(x, C_prev=GDNStateCache())
    with pytest.raises(TypeError):
        mk("SWA")(x, C_prev=KVCache())
    with pytest.raises(ValueError):
        mk("GQA")(x, C_prev=KVCache(), kv_cache=KVCache())
    with pytest.raises(ValueError):
        mk("FFN")(x, C_prev=KVCache())


# ------------------------------------------------------------------ GDN
@torch.no_grad()
def test_gdn_state_flow_and_purity():
    b = mk("GDN", dim=768)
    h0, R = rnd(1, 1, 768, seed=2), rnd(1, 1, 768, seed=3)
    x1, x2 = rnd(1, 1, 768, seed=4), rnd(1, 1, 768, seed=5)

    cache = GDNStateCache()
    out = b(x1, h0=h0, R=R, C_prev=cache)
    assert isinstance(out, tuple) and len(out) == 2
    y1, s1 = out
    assert y1.shape == x1.shape and torch.isfinite(y1).all()
    assert cache.version == 0 and cache.committed_state is None   # nothing written

    cache.commit_terminal_context(s1, None)                       # explicit commit only
    before, v = snap(cache.committed_state), cache.version

    ya, _ = b(x2, h0=h0, R=R, C_prev=cache)
    yb, _ = b(x2, h0=h0, R=R, C_prev=cache)                       # revisit
    yn, _ = b(x2, h0=h0, R=R, C_prev=GDNStateCache())             # no committed state
    assert torch.equal(ya, yb)
    assert cache.version == v and same(cache.committed_state, before)
    assert not torch.allclose(ya, yn, atol=1e-6), "committed GDN state has no influence"


@torch.no_grad()
def test_gdn_legacy_call_still_works():
    b = mk("GDN", dim=768)
    out = b(rnd(1, 1, 768))
    assert isinstance(out, tuple) and len(out) == 2


# ------------------------------------------------------- bank, executor form
@torch.no_grad()
def test_bank_executor_form_all_species_pure():
    bank = HeterogeneousBlockBank(dim=768, swa_window=8, max_seq_len=64,
                                  ffn_hidden_dim=3072).eval()
    assert bank.species_order == ["SWA", "SWA", "GQA", "GQA", "GDN", "GDN", "FFN", "FFN"]
    ec = ExecutionCache(swa_window=8)
    h, h0, R = rnd(1, 1, 768, seed=1), rnd(1, 1, 768, seed=2), rnd(1, 1, 768, seed=3)
    for j, sp in enumerate(bank.species_order):
        ctx = {"GQA": lambda: ec.get_kv_cache(j), "SWA": lambda: ec.get_swa_cache(j),
               "GDN": lambda: ec.get_gdn_cache(j), "FFN": lambda: None}[sp]()
        out = bank(j, h, h0=h0, R=R, C_prev=ctx)
        y = out[0] if sp == "GDN" else out
        assert y.shape == h.shape and torch.isfinite(y).all(), f"block {j} ({sp})"
        assert bank(j, h, h0=h0, R=R, C_prev=ctx) is not None          # REPEAT
    for caches in (ec.kv, ec.swa, ec.gdn):
        for c in caches.values():
            assert c.version == 0
