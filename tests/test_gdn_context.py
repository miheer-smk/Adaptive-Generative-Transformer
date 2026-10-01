import os, sys
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gdn_v3 import GDNv3, GDNStep
from src.blocks import AGTBlock
from src.caches import GDNStateCache, KVCache

DIM, H, DK, DV = 32, 2, 8, 16
C = 2 * H * DK + H * DV          # 64 conv channels
K1 = 3


def mk(seed=0, boost=True):
    torch.manual_seed(seed)
    m = GDNv3(dim=DIM, num_heads=H, dk=DK, dv=DV).eval()
    if boost:                      # make every path numerically significant
        g = torch.Generator().manual_seed(seed + 100)
        with torch.no_grad():
            for p in m.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return m


def rnd(*shape, seed=1):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def commit_tokens(m, xs):
    cache = GDNStateCache()
    for x in xs:
        st, cv = cache.get_committed_context()
        S, buf = m.commit(x, state=st, conv_buffer=cv)
        cache.commit_terminal_context(S, buf)
    return cache


@torch.no_grad()
def ref_forward(m, u, S0=None, buf0=None):
    """Independent re-statement of the blueprint recurrence (explicit matrices,
    explicit conv windows)."""
    B, T, _ = u.shape
    pre = torch.cat([m.q_proj(u), m.k_proj(u), m.v_proj(u)], -1)
    buf = torch.zeros(B, K1, C) if buf0 is None else buf0
    seq = torch.cat([buf, pre], 1)
    Wc = m.conv.weight[:, 0, :]                                # [C, 4]
    conv = torch.stack([(seq[:, t:t + 4, :] * Wc.t()).sum(1) for t in range(T)], 1)
    conv = F.silu(conv)
    q, k, v = conv.split([H * DK, H * DK, H * DV], -1)
    q = F.normalize(q.reshape(B, T, H, DK), dim=-1)
    k = F.normalize(k.reshape(B, T, H, DK), dim=-1)
    v = v.reshape(B, T, H, DV)
    alpha = torch.exp(-torch.exp(m.A_log) * F.softplus(m.w_alpha(u) + m.b_h))
    beta = torch.sigmoid(m.w_beta(u))
    S = torch.zeros(B, H, DV, DK) if S0 is None else S0
    I = torch.eye(DK)
    outs = []
    for t in range(T):
        kt = k[:, t]
        kkT = kt.unsqueeze(-1) * kt.unsqueeze(-2)
        a = alpha[:, t][..., None, None]
        b = beta[:, t][..., None, None]
        S = a * (S @ (I - b * kkT)) + b * (v[:, t].unsqueeze(-1) * kt.unsqueeze(-2))
        outs.append((S @ q[:, t].unsqueeze(-1)).squeeze(-1))
    o = m.head_norm(torch.stack(outs, 1))
    g = F.silu(m.g_proj(u)).reshape(B, T, H, DV)
    y = m.o_proj((o * g).reshape(B, T, H * DV))
    return y, S, seq[:, -K1:, :]


def test_param_count_matches_blueprint():
    m = GDNv3(dim=768)                                         # defaults: 6 heads, dk 64, dv 128
    n = sum(p.numel() for p in m.parameters())
    assert n == 2_374_796, n                                    # blueprint 18.1 GDN mixer


@torch.no_grad()
def test_gate_ranges():
    m = mk()
    alpha, beta = m._gates(rnd(2, 5, DIM))
    assert (alpha > 0).all() and (alpha <= 1).all()
    assert (beta > 0).all() and (beta < 1).all()


@torch.no_grad()
def test_matches_independent_reference():
    m = mk()
    u = rnd(2, 7, DIM)
    step = m.forward_ephemeral(u)
    yr, Sr, br = ref_forward(m, u)
    assert isinstance(step, GDNStep)
    assert torch.allclose(step.y, yr, atol=1e-5)
    assert torch.allclose(step.state, Sr, atol=1e-5)
    assert torch.allclose(step.conv_buffer, br, atol=1e-6)
    S0, b0 = rnd(2, H, DV, DK, seed=7) * 0.3, rnd(2, K1, C, seed=8)
    step = m.forward_ephemeral(u, state=S0, conv_buffer=b0)
    yr, Sr, br = ref_forward(m, u, S0, b0)
    assert torch.allclose(step.y, yr, atol=1e-5)
    assert torch.allclose(step.state, Sr, atol=1e-5)
    assert torch.allclose(step.conv_buffer, br, atol=1e-6)


@torch.no_grad()
def test_tokenwise_commit_equals_full_sequence():
    m = mk()
    xs = [rnd(1, 1, DIM, seed=100 + t) for t in range(9)]
    full = m.forward_ephemeral(torch.cat(xs, 1))
    cache = GDNStateCache()
    ys = []
    for x in xs:
        st, cv = cache.get_committed_context()
        ys.append(m.forward_ephemeral(x, state=st, conv_buffer=cv).y)
        S, buf = m.commit(x, state=st, conv_buffer=cv)
        cache.commit_terminal_context(S, buf)
    assert cache.version == len(xs)
    assert torch.allclose(torch.cat(ys, 1), full.y, atol=1e-5)
    assert torch.allclose(cache.committed_state, full.state, atol=1e-5)
    assert torch.allclose(cache.committed_conv_buffer, full.conv_buffer, atol=1e-6)


@torch.no_grad()
def test_conv_buffer_contents():
    m = mk()
    xs = [rnd(1, 1, DIM, seed=200 + t) for t in range(5)]
    pre = [torch.cat([m.q_proj(x), m.k_proj(x), m.v_proj(x)], -1) for x in xs]
    c2 = commit_tokens(m, xs[:2]).committed_conv_buffer
    assert c2.shape == (1, K1, C)
    assert torch.equal(c2[:, 0], torch.zeros(1, C))              # zero-initialised history
    assert torch.equal(c2[:, 1:], torch.cat(pre[:2], 1))
    c5 = commit_tokens(m, xs).committed_conv_buffer
    assert torch.equal(c5, torch.cat(pre[2:5], 1))               # last 3 pre-conv inputs


@torch.no_grad()
def test_inflight_purity_and_no_aliasing():
    m = mk()
    cache = commit_tokens(m, [rnd(1, 1, DIM, seed=300 + t) for t in range(4)])
    S_snap, B_snap = cache.committed_state.clone(), cache.committed_conv_buffer.clone()
    v0 = cache.version
    st, cv = cache.get_committed_context()
    for i in range(5):
        x = rnd(1, 1, DIM, seed=400 + i)
        step = m.forward_ephemeral(x, state=st, conv_buffer=cv)
        y, S = m.forward(x, state=st, conv_buffer=cv)
        S2, b2 = m.commit(x, state=st, conv_buffer=cv)            # pure compute, no write
        for t in (step.state, S, S2):
            assert t.data_ptr() != st.data_ptr()
        for t in (step.conv_buffer, b2):
            assert t.data_ptr() != cv.data_ptr()
    assert cache.version == v0
    assert torch.equal(cache.committed_state, S_snap)
    assert torch.equal(cache.committed_conv_buffer, B_snap)


@torch.no_grad()
def test_committed_context_influences_output():
    m = mk()
    cache = commit_tokens(m, [rnd(1, 1, DIM, seed=500 + t) for t in range(4)])
    x = rnd(1, 1, DIM, seed=9)
    st, cv = cache.get_committed_context()
    y_both = m.forward_ephemeral(x, state=st, conv_buffer=cv).y
    y_none = m.forward_ephemeral(x).y
    y_state = m.forward_ephemeral(x, state=st).y
    y_conv = m.forward_ephemeral(x, conv_buffer=cv).y
    assert not torch.allclose(y_both, y_none, atol=1e-5)
    assert not torch.allclose(y_state, y_none, atol=1e-5)      # S matters
    assert not torch.allclose(y_conv, y_none, atol=1e-5)       # conv buffer matters


@torch.no_grad()
def test_version_changes_only_on_explicit_commit():
    m = mk()
    cache = GDNStateCache()
    for i in range(3):
        x = rnd(1, 1, DIM, seed=600 + i)
        st, cv = cache.get_committed_context()
        m.forward_ephemeral(x, state=st, conv_buffer=cv)
        m.forward_ephemeral(x, state=st, conv_buffer=cv)       # REPEAT-style revisit
        S, buf = m.commit(x, state=st, conv_buffer=cv)
        assert cache.version == i
        cache.commit_terminal_context(S, buf)
        assert cache.version == i + 1


def test_all_parameters_receive_gradients():
    m = mk()
    m.train()
    u = rnd(2, 6, DIM)
    y, _ = m(u)
    y.pow(2).sum().backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
        assert p.grad.abs().sum() > 0, f"zero gradient: {n}"


@torch.no_grad()
def test_non_expansive_step_prop_7_1():
    torch.manual_seed(0)
    S = torch.randn(2, H, DV, DK)
    q = F.normalize(torch.randn(2, H, DK), dim=-1)
    k = F.normalize(torch.randn(2, H, DK), dim=-1)
    v = torch.randn(2, H, DV)
    alpha, beta = torch.rand(2, H) * 0.9 + 0.05, torch.rand(2, H)
    S2, _ = GDNv3._step(S, q, k, v, alpha, beta)
    lhs = torch.linalg.matrix_norm(S2, ord=2)
    rhs = alpha * torch.linalg.matrix_norm(S, ord=2) + beta * v.norm(dim=-1)
    assert (lhs <= rhs + 1e-5).all()


# ------------------------------------------------------------ block level
def mkblock(seed=0):
    torch.manual_seed(seed)
    b = AGTBlock(dim=64, species="GDN").eval()
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        b.anchor_gate.copy_(torch.randn(64, generator=g) * 0.5)
        b.memory_gate.copy_(torch.randn(64, generator=g) * 0.5)
    return b


@torch.no_grad()
def test_block_uses_v3_core_and_commit_semantics():
    b = mkblock()
    assert isinstance(b.core, GDNv3)
    h0, R = rnd(1, 1, 64, seed=2), rnd(1, 1, 64, seed=3)
    cache = GDNStateCache()
    for t in range(4):
        ht = rnd(1, 1, 64, seed=10 + t)
        S, buf = b.gdn_commit_entry(ht, h0=h0, R_c=R, C_prev=cache)
        assert cache.version == t                                # pure
        cache.commit_terminal_context(S, buf)
        assert cache.version == t + 1
    assert S.shape == (1, 6, 128, 64) and buf.shape == (1, 3, 1536)

    S_snap, B_snap, v0 = cache.committed_state.clone(), cache.committed_conv_buffer.clone(), cache.version
    x = rnd(1, 1, 64, seed=50)
    y1, s1 = b(x, h0=h0, R=R, C_prev=cache)
    y2, s2 = b(x, h0=h0, R=R, C_prev=cache)                      # revisit
    assert torch.equal(y1, y2) and torch.equal(s1, s2)
    assert cache.version == v0
    assert torch.equal(cache.committed_state, S_snap)
    assert torch.equal(cache.committed_conv_buffer, B_snap)

    # I4: commit entry == in-flight revisit at the terminal state
    S_c, buf_c = b.gdn_commit_entry(x, h0=h0, R_c=R, C_prev=cache)
    assert torch.allclose(s1, S_c, atol=1e-6)
    u = b.norm(x + b.injection(x, h0, R))
    step = b.core.forward_ephemeral(u, state=S_snap, conv_buffer=B_snap)
    assert torch.allclose(step.conv_buffer, buf_c, atol=1e-6)
    assert torch.allclose(step.state, S_c, atol=1e-6)

    # unvisited block (empty committed context) still produces an entry
    S_n, buf_n = b.gdn_commit_entry(x, h0=h0, R_c=R, C_prev=None)
    assert S_n.shape == S_c.shape and buf_n.shape == buf_c.shape

    # injection reaches the commit entry
    S_noinj, _ = b.gdn_commit_entry(x, h0=None, R_c=None, C_prev=cache)
    assert not torch.allclose(S_noinj, S_c, atol=1e-6)


def test_block_commit_guards():
    with pytest.raises(ValueError):
        AGTBlock(dim=64, species="SWA", swa_window=4, max_seq_len=16).gdn_commit_entry(rnd(1, 1, 64))
    with pytest.raises(TypeError):
        mkblock().gdn_commit_entry(rnd(1, 1, 64), C_prev=KVCache())


@torch.no_grad()
def test_forward_compat_two_tuple():
    out = mk()(rnd(1, 1, DIM))
    assert isinstance(out, tuple) and len(out) == 2
    assert out[1].shape == (1, H, DV, DK)
