import os, sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gqa import GQA
from src.blocks import SlidingWindowAttention
from src.caches import KVCache, SWARingCache

DIM, HQ, HKV, W, L, MAXLEN = 64, 4, 2, 4, 12, 32   # L > W so the window binds


def make_swa(seed=0):
    torch.manual_seed(seed)
    return SlidingWindowAttention(
        dim=DIM, num_query_heads=HQ, num_kv_heads=HKV,
        window_size=W, max_seq_len=MAXLEN,
    ).eval()


def make_x(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, L, DIM, generator=g)


@torch.no_grad()
def ref_window_attention(m, x, window):
    # Independent mask/softmax math; shares only the verified projection helper.
    q, k, v = m._project_qkv_rope(x, 0)
    k, v = m._repeat_kv_heads(k, v)
    T = x.shape[1]
    scores = (q @ k.transpose(-1, -2)) / (m.head_dim ** 0.5)
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    allowed = ((i - j) >= 0) & ((i - j) < window)
    scores = scores.masked_fill(~allowed, float("-inf"))
    y = torch.softmax(scores, dim=-1) @ v
    y = y.transpose(1, 2).contiguous().view(1, T, -1)
    return m.o_proj(y)


@torch.no_grad()
def run_cached(m, x):
    cache = SWARingCache(window_size=W)
    outs = []
    for t in range(x.shape[1]):
        xt = x[:, t:t + 1]
        outs.append(m(xt, kv_cache=cache))
        k, v = m.project_kv(xt, cache.next_position)
        cache.commit_terminal_context(k, v)
    return torch.cat(outs, dim=1), cache


@torch.no_grad()
def test_nocache_matches_independent_reference():
    m, x = make_swa(), make_x()
    assert torch.allclose(m(x), ref_window_attention(m, x, W), atol=1e-5)


@torch.no_grad()
def test_window_actually_masks():
    m, x = make_swa(), make_x()
    gqa = GQA(dim=DIM, num_query_heads=HQ, num_kv_heads=HKV, max_seq_len=MAXLEN).eval()
    gqa.load_state_dict(m.state_dict())
    ys, yg = m(x), gqa(x)
    assert torch.allclose(ys[:, :W], yg[:, :W], atol=1e-5)       # inside window: same as causal
    assert not torch.allclose(ys[:, W:], yg[:, W:], atol=1e-4)   # beyond: must differ


@torch.no_grad()
def test_cached_equals_full_window_forward_and_visible_bound():
    m, x = make_swa(), make_x()
    full = m(x)
    cache = SWARingCache(window_size=W)
    outs = []
    for t in range(L):
        xt = x[:, t:t + 1]
        y1 = m(xt, kv_cache=cache)
        y2 = m(xt, kv_cache=cache)                 # repeated in-flight call
        assert torch.equal(y1, y2)
        outs.append(y1)
        k, v = m.project_kv(xt, cache.next_position)
        kc, _ = cache.make_ephemeral_context(k, v)
        assert kc.shape[2] == min(t, W - 1) + 1 <= W
        cache.commit_terminal_context(k, v)
    cached = torch.cat(outs, dim=1)
    err = (cached - full).abs().max().item()
    assert torch.allclose(cached, full, atol=1e-5), f"max abs diff {err}"


@torch.no_grad()
@pytest.mark.parametrize("cached", [False, True])
def test_window_boundary_and_future_invariance(cached):
    m, x = make_swa(), make_x()
    fwd = (lambda z: run_cached(m, z)[0]) if cached else m
    base = fwd(x)
    t = 9
    out_j, in_j = t - W, t - W + 1                 # 5 (outside), 6 (inside)
    x2 = x.clone(); x2[:, out_j] += 1.0
    assert torch.allclose(fwd(x2)[:, t], base[:, t], atol=1e-6)
    x3 = x.clone(); x3[:, in_j] += 1.0
    assert not torch.allclose(fwd(x3)[:, t], base[:, t], atol=1e-4)
    x4 = x.clone(); x4[:, t + 1:] += 1.0           # future must not matter
    assert torch.allclose(fwd(x4)[:, :t + 1], base[:, :t + 1], atol=1e-6)


@torch.no_grad()
def test_inflight_purity_with_varied_inputs():
    m, x = make_swa(), make_x()
    cache = SWARingCache(window_size=W)
    for t in range(6):
        xt = x[:, t:t + 1]
        k, v = m.project_kv(xt, cache.next_position)
        cache.commit_terminal_context(k, v)
    snap = (cache.version, cache.committed_length, cache.next_position,
            cache.committed_keys.clone(), cache.committed_values.clone())
    g = torch.Generator().manual_seed(7)
    for _ in range(5):
        m(torch.randn(1, 1, DIM, generator=g), kv_cache=cache)
    assert (cache.version, cache.committed_length, cache.next_position) == snap[:3]
    assert torch.equal(cache.committed_keys, snap[3])
    assert torch.equal(cache.committed_values, snap[4])


@torch.no_grad()
def test_cardinality_positions_and_recency():
    m, x = make_swa(), make_x()
    cache = SWARingCache(window_size=W)
    all_k = []
    for t in range(1, L + 1):
        xt = x[:, t - 1:t]
        k, v = m.project_kv(xt, cache.next_position)
        all_k.append(k)
        cache.commit_terminal_context(k, v)
        assert cache.committed_length == min(t, W - 1)
        assert cache.next_position == t and cache.version == t
        keep = torch.cat(all_k, dim=2)[:, :, -(W - 1):]
        assert torch.equal(cache.committed_keys, keep)


@torch.no_grad()
def test_reset_isolation():
    m, x = make_swa(), make_x()
    _, cache = run_cached(m, x)
    cache.clear()
    assert (cache.committed_length, cache.next_position, cache.version) == (0, 0, 0)
    fresh = m(x[:, :1], kv_cache=SWARingCache(window_size=W))
    assert torch.equal(m(x[:, :1], kv_cache=cache), fresh)


@torch.no_grad()
def test_guards():
    m, x = make_swa(), make_x()
    with pytest.raises(ValueError):
        SWARingCache(window_size=1)
    with pytest.raises(ValueError):
        m(x[:, :1], kv_cache=SWARingCache(window_size=W + 1))
    with pytest.raises(ValueError):
        m(x[:, :2], kv_cache=SWARingCache(window_size=W))      # T != 1
    with pytest.raises(TypeError):
        m(x[:, :1], kv_cache=KVCache())


@torch.no_grad()
def test_gqa_helper_consistency():
    # project_kv-committed K/V must reproduce GQA.forward (guards the duplication).
    torch.manual_seed(0)
    gqa = GQA(dim=DIM, num_query_heads=HQ, num_kv_heads=HKV, max_seq_len=MAXLEN).eval()
    x = make_x()
    full = gqa(x)
    cache = KVCache()
    outs = []
    for t in range(L):
        xt = x[:, t:t + 1]
        outs.append(gqa(xt, kv_cache=cache))
        k, v = gqa.project_kv(xt, cache.next_position)
        cache.commit_terminal_context(k, v)
    assert torch.allclose(torch.cat(outs, dim=1), full, atol=1e-5)
