import pytest
import torch
from src.agt import AGTv3
from experiments.diagnose_stage3_taskA_gradient_flow import capture, measured_example, matrix, route_sensitivity


def test_proxy_preserves_real_forward_noise_and_exposes_boundary():
    torch.manual_seed(1)
    model = AGTv3(dim=64,num_memory_slots=16,vocab_size=2)
    model.executor.max_execution_depth = 8
    from experiments.run_stage3_taskA_controller import run_diff_sequence
    x = torch.randn(1,2,64)
    torch.manual_seed(5)
    expected, route = run_diff_sequence(model,x,2.0,True)
    torch.manual_seed(5)
    with capture(model) as proxy:
        actual, captured_route = run_diff_sequence(model,x,2.0,True)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert captured_route == route
    assert len(proxy.rows) == sum(map(len, route))
    assert [(r['token'], r['decision']) for r in proxy.rows] == [
        (t, d) for t, history in enumerate(route) for d in range(len(history))]
    assert [r['history'] for r in proxy.tokens] == route
    assert all(r['execution_count'] == 8 for r in proxy.tokens)
    assert sum(r['terminal_decision'] for r in proxy.rows) == len(route)
    assert all(not r['next_pointer_requires_grad'] and not r['next_pointer_grad_fn'] for r in proxy.rows)
    actual.sum().backward()
    assert all(r['logits'].grad is not None for r in proxy.rows)
    assert all(r['logits'].grad[0,10] == 0 for r in proxy.rows)


def test_scripted_replacements_have_one_difference_and_legal_continuation(monkeypatch):
    """One prescribed difference within D=8 prefixes, never fallback completion."""
    from experiments.audit_gatec_canonical_replay import snapshot, unchanged
    from experiments.run_stage3_taskA_controller import run_fixed_sequence
    import src.agt as agt_module
    model = AGTv3(dim=64,num_memory_slots=16,vocab_size=2)
    model.executor.max_execution_depth=8
    x=torch.randn(1,8,64)
    with torch.no_grad():
        reference = run_fixed_sequence(model, x)
    weights = snapshot(model.state_dict())
    commits = []
    original_commit = agt_module.terminal_commit
    def reference_commit(state, **kwargs):
        assert state.tokens_committed < 7, 'Observed token prefix performed terminal commit'
        commits.append(state.tokens_committed)
        return original_commit(state, **kwargs)
    monkeypatch.setattr(agt_module, 'terminal_commit', reference_commit)
    monkeypatch.setattr(model.executor, 'execute_differentiable',
                        lambda *a, **k: pytest.fail('Prefix requested production completion'))
    original_controller = model.executor.controller
    results = route_sensitivity(model, x, torch.tensor([0]))
    assert model.executor.controller is original_controller
    assert commits == list(range(7))  # Only preceding completed reference tokens.
    torch.testing.assert_close(torch.tensor(results[0]['prefix_readout_logits']), reference, atol=0, rtol=0)
    assert results[0]['prefix_readout_ce'] == torch.nn.functional.cross_entropy(reference, torch.tensor([0])).item()
    assert results[0]['prefix_readout_prediction'] == int(reference.argmax(-1))
    for result, action in zip(results[1:], [8, 9]):
        histories = result['histories']
        differences=[(t,d) for t,h in enumerate(histories) for d,a in enumerate(h) if a != ('ADVANCE',d)]
        assert differences == [(7,4)]
        assert histories[7][5:] == [('ADVANCE',5),('ADVANCE',6),('ADVANCE',7)]
        assert result['prefix_decision_count'] == 8
        assert result['prefix_execution_count'] == (8 if action == 8 else 7)
        assert not result['prefix_committed'] and result['termination_reason'] == 'observation_horizon'
        assert len(histories) == 8 and all(len(history) == 8 for history in histories)
        rows = result['prefix_rows']
        assert [(r['token'], r['decision']) for r in rows] == [(7, d) for d in range(8)]
        assert all(r['selected_action'] in r['legal_actions'] for r in rows)
        assert rows[4]['execution_before'] == 4
        assert rows[4]['execution_after'] == (5 if action == 8 else 4)
        assert rows[5]['just_restarted_before'] == (action == 9)
        assert result['additional_blocks_in_token'] == 3
    assert unchanged(weights, snapshot(model.state_dict()))


@pytest.mark.parametrize('restart_second_token', [False, True])
def test_proxy_and_gradient_tables_keep_actual_observed_decisions(restart_second_token):
    from experiments.diagnose_stage3_taskA_gradient_flow import aggregate, markdown
    from experiments.run_stage3_taskA_controller import run_diff_sequence
    torch.manual_seed(3401)
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=2)
    model.executor.max_execution_depth = 8
    canonical = [('ADVANCE', i) for i in range(8)]
    alternating = [('ADVANCE', 0), ('RESTART', None)] * 7 + [('ADVANCE', 0)]
    histories = [canonical, alternating if restart_second_token else canonical]
    actions = [block if kind == 'ADVANCE' else 9 for history in histories for kind, block in history]
    calls = []
    def observed_policy(**kwargs):
        output = model.controller(**kwargs)
        bias = output['logits'].new_full((1, 11), -100.)
        bias[:, actions[len(calls) % len(actions)]] = 100.
        calls.append(1)
        return dict(output, logits=output['logits'] + bias)
    model.executor.controller = observed_policy
    before = {name: value.clone() for name, value in model.state_dict().items()}
    x = torch.randn(1, 2, 64)
    original_execute = model.executor.execute_differentiable
    torch.manual_seed(5)
    expected, expected_histories = run_diff_sequence(model, x, 2., True)
    torch.manual_seed(5)
    with capture(model) as proxy:
        actual, actual_histories = run_diff_sequence(model, x, 2., True)
    assert model.executor.execute_differentiable == original_execute
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual_histories == expected_histories == histories
    assert [t['decision_count'] for t in proxy.tokens] == list(map(len, histories))
    assert [(r['token'], r['decision']) for r in proxy.rows] == [
        (t, d) for t, history in enumerate(histories) for d in range(len(history))]
    example = measured_example(model, x, torch.tensor([0]), 0)
    assert example['histories'] == histories
    rows = example['decisions']
    width = max(map(len, histories))
    values = matrix(rows, 'gradient_l2')
    assert len(values) == 2 and all(len(row) == width for row in values)
    assert all(value is not None for value in values[1])
    assert all(value is None for value in values[0][len(canonical):])
    statistics = aggregate(rows)
    assert sum(d['observations'] for d in statistics['by_decision']) == len(rows)
    assert [d['decision'] for d in statistics['by_decision']] == list(range(width))
    assert [d['observations'] for d in statistics['by_decision']] == [sum(len(h) > d for h in histories) for d in range(width)]
    assert statistics['terminal_decision_observations'] == 2
    assert statistics['halt_exact_zero_count'] == len(rows)
    assert all(r['execution_after'] == 8 for r in rows if r['terminal_decision'])
    text = markdown({'aggregate': statistics, 'matrices': {'gradient_l2': values}, 'counterfactuals': []})
    assert 'eight block executions/token with variable decision counts' in text
    assert 'final observed decision of each token' in text
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
