import pytest
import torch

from data.task_a import TaskAParity
from experiments.evaluate_stage3_stochastic_continuation import assess
from src.agt import AGTv3


def test_paired_stochastic_assessment_replays_without_parameter_updates():
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=2)
    model.executor.max_execution_depth = 8
    before = {name:value.clone() for name,value in model.state_dict().items()}
    batch = TaskAParity(64).make_fixed_set("val",4,1)
    first = assess(model,[batch],noise_seed=40,repeats=2)
    second = assess(model,[batch],noise_seed=40,repeats=2)
    assert first == second
    assert first["trajectories"] == 2
    decisions = sum(sum(row['token_decision_counts']) for row in first['rows'])
    assert first['routing']['decisions'] == decisions
    assert first['routing']['executions'] == 8 * batch.length * first['trajectories']
    for row in first['rows']:
        assert len(row['histories']) == batch.length
        assert row['token_decision_counts'] == list(map(len, row['histories']))
        assert row['token_execution_counts'] == [8] * batch.length
        for history, execution in zip(row['histories'], row['token_execution_counts']):
            restarts = sum(k == 'RESTART' for k, _ in history)
            assert len(history) == execution + restarts
            assert execution <= len(history) <= 2 * execution - 1
        assert row['actual_candidate_block_calls'] == 9 * sum(row['token_decision_counts'])
    assert first['actual_candidate_block_calls'] == 9 * decisions
    assert first["routing"]["illegal_decisions"] == 0
    assert all(torch.equal(before[name],value) for name,value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize('restart_second_token', [False, True])
def test_stochastic_assessment_retains_canonical_and_restart_token_boundaries(restart_second_token):
    torch.manual_seed(3401)
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=2)
    model.executor.max_execution_depth = 8
    canonical = [('ADVANCE', i) for i in range(8)]
    alternating = [('ADVANCE', 0), ('RESTART', None)] * 7 + [('ADVANCE', 0)]
    histories = [canonical, alternating if restart_second_token else canonical, canonical, canonical]
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
    batch = TaskAParity(64).make_fixed_set('val', 4, 1)
    first = assess(model, [batch], noise_seed=40, repeats=2)
    second = assess(model, [batch], noise_seed=40, repeats=2)
    assert first == second
    assert [row['noise_seed'] for row in first['rows']] == [40, 41]
    assert all(row['histories'] == histories for row in first['rows'])
    assert all(row['token_execution_counts'] == [8] * len(histories) for row in first['rows'])
    assert first['routing']['decisions'] == 2 * sum(map(len, histories))
    assert first['actual_candidate_block_calls'] == 9 * first['routing']['decisions']
    assert first['routing']['illegal_decisions'] == 0
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert all(parameter.grad is None for parameter in model.parameters())
