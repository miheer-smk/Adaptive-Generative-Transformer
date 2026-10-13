"""Action-history measurements accept complete trajectories and supplied prefixes."""
import pytest

from experiments.stage3_measurements import route_metrics


CANONICAL = [('ADVANCE', i) for i in range(8)]
ALTERNATING = [('ADVANCE', 0), ('RESTART', None)] * 7 + [('ADVANCE', 0)]


def test_canonical_history_preserves_existing_metrics():
    result = route_metrics([CANONICAL])
    assert result['tokens'] == result['exact_tokens'] == 1
    assert result['decisions'] == result['selected_block_calls'] == 8
    assert result['executions'] == 8
    assert result['restarts'] == result['repeats'] == 0
    assert result['exact_route_fraction'] == 1
    assert result['restart_rate'] == result['repeat_rate'] == 0
    assert result['selected_block_calls_per_token'] == result['unique_blocks_per_token'] == 8
    assert result['incorrect_decisions'] == result['illegal_decisions'] == 0
    assert result['first_error_decision'] == {}
    assert result['route_histogram'] == {' | '.join(map(str, CANONICAL)): 1}
    assert result['block_execution_counts'] == [1] * 8
    assert result['block_execution_frequencies'] == [1/8] * 8
    assert result['per_decision_observations'] == [1] * 8
    for decision in range(8):
        assert result['per_decision_actions'][decision] == {str(CANONICAL[decision]): 1}
        assert result['confusion'][decision][decision] == 1
        assert sum(result['confusion'][decision]) == 1


def test_fifteen_decision_restart_route_counts_eight_executions():
    result = route_metrics([ALTERNATING])
    assert result['decisions'] == 15
    assert result['executions'] == result['selected_block_calls'] == 8
    assert result['restarts'] == 7 and result['repeats'] == 0
    assert result['restart_rate'] == 7/15
    assert result['selected_block_calls_per_token'] == 8
    assert result['unique_blocks_per_token'] == 1
    assert result['exact_tokens'] == result['illegal_decisions'] == 0
    assert len(result['confusion']) == len(result['per_decision_actions']) == 15
    assert result['block_execution_counts'] == [8] + [0] * 7
    assert result['block_execution_frequencies'] == [1.] + [0.] * 7
    assert sum(sum(row) for row in result['confusion']) == 15


@pytest.mark.parametrize('prefix', [CANONICAL[:3], ALTERNATING[:8]])
def test_supplied_prefix_is_analyzed_without_claiming_canonical_completion(prefix):
    result = route_metrics([prefix])
    assert result['decisions'] == len(prefix)
    assert result['executions'] == sum(kind in ('ADVANCE', 'REPEAT') for kind, _ in prefix)
    assert result['exact_tokens'] == 0
    assert len(result['confusion']) == len(prefix)
    assert result['route_histogram'] == {' | '.join(map(str, prefix)): 1}
    assert result['illegal_decisions'] == 0


def test_mixed_history_lengths_do_not_fabricate_decisions():
    result = route_metrics([CANONICAL, ALTERNATING, CANONICAL[:3]])
    assert result['tokens'] == 3 and result['exact_tokens'] == 1
    assert result['decisions'] == 26 and result['executions'] == 19
    assert result['per_decision_observations'] == [3] * 3 + [2] * 5 + [1] * 7
    assert sum(result['per_decision_observations']) == result['decisions']
    assert sum(sum(row) for row in result['confusion']) == 26
    assert sum(sum(row.values()) for row in result['per_decision_actions']) == 26
    assert result['restart_rate'] == 7/26
    assert result['selected_block_calls_per_token'] == 19/3
    assert result['exact_route_fraction'] == 1/3
    assert sum(result['route_histogram'].values()) == 3
    assert result['block_execution_counts'] == [10, 2, 2, 1, 1, 1, 1, 1]
    assert result['block_execution_frequencies'] == [n/19 for n in result['block_execution_counts']]


def test_repeat_and_restart_rates_keep_decision_denominator_and_block_use_execution_denominator():
    history = [('ADVANCE', 4), ('REPEAT', 4), ('RESTART', None), ('ADVANCE', 1)]
    result = route_metrics([history])
    assert result['decisions'] == 4 and result['executions'] == 3
    assert result['restarts'] == result['repeats'] == 1
    assert result['restart_rate'] == result['repeat_rate'] == 1/4
    assert result['block_execution_counts'] == [0, 1, 0, 0, 2, 0, 0, 0]
    assert result['block_execution_frequencies'][4] == 2/3
    assert result['unique_blocks_per_token'] == 2
    assert result['illegal_decisions'] == 0


def test_legality_reconstruction_passes_actual_execution_count(monkeypatch):
    import experiments.stage3_measurements as measurements
    original = measurements.build_execution_mask
    counts = []
    def observed(pointer, execution_count, restarted, num_blocks):
        counts.append(execution_count)
        return original(pointer, execution_count, restarted, num_blocks)
    monkeypatch.setattr(measurements, 'build_execution_mask', observed)
    route_metrics([ALTERNATING])
    assert counts == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]


def test_empty_histories_do_not_fabricate_rows_or_canonical_matches():
    for histories in ([], [[]]):
        result = route_metrics(histories)
        assert result['decisions'] == result['executions'] == result['exact_tokens'] == 0
        assert result['confusion'] == result['per_decision_actions'] == []
        assert result['per_decision_observations'] == []
        assert result['block_execution_counts'] == [0] * 8
        assert result['block_execution_frequencies'] == [0.] * 8
