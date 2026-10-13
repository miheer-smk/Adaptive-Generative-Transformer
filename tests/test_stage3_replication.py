from types import SimpleNamespace

import pytest
import torch

from experiments.diagnose_stage3_gradients import ScriptedSweep
from experiments.run_stage3_replication import main, measured_evaluation, summarize
from src.agt import AGTv3
from experiments.run_stage3_taskA_controller import main as pilot_main


def test_compute_measurement_counts_only_inference_selected_blocks(monkeypatch):
    """Eight supplied decisions: seven selected bank executions, no commit."""
    from experiments.stage3_decision_prefix import observe_token_prefix
    from experiments.stage3_measurements import count_block_calls, route_metrics
    from experiments.audit_gatec_canonical_replay import snapshot, unchanged
    import src.agt as agt_module
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=2)
    model.executor.max_execution_depth = 8
    model.executor.controller = ScriptedSweep([0,9,0,1,2,3,4,5])
    batch = SimpleNamespace(x=torch.randn(1,1,64), target=torch.tensor([0]))
    state = model.initial_persistent_state(1, 'cpu')
    before = snapshot(state)
    selected = []
    handles = [block.register_forward_hook(lambda module, args, output, i=i: selected.append(i))
               for i, block in enumerate(model.blocks.blocks)]
    monkeypatch.setattr(agt_module, 'terminal_commit', lambda *a, **k: pytest.fail('Prefix committed'))
    monkeypatch.setattr(model.executor, 'candidate_successors',
                        lambda *a, **k: pytest.fail('Inference evaluated candidates'))
    try:
        with count_block_calls(model) as counter:
            prefix = observe_token_prefix(model, batch.x[:, 0], state, torch.zeros(1, 64),
                                           decision_horizon=8, training=False)
    finally:
        for handle in handles:
            handle.remove()
    result = dict(actual_executor_block_calls=counter['block_calls'],
                  routing=route_metrics([prefix.action_history]))
    assert result["actual_executor_block_calls"] == 7
    assert result["routing"]["decisions"] == 8
    assert result["routing"]["selected_block_calls"] == 7
    assert selected == [0, 0, 1, 2, 3, 4, 5]
    assert prefix.action_history == [('ADVANCE', 0), ('RESTART', None)] + [('ADVANCE', i) for i in range(6)]
    assert prefix.decision_count == 8 and prefix.execution_count == 7
    assert model.executor.controller.calls == 8
    assert not prefix.committed and prefix.termination_reason == 'observation_horizon'
    assert prefix.rows[1]['execution_before'] == prefix.rows[1]['execution_after'] == 1
    assert all(row['selected_action'] in row['legal_actions'] for row in prefix.rows)
    assert unchanged(before, snapshot(state))


def test_replication_summary_uses_sample_standard_deviation():
    runs = []
    for seed, accuracy in ((3305, 0.0), (3306, 0.5), (3307, 1.0)):
        routing = dict(exact_route_fraction=accuracy, repeat_rate=0., restart_rate=0.,
                       unique_blocks_per_token=8., selected_block_calls_per_token=8.)
        runs.append(dict(seed=seed, initialization="random", numerically_stable=True,
                         after={"val": dict(accuracy=accuracy, ce=1., routing=routing)}))
    result = summarize(runs)["random"]["val"]
    assert result["accuracy"]["mean"] == 0.5
    assert result["accuracy"]["sample_sd"] == 0.5
    assert result["perfect_task_seed_count"] == 1


def test_existing_run_is_rejected_before_training(tmp_path, monkeypatch):
    output = tmp_path / "prior.json"
    output.write_text("preserve me")
    monkeypatch.setattr("sys.argv", ["replication", "--warmstart-template", "unused_{seed}.pt",
                                     "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert output.read_text() == "preserve me"


def test_pilot_rejects_unsupported_batch_before_loading_checkpoint(monkeypatch):
    monkeypatch.setattr("sys.argv", ["pilot", "--batch-size", "2"])
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Checkpoint loaded before validation"))
    with pytest.raises(SystemExit) as error:
        pilot_main()
    assert error.value.code == 2
