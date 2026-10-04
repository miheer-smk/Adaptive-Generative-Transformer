from types import SimpleNamespace

import pytest
import torch

from experiments.diagnose_stage3_gradients import ScriptedSweep
from experiments.run_stage3_replication import main, measured_evaluation, summarize
from src.agt import AGTv3
from experiments.run_stage3_taskA_controller import main as pilot_main


def test_compute_measurement_counts_only_inference_selected_blocks():
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=2)
    model.executor.max_execution_depth = 8
    model.executor.controller = ScriptedSweep([0,9,0,1,2,3,4,5])
    batch = SimpleNamespace(x=torch.randn(1,1,64), target=torch.tensor([0]))
    result = measured_evaluation(model, {"val": [batch]})["val"]
    assert result["actual_executor_block_calls"] == 7
    assert result["routing"]["decisions"] == 8
    assert result["routing"]["selected_block_calls"] == 7


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
