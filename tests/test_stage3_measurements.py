import torch

from experiments.diagnose_stage3_gradients import gradient_trial, check_observation_alignment, metadata_credit_probe, metadata_dependency_profile
from experiments.stage3_measurements import route_metrics
from src.agt import AGTv3


def test_route_metrics_distinguish_routing_budget_from_block_compute():
    canonical = [("ADVANCE", i) for i in range(8)]
    restarted = [("ADVANCE", 0), ("RESTART", None)] + [("ADVANCE", i) for i in range(6)]
    result = route_metrics([canonical, restarted])
    assert result["decisions"] == 16
    assert result["selected_block_calls"] == 15
    assert result["exact_tokens"] == 1
    assert result["illegal_decisions"] == 0
    assert result["restart_rate"] == 1/16
    assert result["first_error_decision"] == {"1": 1}
    assert result["confusion"][1][9] == 1


def test_real_gradient_diagnostic_preserves_forward_and_mask_contract():
    torch.manual_seed(12)
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=2)
    model.executor.max_execution_depth = 8
    x = torch.randn(1, 1, 64)
    trial, vector = gradient_trial(model, x, torch.tensor([0]), noise_seed=45)
    assert torch.isfinite(vector).all() and vector.norm() > 0
    assert len(trial["steps"]) == 8
    for step in trial["steps"]:
        assert step["hard_forward_exact"]
        assert step["illegal_forward_mass"] == 0
        assert step["illegal_logit_gradient_max"] == 0
        assert step["halt_logit_gradient_max"] == 0
        assert step["successor_count"] == 10
        assert step["selected_successor_gradient"]["finite"]
        assert step["unselected_successor_gradient"]["l2"] == 0
    repeat, repeated_vector = gradient_trial(model, x, torch.tensor([0]), noise_seed=45)
    assert trial["routes"] == repeat["routes"]
    torch.testing.assert_close(vector, repeated_vector, rtol=0, atol=0)


def test_teacher_observation_tuple_matches_real_executor(monkeypatch):
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=2)
    model.executor.max_execution_depth = 8
    calls = []

    def canonical(**kwargs):
        logits = kwargs["h"].new_full((1, 11), -100.0)
        logits[:, len(calls) % 8] = 100.0
        calls.append(1)
        return {"logits": logits}

    monkeypatch.setattr(model.controller, "forward", canonical)
    result = check_observation_alignment(model, torch.randn(1, 2, 64))
    assert result["rows"] == 16
    assert result["exact_alignment"]
    assert result["routes"]["exact_tokens"] == 2
    assert result["nominal_teacher_trajectory_probability"] == 1.0


def test_metadata_credit_blindspot_has_causal_forward_effect_and_zero_gradient():
    original = torch.rand_like
    result = metadata_credit_probe()["results"]
    assert torch.rand_like is original
    assert result[0]["loss"] == 1.0
    assert result[1]["loss"] == 10000.0
    assert result[0]["history"][1] == ("REPEAT", 6)
    assert result[1]["history"][1] == ("RESTART", None)
    assert result[0]["branch_gradient"] == result[1]["branch_gradient"] == [0.0,0.0]
    for row in result:
        reference = row["expected_categorical_gradient_resampling_branch_only"]
        assert reference[0] < -1000 and reference[1] > 1000


def test_real_metadata_profile_preserves_weights_and_commit_versions():
    model = AGTv3(dim=64,num_memory_slots=16,vocab_size=2)
    before = {name:value.clone() for name,value in model.state_dict().items()}
    result = metadata_dependency_profile(model,torch.randn(1,2,64))
    assert result["observations"] == 12
    assert result["context_versions_unchanged"]
    assert all(row["future_decisions"] > 0 for row in result["rows"])
    assert all(row["mask_symmetric_difference"] >= 3 for row in result["rows"])
    assert all(torch.equal(before[name],value) for name,value in model.state_dict().items())
