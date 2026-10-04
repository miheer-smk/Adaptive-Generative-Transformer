from types import SimpleNamespace

import torch

from experiments import run_stage3_controlled_audit as audit


def test_paired_metrics_detect_route_errors_despite_correct_task_output(monkeypatch):
    canonical = [("ADVANCE", i) for i in range(8)]
    error = canonical.copy()
    error[3] = ("RESTART", None)
    logits = torch.tensor([[4.0, -4.0]])
    monkeypatch.setattr(audit, "run_fixed_sequence", lambda model, x: logits)
    monkeypatch.setattr(audit, "run_diff_sequence",
                        lambda model, x, tau, training: (logits, [canonical, error]))
    batch = SimpleNamespace(x=torch.zeros(1, 2, 4), target=torch.tensor([0]))
    result = audit.evaluate_cases(None, {"val": [batch]})["val"]
    assert result["accuracy"] == result["fixed_accuracy"] == 1.0
    assert result["max_logit_delta"] == 0.0
    assert result["tokens"] == 2
    assert result["exact_route_tokens"] == 1
    assert result["decision_accuracy"] == [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]
    assert result["actions"]["('RESTART', None)"] == 1
    assert sum(result["actions"].values()) == 16
