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
    assert first["routing"]["decisions"] == 64
    assert first["actual_candidate_block_calls"] == 9*64
    assert first["routing"]["illegal_decisions"] == 0
    assert all(torch.equal(before[name],value) for name,value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
