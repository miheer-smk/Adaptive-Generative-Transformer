import torch

from data.task_a import TaskAParity
from experiments.run_stage3_variance_ablation import averaged_gradients
from src.agt import AGTv3


def test_gradient_averaging_matches_independent_draws_without_updating_weights():
    model = AGTv3(dim=64,num_memory_slots=16,vocab_size=2)
    model.executor.max_execution_depth = 8
    batch = TaskAParity(64).make_fixed_set("val",4,1)
    before = {name:value.clone() for name,value in model.state_dict().items()}
    first,_ = averaged_gradients(model,batch,[7])
    second,_ = averaged_gradients(model,batch,[8])
    mean,record = averaged_gradients(model,batch,[7,8])
    torch.testing.assert_close(mean,(first+second)/2,rtol=0,atol=0)
    assigned = torch.cat([p.grad.reshape(-1) for p in model.controller.parameters()])
    torch.testing.assert_close(assigned,mean,rtol=0,atol=0)
    assert len(record["samples"]) == 2
    assert all(torch.equal(before[name],value) for name,value in model.state_dict().items())
