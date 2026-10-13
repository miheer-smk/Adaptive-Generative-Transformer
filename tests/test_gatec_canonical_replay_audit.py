import torch
from experiments.audit_gatec_canonical_replay import snapshot,first_difference,Alternating,source_budget_replay
from src.executor import AdaptiveExecutor


def test_production_budget_matches_independent_restart_replay():
    class Block:
        def __call__(self,x,**kwargs):return x+1
    class Bank:
        blocks=[Block() for _ in range(8)]
        def __call__(self,index,x,**kwargs):return self.blocks[index](x,**kwargs)
    executor=AdaptiveExecutor(Bank(),Alternating(),max_execution_depth=8)
    h=torch.zeros(1,64);args=dict(h0=h,memory=h,context=None,memory_mean=torch.zeros(1,256),budget=torch.zeros(1,2))
    production=executor.execute_differentiable(**args,training=False)
    assert production.execution_count==8 and len(production.action_history)==15 and production.h_terminal[0,0]==8
    executor.controller=Alternating()
    corrected=source_budget_replay(executor,**args)
    assert corrected.execution_count==8 and len(corrected.action_history)==15 and corrected.h_terminal[0,0]==8
    assert production.action_history == corrected.action_history
    assert torch.equal(production.h_terminal, corrected.h_terminal)


def test_first_divergence_uses_tolerances_and_exact_discrete_metadata():
    a=snapshot(dict(h=torch.ones(1,2),pointer=0));b=snapshot(dict(h=torch.ones(1,2)+1e-7,pointer=0))
    assert first_difference(a,b) is None
    b['/pointer']=1
    assert first_difference(a,b)['field']=='/pointer'


def test_learned_first_mismatch_distinguishes_equal_features_wrong_action(tmp_path):
    from src.agt import AGTv3
    from experiments.audit_gatec_canonical_replay import Trace,learned_replay
    model=AGTv3(dim=64,num_memory_slots=16,vocab_size=2)
    model.executor.max_execution_depth=8
    x=torch.randn(1,1,64);target=torch.zeros(1,1,dtype=torch.long)
    with torch.no_grad():reference=Trace(model,'A').run(x,target)
    before={k:v.clone() for k,v in model.state_dict().items()}
    synthetic={k:torch.zeros_like(v) for k,v in model.controller.state_dict().items()}
    synthetic['mlp.2.bias'][7]=1
    path=tmp_path/'synthetic_unit_controller.pt'
    torch.save({'controller_state_dict':synthetic},path)
    result=learned_replay(model,reference,x,target,path)
    assert result['first_policy_mismatch']['selected_action']==7
    assert result['first_policy_mismatch']['raw_input_first_difference'] is None
    assert result['first_policy_mismatch']['normalized_input_first_difference'] is None
    assert result['final_ce'] is None
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in before.items())


def test_purity_check_rejects_mutation_below_replay_tolerance():
    from experiments.audit_gatec_canonical_replay import unchanged
    a=snapshot(torch.ones(1,2));b=snapshot(torch.ones(1,2)+1e-7)
    assert first_difference(a,b) is None
    assert not unchanged(a,b)
