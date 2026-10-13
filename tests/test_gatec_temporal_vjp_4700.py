"""Synthetic graph/path-isolation tests; no Task-A replays or training."""
from types import SimpleNamespace
import pytest
import torch
from experiments.diagnose_gatec_temporal_vjp_4700 import ControllerOutputVJPs,restore_state,vector_variance,status
from experiments.run_gatec_paired_task_maintenance import gpu_snapshot

def toy_graph(tokens):
    first=torch.tensor([.2,.1],dtype=torch.float64,requires_grad=True)
    h=(first*torch.tensor([1.,2.],dtype=torch.float64)).sum()
    logits=[first]
    for i in range(1,len(tokens)):
        # Direct execution contribution and a state-conditioned controller branch.
        out=torch.stack([h*(i+1),h*.25]);logits.append(out)
        h=h+out[0]*.15-out[1]*.3
    return h.square(),logits

def test_hooks_preserve_forward_and_keep_probe_differentiable_in_all_modes():
    tokens=[0,0,1,2]
    loss,logits=toy_graph(tokens)
    forward=loss.detach().clone()
    original=torch.autograd.grad(loss,logits[0],retain_graph=True)[0]
    values={}
    with ControllerOutputVJPs(logits,tokens,0) as gate:
        for mode in ['exec','token','next','full']:
            gate.set_mode(mode)
            values[mode]=torch.autograd.grad(loss,logits[0],retain_graph=True)[0]
            assert gate.allowed(0) and torch.equal(loss.detach(),forward)
        torch.testing.assert_close(values['full'],original,atol=0,rtol=0)
    assert values['exec'].norm()>0 and not torch.equal(values['exec'],values['full'])
    parts=[values['exec'],values['token']-values['exec'],values['next']-values['token'],values['full']-values['next']]
    torch.testing.assert_close(sum(parts),values['full'],atol=1e-12,rtol=1e-12)
    # Removing diagnostic hooks restores the unchanged derivative.
    torch.testing.assert_close(torch.autograd.grad(loss,logits[0])[0],original,atol=0,rtol=0)

def test_token_boundary_gating_uses_actual_token_ID_not_call_index():
    logits=[torch.ones(2,requires_grad=True) for _ in range(6)]
    tokens=[4,4,4,5,5,6]
    with ControllerOutputVJPs(logits,tokens,4) as gate:
        expected={'exec':[0],'token':[0,1,2],'next':[0,1,2,3,4],'full':list(range(6))}
        for mode,indices in expected.items():
            gate.set_mode(mode)
            assert [i for i in range(6) if gate.allowed(i)]==indices

def test_final_token_gates_are_structurally_identical_without_new_derivatives():
    logits=[torch.ones(2,requires_grad=True) for _ in range(5)]
    with ControllerOutputVJPs(logits,[31]*5,31) as gate:
        for mode in ['token','next','full']:
            gate.set_mode(mode);assert all(gate.allowed(i) for i in range(5))
        gate.set_mode('exec');assert not any(gate.allowed(i) for i in range(1,5))

def test_full_multi_target_VJP_matches_original_single_target_VJP():
    loss,logits=toy_graph([0,0,1])
    original=torch.autograd.grad(loss,logits[0],retain_graph=True)[0]
    with ControllerOutputVJPs(logits,[0,0,1],0) as gate:
        gate.set_mode('full')
        multi=torch.autograd.grad(loss,logits,retain_graph=True)
    torch.testing.assert_close(multi[0],original,atol=0,rtol=0)

def test_saved_state_restore_preserves_full_tensor_bytes_topology_and_metadata():
    source=SimpleNamespace(s=torch.arange(3.).view(1,3),memory=torch.arange(8.).view(1,2,4),
        ctx=SimpleNamespace(gdn={4:SimpleNamespace(committed_state=torch.ones(2,3),version=7)},swa_window=256),tokens_committed=7,_consumed=False)
    target=SimpleNamespace(s=torch.zeros(1,3),memory=torch.zeros(1,2,4),
        ctx=SimpleNamespace(gdn={4:SimpleNamespace(committed_state=None,version=0)},swa_window=256),tokens_committed=0,_consumed=False)
    saved=gpu_snapshot(source);before={k:v.clone() if torch.is_tensor(v) else v for k,v in saved.items()}
    restored=restore_state(target,saved,'cpu')
    assert restored.tokens_committed==7 and restored.ctx.gdn[4].version==7
    assert all(torch.equal(v,before[k]) if torch.is_tensor(v) else v==before[k] for k,v in saved.items())
    restored.s.add_(1);assert torch.equal(saved['/s'],source.s)
    with pytest.raises(AssertionError):restore_state(target,{k:v for k,v in saved.items() if k!='/tokens_committed'},'cpu')

def test_variance_covariance_identity_and_joint_component_resampling():
    exe=torch.arange(64,dtype=torch.float64).view(8,8)/20
    rest=-.3*exe+torch.arange(8,dtype=torch.float64).view(8,1).sin()
    full=exe+rest
    cov=((exe-exe.mean(0))*(rest-rest.mean(0))).sum()/7
    torch.testing.assert_close(vector_variance(full),vector_variance(exe)+vector_variance(rest)+2*cov)
    indices=torch.tensor([[0,0,3,1,2,4,7,6],[7,6,5,4,3,2,1,0]])
    torch.testing.assert_close(full[indices],exe[indices]+rest[indices])
    assert vector_variance(full[indices]).shape==(2,)

def test_connected_zero_small_and_disconnected_are_distinct():
    assert status(None)=='disconnected'
    assert status(None,False)=='fixed_tensor_no_autograd_input'
    assert status(torch.zeros(11))=='connected_exact_zero'
    assert status(torch.full((11,),1e-10))=='connected_small'
    assert status(torch.ones(11))=='connected_nonzero'
