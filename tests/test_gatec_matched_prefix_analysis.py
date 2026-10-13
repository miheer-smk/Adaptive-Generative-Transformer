"""Array-only fixtures: no model, backward or experimental noise."""
import torch
from experiments.analyze_gatec_matched_prefix_4700 import rare_group,pair_contrasts,summarize_tapes,hard_and_relaxed,local_credit

def indices():return torch.arange(8).expand(2000,8).clone()

def test_coverage_group_is_probability_selected_and_ties_use_action_ID():
    pi=torch.tensor([.004,.004,.002,.1,.2,.2,.2,.29],dtype=torch.float64)
    group,order,mass=rare_group(pi)
    assert order[:3]==[2,0,1] and group==[2,0,1] and mass==.01
    assert rare_group(torch.full((8,),.125))[0]==[]

def test_state_effect_fixes_pi_B_and_does_not_use_teacher_distribution():
    B=dict(pi=torch.tensor([.8,.2]+[0.]*6,dtype=torch.float64),R=torch.zeros(8,8),q=torch.full((8,8),2.))
    C=dict(pi=torch.tensor([.2,.8]+[0.]*6,dtype=torch.float64),R=torch.tensor([[1.,0.]+[0.]*6]*8),q=torch.ones(8,8))
    for state in [B,C]:
        state.update(R=state['R'].double(),q=state['q'].double(),gs=torch.zeros(8,8,dtype=torch.float64),z=torch.full((8,8),.125,dtype=torch.float64))
    result=pair_contrasts(B,C,indices(),indices())
    assert abs(result['contrasts']['state_parity_C_minus_B_under_pi_B']['mean']-.8)<1e-12
    assert abs(result['contrasts']['state_parity_C_minus_B_under_pi_C']['mean']-.2)<1e-12
    assert result['contrasts']['state_CE_C_minus_B_under_pi_B']['mean']==-1

def test_joint_prefix_tape_pairing_keeps_identical_binary_outcomes_zero():
    r=torch.eye(8,dtype=torch.float64)
    state=dict(pi=torch.full((8,),.125,dtype=torch.float64),R=r,q=1-r,gs=r,z=r)
    result=pair_contrasts(state,state,indices(),indices())
    c=result['contrasts']['state_parity_C_minus_B_under_pi_B']
    assert c['mean']==0 and c['bootstrap_95']==[0,0] and c['resolved_sign']=='unresolved'
    assert result['binary_discordance']['zero_discordance_flag']

def test_half_bank_sign_disagreement_prevents_resolved_claim():
    summary=summarize_tapes([1]*4+[-.1]*4,indices())
    assert summary['bootstrap_95'][0]>0 and summary['resolved_sign']=='unresolved'

def test_sampler_reconstruction_hard_probability_has_no_tau_scaling():
    raw=[float(i)/5 for i in range(11)]
    u=torch.linspace(.1,.9,10,dtype=torch.float32)
    pi,lp,z,selected=hard_and_relaxed(raw,u,[True]*8+[False]*2,2.)
    torch.testing.assert_close(pi,torch.tensor(raw[:8],dtype=torch.float64).softmax(-1))
    assert not torch.allclose(pi,(torch.tensor(raw[:8],dtype=torch.float64)/2).softmax(-1))
    assert pi.gt(0).all() and lp.isfinite().all() and z[8:].eq(0).all() and selected<8

def test_CE_credit_can_improve_loss_while_parity_credit_is_opposed():
    pi=torch.tensor([.5,.5],dtype=torch.float64)
    R=torch.tensor([1.,0.],dtype=torch.float64);Q=torch.tensor([2.,1.],dtype=torch.float64)
    bpar,gamma,ce=local_credit(pi,R,Q,torch.tensor([1.,-1.],dtype=torch.float64))
    torch.testing.assert_close(bpar,torch.tensor([.25,-.25],dtype=torch.float64))
    assert gamma==-.5 and ce==.5

def test_constant_parity_values_have_exactly_zero_credit_and_allocation_effect():
    pi=torch.arange(1,9,dtype=torch.float64).softmax(-1)
    b,g,_=local_credit(pi,torch.ones(8,dtype=torch.float64),torch.arange(8,dtype=torch.float64),torch.ones(8,dtype=torch.float64))
    assert b.eq(0).all() and g==0
    B=dict(pi=pi,R=torch.ones(8,8,dtype=torch.float64),q=torch.ones(8,8,dtype=torch.float64),
        gs=torch.ones(8,8,dtype=torch.float64),z=torch.full((8,8),.125,dtype=torch.float64))
    C=dict(B,pi=pi.flip(0))
    result=pair_contrasts(B,C,indices(),indices())
    assert result['contrasts']['within_B_parity_pi_C_minus_pi_B']['mean']==0
