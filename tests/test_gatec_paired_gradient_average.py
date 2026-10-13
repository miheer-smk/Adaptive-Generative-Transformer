"""Diagnostic-local preparation and gradient arithmetic; no Task-A training."""
import torch
from collections import Counter
from experiments.run_gatec_paired_gradient_average_4900 import (
    training_lanes,LaneNoise,average_gradients,gradient_pair,confirmation_panel,stratified_indices,paired_interval)

def test_noise_lanes_are_sealed_and_independent_of_call_order():
    rows=[dict(sequence_id='epoch01/a',length=4),dict(sequence_id='epoch01/b',length=5)]
    lanes=training_lanes(rows);again=training_lanes(rows)
    a=LaneNoise(lanes[0],0);b=LaneNoise(lanes[1],1)
    expected=a.get('epoch01/a',1,2).clone()
    a.get('epoch01/b',4,14);b.get('epoch01/a',0,0)
    assert torch.equal(expected,a.get('epoch01/a',1,2))
    assert torch.equal(expected,LaneNoise(again[0],0).get('epoch01/a',1,2))
    assert not torch.equal(expected,b.get('epoch01/a',1,2))

def test_duplicate_mean_is_the_original_gradient_not_sum():
    first={'a':torch.tensor([2.,-3.]),'b':torch.tensor([.5])}
    avg=average_gradients(first,first)
    assert all(torch.equal(v,first[k]) for k,v in avg.items())
    stats=gradient_pair(first,first)
    assert stats['difference_l2']==stats['max_absolute_difference']==0
    assert stats['averaged_l2']==stats['first_l2']

def test_gradient_mean_equals_mean_loss_derivative_with_parameters_fixed():
    p=torch.tensor([1.,2.],requires_grad=True)
    a=torch.autograd.grad((p**2).sum(),p)[0]
    b=torch.autograd.grad((p**3).sum(),p)[0]
    combined=torch.autograd.grad(((p**2).sum()+(p**3).sum())/2,p)[0]
    torch.testing.assert_close(average_gradients({'p':a},{'p':b})['p'],combined)
    assert torch.equal(p.detach(),torch.tensor([1.,2.]))

def test_noise_injection_returns_exact_lane_and_restores_global_sampler():
    noise=training_lanes([dict(sequence_id='a',length=4)])
    tape=LaneNoise(noise[0],0);original=torch.rand_like
    with tape.inject('a',0,3):
        assert torch.equal(torch.rand_like(torch.empty(1,10)),tape.get('a',0,3))
    assert torch.rand_like is original

def test_confirmation_panels_use_declared_seeds_balance_distinctness_and_exclusion():
    baseline,_=confirmation_panel(set());excluded={tuple(s['bits']) for s in baseline[:3]}
    rows,attempts=confirmation_panel(excluded)
    assert len(rows)==len({tuple(s['bits']) for s in rows})==96
    assert not ({tuple(s['bits']) for s in rows}&excluded)
    for L in range(16,33):
        quota=16 if L==16 else 2
        assert Counter(s['label'] for s in rows if s['length']==L)==Counter({0:quota,1:quota})
    assert rows==confirmation_panel(excluded)[0]
    assert any(s['reason']=='known_input_overlap' for s in attempts)

def test_stratified_bootstrap_preserves_each_length_parity_and_all_tapes_as_unit():
    rows=[dict(length=L,label=p) for L in [17,18] for p in [0,1] for _ in range(2)]
    idx=stratified_indices(rows,torch.Generator().manual_seed(4904))
    assert idx.shape==(10000,8)
    counts=Counter((s['length'],s['label']) for s in rows)
    for sample in idx[:20]:assert Counter((rows[i]['length'],rows[i]['label']) for i in sample.tolist())==counts
    a=torch.tensor([[0.,1.,0.,1.,0.,1.,0.,1.]]*4)
    summary=paired_interval(a.mean(0).tolist(),idx)
    assert summary['mean']==.5

def test_sequence_bootstrap_does_not_treat_four_tapes_as_independent_rows():
    rows=[dict(length=16,label=p) for p in [0,1] for _ in range(16)]
    idx=stratified_indices(rows,torch.Generator().manual_seed(4904))
    assert idx.shape==(10000,32)
    result=paired_interval([.25]*32,idx)
    assert result['mean']==.25 and result['bootstrap_95']==[.25,.25]
