"""Synthetic comparator/indexing checks; no Task-A model runs or training."""
import pytest
import torch
from experiments.diagnose_gatec_st_action_values_4700 import (
    banks,extend_prefix_noise,tape,categorical_gradient,compare_banks,force_advance_candidate,
)

def test_append_only_reconstruction_preserves_exact_ID_association():
    rows=[dict(sequence_id='eval/a',length=2),dict(sequence_id='eval/b',length=3)]
    extra=[dict(sequence_id='train/a',length=2),dict(sequence_id='train/b',length=2)]
    generator=torch.Generator().manual_seed(924088217)
    original={r['sequence_id']:torch.rand(r['length'],15,10,generator=generator) for r in rows}
    expected={r['sequence_id']:torch.rand(r['length'],15,10,generator=generator) for r in extra}
    before={k:v.clone() for k,v in original.items()}
    actual=extend_prefix_noise(rows,extra,original,924088217)
    assert list(actual)==['train/a','train/b']
    assert all(torch.equal(actual[k],v) for k,v in expected.items())
    assert all(torch.equal(original[k],v) for k,v in before.items())
    with pytest.raises(AssertionError):extend_prefix_noise(rows[::-1],extra,original,924088217)

def test_independent_banks_preserve_logical_coordinates_despite_route_lengths():
    rows=[dict(sequence_id='unit/a',length=2),dict(sequence_id='unit/b',length=3)]
    first,other=banks(rows,4800),banks(rows,4801)
    assert len(first)==len(other)==8
    noise=tape(first[0],4800);before=noise.get('unit/b',1,5).clone()
    for decision in range(15):noise.get('unit/a',0,decision)
    assert torch.equal(before,noise.get('unit/b',1,5))
    assert not torch.equal(first[0]['unit/a'],other[0]['unit/a'])
    assert all(torch.equal(a['unit/a'],b['unit/a']) for a,b in zip(first,banks(rows,4800)))

def test_categorical_benchmark_uses_hard_sampler_probability_not_relaxed_tau():
    logits=torch.arange(8,dtype=torch.float64)/3
    q=torch.arange(8,dtype=torch.float64).flip(0)
    pi,gradient=categorical_gradient(logits,q)
    assert torch.equal(pi,logits.softmax(-1))
    assert not torch.allclose(pi,(logits/2).softmax(-1))
    assert abs(float(gradient.sum()))<1e-12

def comparison(sign):
    logits=[.2,-.3,.8,.1,-.8,.5,-.2,.4]
    q=[[float(a)+float(k) for a in range(8)] for k in range(8)]
    pi,g=categorical_gradient(logits,torch.tensor(q,dtype=torch.float64).mean(0))
    grads=[(sign*g).tolist()+[0.,0.,0.] for _ in range(8)]
    generator=torch.Generator().manual_seed(4802)
    cf=torch.randint(0,8,(2000,8),generator=generator);st=torch.randint(0,8,(2000,8),generator=generator)
    return compare_banks(logits,q,grads,[[False]*8 for _ in range(8)],cf,st,[pi.tolist(),pi.tolist()])

def test_paired_bootstrap_and_half_bank_rule_resolve_correct_sign():
    aligned,opposed=comparison(1),comparison(-1)
    assert aligned['classification']=='informative_aligned_ST'
    assert opposed['classification']=='informative_opposed_ST'
    assert aligned['inner_product_bootstrap_95'][0]>0
    assert opposed['inner_product_bootstrap_95'][1]<0
    assert all(p['resolved'] for p in aligned['paired_action_loss_differences'])
    assert aligned['checkpoint_probability_shift']['value']==0

def test_force_changes_only_hard_first_ADVANCE_result_without_mutating_original():
    candidates=torch.arange(640,dtype=torch.float32).view(1,10,64)
    original=dict(candidates=candidates,action=torch.nn.functional.one_hot(torch.tensor([3]),10).float(),
                  h_next=candidates[:,3],next_pointer=torch.tensor([3]),selected_action=torch.tensor([3]),probs=torch.full((1,10),.1))
    forced=force_advance_candidate(original,5)
    assert torch.equal(forced['h_next'],candidates[:,5]) and int(forced['next_pointer'])==5
    assert int(original['selected_action'])==3
    assert forced['candidates'] is original['candidates'] and forced['probs'] is original['probs']
    assert torch.equal(forced['action'],torch.nn.functional.one_hot(torch.tensor([5]),10).float())
    with pytest.raises(AssertionError):force_advance_candidate(original,9)
