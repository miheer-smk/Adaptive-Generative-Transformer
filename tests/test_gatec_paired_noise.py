"""Diagnostic noise coupling tests; no Task-A training or trajectory intervention."""
import torch
from src.routing import st_gumbel_action
from experiments.run_gatec_paired_task_maintenance import IndexedNoise, action_probability_statistics


def test_indexed_noise_is_independent_of_route_call_order():
    sequences=[dict(sequence_id='a',length=2),dict(sequence_id='b',length=1)]
    tape=IndexedNoise(sequences,4601)
    expected=tape.get('a',1,2).clone()
    tape.get('b',0,9);tape.get('a',0,14)
    assert torch.equal(expected,tape.get('a',1,2))
    assert not torch.equal(expected,tape.get('a',1,3))
    assert torch.equal(expected,IndexedNoise(sequences,4601).get('a',1,2))


def test_indexed_noise_preserves_original_st_forward_and_gradient(monkeypatch):
    tape=IndexedNoise([dict(sequence_id='a',length=1)],4601)
    mask=torch.tensor([[True,False,True,True,False,True,True,True,False,True]])
    logits=torch.linspace(-2,2,10).unsqueeze(0).requires_grad_()
    costs=torch.arange(10.).unsqueeze(0)
    original=torch.rand_like
    with tape.inject('a',0,0):
        action,probabilities=st_gumbel_action(logits,mask,2.,True)
        gradient,=torch.autograd.grad((action*costs).sum(),logits)
    assert torch.rand_like is original
    reference=logits.detach().clone().requires_grad_()
    monkeypatch.setattr(torch,'rand_like',lambda tensor:tape.get('a',0,0).to(tensor).clone())
    expected_action,expected_probabilities=st_gumbel_action(reference,mask,2.,True)
    expected_gradient,=torch.autograd.grad((expected_action*costs).sum(),reference)
    assert torch.equal(action,expected_action) and torch.equal(probabilities,expected_probabilities)
    assert torch.equal(gradient,expected_gradient)
    assert gradient[~mask].eq(0).all() and gradient[mask].abs().sum()>0


def test_existing_deterministic_probability_tensor_gets_derived_statistics():
    row=dict(raw_controller_logits=torch.tensor([[0.,1.]+[0.]*9]),
             legal_successor_mask=torch.tensor([True,True]+[False]*8),selected_action=1,
             probabilities=torch.tensor([[.2,.8]+[0.]*8]))
    action_probability_statistics([row])
    assert row['probabilities'].shape==(10,)
    assert row['selected_probability']==float(torch.tensor(.8))
    assert row['maximum_probability']==float(torch.tensor(.8))
    assert row['selected_raw_logit_margin']==1.
