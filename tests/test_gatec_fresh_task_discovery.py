"""Preparation-contract tests; no training or checkpoint selection."""
from collections import Counter
import torch
from experiments.run_gatec_fresh_task_discovery import (
    training_visits, fresh_panel, recorded_bits, IndexedNoise,
    VerifiedStochasticPath, anchor_guard, gradient_coverage,
)


def training_rows():
    return [dict(sequence_id=f'train/{i}',length=4,split='train',bits=[0,0,0,0],label=0)
            for i in range(104)]


def test_each_epoch_contains_all_training_rows_once_with_new_indexed_noise():
    rows=training_rows()
    visits,permutations=training_visits(rows)
    assert len(visits)==1040 and [v['update'] for v in visits]==list(range(1,1041))
    for epoch in range(1,11):
        selected=[v for v in visits if v['epoch']==epoch]
        assert Counter(v['source_sequence_id'] for v in selected)==Counter(v['sequence_id'] for v in rows)
    assert visits==training_visits(rows)[0]
    assert len({tuple(p) for p in permutations})==10
    tape=IndexedNoise(visits,4702)
    sid=rows[0]['sequence_id']
    a=tape.get('epoch01/'+sid,0,0)
    b=tape.get('epoch02/'+sid,0,0)
    assert not torch.equal(a,b)
    tape.get('epoch01/'+sid,1,14)
    assert torch.equal(a,tape.get('epoch01/'+sid,0,0))


def test_final_panel_is_distinct_balanced_and_rejects_exposed_inputs():
    first,_=fresh_panel(set())
    excluded={tuple(row['bits']) for row in first[:2]}
    panel,attempts=fresh_panel(excluded)
    assert len(panel)==len({tuple(row['bits']) for row in panel})==96
    assert not ({tuple(row['bits']) for row in panel}&excluded)
    assert Counter(row['length'] for row in panel)==Counter({16:32,**{n:4 for n in range(17,33)}})
    for length in range(16,33):
        expected=16 if length==16 else 2
        assert Counter(row['label'] for row in panel if row['length']==length)==Counter({0:expected,1:expected})
    assert any(row['reason']=='prior_or_training_exact_match' for row in attempts)
    assert panel==fresh_panel(excluded)[0]
    assert all(sum(row['bits'])%2==row['label'] for row in panel)


def test_record_inventory_reads_nested_binary_inputs_without_confusing_features():
    bits=[1,0,0,1]
    value=dict(inputs=dict(bits=[bits,[1,1,1,0,0]]),rows=[dict(bits=bits)],
               features=[0.,1.,.2,.3],budget=[0,0],logits=[1.1,2.2,3.3,4.4])
    found=recorded_bits(value)
    assert set(sequence for _,sequence in found)=={tuple(bits),(1,1,1,0,0)}
    assert len(found)==3


def test_real_frozen_backbone_observers_preserve_gradient_and_execution_contract():
    from src.agt import AGTv3
    from data.task_a import TaskAParity
    from experiments.replay_gatec_learned_policy import BACKBONE
    torch.set_num_threads(4)
    torch.manual_seed(4700)
    model=AGTv3(dim=64,num_memory_slots=16,vocab_size=2)
    weights={k:v for k,v in torch.load(BACKBONE,map_location='cpu',weights_only=True)['state_dict'].items()
             if not k.startswith('controller.')}
    missing,unexpected=model.load_state_dict(weights,strict=False)
    assert not unexpected and all(k.startswith('controller.') for k in missing)
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('controller.'))
    model.executor.max_execution_depth=8
    task=TaskAParity(64,'cpu');bits=torch.tensor([[0,1,1,0]])
    x,targets=task.token_features[bits],bits.cumsum(1).remainder(2)
    tape=IndexedNoise([dict(sequence_id='unit',length=4)],4702)
    with anchor_guard(model):path=VerifiedStochasticPath(model,tape,'unit',True).run(x,targets)
    loss=torch.nn.functional.cross_entropy(path.final_logits,targets[:,-1])
    loss.backward()
    coverage=gradient_coverage(path)
    assert coverage['HALT_exact_zero'] and coverage['illegal_exact_zero'] and coverage['nonzero_decisions']>0
    assert all(token['P']==8 and token['D']==8+token['restart_count'] and token['commit_count']==1 for token in path.tokens)
    assert all(row['noise_reconstruction_verified'] for row in path.rows)
    assert path.tokens[0]['graph_state']['/s']['requires_grad']
    assert all(p.grad is None for name,p in model.named_parameters() if not name.startswith('controller.'))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.controller.parameters())
    assert all(torch.equal(model.state_dict()[key],value) for key,value in weights.items())
