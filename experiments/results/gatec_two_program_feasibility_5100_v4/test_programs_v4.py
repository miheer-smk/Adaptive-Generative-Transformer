import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
"""Research-only complete-program contracts; production semantics stay unchanged."""
from dataclasses import FrozenInstanceError
import json

import pytest
import torch

from research_programs_v4 import (
    Program, P0, P1, ScriptedProvider, CompleteToken, MechanicalViolation,
    assert_disjoint, close_difference, gpu_snapshot, exact_state,
    bootstrap_prefix_worst, state_digest, sha,
)
from src.agt import AGTv3


def test_immutable_descriptors_and_exact_execution_validation():
    assert P0.actions == tuple(range(8))
    assert P1.actions == (0,1,2,3,4,5,8,7)
    assert [r['execution_after'] for r in P1.validate()] == list(range(1,9))
    assert P1.validate()[6]['pointer_after'] == 5
    with pytest.raises(FrozenInstanceError):
        P0.name = 'changed'
    with pytest.raises(ValueError):
        Program('mutable', list(range(8)))


@pytest.mark.parametrize('actions', [
    (8,)*8, (9,0,1,2,3,4,5,6,7), (0,1,0,3,4,5,6,7),
    (0,1,2,3,4,5,6,10), tuple(range(7)), tuple(range(8))+(9,),
])
def test_fail_closed_invalid_programs(actions):
    with pytest.raises(ValueError):
        Program('invalid',actions)


def test_restart_descriptor_has_fifteen_decisions_eight_executions():
    program=Program('restart',(0,9)*7+(0,))
    rows=program.validate()
    assert len(rows)==15 and rows[-1]['execution_after']==8
    assert sum(r['execution_before']==r['execution_after'] for r in rows)==7
    assert rows[7]['execution_after']==4


def test_exhausted_provider_never_falls_back():
    provider=ScriptedProvider(P1)
    for action in P1.actions:
        assert int(provider(h=torch.zeros(1,64))['logits'].argmax(-1))==action
    with pytest.raises(MechanicalViolation,match='Exhausted'):
        provider(h=torch.zeros(1,64))


@pytest.fixture(scope='module')
def model():
    torch.set_num_threads(4)
    torch.manual_seed(99)
    model=AGTv3(dim=64,num_memory_slots=16,vocab_size=2).eval()
    model.executor.max_execution_depth=8
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def test_complete_program_reference_branch_isolation_and_supplied_continuation(model):
    runner=CompleteToken(model)
    x=torch.randn(1,64)
    parent=model.initial_persistent_state(1,'cpu',torch.float32)
    parent_values=gpu_snapshot(parent)
    weights=gpu_snapshot(model.state_dict())
    ref,_,rr=runner.run(x,parent,P0,reference=True)
    a,ma,ra=runner.run(x,parent,P0)
    b,mb,rb=runner.run(x,parent,P1)
    assert close_difference(ref['state'],a['state']) is None
    assert close_difference(ref['logits'],a['logits']) is None
    for left,right in zip(rr,ra):
        assert close_difference(left['h_after'],right['h_after']) is None
    assert ma['P']==ma['D']==mb['P']==mb['D']==8
    assert mb['executed_blocks']==[0,1,2,3,4,5,5,7]
    assert mb['repeat_count']==1 and mb['restart_count']==0
    assert ma['commit_count']==mb['commit_count']==1
    assert_disjoint(a['state'],b['state'])
    sibling_values=gpu_snapshot(b['state'])
    saved=gpu_snapshot(a['state'])
    following,sm,_=runner.run(x,a['state'],P1)
    exact_state(saved,a['state'])
    exact_state(sibling_values,b['state'])
    exact_state(parent_values,parent)
    exact_state(weights,model.state_dict())
    assert following['state'].tokens_committed==2 and sm['token_index']==1
    assert runner.completed==4 and runner.block_calls==32 and runner.commit_calls==4
    assert runner.plan_purity_checks==4 and sum(runner.gdn_checks.values())==10
    assert model.executor.controller is model.controller
    assert all(p.grad is None for p in model.parameters())


def test_wrong_requested_action_stops_before_commit_and_restores_observers(model,monkeypatch):
    original=ScriptedProvider.__call__
    def corrupted(self,**features):
        result=original(self,**features)
        # A1 is legal initially but differs from the requested A0.
        result['logits'].fill_(-100)
        result['logits'][:,1]=100
        return result
    monkeypatch.setattr(ScriptedProvider,'__call__',corrupted)
    state=model.initial_persistent_state(1,'cpu',torch.float32)
    before=gpu_snapshot(state)
    runner=CompleteToken(model)
    with pytest.raises(MechanicalViolation,match='fallback'):
        runner.run(torch.zeros(1,64),state,P0)
    assert runner.commit_calls==runner.block_calls==0
    exact_state(before,state)
    assert model.executor.controller is model.controller


def test_worst_prefix_statistic_keeps_suffix_pairing_and_primary_horizons():
    prefix={'sequence_id':'example','label':0}
    rows=[]
    for hist in [0,1]:
        rows.append(dict(sequence_id='example',horizon=0,history_program=hist,CE_delta=.005))
        for schedule in ['P0','P1','alternating_P0_first']:
            for horizon in range(1,9):
                for suffix in range(4):
                    rows.append(dict(sequence_id='example',history_program=hist,schedule=schedule,
                        horizon=horizon,CE_delta=.02 if horizon==4 else .001,suffix=suffix))
    assert bootstrap_prefix_worst(rows,[prefix])==[.02]
    with pytest.raises(MechanicalViolation,match='Incomplete'):
        bootstrap_prefix_worst(rows[:-1],[prefix])


def test_all_protected_production_sources_and_archives_still_match_baseline():
    from research_programs_v4 import ROOT,OUT
    baseline=json.loads((OUT/'integrity_before.json').read_text())
    for path,hash_value in baseline['protected_files'].items():
        if path.startswith('src/') or path.startswith('experiments/runs/'):
            assert sha(ROOT/path)==hash_value,path
