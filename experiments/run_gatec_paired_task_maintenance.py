"""One 104-update warm task-loss pass, with an indexed-noise no-update arm."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from data.task_a import TaskAParity
from src.masks import build_execution_mask
import src.agt as agt_module
from experiments.replay_gatec_learned_policy import (
    ObservedPath, BACKBONE, BACKBONE_SHA, LABELS, sha, copy, snapshot,
    unchanged, fingerprint, first_difference, state_record, git_identity, encode,
)
from experiments.evaluate_gatec_locked_policy import sequence_metrics, aggregate, write_json, record, json_hash
from experiments.replicate_gatec_locked_imitation import make_model, teacher_view
from experiments.stage3_measurements import parameter_gradients

PARENT=ROOT/'experiments/results/gatec_imitation_seed_replication_4401_4402_v1'
WARM=ROOT/'experiments/runs/stage3_imitation_replication_4401_locked_v1.pt'
WARM_SHA='4e14f56d3762aac1442a8dcc6fd86eb950e0013986409ff31b48d514fd972fcd'
PARENT_SHA='b661d0042c56880fc324a7efd758ab744753d7fc63fa6770838cb50408c6fb0c'
OUT=ROOT/'experiments/results/gatec_paired_task_maintenance_4401_v1'
FINAL=ROOT/'experiments/runs/stage3_task_maintenance_4401_104_locked_v1.pt'
TAU=2.0


class IndexedNoise:
    """Pre-recorded uniforms; coordinates do not advance with another route."""
    def __init__(self, sequences, seed):
        generator=torch.Generator(device='cpu').manual_seed(seed)
        self.seed=seed
        # Fifteen is the legal P=8 trajectory bound, not a stopping rule.
        self.draws={s['sequence_id']:torch.rand(s['length'],15,10,generator=generator,dtype=torch.float32)
                    for s in sequences}

    def get(self, sequence, token, decision):
        return self.draws[sequence][token,decision].unsqueeze(0)

    @contextmanager
    def inject(self, sequence, token, decision):
        original=torch.rand_like; calls=[]
        def draw(tensor,*args,**kwargs):
            assert tensor.shape==(1,10) and tensor.dtype==torch.float32 and not args and not kwargs
            calls.append(1)
            return self.get(sequence,token,decision).to(tensor).clone()
        torch.rand_like=draw
        try:
            yield
            assert len(calls)==1,'Expected exactly one original ST-Gumbel uniform request'
        finally:
            torch.rand_like=original


def refs(value,prefix=''):
    if torch.is_tensor(value):return {prefix:value}
    if isinstance(value,dict):items=value.items()
    elif isinstance(value,(tuple,list)):items=enumerate(value)
    elif hasattr(value,'__dict__'):items=value.__dict__.items()
    else:return {prefix:value}
    result={}
    for key,item in items:result.update(refs(item,f'{prefix}/{key}'))
    return result


def versions(value):
    return {key:(id(v),v._version,tuple(v.shape)) if torch.is_tensor(v) else v for key,v in refs(value).items()}


def gpu_snapshot(value):
    return {key:v.detach().clone() if torch.is_tensor(v) else v for key,v in refs(value).items()}


def assert_same(values,value):
    current=refs(value)
    assert set(values)==set(current),'Persistent topology changed in flight'
    assert all(torch.equal(v,current[k]) if torch.is_tensor(v) else v==current[k] for k,v in values.items()),'Persistent values changed in flight'


def graph_state(state):
    return {key:dict(requires_grad=v.requires_grad,has_grad_fn=v.grad_fn is not None)
            for key,v in refs(state).items() if torch.is_tensor(v)}


class StochasticPath:
    """Observers call real methods unchanged; no routing/execution reimplementation."""
    def __init__(self,model,tape,sequence,retain_logits=False):
        self.model,self.tape,self.sequence,self.retain_logits=model,tape,sequence,retain_logits
        self.rows,self.tokens=[],[]
        self.block_purity_checks,self.gdn_purity_checks=0,Counter()
        self.candidate_constructions=0

    @contextmanager
    def observers(self):
        model=self.model; ex=model.executor
        controller,step,block,candidates,commit,read=(ex.controller,ex.differentiable_route_step,
            ex._run_block,ex.candidate_successors,agt_module.terminal_commit,model.memory.read)
        block_signature=inspect.signature(block)
        def observed_controller(**features):
            output=controller(**features)
            logits=output['logits']
            assert torch.isfinite(logits).all()
            assert features['budget'].eq(0).all()
            if self.retain_logits:
                assert logits.requires_grad
                logits.retain_grad()
            self.current.append(dict(token=self.token,decision=len(self.current),
                h_before=copy(features['h']),memory_read=copy(features['memory_read']),
                memory_mean=copy(features['memory_mean']),pointer_one_hot=copy(features['pointer_one_hot']),
                budget=copy(features['budget']),normalized_features=copy(model.controller._prepare_inputs(**features)),
                raw_controller_logits=copy(logits),candidate_block_calls=[],_logits=logits if self.retain_logits else None))
            return output
        def observed_block(*args,**kwargs):
            bound=block_signature.bind(*args,**kwargs).arguments
            inputs={k:bound[k] for k in ['h','h0','memory','context']}
            guard=versions(inputs)
            result=block(*args,**kwargs)
            assert guard==versions(inputs),'Candidate mutated its shared pre-action inputs'
            assert self.input_versions==versions(self.input_state),'Caller state changed during candidate execution'
            assert torch.isfinite(result).all(),'Nonfinite candidate block output'
            i=int(bound['block_index']);self.current[-1]['candidate_block_calls'].append(i)
            self.block_purity_checks+=1
            if model.blocks.blocks[i].species=='GDN':self.gdn_purity_checks[i]+=1
            return result
        def observed_candidates(*args,**kwargs):
            before=gpu_snapshot(kwargs['context'])
            result=candidates(*args,**kwargs)
            assert_same(before,kwargs['context'])
            assert self.current[-1]['candidate_block_calls']==list(range(8))+[kwargs['pointer']]
            assert result['h'].shape==(1,10,64) and torch.isfinite(result['h']).all()
            assert torch.equal(result['h'][:,9].detach(),kwargs['h'].detach())
            self.candidate_constructions+=1
            return result
        def observed_step(*args,**kwargs):
            row=self.current[-1];d=row['decision']
            assert kwargs['training'] is True and kwargs['tau']==TAU
            with self.tape.inject(self.sequence,self.token,d):result=step(*args,**kwargs)
            selected=int(result['selected_action'][0]);p=kwargs['pointer'];e=kwargs['execution_count'];r=kwargs['just_restarted']
            mask=build_execution_mask(p,e,r,8)
            assert selected<10 and mask[selected]
            assert int(row['pointer_one_hot'].argmax(-1))==p
            assert e==sum(x['selected_action']!=9 for x in self.current[:-1])
            if d:
                prev=self.current[-2]
                assert (p,e,r)==(prev['pointer_after'],prev['execution_after'],prev['just_restarted_after'])
            else:assert (p,e,r)==(0,0,False)
            hard=F.one_hot(result['selected_action'],10).to(result['action'])
            assert torch.equal(result['action'].detach(),hard),'ST forward action was not hard one-hot'
            assert torch.equal(result['h_next'].detach(),result['candidates'][:,selected].detach()),'Hard successor mismatch'
            probabilities=copy(result['probs'])[0]
            assert torch.isfinite(probabilities).all() and probabilities[~mask[:10]].eq(0).all()
            pointer_after=selected if selected<8 else p if selected==8 else 0
            assert int(result['next_pointer'][0])==pointer_after
            logits=row['raw_controller_logits'][0,:10]
            legal_logits=logits.masked_fill(~mask[:10],-torch.inf)
            other=legal_logits.clone();other[selected]=-torch.inf
            positive=probabilities>0
            row.update(selected_action=selected,selected_action_label=LABELS[selected],
                decoded_block_target=selected if selected<8 else p if selected==8 else None,
                actual_executed_block=selected if selected<8 else p if selected==8 else None,
                executed_blocks=[] if selected==9 else [selected if selected<8 else p],
                execution_before=e,execution_after=e+int(selected!=9),pointer_before=p,pointer_after=pointer_after,
                just_restarted_before=r,just_restarted_after=selected==9,repeat=selected==8,restart=selected==9,
                legal_action_mask=mask,legal_successor_mask=mask[:10],h_after=copy(result['h_next']),
                probabilities=probabilities,selected_probability=float(probabilities[selected]),
                maximum_probability=float(probabilities.max()),selected_raw_logit_margin=float(legal_logits[selected]-other.max()),
                entropy=float(-(probabilities[positive]*probabilities[positive].log()).sum()),
                noise_coordinate=[self.tape.seed,self.sequence,self.token,d],candidate_constructions=1,
                hard_metadata_requires_grad=result['next_pointer'].requires_grad)
            return result
        def observed_read(*args,**kwargs):
            result=read(*args,**kwargs);self.read_count+=1;return result
        def observed_commit(work,**kwargs):
            assert_same(self.input_snapshot,work)
            assert_same(self.input_snapshot,self.input_state)
            result=commit(work,**kwargs);new,plan=result
            assert len(plan.entries)==7 and plan.token_index==self.token
            assert torch.equal(new.s.detach(),plan.s.detach()) and torch.equal(new.memory.detach(),plan.memory.detach())
            for cache,version,payload in plan.entries:
                assert version==self.token and cache.version==version+1
                for actual,expected in zip(cache.get_committed_context(),payload):
                    if actual is not None:
                        if actual.shape!=expected.shape:actual=actual[:,:,-1:,:]
                        assert torch.equal(actual.detach(),expected.detach())
            self.commit_count+=1;return result
        try:
            ex.controller=observed_controller;ex._run_block=observed_block
            ex.candidate_successors=observed_candidates;ex.differentiable_route_step=observed_step
            agt_module.terminal_commit=observed_commit;model.memory.read=observed_read
            yield
        finally:
            ex.controller=controller;ex.differentiable_route_step=step;ex._run_block=block
            ex.candidate_successors=candidates;agt_module.terminal_commit=commit;model.memory.read=read

    def run(self,x,targets):
        state=self.model.initial_persistent_state(1,x.device,x.dtype)
        compatibility=x.new_zeros(1,64)
        with self.observers():
            for self.token in range(x.shape[1]):
                self.current=[];self.commit_count=self.read_count=0
                self.input_state=state;self.input_snapshot=gpu_snapshot(state);self.input_versions=versions(state)
                incoming=state_record(state);before=snapshot(state)
                out=self.model.adaptive_differentiable_step(x[:,self.token],state,compatibility,
                    budget=x.new_zeros(1,2),tau=TAU,training=True)
                assert torch.isfinite(out['logits']).all() and torch.isfinite(out['h_terminal']).all()
                history=out['action_history'];executions=sum(r['selected_action']!=9 for r in self.current)
                restarts=sum(r['restart'] for r in self.current)
                assert out['execution_count']==executions==8 and len(history)==len(self.current)==8+restarts
                assert 8<=len(history)<=15 and not out['halted'] and not out['safety_cap_hit']
                assert history==[('ADVANCE',r['selected_action']) if r['selected_action']<8 else ('REPEAT',r['pointer_before'])
                    if r['selected_action']==8 else ('RESTART',None) for r in self.current]
                assert self.commit_count==self.read_count==1
                assert_same(self.input_snapshot,state)
                self.rows.extend(self.current)
                self.tokens.append(dict(token=self.token,P=executions,D=len(history),history=history,
                    restart_count=restarts,repeat_count=sum(r['repeat'] for r in self.current),
                    executed_blocks=[r['actual_executed_block'] for r in self.current if r['actual_executed_block'] is not None],
                    termination_reason='execution_budget_reached',commit_count=self.commit_count,commit_entries=7,
                    incoming_persistent=incoming,outgoing_persistent=state_record(out['state']),
                    _incoming=before,_outgoing=snapshot(out['state']),graph_state=graph_state(out['state']),
                    output_logits=copy(out['logits']),prediction=int(out['logits'].argmax(-1)),target=int(targets[:,self.token]),
                    prefix_ce=float(F.cross_entropy(out['logits'].detach(),targets[:,self.token]))))
                state=out['state']  # Original full-sequence graph; no detach/truncation.
            self.final_logits=out['logits']
        return self


def canonical_log_probability(view):
    by_token=Counter()
    for row in view.rows:
        logits=row['raw_controller_logits'][0,:10]
        lp=F.log_softmax(logits.masked_fill(~row['legal_successor_mask'],-torch.inf),-1)
        by_token[row['token']]+=float(lp[row['decision']])
    return dict(sequence_log_probability=sum(by_token.values()),token_log_probabilities=dict(by_token),
        minimum_teacher_probability=min(float(F.softmax(row['raw_controller_logits'][0,:10].masked_fill(~row['legal_successor_mask'],-torch.inf),-1)[row['decision']]) for row in view.rows),
        interpretation='nominal conditional full canonical-route probability; diagnostic only, never added to loss')


def action_probability_statistics(rows):
    for row in rows:
        if 'selected_probability' not in row:
            logits=row['raw_controller_logits'][0,:10]
            masked=logits.masked_fill(~row['legal_successor_mask'],-torch.inf)
            probabilities=row['probabilities'].reshape(-1) if 'probabilities' in row else F.softmax(masked,-1)
            selected=row['selected_action'];other=masked.clone();other[selected]=-torch.inf
            positive=probabilities>0
            row.update(probabilities=probabilities,selected_probability=float(probabilities[selected]),
                maximum_probability=float(probabilities.max()),selected_raw_logit_margin=float(masked[selected]-other.max()),
                entropy=float(-(probabilities[positive]*probabilities[positive].log()).sum()))


def add_measurements(spec,reference,path,controller):
    view=teacher_view(reference,controller,'cuda')
    action_probability_statistics(path.rows)
    result=sequence_metrics(spec,view,path)
    result['canonical_log_probability']=canonical_log_probability(view)
    result['candidate_constructions']=getattr(path,'candidate_constructions',0)
    result['candidate_block_calls']=sum(len(r.get('candidate_block_calls',[])) for r in path.rows)
    result['realized_block_executions']=sum(t['P'] for t in path.tokens)
    result['mean_selected_probability']=sum(r['selected_probability'] for r in path.rows)/len(path.rows)
    result['minimum_selected_logit_margin']=min((r['selected_raw_logit_margin'] for r in path.rows if 'selected_raw_logit_margin' in r),default=None)
    return result


def trace_path(handles,phase,arm,spec,path):
    fields=['token','decision','execution_before','execution_after','pointer_before','pointer_after',
        'just_restarted_before','just_restarted_after','legal_action_mask','legal_successor_mask','raw_controller_logits',
        'selected_action','selected_action_label','decoded_block_target','actual_executed_block','executed_blocks',
        'restart','repeat','budget','candidate_block_calls','probabilities','selected_probability','maximum_probability',
        'selected_raw_logit_margin','entropy','noise_coordinate','candidate_constructions','retained_logit_gradient']
    for row in path.rows:record(handles['decisions'],dict(phase=phase,arm=arm,sequence_id=spec['sequence_id'],**{k:row[k] for k in fields if k in row}))
    for token in path.tokens:record(handles['tokens'],dict(phase=phase,arm=arm,sequence_id=spec['sequence_id'],**encode(token)))


def summary(records):
    result=aggregate(records)
    result['candidate_constructions']=sum(r['candidate_constructions'] for r in records)
    result['candidate_block_calls']=sum(r['candidate_block_calls'] for r in records)
    result['mean_canonical_sequence_log_probability']=sum(r['canonical_log_probability']['sequence_log_probability'] for r in records)/len(records)
    result['mean_canonical_token_log_probability']=sum(sum(r['canonical_log_probability']['token_log_probabilities'].values()) for r in records)/result['tokens']
    result['minimum_teacher_probability']=min(r['canonical_log_probability']['minimum_teacher_probability'] for r in records)
    result['per_length']={str(length):aggregate([r for r in records if r['length']==length]) for length in sorted({r['length'] for r in records})}
    return result


def paired_divergence(u,t):
    first={k:None for k in ['first_action','first_raw_state','first_controller_logits','first_execution','first_task']}
    common=True
    for ut,tt in zip(u.tokens,t.tokens):
        token=ut['token'];ur=[r for r in u.rows if r['token']==token];tr=[r for r in t.rows if r['token']==token]
        if common:
            for d,(a,b) in enumerate(zip(ur,tr)):
                raw=['h_before','memory_read','memory_mean','pointer_one_hot','budget']
                diff=first_difference(snapshot({k:a[k] for k in raw}),snapshot({k:b[k] for k in raw}))
                if diff and first['first_raw_state'] is None:first['first_raw_state']=dict(token=token,decision=d,**diff)
                diff=first_difference(snapshot(a['raw_controller_logits']),snapshot(b['raw_controller_logits']))
                if diff and first['first_controller_logits'] is None:first['first_controller_logits']=dict(token=token,decision=d,**diff)
                if a['selected_action']!=b['selected_action']:
                    first['first_action']=dict(token=token,decision=d,U=a['selected_action'],T=b['selected_action'])
                    diff=first_difference(snapshot(a['h_after']),snapshot(b['h_after']))
                    if diff and first['first_raw_state'] is None:
                        first['first_raw_state']=dict(token=token,decision=d,scope='successors of different hard actions',**diff)
                    common=False;break
        if first['first_execution'] is None:
            for p,(a,b) in enumerate(zip(ut['executed_blocks'],tt['executed_blocks'])):
                if a!=b:first['first_execution']=dict(token=token,execution_position=p,U=a,T=b);break
        if first['first_task'] is None and token==len(u.tokens)-1 and ut['prediction']!=tt['prediction']:
            first['first_task']=dict(token=token,U=ut['prediction'],T=tt['prediction'])
    return first


def parameter_change(before,controller):
    differences={key:controller.state_dict()[key].detach().cpu()-value for key,value in before.items()}
    return dict(l2=float(torch.cat([v.flatten() for v in differences.values()]).norm()),
        changed_values=sum(int(v.ne(0).sum()) for v in differences.values()),
        per_parameter={k:dict(l2=float(v.norm()),max_abs=float(v.abs().max()),changed_values=int(v.ne(0).sum())) for k,v in differences.items()})


def main():
    if FINAL.exists():raise RuntimeError('Refusing to overwrite checkpoint or repeat training')
    recovery=OUT.exists()
    if recovery:
        failure=json.loads((OUT/'failure.json').read_text())
        assert failure['type']=='KeyError' and failure['error']=="'selected_probability'" and failure['completed_updates']==0
        assert all((OUT/f'{key}.jsonl').stat().st_size==0 for key in ['decisions','tokens','sequences','updates'])
        assert not (OUT/'manifest_reporting_correction.json').exists(),'Recovery already attempted; no automatic repeat'
        prior_manifest=json.loads((OUT/'manifest.json').read_text())
    assert sha(WARM)==WARM_SHA and sha(BACKBONE)==BACKBONE_SHA and sha(PARENT/'manifest.json')==PARENT_SHA
    parent=json.loads((PARENT/'manifest.json').read_text())
    assert all(sha(ROOT/p)==digest for p,digest in parent['source_sha256'].items()),'Production source changed since locked imitation replication'
    assert sha(PARENT/'training_sequences.json')==parent['common_training']['sequence_file_sha256']
    assert sha(PARENT/'panel.json')==parent['panel']['sha256']
    train_data=json.loads((PARENT/'training_sequences.json').read_text())
    train=[]
    for batch in train_data['batches']:
        for i,(bits,label) in enumerate(zip(batch['bits'],batch['targets'])):
            assert len(bits)==batch['length'] and sum(bits)%2==label
            train.append(dict(sequence_id=f'train/L{len(bits)}/row{i}',length=len(bits),bits=bits,label=label,split='train'))
    assert len(train)==104 and Counter(s['length'] for s in train)==Counter({i:8 for i in range(4,17)})
    order=torch.randperm(104,generator=torch.Generator().manual_seed(4600)).tolist()
    training=[dict(train[i],update=position+1,source_index=i) for position,i in enumerate(order)]
    panel=json.loads((PARENT/'panel.json').read_text())['rows']
    panels={split:[s for s in panel if s['split']==split] for split in ['val','ood_test']}
    tapes={'train':IndexedNoise(training,4601),'val':IndexedNoise(panels['val'],4602),'ood_test':IndexedNoise(panels['ood_test'],4603)}
    if recovery:
        saved_order=json.loads((OUT/'training_order.json').read_text())
        assert training==saved_order['rows'] and order==saved_order['source_indices']
        assert sha(OUT/'training_order.json')==prior_manifest['inputs']['training_order_sha256']
        for key,tape in tapes.items():
            path=OUT/prior_manifest['randomness']['files'][key]['path']
            assert sha(path)==prior_manifest['randomness']['files'][key]['sha256']
            saved=torch.load(path,map_location='cpu',weights_only=True)
            assert saved['seed']==tape.seed and unchanged(snapshot(tape.draws),snapshot(saved['uniforms']))
            tape.draws=saved['uniforms']  # Exact already-sealed values; no new draw panel.
    identity=git_identity()
    protected={str(p):sha(p) for p in list((ROOT/'src').rglob('*'))+list((ROOT/'experiments/runs').rglob('*'))+list(PARENT.rglob('*')) if p.is_file()}
    torch.set_num_threads(4);device=torch.device('cuda')
    stage1=torch.load(BACKBONE,map_location='cpu',weights_only=True)
    weights={k:v for k,v in stage1['state_dict'].items() if not k.startswith('controller.')}
    checkpoint=torch.load(WARM,map_location='cpu',weights_only=True)
    models={arm:make_model(weights,device,3305,arm=='T')[0] for arm in ['U','T']}
    for model in models.values():model.controller.load_state_dict(checkpoint['controller_state_dict'],strict=True)
    reference=make_model(weights,device,3305,False)[0]
    assert unchanged(snapshot(models['U'].state_dict()),snapshot(models['T'].state_dict()))
    initial_controller={k:copy(v) for k,v in models['U'].controller.state_dict().items()}
    initial_models={arm:snapshot(model.state_dict()) for arm,model in models.items()}
    optimizer=torch.optim.AdamW(models['T'].controller.parameters(),lr=.0001,weight_decay=0.,betas=(.9,.999),eps=1e-8)
    assert not optimizer.state
    if not recovery:
        OUT.mkdir(parents=True,exist_ok=False)
        write_json(OUT/'training_order.json',dict(seed=4600,rows=training,source_indices=order))
        for split,tape in tapes.items():
            with (OUT/f'noise_{split}.pt').open('xb') as handle:torch.save(dict(seed=tape.seed,uniforms=tape.draws),handle)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),experiment='one bounded warm task-loss maintenance pass with matched no-update ST-Gumbel control',
        checkpoint=dict(path=str(WARM.relative_to(ROOT)),sha256=WARM_SHA),backbone=dict(path=str(BACKBONE.relative_to(ROOT)),sha256=BACKBONE_SHA),
        parent_manifest=dict(path=str((PARENT/'manifest.json').relative_to(ROOT)),sha256=PARENT_SHA),git=identity,script_sha256=sha(__file__),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src').glob('*.py')},
        inputs=dict(training_file=str((PARENT/'training_sequences.json').relative_to(ROOT)),training_sha256=sha(PARENT/'training_sequences.json'),
            training_order_file='training_order.json',training_order_sha256=sha(OUT/'training_order.json'),
            panel_file=str((PARENT/'panel.json').relative_to(ROOT)),panel_sha256=sha(PARENT/'panel.json'),panel_identity=parent['panel']['identity'],
            overlap=parent['panel']['overlap'],prior_unknowns=parent['panel']['unresolved_prior_evaluation_inputs'],
            raw_training_sequences=104,cached_imitation_features_used=False),
        training=dict(updates=104,passes=1,batch=1,lengths=[4,16],loss='final-sequence parity CE only',
            optimizer='fresh AdamW on T controller only',optimizer_defaults=optimizer.defaults,tau=TAU,clipping=None,scheduler=None,
            U_optimizer=None,U_updates=0,T_updates=104,selection='final update only; no evaluation-based stopping or tuning',
            trainable_parameters=[k for k,p in models['T'].named_parameters() if p.requires_grad],
            gradient_policy='unchanged run_diff_sequence full-sequence BPTT; no token/state/cache detach; reset at independent-sequence boundary',
            graph_exceptions='existing ST relaxed stop-gradient and hard pointer/mask metadata; existing SWA window slicing; no added truncation',
            frozen='every non-controller parameter and fixed token feature',budget=[0.,0.],budget_fraction=0.,budget_presence_flag=0.),
        randomness=dict(sequence_order_seed=4600,training_seed=4601,validation_seed=4602,ood_seed=4603,
            scheme='independent CPU torch.Generator per phase; ordered ragged FP32 uniform tables indexed by phase/sequence ID (training update mapping in order file)/token/decision/action slot',
            action_slots='0..9 successor slots; HALT slot 10 has no sampled successor',
            fifteen_positions='coverage of mathematical legal D<=15 bound, not a decision stopping cap',
            injection='temporary torch.rand_like hook only inside original differentiable_route_step; original clamp, Gumbel transform, log-softmax, tau and ST hard/backward computation unchanged',
            validation_reuse='exact same validation tensors before and after all updates',
            files={key:dict(path=f'noise_{key}.pt',sha256=sha(OUT/f'noise_{key}.pt'),tensor_fingerprint=fingerprint(snapshot(tape.draws))) for key,tape in tapes.items()}),
        execution=dict(device=str(device),dtype='float32',dim=64,batch=1,P=8,D='8+RESTART, 8..15',
            halt_excluded=True,convergence_unused=True,adaptive_depth=False,
            stochastic='actual production training=True: all 10 successors, nine speculative bank calls per decision, hard selected forward state',
            deterministic='actual production training=False: selected-only legal argmax',
            no_repair=True,no_teacher_forcing=True),
        schedule=['starting deterministic validation','unchanged stochastic validation','paired shuffled 104-sequence pass',
            'final deterministic validation and OOD U/T','final paired no-update stochastic validation and OOD U/T'],
        stopping='mechanical invariant violation or nonfinite result only; ordinary poor performance is recorded',
        strong_criterion='finite real controller updates, valid execution, preserved deterministic parity, sampled T accuracy>=U and mean CE<=U separately on each fixed panel',
        provenance_limitations=parent['limitations']+['panels were exposed in previous imitation replication; not untouched generalization',
            'single indexed-noise draw per sequence; no population stochastic robustness or causal trainability claim'],
        numerical=dict(torch_version=str(torch.__version__),device_name=torch.cuda.get_device_name(),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    manifest_path=OUT/'manifest.json'
    if recovery:
        for key in ['checkpoint','backbone','parent_manifest','inputs','training','randomness','execution','schedule','strong_criterion']:
            assert encode(manifest[key])==prior_manifest[key],f'Locked protocol changed: {key}'
        manifest['pretraining_reporting_correction']=dict(original_manifest_sha256=sha(manifest_path),
            preserved_failure_file='failure.json',completed_optimizer_updates_before_correction=0,
            correction='derive selected-probability statistics when existing observer supplies only a probability tensor; complete first-divergence bookkeeping',
            locked_inputs_order_and_uniforms_reused_exactly=True,protocol_changed=False)
        manifest_path=OUT/'manifest_reporting_correction.json'
    write_json(manifest_path,manifest);manifest_sha=sha(manifest_path)
    print(f'MANIFEST LOCKED {manifest_sha}',flush=True)
    task=TaskAParity(64,device);canonical={};stages={};updates=[]
    suffix='_verified' if recovery else ''
    handles={key:(OUT/f'{key}{suffix}.jsonl').open('x') for key in ['decisions','tokens','sequences','updates']}
    def inputs(spec):
        bits=torch.tensor([spec['bits']],device=device,dtype=torch.long)
        return task.token_features[bits],bits.cumsum(1).remainder(2)
    def reference_for(spec):
        key=tuple(spec['bits'])
        if key not in canonical:
            x,targets=inputs(spec)
            with torch.no_grad():canonical[key]=ObservedPath(reference,models['U'].controller,False).run(x,targets)
        return canonical[key]
    def measure(phase,arm,spec,stochastic):
        x,targets=inputs(spec);ref=reference_for(spec);model=models[arm]
        with torch.no_grad():
            path=StochasticPath(model,tapes[spec['split']],spec['sequence_id']).run(x,targets) if stochastic else ObservedPath(model,model.controller,True).run(x,targets)
            result=add_measurements(spec,ref,path,model.controller)
        trace_path(handles,phase,arm,spec,path);record(handles['sequences'],dict(phase=phase,arm=arm,**result))
        return result
    try:
        stages['initial_deterministic_val']={'U':[measure('initial_deterministic_val','U',s,False) for s in panels['val']]}
        print('INITIAL DETERMINISTIC VALIDATION COMPLETE',flush=True)
        stages['initial_stochastic_val']={'U':[measure('initial_stochastic_val','U',s,True) for s in panels['val']]}
        print('INITIAL STOCHASTIC VALIDATION COMPLETE',flush=True)
        for spec in training:
            ref=reference_for(spec);x,targets=inputs(spec)
            models['U'].train();models['T'].train()
            with torch.no_grad():
                u=StochasticPath(models['U'],tapes['train'],spec['sequence_id']).run(x,targets)
                um=add_measurements(spec,ref,u,models['U'].controller)
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                t=StochasticPath(models['T'],tapes['train'],spec['sequence_id'],True).run(x,targets)
                loss=F.cross_entropy(t.final_logits,torch.tensor([spec['label']],device=device))
                assert torch.isfinite(loss) and loss.requires_grad
                loss.backward()
            gradients=parameter_gradients(models['T'])
            assert all(g['present'] and g['finite'] for g in gradients.values()),'Missing/nonfinite controller gradient'
            assert all(p.grad is None for name,p in models['T'].named_parameters() if not name.startswith('controller.'))
            for row in t.rows:
                gradient=row.pop('_logits').grad
                assert gradient is not None and torch.isfinite(gradient).all()
                assert gradient[0,10]==0
                row['retained_logit_gradient']=copy(gradient)
            with torch.no_grad():tm=add_measurements(spec,ref,t,models['T'].controller)
            delta_before={k:copy(v) for k,v in models['T'].controller.state_dict().items()}
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in models['T'].controller.parameters())
            row=dict(update=spec['update'],sequence_id=spec['sequence_id'],length=spec['length'],U=um,T=tm,
                gradients=gradients,gradient_l2=sum(g['l2']**2 for g in gradients.values())**.5,
                controller_update=parameter_change(delta_before,models['T'].controller),paired_divergence=paired_divergence(u,t))
            updates.append(row);record(handles['updates'],row)
            for arm,path,result in [('U',u,um),('T',t,tm)]:
                trace_path(handles,'training',arm,spec,path);record(handles['sequences'],dict(phase='training',arm=arm,**result))
            print(f'UPDATE {spec["update"]}/104 L={spec["length"]} CE U/T={um["learned_ce"]:.9g}/{tm["learned_ce"]:.9g} grad={row["gradient_l2"]:.7g} change={row["controller_update"]["l2"]:.7g}',flush=True)
            del u,t,loss,ref,x,targets
        optimizer.zero_grad(set_to_none=True)
        change=parameter_change(initial_controller,models['T'].controller)
        assert unchanged(initial_models['U'],snapshot(models['U'].state_dict()))
        assert all(torch.equal(model.state_dict()[k].detach().cpu(),weights[k]) for model in models.values() for k in weights)
        with FINAL.open('xb') as handle:torch.save(dict(controller_state_dict={k:copy(v) for k,v in models['T'].controller.state_dict().items()},
            warmstart_sha256=WARM_SHA,backbone_sha256=BACKBONE_SHA,updates=104,optimizer='fresh AdamW',optimizer_defaults=optimizer.defaults,
            tau=TAU,dim=64,budget=[0.,0.],manifest_sha256=manifest_sha,selection_rule='final update 104 only'),handle)
        final_sha=sha(FINAL);del optimizer
        for model in models.values():
            model.eval()
            for p in model.parameters():p.requires_grad_(False)
        print(f'FINAL CHECKPOINT FIXED {final_sha}',flush=True)
        for split in ['val','ood_test']:
            phase='final_deterministic_'+split
            stages[phase]={arm:[measure(phase,arm,s,False) for s in panels[split]] for arm in ['U','T']}
            print(f'{phase} COMPLETE',flush=True)
        for split in ['val','ood_test']:
            phase='final_stochastic_'+split
            stages[phase]={arm:[] for arm in ['U','T']}
            for i,spec in enumerate(panels[split]):
                for arm in ['U','T']:stages[phase][arm].append(measure(phase,arm,spec,True))
                print(f'{phase} {i+1}/{len(panels[split])} COMPLETE',flush=True)
        assert stages['initial_stochastic_val']['U']==stages['final_stochastic_val']['U'],'Unchanged U validation did not reproduce with identical indexed noise'
        assert unchanged(initial_models['U'],snapshot(models['U'].state_dict()))
        assert sha(FINAL)==final_sha and all(sha(p)==h for p,h in protected.items())
        assert git_identity()['tracked_diff_sha256']==identity['tracked_diff_sha256']
        assert all(torch.equal(model.state_dict()[k].detach().cpu(),weights[k]) for model in [reference,*models.values()] for k in weights)
        assert all(p.grad is None and not p.requires_grad for model in models.values() for p in model.parameters())
        assert subprocess.run(['git','diff','--check'],capture_output=True).returncode==0
    except Exception as error:
        write_json(OUT/('failure_after_reporting_correction.json' if recovery else 'failure.json'),dict(error=str(error),type=type(error).__name__,traceback=traceback.format_exc(),
            completed_updates=len(updates),last_sequence=locals().get('spec'),instruction='STOP: no tuning, extension, repair or replacement'))
        raise
    finally:
        for handle in handles.values():handle.close()
    stage_summaries={phase:{arm:summary(records) for arm,records in arms.items()} for phase,arms in stages.items()}
    training_summaries={arm:summary([u[arm] for u in updates]) for arm in ['U','T']}
    prior_metrics=json.loads((PARENT/'results.json').read_text())['evaluation']['fresh_4401']['val']
    reproduced_prior=all(stage_summaries['initial_deterministic_val']['U'][k]==prior_metrics[k]
        for k in ['learned_accuracy','learned_mean_ce','canonical_action_route_fraction','exact_sequence_canonical_fraction'])
    strong=change['changed_values']>0 and all(
        stage_summaries['final_deterministic_'+split]['T']['learned_accuracy']>=stage_summaries['final_deterministic_'+split]['U']['learned_accuracy'] and
        stage_summaries['final_stochastic_'+split]['T']['learned_accuracy']>=stage_summaries['final_stochastic_'+split]['U']['learned_accuracy'] and
        stage_summaries['final_stochastic_'+split]['T']['learned_mean_ce']<=stage_summaries['final_stochastic_'+split]['U']['learned_mean_ce']
        for split in ['val','ood_test'])
    report=dict(manifest_sha256=manifest_sha,final_checkpoint=dict(path=str(FINAL.relative_to(ROOT)),sha256=final_sha),
        strong_bounded_maintenance=strong,starting_deterministic_validation_matches_prior_replication=reproduced_prior,
        stages=stage_summaries,training=training_summaries,
        controller_change=change,gradient_statistics=dict(mean_l2=sum(u['gradient_l2'] for u in updates)/104,
            minimum_l2=min(u['gradient_l2'] for u in updates),maximum_l2=max(u['gradient_l2'] for u in updates),
            finite_updates=104,nonzero_update_count=sum(u['controller_update']['changed_values']>0 for u in updates)),
        sampled_training_noncanonical_sequence_fraction={arm:1-training_summaries[arm]['exact_sequence_canonical_fraction'] for arm in ['U','T']},
        sampled_training_noncanonical_token_fraction={arm:1-training_summaries[arm]['canonical_action_route_fraction'] for arm in ['U','T']},
        invariants=dict(mechanical_validity=True,source_backbone_and_historical_artifacts_unchanged=True,
            U_unchanged=True,U_validation_noise_replay_exact=True,T_updates=104,U_updates=0,
            no_additional_detach=True,all_final_checkpoint_and_evaluation_parameters_unchanged=True,
            full_sequence_BPTT=True,no_clipping_scheduler_auxiliary_loss=True,diff_check_clean=True),
        limitations=manifest['provenance_limitations'])
    write_json(OUT/'results.json',report)
    lines=['# Bounded warm task-loss routing with matched ST-Gumbel control','',
        f'Strong bounded-maintenance criterion: **{strong}**.',
        f'Manifest SHA-256 `{manifest_sha}`. T: exactly 104 final-parity CE updates from seed4401 imitation; U: zero updates.',
        '', '| Phase | Arm | Accuracy | Mean CE | Canonical token / sequence fraction | RESTART / REPEAT | Candidate constructions |',
        '|---|---|---|---|---|---|---|']
    for phase,arms in stage_summaries.items():
        for arm,m in arms.items():lines.append(f'| {phase} | {arm} | {m["learned_accuracy"]} | {m["learned_mean_ce"]:.12g} | {m["canonical_action_route_fraction"]} / {m["exact_sequence_canonical_fraction"]} | {m["restart_count"]} / {m["repeat_count"]} | {m["candidate_constructions"]} |')
    lines.extend(['',f'Controller parameter change L2: {change["l2"]:.12g}; changed scalar values: {change["changed_values"]}.',
        f'Controller gradient L2 mean/min/max: {report["gradient_statistics"]}.',
        '', 'Raw sequences were recomputed under each arm. Full BPTT across the sequence and committed state/caches was retained; frozen backbone computations remained in T autograd. No extra detach, clipping, scheduler or auxiliary loss.',
        '', 'Noise was indexed by sequence/update, token, decision and ten successor slots. Validation used the same recorded noise before and after updates. U re-evaluation matched exactly. Nine physical speculative bank calls per stochastic decision are distinct from P=8 realized executions; deterministic paths construct no candidates.',
        '', 'Canonical full-route log-probabilities are diagnostics on independent reference states, never an optimization term. Controller probability, margin, per-sequence loss, gradients, parameter deltas, routes and first divergences are retained in JSON/JSONL.',
        '', 'Interpret sampling effects using initial U stochastic versus deterministic validation; update effects using final T versus matched U. Preserved task performance alone does not establish informative route learning. If routes and task losses remain unchanged, finite parameter updates and confidence changes do not demonstrate route discovery or task improvement.',
        '', 'Single fixed-noise panels limit stochastic conclusions. Six validation samples overlap reconstructed imitation training, two are duplicates, and prior panel exposure is recorded. No untouched-generalization claim.',
        '', 'This experiment does not establish fresh-init task-loss trainability, useful diversity, state-conditioned topology, adaptive depth or learned HALT. Production code, frozen backbone and historical checkpoints were not changed. No commit.', ''])
    (OUT/'report.md').write_text('\n'.join(lines))
    print(f'EXPERIMENT COMPLETE strong_bounded_maintenance={strong} change_l2={change["l2"]:.9g} {OUT}',flush=True)


if __name__=='__main__':main()
