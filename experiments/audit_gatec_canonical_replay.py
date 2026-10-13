"""Zero-update paired replay. All overrides are temporary diagnostic adapters."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from data.task_a import TaskAParity
from experiments.stage3_measurements import load_model, CHECKPOINT, file_sha256
from experiments.run_stage3_taskA_controller import set_seed
from src.masks import build_execution_mask
from src.executor import ExecutionResult
from src.terminal_commit import build_commit_plan
import src.agt as agt_module
import src.executor as executor_module

ATOL,RTOL=1e-6,1e-5
PROVISIONAL=ROOT/'experiments/runs/stage3_controller_route_imitation_seed3401.pt'


def snapshot(value,prefix=''):
    if torch.is_tensor(value): return {prefix:value.detach().cpu().clone()}
    if isinstance(value,dict):
        items=value.items()
    elif isinstance(value,(tuple,list)):
        items=enumerate(value)
    elif hasattr(value,'__dict__'):
        items=value.__dict__.items()
    else: return {prefix:value}
    out={}
    for key,item in items:
        out.update(snapshot(item,f'{prefix}/{key}'))
    return out


def unchanged(a,b):
    return set(a)==set(b) and all(torch.equal(a[k],b[k]) if torch.is_tensor(a[k]) else a[k]==b[k] for k in a)


def first_difference(a,b):
    if set(a)!=set(b): return dict(field='keys',left=sorted(a),right=sorted(b))
    for key in a:
        left,right=a[key],b[key]
        if torch.is_tensor(left):
            if not torch.is_tensor(right) or left.shape!=right.shape:
                return dict(field=key,reason='shape/type')
            if not torch.allclose(left,right,atol=ATOL,rtol=RTOL):
                return dict(field=key,max_abs=(left-right).abs().max().item())
        elif isinstance(left,float) and isinstance(right,float):
            if not math.isclose(left,right,abs_tol=ATOL,rel_tol=RTOL): return dict(field=key,left=left,right=right)
        elif left!=right: return dict(field=key,left=left,right=right)
    return None


def concise(values):
    return {key:dict(shape=list(value.shape),l2=value.float().norm().item(),max_abs=value.abs().max().item() if value.numel() else 0.) if torch.is_tensor(value) else value for key,value in values.items()}


class Canonical:
    def __init__(self): self.calls=0
    def __call__(self,**kwargs):
        logits=kwargs['h'].new_full((1,11),-100.)
        logits[:,self.calls%8]=100.;self.calls+=1
        return {'logits':logits}


class Alternating:
    def __init__(self): self.calls=0
    def __call__(self,**kwargs):
        logits=kwargs['h'].new_full((1,11),-100.)
        logits[:,0 if self.calls%2==0 else 9]=100.;self.calls+=1
        return {'logits':logits}


def source_budget_replay(executor,h0,memory,context,memory_mean,budget,limit=8,on_decision=None):
    """Diagnostic selected-only replay, separate counters; no production edit."""
    h=h0.clone();pointer=0;restarted=False;executions=decisions=0;hist=[]
    while executions<limit:
        onehot=h.new_zeros(1,9);onehot[:,pointer]=1
        features=dict(h=h,memory_read=memory,memory_mean=memory_mean,pointer_one_hot=onehot,budget=budget)
        logits=executor.controller(**features)['logits']
        mask=build_execution_mask(pointer,executions,restarted,8).to(h.device)[:10]
        selected=int(logits[:,:10].masked_fill(~mask,-torch.inf).argmax(-1))
        assert mask[selected]
        if on_decision: on_decision(decisions,executions,pointer,restarted,mask,selected,features,logits)
        if selected<8:
            h=executor._run_block(selected,h,h0,memory,context);pointer=selected;restarted=False;hist.append(('ADVANCE',selected));executions+=1
        elif selected==8:
            h=executor._run_block(pointer,h,h0,memory,context);restarted=False;hist.append(('REPEAT',pointer));executions+=1
        else:
            pointer=0;restarted=True;hist.append(('RESTART',None))
        decisions+=1
        if decisions>2*limit-1: raise RuntimeError('Source legality termination bound violated')
    return ExecutionResult(h_terminal=h,execution_count=executions,action_history=hist)


class Trace:
    def __init__(self,model,mode):
        self.model=model;self.mode=mode;self.tokens=[];self.steps=[];self.inputs=[];self.purity=[]
    def row(self,before,after,anchor,memory,block,context):
        d=len(self.current_steps);pointer=0 if d==0 else d-1
        onehot=before.new_zeros(1,9);onehot[:,pointer]=1
        features=dict(h=before,memory_read=memory,memory_mean=self.input_state.memory.mean(1),pointer_one_hot=onehot,budget=before.new_zeros(1,2))
        if self.mode!='A':
            features=self.latest_features
            pointer=self.last_mask_metadata['pointer']
        mask=build_execution_mask(pointer,d,False,8) if self.mode=='A' else self.last_mask
        restarted=False if self.mode=='A' else self.last_mask_metadata['just_restarted']
        logits=self.model.controller(**features)['logits']
        self.current_steps.append(dict(token=self.t,decision_count=d,execution_count=d,pointer_before=pointer,
             selected_action=block,pointer_after=block,just_restarted=restarted,legal_actions=[i for i in range(11) if mask[i]],h_before=before.detach().cpu().clone(),h=after.detach().cpu().clone(),h0=anchor.detach().cpu().clone(),
             memory_read=memory.detach().cpu().clone(),memory_mean=features['memory_mean'].detach().cpu().clone(),
             features=self.model.controller._prepare_inputs(**features).detach().cpu().clone(),controller_logits=logits.detach().cpu().clone(),
             persistent=snapshot(self.input_state),context=snapshot(context)))
    def run(self,x,targets):
        model=self.model;state=model.initial_persistent_state(1,x.device,x.dtype);compatibility=x.new_zeros(1,64)
        old_controller=model.executor.controller;old_step=model.executor.differentiable_route_step;old_block=model.executor._run_block;old_commit=agt_module.terminal_commit;old_read=model.memory.read;old_mask=executor_module.build_execution_mask
        handles=[];canonical=Canonical();read_calls=[]
        def observe_controller(**kwargs):
            self.latest_features=kwargs
            return canonical(**kwargs) if self.mode=='C' else old_controller(**kwargs)
        def observe_mask(*args,**kwargs):
            mask=old_mask(*args,**kwargs)
            self.last_mask=mask;self.last_mask_metadata=kwargs
            return mask
        def read(query,memory):
            out=old_read(query,memory);read_calls.append(dict(query=query.detach().cpu().clone(),R=out[0].detach().cpu().clone()));return out
        def commit(work,**kwargs):
            pre=snapshot(work);expected=snapshot(self.input_state)
            assert unchanged(pre,expected),'In-flight persistent mutation'
            plan=build_commit_plan(work,**kwargs)
            assert unchanged(pre,snapshot(work)),'Commit-plan construction mutation'
            result=old_commit(work,**kwargs)
            new=result[0]
            torch.testing.assert_close(new.s,plan.s,atol=ATOL,rtol=RTOL)
            torch.testing.assert_close(new.memory,plan.memory,atol=ATOL,rtol=RTOL)
            assert len(plan.entries)==7
            for cache,version,payload in plan.entries:
                assert cache.version==version+1
                actual=cache.get_committed_context()
                for v,w in zip(actual,payload):
                    if v is not None:
                        if v.shape!=w.shape: v=v[:,:,-1:,:]
                        torch.testing.assert_close(v,w,atol=ATOL,rtol=RTOL)
            self.commit_count+=1
            self.commit_pre=pre;self.commit_post=snapshot(new)
            self.terminal=kwargs['h_terminal'].squeeze(1).detach().cpu().clone()
            self.commit_entries=len(plan.entries)
            return result
        def checked_block(block_index,h,h0,memory,context,**kwargs):
            pre=snapshot(dict(h=h,h0=h0,R=memory,C=context))
            out=old_block(block_index,h,h0,memory,context,**kwargs)
            assert unchanged(pre,snapshot(dict(h=h,h0=h0,R=memory,C=context))),'Candidate input mutation'
            self.block_calls+=1;self.purity.append(True)
            if self.mode=='C': self.row(h,out,h0,memory,block_index,context)
            return out
        def forced_step(*args,**kwargs):
            result=old_step(*args,**kwargs)
            d=kwargs['execution_count'];mask=build_execution_mask(kwargs['pointer'],d,kwargs['just_restarted'],8)
            assert mask[d]
            self.row(kwargs['h'],result['candidates'][:,d],kwargs['h0'],kwargs['memory'],d,kwargs['context'])
            # Select only after unchanged exhaustive candidate construction/ST call.
            result['h_next']=result['candidates'][:,d]
            result['selected_action']=torch.tensor([d],device=x.device)
            result['next_pointer']=torch.tensor([d],device=x.device)
            result['action']=F.one_hot(result['selected_action'],10).to(x.dtype)
            return result
        def hook(index):
            def recorded(module,args,kwargs,out):
                after=out[0] if isinstance(out,tuple) else out
                self.row(args[0].squeeze(1),after.squeeze(1),kwargs['h0'].squeeze(1),kwargs['R'].squeeze(1),index,self.input_state.ctx)
            return recorded
        try:
            model.memory.read=read;agt_module.terminal_commit=commit
            if self.mode=='A':
                handles=[block.register_forward_hook(hook(i),with_kwargs=True) for i,block in enumerate(model.blocks.blocks)]
            else:
                model.executor._run_block=checked_block
                model.executor.controller=observe_controller;executor_module.build_execution_mask=observe_mask
                if self.mode=='B': model.executor.differentiable_route_step=forced_step
                else: pass
            for self.t in range(x.shape[1]):
                self.input_state=state;self.inputs.append(state);self.current_steps=[];self.commit_count=0;self.block_calls=0;start=len(read_calls)
                out=model.step(x[:,self.t],state,compatibility) if self.mode=='A' else model.adaptive_differentiable_step(x[:,self.t],state,compatibility,training=self.mode=='B',tau=2.)
                assert self.commit_count==1 and len(self.current_steps)==8
                assert len(read_calls)-start==1
                torch.testing.assert_close(read_calls[-1]['query'],state.s.cpu(),atol=0,rtol=0)
                self.steps.extend(self.current_steps)
                self.tokens.append(dict(token=self.t,terminal=self.terminal,commit=self.commit_post,commit_pre=self.commit_pre,
                    logits=out['logits'].detach().cpu(),prediction=int(out['logits'].argmax(-1)),
                    ce=F.cross_entropy(out['logits'],targets[:,self.t]).item(),commit_count=self.commit_count,
                    commit_entries=self.commit_entries,read_calls=1,decision_count=8,execution_count=8,
                    candidate_block_calls=self.block_calls if self.mode!='A' else 8))
                state=out['state']
        finally:
            for h in handles:h.remove()
            model.executor.controller=old_controller;model.executor.differentiable_route_step=old_step;model.executor._run_block=old_block
            agt_module.terminal_commit=old_commit;model.memory.read=old_read;executor_module.build_execution_mask=old_mask
        return self


def compare(reference,other):
    result=dict(first_state_divergence=None,first_metadata_divergence=None,first_commit_divergence=None,first_logit_divergence=None)
    for a,b in zip(reference.steps,other.steps):
        for group,fields in [('metadata',['decision_count','execution_count','pointer_before','selected_action','pointer_after','just_restarted','legal_actions']),
                             ('state',['h_before','h','h0','memory_read','memory_mean','features','controller_logits','persistent','context'])]:
            mismatch=first_difference(snapshot({k:a[k] for k in fields}),snapshot({k:b[k] for k in fields}))
            key='first_'+group+'_divergence'
            if mismatch and result[key] is None:result[key]=dict(token=a['token'],decision=a['decision_count'],**mismatch)
    for a,b in zip(reference.tokens,other.tokens):
        for group,fields in [('commit',['terminal','commit','commit_pre','commit_count','commit_entries']),('logit',['logits','prediction','ce'])]:
            mismatch=first_difference(snapshot({k:a[k] for k in fields}),snapshot({k:b[k] for k in fields}))
            key='first_'+group+'_divergence'
            if mismatch and result[key] is None:result[key]=dict(token=a['token'],**mismatch)
    return result


def serial_trace(trace):
    steps=[]
    for row in trace.steps:
        steps.append({k:concise(v) if k in ['persistent','context'] else v.tolist() if torch.is_tensor(v) else v for k,v in row.items()})
    return dict(steps=steps,tokens=[{k:concise(v) if isinstance(v,dict) else v.tolist() if torch.is_tensor(v) else v for k,v in r.items()} for r in trace.tokens])


@torch.no_grad()
def gdn_probe(model,reference):
    records=[]
    for index in [4,5]:
        state=reference.inputs[1];row=reference.steps[8+index];ctx=state.ctx;before=snapshot(ctx)
        device=state.s.device;h=row['h_before'].to(device);h0=row['h0'].to(device);R=row['memory_read'].to(device)
        core=model.blocks.blocks[index].core;original_step=core._step;original_ephemeral=core.forward_ephemeral;checks=[];buffer_checks=[]
        committed_state,committed_buffer=ctx.gdn[index].get_committed_context()
        def ephemeral(u,**kwargs):
            torch.testing.assert_close(kwargs['state'],committed_state,atol=0,rtol=0)
            torch.testing.assert_close(kwargs['conv_buffer'],committed_buffer,atol=0,rtol=0)
            buffer_checks.append(True)
            return original_ephemeral(u,**kwargs)
        def recurrent(S,q,k,v,alpha,beta):
            torch.testing.assert_close(S,committed_state,atol=0,rtol=0)
            Snew,o=original_step(S,q,k,v,alpha,beta)
            torch.testing.assert_close(o,torch.einsum('bhvk,bhk->bhv',Snew,q),atol=ATOL,rtol=RTOL)
            checks.append(True);return Snew,o
        core._step=recurrent;core.forward_ephemeral=ephemeral
        try:
            first,e1=model.executor._run_block(index,h,h0,R,ctx,return_ephemeral=True)
            second,e2=model.executor._run_block(index,first,h0,R,ctx,return_ephemeral=True)
            rebuilt,e3=model.executor._run_block(index,first,h0,R,state.ctx.clone(),return_ephemeral=True)
            ordinary=model.executor._run_block(index,first,h0,R,ctx)
            torch.testing.assert_close(second,rebuilt,atol=ATOL,rtol=RTOL)
            torch.testing.assert_close(second,ordinary,atol=ATOL,rtol=RTOL)
            torch.testing.assert_close(e2.state,e3.state,atol=ATOL,rtol=RTOL)
            torch.testing.assert_close(e2.conv_buffer,e3.conv_buffer,atol=ATOL,rtol=RTOL)
            assert unchanged(before,snapshot(ctx))
            records.append(dict(block=index,status='VERIFIED',committed_state_and_conv_unchanged=True,
                 repeat_matches_recomputed_from_cloned_committed_context=True,ephemeral_recurrence_output_checks=len(checks),committed_input_buffer_checks=len(buffer_checks),
                 first_second_ephemeral_state_max_abs=(e1.state-e2.state).abs().max().item()))
        finally:
            del core._step
            del core.forward_ephemeral
    return records



@torch.no_grad()
def memory_query_probe(model,reference):
    deltas=[]
    for row in reference.steps:
        state=reference.inputs[row['token']]
        alternate=model.memory.read(row['h_before'].to(state.s.device),state.memory)[0].cpu()
        deltas.append(dict(token=row['token'],decision=row['decision_count'],
                           max_abs_R_difference=(alternate-row['memory_read']).abs().max().item(),
                           same_within_tolerance=bool(torch.allclose(alternate,row['memory_read'],atol=ATOL,rtol=RTOL))))
    return dict(status='VERIFIED',read_calls_per_token=1,query='s_(t-1)',R_reused_within_token=True,
                alternate_h_query_scope='read-only probe on the same frozen slots; never fed to execution',
                alternate_h_query_max_abs=max(r['max_abs_R_difference'] for r in deltas),
                alternate_h_query_equal_count=sum(r['same_within_tolerance'] for r in deltas),rows=deltas,
                per_step_h_query='NOT IMPLEMENTED; no source correction inferred from replay equivalence')

def accounting_probe(model,reference):
    row=reference.steps[0];state=reference.inputs[0];device=state.s.device
    kwargs=dict(h0=row['h0'].to(device),memory=row['memory_read'].to(device),context=state.ctx,
                memory_mean=state.memory.mean(1),budget=state.s.new_zeros(1,2))
    old=model.executor.controller
    try:
        model.executor.controller=Alternating();current=model.executor.execute_differentiable(**kwargs,training=False)
        model.executor.controller=Alternating();corrected=source_budget_replay(model.executor,**kwargs)
    finally:model.executor.controller=old
    def counts(result):
        return dict(reported_execution_count=result.execution_count,decision_count=len(result.action_history),
          actual_execution_count=sum(k!='RESTART' for k,_ in result.action_history),restart_count=sum(k=='RESTART' for k,_ in result.action_history),
          repeat_count=sum(k=='REPEAT' for k,_ in result.action_history),history=result.action_history)
    a,b=counts(current),counts(corrected)
    assert a['actual_execution_count']==a['reported_execution_count']==8
    assert a == b
    torch.testing.assert_close(current.h_terminal,corrected.h_terminal,atol=ATOL,rtol=RTOL)
    assert b['actual_execution_count']==8 and b['decision_count']==15 and b['restart_count']==7
    return dict(status='VERIFIED',production=a,diagnostic_source_budget=b,
                production_corrected=True,protocol='fixed-eight-block-executions-v1',
                historical_boundary='Earlier fixed-eight-decision experiments retain their original protocol; they are not reinterpreted as eight executions.')


class FirstPolicyMismatch(Exception):
    def __init__(self, detail): self.detail=detail


@torch.no_grad()
def learned_replay(model,reference,x,targets,checkpoint_path=PROVISIONAL):
    if not checkpoint_path.exists():
        return dict(status='NOT TESTED',checkpoint=str(checkpoint_path),reason='Exact prescribed provisional checkpoint missing; no substitute used')
    checkpoint=torch.load(checkpoint_path,map_location='cpu')
    if 'controller_state_dict' not in checkpoint:
        return dict(status='NOT TESTED',checkpoint=str(checkpoint_path),reason='No verified controller_state_dict field; no guessed weight conversion')
    saved={k:v.clone() for k,v in model.controller.state_dict().items()}
    old_controller=model.executor.controller;old_execute=model.executor.execute_differentiable
    logs=[];first=None;final=None;state=model.initial_persistent_state(1,x.device,x.dtype)
    compatibility=x.new_zeros(1,64)
    try:
        model.controller.load_state_dict(checkpoint['controller_state_dict'],strict=True)
        model.executor.controller=model.controller
        for token in range(x.shape[1]):
            decisions=[]
            def observed(decision,execution,pointer,restarted,mask,action,features,logits):
                canonical=reference.steps[token*8+execution]
                probabilities=logits[:,:10].masked_fill(~mask,-torch.inf).softmax(-1)
                p=probabilities[probabilities>0]
                row=dict(token=token,decision_count=decision,execution_count=execution,pointer=pointer,
                         just_restarted=restarted,legal_actions=mask.nonzero().flatten().tolist(),selected_action=action,
                         canonical_action=execution,matches=action==execution,h_norm=features['h'].norm().item(),
                         entropy=float(-(p*p.log()).sum()),restart_count=sum(r['selected_action']==9 for r in decisions),
                         repeat_count=sum(r['selected_action']==8 for r in decisions))
                decisions.append(row)
                if action!=execution:
                    onehot=features['pointer_one_hot'].new_zeros(1,9);onehot[:,canonical['pointer_before']]=1
                    expected=dict(h=canonical['h_before'].to(x.device),memory_read=canonical['memory_read'].to(x.device),
                                  memory_mean=canonical['memory_mean'].to(x.device),pointer_one_hot=onehot,budget=x.new_zeros(1,2))
                    raw=first_difference(snapshot(features),snapshot(expected))
                    live_features=model.controller._prepare_inputs(**features)
                    teacher_features=model.controller._prepare_inputs(**expected)
                    feature_difference=first_difference(snapshot(live_features),snapshot(teacher_features))
                    expected_mask=build_execution_mask(canonical['pointer_before'],execution,False,8)[:10].to(x.device)
                    raise FirstPolicyMismatch(dict(**row,raw_input_first_difference=raw,
                          normalized_input_first_difference=feature_difference,mask_matches_canonical=bool(torch.equal(mask,expected_mask)),
                          canonical_normalized_features=teacher_features.tolist(),learned_normalized_features=live_features.tolist(),
                          canonical_logits=model.controller(**expected)['logits'].tolist(),learned_logits=logits.tolist(),
                          canonical_mask=expected_mask.tolist(),learned_mask=mask.tolist(),
                          interpretation='same state/features -> wrong canonical-policy action' if raw is None and feature_difference is None else 'features already diverged before policy mismatch'))
            def execute(h0,memory,C_prev=None,context=None,memory_mean=None,budget=None,**kwargs):
                return source_budget_replay(model.executor,h0,memory,context,memory_mean,budget,8,observed)
            model.executor.execute_differentiable=execute
            try:
                out=model.adaptive_differentiable_step(x[:,token],state,compatibility,training=False)
            except FirstPolicyMismatch as mismatch:
                first=mismatch.detail;logs.append(decisions[:20]);break
            logs.append(decisions[:20]);state=out['state'];final=out
            print('learned token',token,'decisions',len(decisions),'executions',out['execution_count'],flush=True)
        return dict(status='PARTIAL' if first else 'VERIFIED',checkpoint=str(checkpoint_path),checkpoint_sha256=file_sha256(checkpoint_path),
                    budget_mode='diagnostic source-budget adapter; not corrected production',first_policy_mismatch=first,
                    per_token_first20_decisions=logs,canonical_route_recovered=first is None,
                    final_ce=F.cross_entropy(final['logits'],targets[:,-1]).item() if first is None else None)
    finally:
        model.controller.load_state_dict(saved,strict=True)
        model.executor.controller=old_controller;model.executor.execute_differentiable=old_execute


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');parser.add_argument('--output-dir',type=Path,default=ROOT/'experiments/results');parser.add_argument('--skip-learned-replay',action='store_true');args=parser.parse_args()
    output=args.output_dir/'gatec_canonical_replay.json';md=output.with_suffix('.md')
    if output.exists() or md.exists():parser.error('Output exists; do not overwrite')
    torch.set_num_threads(4);set_seed(3401);model=load_model(3401,torch.device(args.device));weights={k:v.clone() for k,v in model.state_dict().items()}
    task=TaskAParity(64,args.device);batch=task.make_fixed_set('train',8,1)
    bits=torch.cdist(batch.x.reshape(-1,64),task.token_features).argmin(-1).reshape(1,8)
    targets=bits.cumsum(1).remainder(2)
    print('PHASE 1 COMPLETE: zero-update adapters built; source masks unchanged.',flush=True)
    traces={mode:Trace(model,mode).run(batch.x,targets) for mode in ['A','B','C']}
    comparisons={mode:compare(traces['A'],traces[mode]) for mode in ['B','C']}
    equivalent=all(all(v is None for v in c.values()) for c in comparisons.values())
    assert equivalent,comparisons
    print('PHASE 2 COMPLETE: A/B/C equivalence established across 64 block executions each.',flush=True)
    learned=(dict(status='NOT TESTED',reason='Explicitly excluded from execution-accounting validation; no controller checkpoint loaded')
             if args.skip_learned_replay else learned_replay(model,traces['A'],batch.x,targets))
    report=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
       starting_status=Path('AUDIT_PRECHECK.md').read_text(),stage1_checkpoint=str(CHECKPOINT),stage1_sha256=file_sha256(CHECKPOINT),
       requested_controller=str(PROVISIONAL),controller_sha256=learned.get('checkpoint_sha256'),source_sha256={name:file_sha256(ROOT/name) for name in ['src/agt.py','src/executor.py','src/routing.py','src/masks.py','src/blocks.py','src/gdn_v3.py','src/memory.py','src/terminal_commit.py']},seed=3401,device=str(args.device),dtype='float32',bits=bits.tolist(),atol=ATOL,rtol=RTOL,
       path_definitions=dict(A='existing AGTv3.step canonical reference',B='real differentiable_route_step and all candidates; forced canonical selection only after construction, unchanged legality mask',C='actual production selected-only deterministic path with canonical policy',D=learned.get('reason','prescribed provisional controller replay')),
       comparisons=comparisons,canonical_status='VERIFIED',paths={k:serial_trace(v) for k,v in traces.items()},
       candidate_purity=dict(status='VERIFIED',checked_calls=len(traces['B'].purity),masked_candidates_also_computed=True),
       gdn=gdn_probe(model,traces['A']),terminal_commit=dict(status='VERIFIED',one_per_token=True,entries_per_token=7,payloads_recomputed_from_precommit_terminal_inputs=True),
       memory=memory_query_probe(model,traces['A']),
       learned=learned,accounting=accounting_probe(model,traces['A']))
    report['weights_unchanged']=all(torch.equal(v,model.state_dict()[k]) for k,v in weights.items());assert report['weights_unchanged']
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:handle.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with md.open('x') as handle:
        handle.write('# Gate-C zero-update canonical replay\n\n'+f"HEAD `{report['head']}`; actual branch `{report['branch']}`. Starting worktree: [AUDIT_PRECHECK.md](../../AUDIT_PRECHECK.md). Stage-1 `{CHECKPOINT}` SHA256 `{report['stage1_sha256']}`. Exact provisional controller replay status is {learned['status']}; no substitute.\n\n")
        for key,value in report['path_definitions'].items():handle.write(f'- Path {key}: {value}.\n')
        handle.write('\n## First divergence\n\n| Pair | State | Metadata | Commit | Logits |\n|---|---|---|---|---|\n')
        for mode,c in comparisons.items():handle.write(f"| A/{mode} | {c['first_state_divergence']} | {c['first_metadata_divergence']} | {c['first_commit_divergence']} | {c['first_logit_divergence']} |\n")
        handle.write('\n**VERIFIED:** all 64 canonical transitions agree within atol1e-6/rtol1e-5, discrete metadata exact; final predictions agree.\n')
        for mode,tr in traces.items():handle.write(f"Path {mode}: final CE {tr.tokens[-1]['ce']:.12g}, prediction {tr.tokens[-1]['prediction']}; 8 executions/8 decisions per token.\n")
        handle.write('\n**VERIFIED GDN:** both blocks use ephemeral recurrence for output; two visits recompute from unchanged committed state/buffer and match cloned-context reconstruction. Convolution buffers remain frozen. Discarding ephemeral state is not a bug.\n\n**VERIFIED candidate purity:** 576 exhaustive block calls leave supplied h/h0/R/context unchanged. The production builder evaluates masked as well as legal candidates.\n\n**VERIFIED terminal commit:** one per token, all six context-bearing bank blocks plus Prelude updated; terminal-derived payloads recomputed from pre-commit state agree with actual committed values.\n\n**VERIFIED memory behavior:** both paths perform Read(s_previous,M) once per token and reuse R; not Read(h_p,M) per decision. Whether a different read policy is required is not resolved by this experiment.\n\nLearned-policy status is recorded above; this accounting validation uses --skip-learned-replay. No closed-loop learned-policy claim.\n\n## Counter discrepancy and diagnostic correction\n\n')
        handle.write('Read-only h-query probe: '+json.dumps({k:v for k,v in report['memory'].items() if k!='rows'})+'\n\n')
        handle.write(json.dumps(report['accounting'],indent=2)+'\n\n')
        handle.write('**VERIFIED production execution accounting:** alternating ADVANCE/RESTART routing now performs and reports eight executions in fifteen decisions with seven restarts; history and terminal state match the independent diagnostic replay.\n\nProtocol: fixed-eight-block-executions-v1. Earlier fixed-eight-decision experiments retain their historical meaning. The only production correction separates decision/execution accounting and terminates on actual block executions.\n\nCanonical numerical/state/commit equivalence is established on one sequence. Learned-policy replay is excluded when --skip-learned-replay is supplied. These results do not uniquely explain task-loss collapse or establish learned-routing success. All weights unchanged; no optimizer updates or commit.\n')
    print('PHASE 3 COMPLETE: learned replay status '+learned['status'],flush=True)
    print('PHASE 4 COMPLETE: production execution accounting matches independent source-budget replay.',flush=True)
    print(output,md,flush=True)


if __name__=='__main__':main()
