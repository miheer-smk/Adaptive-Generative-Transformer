"""Frozen-cache instrumentation derived from the sealed diagnostic observer; production token execution unchanged."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import copy,datetime,gzip,hashlib,json,shutil,subprocess,time,traceback
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'gatec_clrs_insertion_sort_6200_v1'))
import torch
from clrs_adapter_v1 import (InsertionSortAdapter, fields, snapshot, assert_equal,
    assert_disjoint, metadata, cpu_snapshot, pointer_loss, pointer_predictions)
import src.agt as agt_module
from src.masks import build_execution_mask

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]

def save(name,x):
    (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def parameter_identity(model):
    h=hashlib.sha256()
    for n,v in sorted(list(model.named_parameters())+list(model.named_buffers())):
        h.update(n.encode());h.update(str((v.shape,v.dtype)).encode())
        h.update(bytes(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).tolist()))
    return h.hexdigest()

class Ledger:
    def __init__(self):
        self.attempted=self.completed=self.failed=0
        self.bank_attempted=self.bank_completed=self.commits=0
        self.backward_attempted=self.backward_completed=0
        self.phase='setup';self.pending=None;self.started=time.monotonic()
        self.optimizer_attempted=self.optimizer_completed=self.optimizer_failed=0
        self.array_attempted=self.array_completed=self.array_failed=0
        self.phase_counts={};self.deadline=self.started+900
        self.trace=gzip.open(OUT/'token_traces_v1.jsonl.gz','wt')
    def summary(self):
        return dict(attempted_tokens=self.attempted,completed_tokens=self.completed,failed_tokens=self.failed,
            selected_bank_attempted=self.bank_attempted,selected_bank_completed=self.bank_completed,
            commits=self.commits,backward_attempted=self.backward_attempted,backward_completed=self.backward_completed,
            optimizer_objects=0,optimizer_updates=0,phase_counts=self.phase_counts,
            contextual_commit_entries=self.commits*7,elapsed_seconds=time.monotonic()-self.started,
            peak_CUDA_allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_CUDA_reserved=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
    def check(self):
        if time.monotonic()>=self.deadline:raise RuntimeError("RESOURCE_STOP: 15-minute extraction deadline")
        if shutil.disk_usage(OUT).free<256*1024**2:raise RuntimeError("RESOURCE_STOP: storage reserve")
    def token(self,body,x,parent,case_id,token,scripted):
        self.check()
        if self.attempted>=4064:raise RuntimeError('RESOURCE_STOP: attempted-token cap')
        if self.bank_attempted+8>32512:raise RuntimeError('RESOURCE_STOP: bank cap')
        if shutil.disk_usage(OUT).free<128*1024**2:raise RuntimeError('RESOURCE_STOP: disk reserve')
        self.attempted+=1
        pc=self.phase_counts.setdefault(self.phase,dict(attempted=0,completed=0,failed=0,selected_attempted=0,selected_completed=0,commits=0))
        pc['attempted']+=1
        row=dict(attempt=self.attempted,phase=self.phase,case=case_id,token=token,
                 device=str(x.device),scripted=scripted,elapsed_seconds=time.monotonic()-self.started,status='attempted',blocks=[],commits=0,
                 incoming_metadata=metadata(parent),selected_actions=[])
        self.pending=row;before_parent=snapshot(parent);handles=[];block_snap={}
        old_commit=agt_module.terminal_commit;ex=body.executor
        old_controller,old_candidates=ex.controller,ex.candidate_successors
        def blocked(*a,**k):raise AssertionError('Controller or exhaustive candidates unexpectedly used')
        handles.append(body.controller.register_forward_pre_hook(lambda *a:blocked()))
        def pre(index):
            def hook(module,args,kwargs):
                assert index==len(row['blocks']), 'Wrong selected block order'
                self.bank_attempted+=1;pc['selected_attempted']+=1
                block_snap[index]=(snapshot(kwargs['C_prev']) if kwargs.get('C_prev') is not None else {},
                                   kwargs.get('h0').detach().clone(),kwargs.get('R').detach().clone())
            return hook
        def post(index):
            def hook(module,args,kwargs,result):
                previous,anchor,memory=block_snap[index]
                assert_equal(previous,snapshot(kwargs['C_prev']) if kwargs.get('C_prev') is not None else {},True)
                assert torch.equal(anchor,kwargs['h0']) and torch.equal(memory,kwargs['R'])
                row['blocks'].append(index);self.bank_completed+=1;pc['selected_completed']+=1
            return hook
        for i,block in enumerate(body.blocks.blocks):
            handles.append(block.register_forward_pre_hook(pre(i),with_kwargs=True))
            handles.append(block.register_forward_hook(post(i),with_kwargs=True))
        def commit(state,**kwargs):
            versions={k:v for k,v in fields(state).items() if k.endswith('/version')}
            result,plan=old_commit(state,**kwargs)
            self.commits+=1;row['commits']+=1;pc['commits']+=1
            after={k:v for k,v in fields(result).items() if k.endswith('/version')}
            row['commit']=dict(token_index=plan.token_index,entry_count=len(plan.entries),versions_before=versions,versions_after=after)
            assert len(plan.entries)==7 and versions.keys()==after.keys()
            assert all(after[k]==versions[k]+1 for k in versions)
            assert state._consumed and not result._consumed
            assert result.tokens_committed==parent.tokens_committed+1
            return result,plan
        def provider(**features):
            action=len(row['selected_actions']);assert action<8
            pointer=int(features['pointer_one_hot'].argmax(-1));assert pointer==max(0,action-1)
            local_mask=build_execution_mask(pointer,action,False,8).to(features['h'].device)
            logits=features['h'].new_full((1,11),-100.);logits[:,action]=100.
            selected=int(logits[:,:10].masked_fill(~local_mask[:10],-torch.inf).argmax(-1))
            assert selected==action
            row['selected_actions'].append(action)
            row.setdefault('legal_masks',[]).append(local_mask.cpu().tolist())
            return {'logits':logits}
        try:
            agt_module.terminal_commit=commit;ex.candidate_successors=blocked
            if scripted:
                ex.controller=provider
                result=body.adaptive_differentiable_step(x,parent,x.new_zeros(1,64),
                    budget=x.new_zeros(1,2),training=False)
                assert result['execution_count']==8
                assert result['action_history']==[('ADVANCE',i) for i in range(8)]
                assert row['selected_actions']==list(range(8))
            else:
                result=body.step(x,parent,x.new_zeros(1,64))
                row['selected_actions']=list(range(8))
                row['controller_calls']=0
            assert row['blocks']==list(range(8)) and row['commits']==1
            assert bool(torch.isfinite(result['logits']).all())
            assert_equal(before_parent,snapshot(parent),True)
            assert_disjoint(parent,result['state'])
            row.update(status='completed',D=8,P=8,program_visits=8,parent_unchanged=True,
                       returned_state_disjoint=True,returned_metadata=metadata(result['state']))
            self.completed+=1;pc['completed']+=1
            return result
        except BaseException as e:
            self.failed+=1;pc['failed']+=1;row.update(status='failed',error=str(e));raise
        finally:
            agt_module.terminal_commit=old_commit
            ex.controller,ex.candidate_successors=old_controller,old_candidates
            for h in handles:h.remove()
            self.trace.write(json.dumps(row,allow_nan=False)+'\n');self.trace.flush();self.pending=None
            save('execution_ledger_v1.json',self.summary())
    def backward(self,loss):
        raise RuntimeError('Backward is not authorized for cache extraction')
