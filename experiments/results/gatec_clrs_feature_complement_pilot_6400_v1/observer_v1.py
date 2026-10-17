"""Versioned paired accounting around the unchanged native production step.

Purity/commit observations follow the sealed CLRS training observer. No candidate
construction, selector, alternate dispatch or state transition is implemented.
"""
import os,sys,gzip,json,time,shutil
from support_v1 import OUT,ENG,save
sys.path.insert(0,str(ENG))
import torch
from clrs_adapter_v1 import InsertionSortAdapter,fields,snapshot,assert_equal,assert_disjoint,metadata,pointer_loss,pointer_predictions
import src.agt as agt_module
from pilot_modules_v1 import active_mixers,EXCLUDED_NAME

class ResourceStop(RuntimeError):pass
class NumericalFailure(RuntimeError):pass

class Ledger:
    def __init__(self,arm,start,deadline):
        self.arm=arm;self.started=start;self.deadline=deadline;self.phase='initialization'
        self.attempted=self.completed=self.failed=0
        self.bank_attempted=self.bank_completed=self.commits=self.contextual_entries=0
        self.backward_attempted=self.backward_completed=self.backward_failed=0
        self.optimizer_attempted=self.optimizer_completed=self.optimizer_failed=0
        self.array_attempted=self.array_completed=self.array_failed=0
        self.phase_counts={};self.current_token=None
        self.trace=gzip.open(OUT/(arm+'_token_traces_v1.jsonl.gz'),'xt')
    def check(self):
        if time.monotonic()>=self.deadline:raise ResourceStop('Six-hour paired deadline')
        if shutil.disk_usage(OUT).free<256*1024**2:raise ResourceStop('Storage reserve')
    def summary(self):
        return dict(arm=self.arm,attempted_tokens=self.attempted,completed_tokens=self.completed,failed_tokens=self.failed,selected_bank_attempted=self.bank_attempted,selected_bank_completed=self.bank_completed,commits=self.commits,contextual_commit_entries=self.contextual_entries,backward_attempted=self.backward_attempted,backward_completed=self.backward_completed,backward_failed=self.backward_failed,optimizer_attempted=self.optimizer_attempted,optimizer_updates=self.optimizer_completed,optimizer_failed=self.optimizer_failed,array_attempted=self.array_attempted,array_completed=self.array_completed,array_failed=self.array_failed,phase_counts=self.phase_counts,elapsed_since_shared_clock=time.monotonic()-self.started)
    def persist(self):save(self.arm+'_execution_ledger_v1.json',self.summary())
    def token(self,body,x,parent,case_id,token,scripted):
        self.check();assert not scripted
        if self.attempted>=41216 or self.bank_attempted+8>329728:raise ResourceStop('Per-arm model-token/bank cap')
        self.attempted+=1;self.current_token=token
        pc=self.phase_counts.setdefault(self.phase,dict(attempted=0,completed=0,failed=0,selected_attempted=0,selected_completed=0,commits=0,contextual_entries=0));pc['attempted']+=1
        row=dict(arm=self.arm,phase=self.phase,attempt=self.attempted,case=case_id,token=token,device=str(x.device),status='attempted',blocks=[],selected_attempted=0,commits=0,selection_source='Unchanged production native canonical bank loop',elapsed_seconds=time.monotonic()-self.started)
        handles=[];block_snap={};counts={i:0 for i in range(8)};old_commit=agt_module.terminal_commit
        ex=body.executor;old_candidates=ex.candidate_successors
        def blocked(*a,**k):raise AssertionError('Controller/exhaustive candidate path forbidden')
        def pre(i):
            def hook(module,args,kwargs):
                self.check();assert i==len(row['blocks']),'Selected block order'
                self.bank_attempted+=1;pc['selected_attempted']+=1;row['selected_attempted']+=1
                block_snap[i]=(snapshot(kwargs['C_prev']) if kwargs.get('C_prev') is not None else {},kwargs['h0'].detach().clone(),kwargs['R'].detach().clone())
            return hook
        def post(i):
            def hook(module,args,kwargs,value):
                previous,anchor,memory=block_snap[i]
                assert_equal(previous,snapshot(kwargs['C_prev']) if kwargs.get('C_prev') is not None else {},True)
                assert torch.equal(anchor,kwargs['h0']) and torch.equal(memory,kwargs['R'])
                assert counts[i]==1,'Each selected block must invoke its exact active mixer once'
                row['blocks'].append(i);self.bank_completed+=1;pc['selected_completed']+=1
            return hook
        def mhc(i):
            def hook(module,args,value):
                assert args[0].shape==value.shape==(1,1,64) and args[0].dtype==value.dtype==torch.float32
                counts[i]+=1
            return hook
        def commit(state,**kwargs):
            versions={k:v for k,v in fields(state).items() if k.endswith('/version')}
            result,plan=old_commit(state,**kwargs)
            self.commits+=1;pc['commits']+=1;row['commits']+=1
            self.contextual_entries+=len(plan.entries);pc['contextual_entries']+=len(plan.entries)
            after={k:v for k,v in fields(result).items() if k.endswith('/version')}
            row['commit']=dict(token_index=plan.token_index,entry_count=len(plan.entries),versions_before=versions,versions_after=after)
            assert len(plan.entries)==7 and versions.keys()==after.keys() and all(after[k]==versions[k]+1 for k in versions)
            assert state._consumed and not result._consumed and result.tokens_committed==parent.tokens_committed+1
            return result,plan
        try:
            row['incoming_metadata']=metadata(parent);before=snapshot(parent)
            handles.append(body.controller.register_forward_pre_hook(blocked))
            handles.append(body.prelude_block.mhc.register_forward_pre_hook(lambda *a:(_ for _ in ()).throw(AssertionError('Unexecuted prelude mixer called'))))
            for i,block in enumerate(body.blocks.blocks):
                handles.append(block.register_forward_pre_hook(pre(i),with_kwargs=True));handles.append(block.register_forward_hook(post(i),with_kwargs=True))
                module=block.mhc if i<6 else block.core.mhc;handles.append(module.register_forward_hook(mhc(i)))
            agt_module.terminal_commit=commit;ex.candidate_successors=blocked
            result=body.step(x,parent,x.new_zeros(1,64))
            assert row['blocks']==list(range(8)) and row['commits']==1 and all(v==1 for v in counts.values())
            assert_equal(before,snapshot(parent),True);assert_disjoint(parent,result['state'])
            if not bool(torch.isfinite(result['logits']).all()):raise NumericalFailure('Nonfinite frozen vocabulary output')
            row.update(status='completed',D=8,P=8,selected_actions=list(row['blocks']),mixer_calls=[counts[i] for i in range(8)],parent_unchanged=True,returned_state_disjoint=True,returned_metadata=metadata(result['state']))
            self.completed+=1;pc['completed']+=1
            return result
        except BaseException as e:
            self.failed+=1;pc['failed']+=1;row.update(status='failed',error_type=type(e).__name__,error=str(e));raise
        finally:
            agt_module.terminal_commit=old_commit;ex.candidate_successors=old_candidates
            for h in handles:h.remove()
            self.trace.write(json.dumps(row,allow_nan=False)+'\n');self.trace.flush();self.current_token=None
            self.persist()
    def backward(self,loss):
        self.check()
        if self.backward_attempted>=1000:raise ResourceStop('Per-arm backward cap')
        self.backward_attempted+=1
        try:loss.backward();self.backward_completed+=1
        except BaseException:self.backward_failed+=1;raise
        self.check()
