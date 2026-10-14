"""Single non-resumable 512-token engineering attempt. No optimizer exists."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import copy,datetime,gzip,hashlib,json,shutil,subprocess,time,traceback
from pathlib import Path
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
        self.trace=gzip.open(OUT/'token_traces_v1.jsonl.gz','wt')
    def summary(self):
        return dict(attempted_tokens=self.attempted,completed_tokens=self.completed,failed_tokens=self.failed,
            selected_bank_attempted=self.bank_attempted,selected_bank_completed=self.bank_completed,
            commits=self.commits,backward_attempted=self.backward_attempted,
            backward_completed=self.backward_completed,optimizer_objects=0,optimizer_updates=0,
            contextual_commit_entries=self.commits*7,elapsed_seconds=time.monotonic()-self.started,
            peak_CUDA_allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_CUDA_reserved=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
    def token(self,body,x,parent,case_id,token,scripted):
        if self.attempted>=512:raise RuntimeError('RESOURCE_STOP: attempted-token cap')
        if self.bank_attempted+8>4096:raise RuntimeError('RESOURCE_STOP: bank cap')
        if shutil.disk_usage(OUT).free<128*1024**2:raise RuntimeError('RESOURCE_STOP: disk reserve')
        self.attempted+=1
        row=dict(attempt=self.attempted,phase=self.phase,case=case_id,token=token,
                 device=str(x.device),scripted=scripted,status='attempted',blocks=[],commits=0,
                 incoming_metadata=metadata(parent),selected_actions=[])
        self.pending=row;before_parent=snapshot(parent);handles=[];block_snap={}
        old_commit=agt_module.terminal_commit;ex=body.executor
        old_controller,old_candidates=ex.controller,ex.candidate_successors
        def blocked(*a,**k):raise AssertionError('Controller or exhaustive candidates unexpectedly used')
        handles.append(body.controller.register_forward_pre_hook(lambda *a:blocked()))
        def pre(index):
            def hook(module,args,kwargs):
                assert index==len(row['blocks']), 'Wrong selected block order'
                self.bank_attempted+=1
                block_snap[index]=(snapshot(kwargs['C_prev']) if kwargs.get('C_prev') is not None else {},
                                   kwargs.get('h0').detach().clone(),kwargs.get('R').detach().clone())
            return hook
        def post(index):
            def hook(module,args,kwargs,result):
                previous,anchor,memory=block_snap[index]
                assert_equal(previous,snapshot(kwargs['C_prev']) if kwargs.get('C_prev') is not None else {},True)
                assert torch.equal(anchor,kwargs['h0']) and torch.equal(memory,kwargs['R'])
                row['blocks'].append(index);self.bank_completed+=1
            return hook
        for i,block in enumerate(body.blocks.blocks):
            handles.append(block.register_forward_pre_hook(pre(i),with_kwargs=True))
            handles.append(block.register_forward_hook(post(i),with_kwargs=True))
        def commit(state,**kwargs):
            versions={k:v for k,v in fields(state).items() if k.endswith('/version')}
            result,plan=old_commit(state,**kwargs)
            self.commits+=1;row['commits']+=1
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
            row.update(status='completed',P=8,program_visits=8,parent_unchanged=True,
                       returned_state_disjoint=True,returned_metadata=metadata(result['state']))
            self.completed+=1
            return result
        except BaseException as e:
            self.failed+=1;row.update(status='failed',error=str(e));raise
        finally:
            agt_module.terminal_commit=old_commit
            ex.controller,ex.candidate_successors=old_controller,old_candidates
            for h in handles:h.remove()
            self.trace.write(json.dumps(row,allow_nan=False)+'\n');self.trace.flush();self.pending=None
            save('execution_ledger_v1.json',self.summary())
    def backward(self,loss):
        if self.backward_attempted>=6:raise RuntimeError('RESOURCE_STOP: backward cap')
        self.backward_attempted+=1
        loss.backward();self.backward_completed+=1

def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()

def gradients(model,result):
    norms={}
    for name,p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(),name
            norms[name]=float(p.grad.norm())
    assert norms and sum(v*v for v in norms.values())>0
    assert any(name.startswith('body.') and norm>0 for name,norm in norms.items()), 'No core parameter credit'
    assert float(model.encoder.weight.grad.norm())>0
    assert float(model.query.weight.grad.norm())>0 and float(model.key.weight.grad.norm())>0
    candidate=[dict(connected=z.grad is not None,norm=float(z.grad.norm()) if z.grad is not None else None)
               for z in result['candidates']]
    assert all(x['connected'] for x in candidate) and any(x['norm']>0 for x in candidate)
    prefix={k:dict(connected=v.grad is not None,norm=float(v.grad.norm()) if v.grad is not None else None)
            for k,v in fields(result['prefix_state']).items() if torch.is_tensor(v) and v.requires_grad}
    assert prefix['/s']['connected']
    assert all(p.grad is None for p in model.body.controller.parameters())
    assert all(p.grad is None for p in model.body.output_head.parameters())
    return dict(parameter_gradient_norms=norms,candidate_gradients=candidate,prefix_adjoints=prefix,
                no_direct_label_input=True,frozen_unused_parameters_have_no_grad=True)

def integrity():
    original=json.loads((OUT/'integrity_initial.json').read_text());bad=[]
    for name,r in original['files'].items():
        f=ROOT/name
        if r['kind']=='symlink':ok=f.is_symlink() and os.readlink(f)==r['target']
        else:ok=f.is_file() and not f.is_symlink() and f.stat().st_size==r['size'] and sha(f)==r['sha256'] and f.stat().st_mode&0o777==r['mode']
        if not ok:bad.append(name)
    env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,env=env).decode().strip()
    diff=hashlib.sha256(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT,env=env)).hexdigest()
    r=dict(protected_count=len(original['files']),changed_or_missing=bad,
           HEAD_unchanged=head==original['HEAD'],tracked_diff_unchanged=diff==original['tracked_diff_sha256'],
           historical_exception_preserved=True)
    r['success']=not bad and r['HEAD_unchanged'] and r['tracked_diff_unchanged']
    save('integrity_final_v1.json',r);assert r['success'];return r

def main():
    if (OUT/'smoke_started_v1.json').exists():raise SystemExit('STOP: no repeat/resume')
    seal=json.loads((OUT/'implementation_seal_v1.json').read_text())
    for name,h in seal['files'].items():assert sha(OUT/name)==h,name
    assert json.loads((OUT/'data_export_verification_v1.json').read_text())['success']
    save('smoke_started_v1.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_authorized=False))
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available(),'CUDA prerequisite unavailable'
    torch.cuda.init();torch.cuda.reset_peak_memory_stats()
    ledger=Ledger();models={};result={'status':'STOPPED','training_executed':False};records=[];retained={}
    try:
        base=InsertionSortAdapter()
        models={'cpu':base,'cuda':copy.deepcopy(base).to('cuda')}
        identities={d:parameter_identity(m) for d,m in models.items()}
        assert identities['cpu']==identities['cuda']
        save('initial_model_identity_v1.json',dict(seed=6200,identities=identities,
            initialization='Fresh AGTv3 constructor, then encoder/query/key; CPU/CUDA independently owned identical copies',
            model_parameters=sum(p.numel() for p in base.parameters()),
            trainable_parameters=sum(p.numel() for p in base.parameters() if p.requires_grad),
            no_archival_checkpoint_loaded=True))
        fixtures=json.loads((OUT/'smoke_fixture_inputs_v1.json').read_text())
        for device,model in models.items():
            ledger.phase='fixtures_'+device;full={};canonical=None
            for row in fixtures:
                need_grad=row['n']==16;model.zero_grad(set_to_none=True)
                sync(device);begin=time.monotonic()
                with torch.set_grad_enabled(need_grad):
                    out=model(row['key_fp32'],row['pos'],ledger,row['id'],
                              retain_gradients=need_grad,retain_states=row['n']==2)
                    loss=pointer_loss(out['pointer_logits'],row['pred'])
                sync(device);forward=time.monotonic()-begin
                predictions=pointer_predictions(out['pointer_logits'])[0]
                record=dict(phase=ledger.phase,device=device,id=row['id'],n=row['n'],
                    loss=float(loss.detach()),predictions=predictions,truth=row['pred'],
                    pointer_logits=out['pointer_logits'].detach().cpu()[0].tolist(),
                    synchronized_forward_seconds=forward,autograd_enabled=need_grad,
                    target_alignment_pass=True)
                if need_grad:
                    begin=time.monotonic();ledger.backward(loss);sync(device)
                    record['synchronized_backward_seconds']=time.monotonic()-begin
                    record['gradients']=gradients(model,out)
                if row['n']==2:canonical=out['retained_states']
                full[f'N{row["n"]}/terminal']=cpu_snapshot(out['state'])
                records.append(record);save('smoke_records_partial_v1.json',records)
            ledger.phase='canonical_executor_comparison_'+device
            row=fixtures[0]
            with torch.no_grad():
                out=model(row['key_fp32'],row['pos'],ledger,row['id'],scripted=True,retain_states=True)
            assert len(out['retained_states'])==len(canonical)==4
            for i,(a,b) in enumerate(zip(canonical,out['retained_states'])):
                assert_equal(a,b)
                full[f'N2/fixed/{i}']=a;full[f'N2/scripted/{i}']=b
            # Same pointer function, same representations and entire returned state.
            fixed=next(r for r in records if r['device']==device and r['n']==2)
            assert torch.allclose(out['pointer_logits'].detach().cpu()[0],torch.tensor(fixed['pointer_logits']),atol=1e-6,rtol=1e-5)
            records.append(dict(phase=ledger.phase,device=device,n=2,
                full_state_and_metadata_equivalence=True,pointer_logits_equivalent=True))
            with (OUT/f'fixture_complete_states_{device}_v1.pt').open('xb') as f:torch.save(full,f)
        # No model optimizer exists. Four independent sequence initializations.
        row=next(r for r in fixtures if r['n']==16);model=models['cuda'];ledger.phase='CUDA_calibration'
        for i in range(4):
            model.zero_grad(set_to_none=True);sync('cuda');begin=time.monotonic()
            out=model(row['key_fp32'],row['pos'],ledger,row['id'],retain_gradients=True)
            loss=pointer_loss(out['pointer_logits'],row['pred']);sync('cuda')
            forward=time.monotonic()-begin;begin_back=time.monotonic()
            ledger.backward(loss);sync('cuda');backward=time.monotonic()-begin_back
            records.append(dict(phase=ledger.phase,rollout=i,warmup=i==0,
                forward_seconds=forward,backward_seconds=backward,total_seconds=forward+backward,
                loss=float(loss.detach()),gradients=gradients(model,out)))
            save('smoke_records_partial_v1.json',records)
        assert ledger.attempted==ledger.completed==512 and ledger.failed==0
        assert ledger.bank_attempted==ledger.bank_completed==4096 and ledger.commits==512
        assert ledger.backward_attempted==ledger.backward_completed==6
        final_id={d:parameter_identity(m) for d,m in models.items()};assert final_id==identities
        result.update(status='SMOKE_PASSED',parameter_and_buffer_identity_unchanged=True,
                      fixture_sizes=[2,4,8,16,64],CPU_and_CUDA_executed=True,
                      canonical_executor_equivalence=True,all_six_backwards_finite=True)
        save('parameter_identity_final_v1.json',dict(initial=identities,final=final_id,equal=True))
    except BaseException as e:
        result['failure']=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),phase=ledger.phase)
        save('failure_v1.json',result['failure'])
    finally:
        ledger.trace.close();save('execution_ledger_v1.json',ledger.summary())
        result['ledger']=ledger.summary();result['records']=records
        try:
            for name,h in seal['files'].items():assert sha(OUT/name)==h,name
            result['integrity']=integrity()
            if models:
                unchanged={d:parameter_identity(m)==identities[d] for d,m in models.items()}
                result['parameter_identity_unchanged']=unchanged;assert all(unchanged.values())
        except BaseException as e:
            result['status']='STOPPED';result['integrity_failure']=str(e)
        save('smoke_results_v1.json',result)
        print(result['status'],json.dumps(ledger.summary()),flush=True)

if __name__=='__main__':main()
