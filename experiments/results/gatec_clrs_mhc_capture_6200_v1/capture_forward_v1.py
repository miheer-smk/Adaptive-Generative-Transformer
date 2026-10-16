"""One frozen approved-adapter trajectory; read-only hooks, no retry or backward."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,hashlib,subprocess,time,traceback
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
LOCAL=OUT.parent/'gatec_clrs_pointer_localization_6200_v1'
ENG=OUT.parent/'gatec_clrs_insertion_sort_6200_v1'
sys.path.insert(0,str(LOCAL))
import torch
import diagnostic_observer_v1 as original
original.OUT=OUT  # Redirect only the existing research observer's output files.
from diagnostic_observer_v1 import (InsertionSortAdapter,parameter_identity,fields,
    metadata,sha,pointer_loss)

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def integrity():
    base=json.loads((OUT/'integrity_initial_v1.json').read_text());bad=[]
    for name,r in base['files'].items():
        p=ROOT/name
        if r['kind']=='symlink':ok=p.is_symlink() and os.readlink(p)==r['target'] and p.lstat().st_mode&0o777==r['mode']
        else:ok=p.is_file() and not p.is_symlink() and p.stat().st_size==r['size'] and sha(p)==r['sha256'] and p.stat().st_mode&0o777==r['mode']
        if not ok:bad.append(name)
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,env=env,text=True).strip()
    diff=hashlib.sha256(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT,env=env)).hexdigest()
    return dict(success=not bad and head==base['HEAD'] and diff==base['tracked_diff_sha256'],protected_files=len(base['files']),changed_or_missing=bad,HEAD_unchanged=head==base['HEAD'],tracked_diff_unchanged=diff==base['tracked_diff_sha256'],historical_exception_preserved=True)

class CaptureLedger(original.Ledger):
    def __init__(self):
        super().__init__();self.phase='final_frozen_capture';self.current_token=None
        self.states=[];self.state_values=[];self.h0=[];self.candidates=[]
    def token(self,body,x,parent,case_id,token,scripted):
        if self.attempted>=32 or self.bank_attempted+8>256:raise RuntimeError('Authorized capture cap reached')
        assert token==self.attempted and not scripted
        self.current_token=token
        out=super().token(body,x,parent,case_id,token,scripted)
        self.h0.append(out['h0'].detach().cpu().clone())
        self.state_values.append(out['state'].s.detach().cpu().clone())
        self.states.append(dict(token=token,kind='candidate' if token<16 else 'query',original_index=token%16,
            metadata=metadata(out['state']),tensor_summaries={k:dict(shape=list(v.shape),dtype=str(v.dtype),finite=bool(torch.isfinite(v).all()),norm=float(v.norm()),mean=float(v.mean()) if v.numel() else None) for k,v in fields(out['state']).items() if torch.is_tensor(v)}))
        for value,frozen in self.candidates:assert torch.equal(value,frozen),'Prior candidate mutated'
        if token<16:self.candidates.append((out['state'].s,out['state'].s.detach().clone()))
        return out

def main():
    if (OUT/'execution_started_v1.json').exists():raise SystemExit('STOP: no retry authorized')
    manifest=json.loads((OUT/'manifest_v1.json').read_text())
    for name,r in manifest['inputs'].items():assert sha(ROOT/name)==r['sha256'],name
    seal=json.loads((OUT/'capture_seal_v1.json').read_text())
    for name,h in seal['files'].items():assert sha(OUT/name)==h,name
    before=integrity();assert before['success']
    row=json.loads((LOCAL/'locked_input_v1.json').read_text());assert row==json.loads((ENG/'train_pool_v1.json').read_text())[0]
    assert torch.cuda.is_available(),'Required previous CUDA device unavailable'
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.init();torch.cuda.reset_peak_memory_stats()
    ledger=CaptureLedger();result=dict(status='STOPPED',model_constructions_attempted=0,model_constructions_completed=0,checkpoint_loads=0,rollouts_attempted=0,rollouts_completed=0,backwards=0,optimizer_objects=0,optimizer_updates=0)
    capture=dict(mhc={str(i):dict(inputs=[],outputs=[]) for i in range(8)},blocks={str(i):dict(inputs=[],outputs=[]) for i in range(8)})
    handles=[];invocations=[];module_counts={}
    save('execution_started_v1.json',dict(pid=os.getpid(),retry=False,token_cap=32,backwards=0))
    try:
        result['model_constructions_attempted']=1
        model=InsertionSortAdapter().to('cuda');result['model_constructions_completed']=1
        checkpoint=torch.load(ENG/'training_results_v1/model_update1000_v1.pt',map_location='cuda',weights_only=True)
        assert checkpoint['optimizer_update_count']==1000
        model.load_state_dict(checkpoint['model_state_dict'],strict=True);result['checkpoint_loads']=1;del checkpoint
        assert parameter_identity(model)==manifest['expected_model_identity']
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        initial_identity=parameter_identity(model)
        prior=torch.load(LOCAL/'final_tensor_evidence_v1.pt',map_location='cpu',weights_only=True)
        expected=json.loads((LOCAL/'final_results_v1.json').read_text())
        def cpu(x):return x.detach().cpu().clone()
        def block_hook(i):
            def hook(module,args,output):
                output=output[0] if isinstance(output,tuple) else output
                capture['blocks'][str(i)]['inputs'].append(cpu(args[0]));capture['blocks'][str(i)]['outputs'].append(cpu(output))
            return hook
        def mhc_hook(i,name):
            def hook(module,args,output):
                token=ledger.current_token
                assert token is not None and len(capture['mhc'][str(i)]['inputs'])==token
                x,y=cpu(args[0]),cpu(output)
                assert x.shape==y.shape==(1,1,64) and x.dtype==y.dtype==torch.float32
                assert torch.isfinite(x).all() and torch.isfinite(y).all()
                capture['mhc'][str(i)]['inputs'].append(x);capture['mhc'][str(i)]['outputs'].append(y)
                module_counts[name]=module_counts.get(name,0)+1
                invocations.append(dict(token=token,kind='candidate' if token<16 else 'query',original_index=token%16,
                    block=i,operator=model.body.blocks.blocks[i].species,module=name,shape=list(x.shape),dtype=str(x.dtype),
                    device=str(args[0].device),input_norm=float(x.norm()),output_norm=float(y.norm()),input_mean=float(x.mean()),output_mean=float(y.mean()),finite=True))
            return hook
        active={}
        for i,b in enumerate(model.body.blocks.blocks):
            module=b.core.mhc if b.species=='FFN' else b.mhc
            name=next(n for n,m in model.named_modules() if m is module);active[name]=i
            handles.append(b.register_forward_hook(block_hook(i)))
            handles.append(module.register_forward_hook(mhc_hook(i,name)))
        def inactive_hook(module,args,output):raise AssertionError('Inactive prelude mixer unexpectedly called')
        for name,module in model.named_modules():
            if module.__class__.__name__=='MHC' and name not in active:
                module_counts[name]=0;handles.append(module.register_forward_hook(inactive_hook))
        save('runtime_identity_v1.json',dict(model_identity=initial_identity,checkpoint_sha256=manifest['checkpoint_sha256'],device='cuda',device_name=torch.cuda.get_device_name(),torch=torch.__version__,CUDA=torch.version.cuda,FP32=True,TF32=False,all_parameters_frozen=True,evaluation_mode=not model.training,grad_enabled_during_trajectory=False,active_mixers=active))
        result['rollouts_attempted']=1
        with torch.no_grad():
            out=model(row['key_fp32'],row['pos'],ledger,row['id'])
            loss=pointer_loss(out['pointer_logits'],row['pred'])
        torch.cuda.synchronize();result['rollouts_completed']=1
        capture['scores']=cpu(out['pointer_logits'][0]);capture['CE']=float(loss);capture['targets']=torch.tensor(row['pred'])
        capture['h0']=torch.cat(ledger.h0);capture['committed_streams']=torch.cat(ledger.state_values)
        for family in ['mhc','blocks']:
            for values in capture[family].values():
                for k,x in values.items():assert len(x)==32;values[k]=torch.cat(x).squeeze(1)
        with (OUT/'captured_tensors_v1.pt').open('xb') as f:torch.save(capture,f)
        save('mhc_invocations_v1.json',invocations);save('committed_state_summaries_v1.json',ledger.states)
        comparisons={}
        def compare(name,x,y):
            assert x.shape==y.shape and x.dtype==y.dtype,name
            ok=bool(torch.allclose(x,y,atol=1e-6,rtol=1e-5))
            comparisons[name]=dict(pass_comparison=ok,shape=list(x.shape),max_absolute_difference=float((x-y).abs().max()),exact=bool(torch.equal(x,y)))
            assert ok,'Retained evidence mismatch: '+name
        compare('pointer_scores',capture['scores'],prior['raw_scores'])
        assert abs(float(loss)-expected['CE'])<=1e-6+1e-5*abs(expected['CE']),'CE mismatch'
        compare('h0',capture['h0'],prior['h0'])
        compare('committed_streams',capture['committed_streams'],prior['all_committed_streams'])
        for i in range(8):
            compare('block_output_'+str(i),capture['blocks'][str(i)]['outputs'],prior['block_outputs'][str(i)])
            residual=capture['blocks'][str(i)]['inputs']+capture['mhc'][str(i)]['outputs']
            assert torch.equal(residual,capture['blocks'][str(i)]['outputs']),'Residual identity mismatch'
        assert ledger.attempted==ledger.completed==32 and ledger.failed==0
        assert ledger.bank_attempted==ledger.bank_completed==256 and ledger.commits==32
        assert len(invocations)==256 and all(module_counts[n]==32 for n in active)
        assert parameter_identity(model)==initial_identity and all(p.grad is None for p in model.parameters())
        save('forward_comparisons_v1.json',dict(success=True,tolerances=dict(atol=1e-6,rtol=1e-5),comparisons=comparisons,CE=float(loss),retained_CE=expected['CE'],absolute_CE_difference=abs(float(loss)-expected['CE']),residual_addition_bitwise=True,parameter_and_buffer_identity_unchanged=True,module_invocation_counts=module_counts))
        result['status']='COMPLETED_ONE_FROZEN_FORWARD'
    except BaseException as e:
        result['failure']=dict(type=type(e).__name__,message=str(e))
        save('failure_v1.json',dict(**result['failure'],traceback=traceback.format_exc(),no_retry=True))
        if not (OUT/'captured_tensors_v1.pt').exists():
            with (OUT/'partial_capture_v1.pt').open('xb') as f:torch.save(capture,f)
        save('mhc_invocations_partial_v1.json',invocations)
    finally:
        for h in handles:h.remove()
        ledger.trace.close();result['ledger']=ledger.summary();save('execution_ledger_v1.json',result['ledger'])
        result['integrity']=integrity();save('integrity_after_forward_v1.json',result['integrity'])
        if not result['integrity']['success']:result['status']='STOPPED_INTEGRITY_FAILURE'
        save('capture_results_v1.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
