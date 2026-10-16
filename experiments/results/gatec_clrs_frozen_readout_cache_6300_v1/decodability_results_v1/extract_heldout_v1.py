"""Only the sealed 32 development and32 transfer arrays, after fit seal."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,time,signal,traceback,hashlib
import torch
from support_v1 import OUT,CACHE,ROOT,ENG,sha,save,check,deadline,integrity
from heldout_observer_v1 import Ledger,InsertionSortAdapter,parameter_identity

class Capture(Ledger):
    def begin(self,row):self.case=row['id'];self.n=row['n'];self.values=[];self.preserved=[]
    def token(self,body,x,parent,case_id,token,scripted):
        assert case_id==self.case and token==len(self.values) and token<2*self.n and parent.tokens_committed==token and not scripted
        result=super().token(body,x,parent,case_id,token,scripted)
        s=result['state'].s;assert s.shape==(1,64) and s.dtype==torch.float32 and torch.isfinite(s).all() and not s.requires_grad
        self.values.append(s.detach().cpu().clone())
        for old,value in self.preserved:assert torch.equal(old,value)
        if token<self.n:self.preserved.append((s,s.detach().clone()))
        self.check();return result

def main():
    check();assert not (OUT/'heldout_started_v1.json').exists(),'No extraction retry'
    seal=json.loads((OUT/'fit_seal_v1.json').read_text())
    for n,h in seal['files'].items():assert sha(OUT/n)==h,n
    assert json.loads((OUT/'fit_results_v1.json').read_text())['status']=='FIT_SEALED'
    assert integrity()['success']
    # Held-out data/targets are opened only after the terminal fit seal check.
    engineering_seal=json.loads((ENG/'implementation_seal_v1.json').read_text())['files']
    pools={};references={};manifest=[]
    for split,n in [('development',16),('size_transfer',64)]:
        name=split+'_pool_v1.json';assert sha(ENG/name)==engineering_seal[name]
        rows=json.loads((ENG/name).read_text());assert len(rows)==32
        refs=[json.loads(x) for x in (ENG/'training_results_v1'/(split+'_records_v1.jsonl')).read_text().splitlines()];assert len(refs)==32
        for i,(row,r) in enumerate(zip(rows,refs)):
            assert row['id']==r['id']==f'official/{split}/row{i:04d}' and row['n']==n and row['pred']==r['truth'] and row['original_input_sha256']==r['original_input_sha256']
            manifest.append(dict(split=split,row_index=i,id=row['id'],n=n,original_input_sha256=row['original_input_sha256'],targets=row['pred'],target_sha256=hashlib.sha256(json.dumps(row['pred'],separators=(',',':')).encode()).hexdigest()))
        pools[split]=rows;references[split]=refs
    save('heldout_input_manifest_v1.json',dict(arrays=manifest,fit_seal_sha256=sha(OUT/'fit_seal_v1.json'),held_out_from_fitting_only=True,previously_examined_historically=True))
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available();torch.cuda.init();torch.cuda.reset_peak_memory_stats()
    ledger=Capture();result=dict(status='INCOMPLETE',arrays_attempted=0,arrays_completed=0,arrays_failed=0,AGT_optimizer_objects=0,AGT_backwards=0)
    save('heldout_started_v1.json',dict(monotonic=time.monotonic(),deadline=deadline(),fit_seal_verified=True,warmups=0,retries=0))
    old=signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(RuntimeError('Remaining-stage deadline')))
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline()-time.monotonic()))
    captured={k:[] for k in pools};records=[];handles=[];model=None;identity=None;current=None
    try:
        model=InsertionSortAdapter().to('cuda')
        cp=ENG/'training_results_v1/model_update1000_v1.pt';assert sha(cp)=='2c0e3d6ad3711d8f0ee209eb3e3b0f8db8964e6eeaf2be77b7bd56120f29e6c8'
        ck=torch.load(cp,map_location='cpu',weights_only=True);assert ck['optimizer_update_count']==1000;model.load_state_dict(ck['model_state_dict'],strict=True);del ck
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        identity=parameter_identity(model);assert identity=='7ab9239da5b1a4957cde546bb95e4c2961800b337f8bf677f8202e662e94f1a0'
        save('heldout_runtime_v1.json',dict(device='cuda',name=torch.cuda.get_device_name(),torch=torch.__version__,CUDA=torch.version.cuda,FP32=True,TF32=False,all_parameters_frozen=True,model_identity=identity))
        hookdata={}
        def qhook(m,args,out):hookdata['queries'].append(args[0].detach().cpu().clone())
        def khook(m,args,out):hookdata['candidate_input']=args[0].detach().cpu().clone()
        handles=[model.query.register_forward_hook(qhook),model.key.register_forward_hook(khook)]
        with torch.no_grad():
            for split,rows in pools.items():
                ledger.phase=split+'_cache'
                for i,row in enumerate(rows):
                    check();current=row;ledger.begin(row);n=row['n'];hookdata.clear();hookdata['queries']=[]
                    result['arrays_attempted']+=1;start=time.monotonic()
                    out=model(row['key_fp32'],row['pos'],ledger,row['id']);torch.cuda.synchronize()
                    assert len(ledger.values)==2*n and out['state'].tokens_committed==2*n
                    C=torch.cat(ledger.values[:n]);Q=torch.cat(ledger.values[n:]);S=out['pointer_logits'][0].detach().cpu().clone()
                    assert torch.equal(C,hookdata['candidate_input'][0]) and torch.equal(Q,torch.cat(hookdata['queries']))
                    for j,z in enumerate(out['candidates']):assert torch.equal(z.detach().cpu(),C[j:j+1])
                    ref=references[split][i];expected=torch.tensor(ref['pointer_logits'],dtype=torch.float32)
                    ok=bool(torch.allclose(S,expected,atol=1e-6,rtol=1e-5));same=S.argmax(1).tolist()==ref['predictions']
                    record=dict(split=split,id=row['id'],n=n,input_sha256=row['original_input_sha256'],scores_within_tolerance=ok,predictions_exact=same,bitwise_scores=torch.equal(S,expected),max_score_error=float((S-expected).abs().max()),elapsed=time.monotonic()-start,commits=2*n,selected_calls=16*n)
                    records.append(record);save('heldout_equivalence_progress_v1.json',records)
                    assert ok and same,'EXTRACTION_INTEGRITY_FAILURE: original head differs from archive'
                    captured[split].append(dict(id=row['id'],candidate_committed_s=C,query_committed_s=Q,targets=torch.tensor(row['pred']),original_pointer_scores=S,input_sha256=row['original_input_sha256']))
                    result['arrays_completed']+=1;check();print('Verified',split,i+1,'/32',flush=True)
                    del out,C,Q,S
        assert ledger.attempted==ledger.completed==ledger.commits==5120 and ledger.failed==0 and ledger.bank_attempted==ledger.bank_completed==40960
        assert all(len(v)==32 for v in captured.values());assert parameter_identity(model)==identity
        result['status']='COMPLETED_VERIFIED_HELDOUT_CACHE'
    except BaseException as e:
        result['arrays_failed']=result['arrays_attempted']-result['arrays_completed']
        result['failure']=dict(type=type(e).__name__,message=str(e),case=current['id'] if current else None)
        save('heldout_failure_v1.json',dict(**result['failure'],traceback=traceback.format_exc(),retry=False))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        for h in handles:h.remove()
        ledger.trace.close();result['ledger']=ledger.summary();save('heldout_execution_ledger_v1.json',result['ledger'])
        if model is not None and identity is not None:
            result['parameter_identity_after']=parameter_identity(model);assert result['parameter_identity_after']==identity
        name='heldout_representations_v1.pt' if result['status'].startswith('COMPLETED') else 'partial_heldout_representations_v1.pt'
        with (OUT/name).open('xb') as f:torch.save(captured,f)
        result['cache_sha256']=sha(OUT/name);result['cache_file']=name;save('heldout_extraction_results_v1.json',result)
        save('heldout_equivalence_v1.json',dict(success=result['status'].startswith('COMPLETED'),atol=1e-6,rtol=1e-5,comparisons=records,fit_sealed_before_extraction=True))
        print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
