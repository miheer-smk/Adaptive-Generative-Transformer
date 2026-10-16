"""127 frozen training trajectories; reuse row0; no fitting, backward or retry."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,hashlib,signal,time,traceback,datetime
from pathlib import Path
import torch
from cache_observer_v1 import (Ledger,InsertionSortAdapter,parameter_identity,sha)
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
ENG=OUT.parent/'gatec_clrs_insertion_sort_6200_v1'
TRAIN=ENG/'training_results_v1'
LOCAL=OUT.parent/'gatec_clrs_pointer_localization_6200_v1'

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def tensor_sha(x):
    assert x.device.type=='cpu' and not x.requires_grad and x.grad_fn is None
    return hashlib.sha256(bytes(x.contiguous().reshape(-1).view(torch.uint8).tolist())).hexdigest()
def finite(x):return bool(torch.isfinite(x).all())

class ExtractionLedger(Ledger):
    def __init__(self):
        super().__init__();self.phase='training_cache_extraction';self.case_id=None
        self.row_values=[];self.row_states=[];self.preserved_candidates=[]
    def begin(self,row):
        self.case_id=row['id'];self.row_values=[];self.row_states=[];self.preserved_candidates=[]
    def token(self,body,x,parent,case_id,token,scripted):
        assert case_id==self.case_id and token==len(self.row_values) and 0<=token<32
        assert parent.tokens_committed==token and not scripted
        out=super().token(body,x,parent,case_id,token,scripted)
        value=out['state'].s;assert value.shape==(1,64) and value.dtype==torch.float32 and finite(value)
        assert not value.requires_grad and value.grad_fn is None
        self.row_values.append(value.detach().cpu().clone())
        for old,frozen in self.preserved_candidates:assert torch.equal(old,frozen),'Stored candidate mutated'
        if token<16:self.preserved_candidates.append((value,value.detach().clone()))
        self.check()
        return out

def main():
    if (OUT/'extraction_started_v1.json').exists():raise SystemExit('STOP: extraction already attempted; no retry')
    manifest=json.loads((OUT/'manifest_v1.json').read_text());inputs=json.loads((OUT/'input_manifest_v1.json').read_text())
    for name,r in manifest['inputs'].items():assert sha(ROOT/name)==r['sha256'],name
    for name,h in json.loads((OUT/'extraction_seal_v1.json').read_text())['files'].items():assert sha(OUT/name)==h,name
    pool=json.loads((ENG/'train_pool_v1.json').read_text());rows=pool[:128]
    refs=[json.loads(line) for line in (TRAIN/'training_subset_records_v1.jsonl').read_text().splitlines()]
    assert len(refs)==128 and [r['id'] for r in rows]==[x['id'] for x in inputs['arrays']]
    assert rows[0]==json.loads((LOCAL/'locked_input_v1.json').read_text())
    assert torch.cuda.is_available(),'Original CUDA runtime is required'
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ExtractionLedger();verified=[];completed_tensors=[];comparisons=[];current=None;model=None;handles=[]
    projected=dict(queries=[],keys=None,key_input=None)
    result=dict(status='INCOMPLETE_CACHE',model_constructions_attempted=0,model_constructions_completed=0,
        checkpoint_loads=0,new_arrays_attempted=0,new_arrays_forward_completed=0,new_arrays_verified=0,new_arrays_failed=0,
        reused_arrays=0,backwards=0,optimizer_objects=0,optimizer_updates=0,held_out_arrays=0,retries=0,warmups=0)
    stream=(OUT/'per_array_extraction_v1.jsonl').open('x')
    def expired(signum,frame):raise RuntimeError('RESOURCE_STOP: 900-second extraction hard deadline')
    old_alarm=signal.signal(signal.SIGALRM,expired)
    save('extraction_started_v1.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),
        no_retry=True,phase_start_monotonic=ledger.started,deadline_monotonic=ledger.deadline,limit_seconds=900,
        scopes='CUDA initialization, construction/load, row0 replay and 127 new frozen trajectories; offline closure/publication outside extraction'))
    signal.setitimer(signal.ITIMER_REAL,max(0.001,ledger.deadline-time.monotonic()))
    Wq=Wk=None;before_identity=None
    def verify_entry(i,C,Q,actual_scores=None,extra=None):
        row=rows[i];ref=refs[i];entry=inputs['arrays'][i]
        assert ref['id']==row['id']==entry['id'] and ref['truth']==row['pred'] and ref['original_input_sha256']==row['original_input_sha256']
        assert C.shape==Q.shape==(16,64) and C.dtype==Q.dtype==torch.float32
        assert C.device.type==Q.device.type=='cpu' and finite(C) and finite(Q)
        assert not C.requires_grad and not Q.requires_grad and C.grad_fn is None and Q.grad_fn is None
        target=torch.tensor(row['pred'],dtype=torch.long);assert target.shape==(16,) and ((target>=0)&(target<16)).all()
        assert hashlib.sha256(json.dumps(row['pred'],separators=(',',':')).encode()).hexdigest()==entry['target_sha256']
        # CPU tensor algebra only: existing trained head, no fitted parameters.
        replay=(Q.double()@Wq.T)@(C.double()@Wk.T).T/8
        expected=torch.tensor(ref['pointer_logits'],dtype=torch.float64)
        assert replay.shape==expected.shape==(16,16) and finite(replay)
        ok=bool(torch.allclose(replay,expected,atol=1e-6,rtol=1e-5));pred=replay.argmax(1).tolist()
        check=dict(row_index=i,id=row['id'],cache_head_max_absolute_error=float((replay-expected).abs().max()),
            cache_head_scores_within_tolerance=ok,cache_head_predictions_exact=pred==ref['predictions'],
            archived_score_shape=[16,16],source=entry['source'])
        if actual_scores is not None:
            check['ordinary_forward_max_absolute_error']=float((actual_scores.double()-expected).abs().max())
            check['ordinary_forward_scores_within_tolerance']=bool(torch.allclose(actual_scores.double(),expected,atol=1e-6,rtol=1e-5))
            check['ordinary_forward_predictions_exact']=actual_scores.argmax(1).tolist()==ref['predictions']
        else:check['new_model_calls']=0
        comparisons.append(check);save('numerical_equivalence_partial_v1.json',comparisons)
        assert ok and check['cache_head_predictions_exact'],'EXTRACTION_INTEGRITY_FAILURE: cached head replay disagrees with archive: '+row['id']
        assert all(check.get(k,True) for k in ['ordinary_forward_scores_within_tolerance','ordinary_forward_predictions_exact']), 'EXTRACTION_INTEGRITY_FAILURE: ordinary forward disagrees with archive: '+row['id']
        ce=float((torch.logsumexp(replay,dim=1)-replay[torch.arange(16),target]).mean())
        assert abs(ce-ref['CE'])<=1e-6+1e-5*abs(ref['CE'])
        record=dict(**entry,status='VERIFIED',candidate_shape=list(C.shape),query_shape=list(Q.shape),dtype=str(C.dtype),
            candidate_tensor_sha256=tensor_sha(C),query_tensor_sha256=tensor_sha(Q),target_tensor_sha256=tensor_sha(target),
            finite=True,graph_retained=False,head_equivalence=check,archived_head_CE_replayed=ce,
            archived_head_predictions=pred,new_model_tokens=0 if i==0 else 32,selected_bank_calls=0 if i==0 else 256,
            commits=0 if i==0 else 32,**(extra or {}))
        completed_tensors.append((C,Q,target));verified.append(record)
        stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
        save('cache_progress_v1.json',dict(verified_arrays=len(verified),reused_arrays=int(bool(verified)),new_arrays_verified=len(verified)-1,last_verified_id=row['id']))
        return record
    try:
        torch.cuda.init();torch.cuda.reset_peak_memory_stats();ledger.check()
        result['model_constructions_attempted']=1
        model=InsertionSortAdapter().to('cuda');result['model_constructions_completed']=1
        checkpoint=torch.load(TRAIN/'model_update1000_v1.pt',map_location='cpu',weights_only=True)
        assert checkpoint['optimizer_update_count']==1000
        model.load_state_dict(checkpoint['model_state_dict'],strict=True);result['checkpoint_loads']=1
        Wq=checkpoint['model_state_dict']['query.weight'].detach().double().clone()
        Wk=checkpoint['model_state_dict']['key.weight'].detach().double().clone();del checkpoint
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        before_identity=parameter_identity(model);assert before_identity==manifest['model_identity']
        save('runtime_identity_v1.json',dict(device='cuda',device_name=torch.cuda.get_device_name(),torch=torch.__version__,CUDA=torch.version.cuda,
            TF32=False,FP32=True,eval_mode=True,all_parameters_frozen=True,model_identity=before_identity,checkpoint_sha256=manifest['checkpoint_sha256'],no_autograd_graph=True))
        old=torch.load(LOCAL/'final_tensor_evidence_v1.pt',map_location='cpu',weights_only=True)
        oldresult=json.loads((LOCAL/'final_results_v1.json').read_text())
        assert oldresult['parameter_identity_before']==oldresult['parameter_identity_after']==before_identity
        assert torch.equal(old['targets'],torch.tensor(rows[0]['pred']))
        assert torch.equal(old['candidate_states'],old['all_committed_streams'][:16]) and torch.equal(old['query_states'],old['all_committed_streams'][16:])
        verify_entry(0,old['candidate_states'].detach().cpu().clone(),old['query_states'].detach().cpu().clone(),old['raw_scores'],dict(reused_capture=True))
        result['reused_arrays']=1;del old
        def query_hook(module,args,output):
            projected['queries'].append(args[0].detach().cpu().clone())
        def key_hook(module,args,output):
            assert projected['keys'] is None
            projected['keys']=output.detach().cpu().clone();projected['key_input']=args[0].detach().cpu().clone()
        handles=[model.query.register_forward_hook(query_hook),model.key.register_forward_hook(key_hook)]
        with torch.no_grad():
            for i in range(1,128):
                ledger.check();current=rows[i];ledger.begin(current)
                projected.update(queries=[],keys=None,key_input=None)
                result['new_arrays_attempted']+=1;ledger.array_attempted+=1;begin=time.monotonic()
                out=model(current['key_fp32'],current['pos'],ledger,current['id'])
                torch.cuda.synchronize();result['new_arrays_forward_completed']+=1
                assert len(ledger.row_values)==32 and out['state'].tokens_committed==32
                C=torch.cat(ledger.row_values[:16]);Q=torch.cat(ledger.row_values[16:])
                assert len(out['candidates'])==16 and len(projected['queries'])==16
                assert torch.equal(torch.cat(projected['queries']),Q)
                assert projected['key_input'].shape==(1,16,64) and torch.equal(projected['key_input'][0],C)
                for j,value in enumerate(out['candidates']):assert torch.equal(value.detach().cpu(),C[j:j+1])
                assert tensor_sha(out['features'][0].detach().cpu())==inputs['arrays'][i]['encoded_features_sha256']
                record=verify_entry(i,C,Q,out['pointer_logits'][0].detach().cpu().clone(),
                    dict(reused_capture=False,fresh_initial_tokens=0,returned_tokens_committed=32,query_and_key_head_inputs_exact=True,elapsed_seconds=time.monotonic()-begin))
                ledger.array_completed+=1;result['new_arrays_verified']+=1
                ledger.check();print('Verified training cache',i+1,'/128',current['id'],flush=True)
                del out,C,Q
        ledger.check();assert result['new_arrays_verified']==127 and result['reused_arrays']==1 and len(verified)==128
        assert ledger.attempted==ledger.completed==ledger.commits==4064 and ledger.failed==0
        assert ledger.bank_attempted==ledger.bank_completed==32512
        assert parameter_identity(model)==before_identity and all(p.grad is None and not p.requires_grad for p in model.parameters())
        ledger.check();result['status']='COMPLETED_VERIFIED_TRAINING_CACHE'
    except BaseException as e:
        result['failure']=dict(type=type(e).__name__,message=str(e),case=current['id'] if current else None)
        result['new_arrays_failed']=max(0,result['new_arrays_attempted']-result['new_arrays_verified'])
        save('failure_v1.json',dict(**result['failure'],traceback=traceback.format_exc(),no_retry=True))
        if current and ledger.row_values:
            with (OUT/'failed_array_partial_states_v1.pt').open('xb') as f:torch.save(dict(id=current['id'],committed_s=torch.cat(ledger.row_values),tokens_observed=len(ledger.row_values)),f)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_alarm)
        result['extraction_elapsed_seconds']=time.monotonic()-ledger.started
        for h in handles:h.remove()
        stream.close();ledger.trace.close();result['ledger']=ledger.summary()
        result['ledger']['new_arrays_attempted']=result['new_arrays_attempted'];result['ledger']['new_arrays_verified']=result['new_arrays_verified']
        result['ledger']['failed_arrays']=result['new_arrays_failed'];result['ledger']['reused_arrays']=result['reused_arrays']
        save('execution_ledger_v1.json',result['ledger'])
        if model is not None and before_identity is not None:
            result['parameter_and_buffer_identity_after']=parameter_identity(model)
            result['parameters_and_buffers_unchanged']=result['parameter_and_buffer_identity_after']==before_identity
            if not result['parameters_and_buffers_unchanged']:result['status']='INCOMPLETE_CACHE_INTEGRITY_FAILURE'
        if completed_tensors:
            cache=dict(schema_version=1,array_ids=[x['id'] for x in verified],row_indices=torch.tensor([x['row_index'] for x in verified]),
                candidate_committed_s=torch.stack([x[0] for x in completed_tensors]),query_committed_s=torch.stack([x[1] for x in completed_tensors]),
                predecessor_targets=torch.stack([x[2] for x in completed_tensors]),original_input_sha256=[x['original_input_sha256'] for x in verified],
                candidate_original_element_indices=torch.arange(16),query_original_element_indices=torch.arange(16),
                checkpoint_sha256=manifest['checkpoint_sha256'],program=list(range(8)),roles='Committed state.s only; candidate input positions and later query positions are separate tables')
            name='training_representation_cache_v1.pt' if result['status']=='COMPLETED_VERIFIED_TRAINING_CACHE' else 'incomplete_training_cache_v1.pt'
            with (OUT/name).open('xb') as f:torch.save(cache,f)
            save('cache_manifest_v1.json',dict(status=result['status'],cache_file=name,sha256=sha(OUT/name),bytes=(OUT/name).stat().st_size,
                verified_arrays=len(verified),new_arrays_verified=result['new_arrays_verified'],reused_arrays=result['reused_arrays'],
                candidate_shape=list(cache['candidate_committed_s'].shape),query_shape=list(cache['query_committed_s'].shape),target_shape=list(cache['predecessor_targets'].shape),
                representation_dtype='torch.float32',target_dtype='torch.int64',device='cpu',gradients_retained=False,
                checkpoint_sha256=manifest['checkpoint_sha256'],input_manifest_sha256=sha(OUT/'input_manifest_v1.json'),
                per_array_payload_hash_recipe='SHA256 raw contiguous CPU tensor bytes; dtype/shape separately recorded',
                candidate_role='state.s committed after original element j',query_role='state.s committed after query for original element i',
                target_role='Official predecessor index in original candidate-element coordinates',no_fitting=True,held_out_rows=0))
        save('numerical_equivalence_v1.json',dict(success=result['status']=='COMPLETED_VERIFIED_TRAINING_CACHE',arrays_compared=len(comparisons),
            all_cached_scores_within_tolerance=all(x['cache_head_scores_within_tolerance'] for x in comparisons),all_cached_predictions_exact=all(x['cache_head_predictions_exact'] for x in comparisons),
            all_ordinary_forward_scores_within_tolerance=all(x.get('ordinary_forward_scores_within_tolerance',True) for x in comparisons),
            max_cached_head_score_error=max([x['cache_head_max_absolute_error'] for x in comparisons],default=None),
            atol=1e-6,rtol=1e-5,discrete_predictions='exact',comparisons= comparisons))
        save('extraction_results_v1.json',result)
        print(json.dumps(dict(status=result['status'],verified_arrays=len(verified),result=result),indent=2),flush=True)

if __name__=='__main__':main()
