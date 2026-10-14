"""Two ordinary rollouts, two original CE backwards, no optimizer or retry."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import copy,gzip,hashlib,json,math,subprocess,time,traceback
from pathlib import Path
import torch
from diagnostic_observer_v1 import (Ledger,OUT,ROOT,save,sha,parameter_identity,fields,
    snapshot,assert_equal,metadata,cpu_snapshot,InsertionSortAdapter,pointer_loss,pointer_predictions)
E=OUT.parent/'gatec_clrs_insertion_sort_6200_v1'
T=E/'training_results_v1'

def finite(t):return bool(torch.isfinite(t).all())
def describe(t):
    if t is None:return dict(connected=False,classification='disconnected')
    x=t.detach().cpu().double().reshape(-1,t.shape[-1])
    n=x.shape[0];norm=x.norm(dim=-1);dist=torch.cdist(x,x)
    off=~torch.eye(n,dtype=torch.bool)
    cosine=(x@x.T)/(norm[:,None]*norm[None,:]).clamp_min(torch.finfo(torch.float64).tiny)
    return dict(connected=True,finite=finite(x),shape=list(t.shape),norms=norm.tolist(),
        mean_row_norm=float(norm.mean()),mean_pairwise_L2=float(dist[off].mean()) if n>1 else None,
        min_pairwise_L2=float(dist[off].min()) if n>1 else None,max_pairwise_L2=float(dist[off].max()) if n>1 else None,
        mean_pairwise_cosine=float(cosine[off].mean()) if n>1 else None,
        relative_pairwise_variation=float(dist[off].mean()/norm.mean()) if n>1 and float(norm.mean()) else None,
        classification='exact_zero' if bool(x.eq(0).all()) else 'connected_nonzero')

class CaptureLedger(Ledger):
    def begin(self,label):
        self.phase=label;self.states=[];self.state_objects=[];self.h0=[];self.memory_reads=[];self.input_candidates=[]
        self.candidate_values=[];self.block_outputs=[[] for _ in range(8)]
    def token(self,body,x,parent,case_id,token,scripted):
        out=super().token(body,x,parent,case_id,token,scripted)
        s=out['state'].s;s.retain_grad()
        out['h0'].retain_grad();out['memory_read'].retain_grad()
        self.h0.append(out['h0']);self.memory_reads.append(out['memory_read'])
        self.state_objects.append(s)
        self.states.append(dict(token_index=token,kind='input' if token<16 else 'query',
            original_element_index=token if token<16 else token-16,
            metadata=metadata(out['state']),fields={k:dict(shape=list(v.shape),norm=float(v.detach().norm()),
            min=float(v.detach().min()) if v.numel() else None,max=float(v.detach().max()) if v.numel() else None,
            finite=finite(v)) for k,v in fields(out['state']).items() if torch.is_tensor(v)}))
        assert all(f['finite'] for f in self.states[-1]['fields'].values())
        for z,value in zip(self.input_candidates,self.candidate_values):assert torch.equal(z.detach(),value)
        if token<16:
            self.input_candidates.append(s);self.candidate_values.append(s.detach().clone())
        return out

def integrity():
    initial=json.loads((OUT/'integrity_initial_v1.json').read_text());bad=[]
    for name,r in initial['files'].items():
        p=ROOT/name
        ok=p.is_symlink() and os.readlink(p)==r['target'] if r['kind']=='symlink' else p.is_file() and not p.is_symlink() and p.stat().st_size==r['size'] and sha(p)==r['sha256'] and p.stat().st_mode&0o777==r['mode']
        if not ok:bad.append(name)
    env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,env=env).decode().strip()
    diff=hashlib.sha256(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT,env=env)).hexdigest()
    r=dict(protected_files=len(initial['files']),changed_or_missing=bad,HEAD_unchanged=head==initial['HEAD'],tracked_diff_unchanged=diff==initial['tracked_diff_sha256'],historical_exception_preserved=True)
    r['success']=not bad and r['HEAD_unchanged'] and r['tracked_diff_unchanged'];save('integrity_final_v1.json',r);return r

def run_model(label,model,row,ledger,expected):
    before=parameter_identity(model);ledger.begin(label);model.zero_grad(set_to_none=True);handles=[];capture={'q':[],'q_input':[]}
    def key_hook(module,args,result):
        result.retain_grad();args[0].retain_grad();capture['K']=result;capture['K_input']=args[0]
    def query_hook(module,args,result):
        result.retain_grad();args[0].retain_grad();capture['q'].append(result);capture['q_input'].append(args[0])
    def encoder_hook(module,args,result):
        result.retain_grad();capture['encoding']=result;capture['features']=args[0]
    def block_hook(i):
        def hook(module,args,result):
            value=result[0] if isinstance(result,tuple) else result
            value.retain_grad();ledger.block_outputs[i].append(value)
        return hook
    handles.extend([model.key.register_forward_hook(key_hook),model.query.register_forward_hook(query_hook),model.encoder.register_forward_hook(encoder_hook)])
    for i,b in enumerate(model.body.blocks.blocks):handles.append(b.register_forward_hook(block_hook(i)))
    try:
        out=model(row['key_fp32'],row['pos'],ledger,row['id'],retain_gradients=True)
        scores=out['pointer_logits'];scores.retain_grad();loss=pointer_loss(scores,row['pred'])
        assert scores.shape==(1,16,16) and finite(loss)
        assert len(capture['q'])==16 and len(ledger.state_objects)==32
        assert capture['K'].shape==capture['K_input'].shape==(1,16,64)
        assert all(torch.equal(q,s) for q,s in zip(capture['q_input'],ledger.state_objects[16:]))
        assert all(torch.equal(z,capture['K_input'][:,j]) for j,z in enumerate(out['candidates']))
        assert torch.equal(capture['features'][0,:16,2],torch.tensor(row['pos'],device='cuda'))
        assert torch.equal(capture['features'][0,16:,2],torch.tensor(row['pos'],device='cuda'))
        assert capture['features'][0,16:,3].eq(0).all()
        expected_scores=torch.tensor(expected['pointer_logits'],device='cuda',dtype=torch.float32)
        assert torch.allclose(scores[0],expected_scores,atol=1e-6,rtol=1e-5),'Retained score replay mismatch'
        assert math.isclose(float(loss.detach()),expected['CE'],abs_tol=1e-6,rel_tol=1e-5),'Retained CE replay mismatch'
        ledger.backward(loss)
        all_grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert all_grads and all(finite(g) for g in all_grads)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        assert parameter_identity(model)==before,'Model parameter/buffer identity changed'
        cp=lambda x:None if x is None else x.detach().cpu().clone()
        stack=lambda xs:torch.cat([cp(x) for x in xs],dim=0)
        gradstack=lambda xs:torch.cat([cp(x.grad) for x in xs],dim=0) if all(x.grad is not None for x in xs) else None
        raw=cp(scores[0]);centered=raw-raw.mean(dim=-1,keepdim=True);prob=torch.softmax(raw,dim=-1)
        data=dict(candidate_states=stack(out['candidates']),query_states=stack(capture['q_input']),
            projected_queries=stack(capture['q']),projected_keys=cp(capture['K'][0]),
            raw_scores=raw,centered_scores=centered,probabilities=prob,targets=torch.tensor(row['pred']),
            all_committed_streams=stack(ledger.state_objects),encoded_tokens=cp(capture['encoding'][0]),
            features=cp(capture['features'][0]),h0=stack(ledger.h0),memory_reads=stack(ledger.memory_reads),
            block_outputs={str(i):torch.cat([cp(v) for v in values],dim=0).squeeze(1) for i,values in enumerate(ledger.block_outputs)},
            gradients=dict(candidate_states=gradstack(out['candidates']),query_states=gradstack(capture['q_input']),
                projected_queries=gradstack(capture['q']),projected_keys=cp(capture['K'].grad[0]) if capture['K'].grad is not None else None,
                stacked_key_inputs=cp(capture['K_input'].grad[0]) if capture['K_input'].grad is not None else None,
                raw_scores=cp(scores.grad[0]) if scores.grad is not None else None,
                encoded_tokens=cp(capture['encoding'].grad[0]) if capture['encoding'].grad is not None else None,
                h0=gradstack(ledger.h0),memory_reads=gradstack(ledger.memory_reads),
                query_weight=cp(model.query.weight.grad),key_weight=cp(model.key.weight.grad),
                block_outputs={str(i):torch.cat([cp(v.grad) for v in values],dim=0).squeeze(1) if all(v.grad is not None for v in values) else None for i,values in enumerate(ledger.block_outputs)}))
        for k in ['candidate_states','query_states','projected_queries','projected_keys']:
            assert data['gradients'][k] is not None and finite(data['gradients'][k]),k
        analytic=(prob-torch.nn.functional.one_hot(data['targets'],16))/16
        assert torch.allclose(analytic,data['gradients']['raw_scores'],atol=1e-6,rtol=1e-5)
        preds=raw.argmax(dim=-1);query_records=[]
        for i,t in enumerate(row['pred']):
            target=float(raw[i,t]);others=torch.cat([raw[i,:t],raw[i,t+1:]])
            query_records.append(dict(query=i,candidate_target=t,prediction=int(preds[i]),correct=int(preds[i])==t,
                target_score=target,target_rank=1+int((raw[i]>target).sum()),strictly_greater_count=int((raw[i]>target).sum()),
                target_ties=int((raw[i]==target).sum()),target_margin=target-float(others.max()),
                logit_span=float(raw[i].max()-raw[i].min()),max_probability=float(prob[i].max()),
                entropy=float(-(prob[i]*prob[i].log()).sum())))
        record=dict(model=label,parameter_identity_before=before,parameter_identity_after=parameter_identity(model),
            CE=float(loss.detach()),pointer_accuracy=float(preds.eq(data['targets']).float().mean()),queries=query_records,
            stages={k:describe(data[k]) for k in ['candidate_states','query_states','projected_queries','projected_keys','h0','memory_reads','encoded_tokens']},
            gradient_stages={k:describe(v) for k,v in data['gradients'].items() if isinstance(v,torch.Tensor) or v is None},
            block_stage_statistics={str(i):dict(input_tokens=describe(v[:16]),query_tokens=describe(v[16:]),
                query_token_adjoints=describe(data['gradients']['block_outputs'][str(i)][16:]) if data['gradients']['block_outputs'][str(i)] is not None else describe(None)) for i,v in enumerate(data['block_outputs'].values())},
            centered_score_statistics=describe(centered),raw_score_statistics=describe(raw),retained_forward_replay_matches=True,
            analytic_score_gradient_matches=True,frozen_parameters_have_no_grad=True,candidate_values_unchanged=True,
            all_parameter_gradients_finite=True,parameter_gradient_norms={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None})
        with (OUT/(label+'_tensor_evidence_v1.pt')).open('xb') as f:torch.save(data,f)
        save(label+'_committed_state_summaries_v1.json',ledger.states)
        save(label+'_results_v1.json',record)
        return record
    finally:
        for h in handles:h.remove()

def main():
    if (OUT/'diagnostic_started_v1.json').exists():raise SystemExit('STOP: no retry')
    manifest=json.loads((OUT/'manifest_v1.json').read_text());seal=json.loads((OUT/'diagnostic_seal_v1.json').read_text())
    for n,h in seal['files'].items():assert sha(OUT/n)==h,n
    assert integrity()['success'];assert sha(T/'model_update1000_v1.pt')==manifest['checkpoint_sha256']
    for directory,name in [(E,'implementation_seal_v1.json'),(T,'training_implementation_seal_v1.json')]:
        assert all(sha(directory/n)==h for n,h in json.loads((directory/name).read_text())['files'].items())
    row=json.loads((OUT/'locked_input_v1.json').read_text());assert row==json.loads((E/'train_pool_v1.json').read_text())[0]
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available();torch.cuda.init();torch.cuda.reset_peak_memory_stats()
    ledger=CaptureLedger();result=dict(status='STOPPED',models={})
    save('diagnostic_started_v1.json',dict(pid=os.getpid(),no_retry=True,models=2,optimizer_objects=0))
    try:
        initial=InsertionSortAdapter().to('cuda');assert parameter_identity(initial)==manifest['initial_identity']
        final=copy.deepcopy(initial)
        c=torch.load(T/'model_update1000_v1.pt',map_location='cuda',weights_only=True)
        assert c['optimizer_update_count']==1000
        final.load_state_dict(c['model_state_dict'],strict=True);del c
        assert parameter_identity(final)==manifest['final_identity']
        assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(initial.parameters(),final.parameters()))
        save('verified_model_identities_v1.json',dict(initial=parameter_identity(initial),final=parameter_identity(final),
            initialization_verified_before_execution=True,checkpoint_loaded_strictly=True,device='cuda',torch=torch.__version__,cuda=torch.version.cuda))
        first_initial=json.loads((T/'initial_diagnostic_records_v1.jsonl').read_text().splitlines()[0])
        first_final=json.loads((T/'training_subset_records_v1.jsonl').read_text().splitlines()[0])
        result['models']['initial']=run_model('initial',initial,row,ledger,first_initial)
        result['models']['final']=run_model('final',final,row,ledger,first_final)
        assert ledger.attempted==ledger.completed==64 and ledger.bank_completed==512 and ledger.commits==64
        assert ledger.backward_attempted==ledger.backward_completed==2 and not ledger.failed
        result['status']='COMPLETED_NO_UPDATE_DIAGNOSTIC'
    except BaseException as e:
        result['failure']=dict(type=type(e).__name__,message=str(e),phase=ledger.phase)
        save('failure_v1.json',dict(**result['failure'],traceback=traceback.format_exc(),ledger=ledger.summary(),retry=False))
    finally:
        ledger.trace.close();result['ledger']=ledger.summary();save('execution_ledger_v1.json',result['ledger'])
        result['integrity']=integrity();result['training_result_unchanged']='COMPLETED_NON_SUCCESS'
        if not result['integrity']['success']:result['status']='STOPPED_INTEGRITY_FAILURE'
        save('diagnostic_results_v1.json',result)
        print(json.dumps(dict(status=result['status'],ledger=result['ledger'],integrity=result['integrity']),indent=2),flush=True)

if __name__=='__main__':main()
