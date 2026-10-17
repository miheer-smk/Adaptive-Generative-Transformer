"""Exactly one locked paired pilot; no diagnostics, retries or resume."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,time,hashlib,signal,traceback,datetime
from support_v1 import OUT,ROOT,ENG,save,sha,integrity
import torch
from observer_v1 import Ledger,InsertionSortAdapter,pointer_loss,pointer_predictions,fields,snapshot,assert_equal,assert_disjoint,ResourceStop,NumericalFailure
from pilot_modules_v1 import install_treatment,active_mixers,ACTIVE_NAMES,EXCLUDED_NAME
from evaluation_v1 import evaluate

def identity(model):
    h=hashlib.sha256()
    for n,v in sorted(list(model.named_parameters())+list(model.named_buffers())):
        h.update(n.encode());h.update(str((v.shape,v.dtype)).encode());h.update(bytes(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).tolist()))
    return h.hexdigest()
def frozen_identity(model):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if not p.requires_grad:h.update(n.encode());h.update(bytes(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).tolist()))
    return h.hexdigest()
def rng():return (torch.random.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
def set_rng(pair):torch.random.set_rng_state(pair[0]);torch.cuda.set_rng_state(pair[1])
def same_rng(a,b):return torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])

def train_update(model,optimizer,ledger,row,update,index,stream):
    ledger.phase='training';ledger.check();ledger.array_attempted+=1;begin=time.monotonic()
    try:
        assert torch.is_grad_enabled() and not model.training
        optimizer.zero_grad(set_to_none=True)
        out=model(row['key_fp32'],row['pos'],ledger,row['id'])
        loss=pointer_loss(out['pointer_logits'],row['pred'])
        if not bool(torch.isfinite(loss)):raise NumericalFailure('Nonfinite training CE')
        torch.cuda.synchronize();forward=time.monotonic()-begin
        scores=out['pointer_logits'][0].detach().cpu();pred=scores.argmax(1).tolist()
        record=dict(arm=ledger.arm,update=update,canonical_pool_index=index,id=row['id'],n=row['n'],input_sha256=row['original_input_sha256'],predictions=pred,targets=row['pred'],pointer_logits=scores.tolist(),pointer_accuracy=sum(x==y for x,y in zip(pred,row['pred']))/row['n'],exact_array_accuracy=int(pred==row['pred']),CE=float(loss.detach()))
        back_start=time.monotonic();ledger.backward(loss)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):raise NumericalFailure('Nonfinite or absent training gradients')
        groups={}
        for prefix in ['body.','encoder.','query.','key.']:
            values=[p.grad.detach().square().sum() for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            groups[prefix]=float(torch.stack(values).sum().sqrt())
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        torch.cuda.synchronize();backward=time.monotonic()-back_start
        before=[p.detach().clone() for p in model.parameters() if p.requires_grad]
        ledger.check();assert ledger.optimizer_attempted<1000;ledger.optimizer_attempted+=1
        try:optimizer.step();ledger.optimizer_completed+=1
        except BaseException:ledger.optimizer_failed+=1;raise
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise NumericalFailure('Nonfinite parameters after optimizer update')
        if not all(bool(torch.isfinite(v).all()) for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)):raise NumericalFailure('Nonfinite optimizer state')
        displacement=float(torch.stack([(p.detach()-b).square().sum() for p,b in zip([p for p in model.parameters() if p.requires_grad],before)]).sum().sqrt())
        torch.cuda.synchronize()
        record.update(unclipped_gradient_norm=float(norm),gradient_norms_by_group=groups,step_parameter_displacement=displacement,forward_seconds=forward,backward_and_clip_seconds=backward,elapsed_update_seconds=time.monotonic()-begin,total_elapsed_seconds=time.monotonic()-ledger.started,all_gradients_finite=True,parameters_optimizer_finite=True)
        stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();ledger.array_completed+=1;ledger.persist()
        del out,loss,before;ledger.check()
        return record
    except BaseException:
        ledger.array_failed+=1;ledger.persist();raise

def reconcile(ledgers):
    results={}
    for arm,l in ledgers.items():
        counts=dict(attempted=0,completed=0,failed=0,selected_attempted=0,selected_completed=0,commits=0,contextual_entries=0);phase={}
        import gzip
        with gzip.open(OUT/(arm+'_token_traces_v1.jsonl.gz'),'rt') as f:
            for s in f:
                r=json.loads(s);counts['attempted']+=1;counts[r['status']]+=1;counts['selected_attempted']+=r['selected_attempted'];counts['selected_completed']+=len(r['blocks']);counts['commits']+=r['commits'];counts['contextual_entries']+=r.get('commit',{}).get('entry_count',0)
                p=phase.setdefault(r['phase'],dict(attempted=0,completed=0,failed=0));p['attempted']+=1;p[r['status']]+=1
                if r['status']=='completed':assert r['D']==r['P']==8 and r['blocks']==r['selected_actions']==list(range(8)) and r['mixer_calls']==[1]*8 and r['commits']==1 and r['parent_unchanged'] and r['returned_state_disjoint']
        assert counts==dict(attempted=l.attempted,completed=l.completed,failed=l.failed,selected_attempted=l.bank_attempted,selected_completed=l.bank_completed,commits=l.commits,contextual_entries=l.contextual_entries)
        results[arm]=dict(success=True,counts=counts,phases=phase)
    save('complete_trace_verification_v1.json',results)
    return results

def main():
    if (OUT/'execution_started_v1.json').exists():raise RuntimeError('No second attempt/resume authorized')
    seal=json.loads((OUT/'implementation_seal_v1.json').read_text())
    for n,h in seal['files'].items():assert sha(ROOT/n)==h,n
    assert json.loads((OUT/'operator_checks_v1.json').read_text())['success']
    assert integrity()['success']
    data={n:json.loads((ENG/(n+'_pool_v1.json')).read_text()) for n in ['train','development','size_transfer']}
    order=json.loads((ENG/'training_results_v1/training_order_v1.json').read_text());assert sorted(order['indices'])==list(range(1000))
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available();torch.cuda.init();torch.cuda.reset_peak_memory_stats()
    models={};optimizers={};ledgers={};logs={};summaries={};updates={};frozen={};current_arm=None
    result=dict(status='INCOMPLETE',phases={},outcomes={},no_diagnostic_forward=True)
    def alarm(*unused):raise ResourceStop('Six-hour paired hard deadline')
    signal.signal(signal.SIGALRM,alarm)
    start=time.monotonic();deadline=start+21600;signal.setitimer(signal.ITIMER_REAL,21600)
    save('execution_started_v1.json',dict(pid=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),start_monotonic=start,deadline_monotonic=deadline,seconds=21600,no_retry_resume=True,first_forward='First prescribed control training array',no_preliminary_full_model_call=True))
    try:
        for arm in ['C','T']:
            current_arm=arm;ledgers[arm]=Ledger(arm,start,deadline)
            models[arm]=InsertionSortAdapter().to('cuda')
            assert identity(models[arm])=='c28e83425e505734e1b6a39f4c6d5ecf414f1b1fb28a77e01ddc31bbfa92280e','Original seed6200 initialization mismatch'
            assert not models[arm].training and all(not m.training for m in models[arm].modules())
            assert sum(p.numel() for p in models[arm].parameters())==1942836 and sum(p.numel() for p in models[arm].parameters() if p.requires_grad)==1838377
        paramsC=dict(models['C'].named_parameters());paramsT=dict(models['T'].named_parameters());buffersC=dict(models['C'].named_buffers());buffersT=dict(models['T'].named_buffers())
        assert paramsC.keys()==paramsT.keys() and buffersC.keys()==buffersT.keys()
        for groupC,groupT in [(paramsC,paramsT),(buffersC,buffersT)]:
            for name,p in groupC.items():
                assert torch.equal(p,groupT[name]) and (not p.numel() or p.untyped_storage().data_ptr()!=groupT[name].untyped_storage().data_ptr()),name
        own=[]
        for arm in ['C','T']:
            model=models[arm];own.append(model.body.initial_persistent_state(1,torch.device('cuda'),torch.float32));frozen[arm]=frozen_identity(model)
            assert frozen[arm]=='6bbea651fb2a44f9581141abc16741a475ddcc1315bebb3833cbf4717c603182'
        assert_equal(snapshot(own[0]),snapshot(own[1]),True);assert_disjoint(own[0],own[1]);del own
        sites=install_treatment(models['T'])
        assert sites==list(ACTIVE_NAMES) and identity(models['T'])==identity(models['C'])=='c28e83425e505734e1b6a39f4c6d5ecf414f1b1fb28a77e01ddc31bbfa92280e'
        for arm in ['C','T']:
            optimizers[arm]=torch.optim.AdamW([p for p in models[arm].parameters() if p.requires_grad],lr=.001,weight_decay=0,betas=(.9,.999),eps=1e-8)
            assert not optimizers[arm].state
            logs[arm]=(OUT/(arm+'_training_metrics_v1.jsonl')).open('x');updates[arm]=[]
        assert optimizers['C'] is not optimizers['T']
        save('starting_equivalence_v1.json',dict(success=True,model_constructions=2,initial_parameter_buffer_identity='c28e83425e505734e1b6a39f4c6d5ecf414f1b1fb28a77e01ddc31bbfa92280e',frozen_parameter_identity=frozen['C'],parameters_equal=True,buffers_equal=True,all_parameter_buffer_storages_independently_owned=True,initial_persistent_states_equal_and_disjoint=True,optimizer_states_fresh_empty_independent=True,trainable_parameters=1838377,parameters_total=1942836,mode='Original adapter eval mode with full gradient recording',treatment_sites=sites,excluded_site=EXCLUDED_NAME,added_trainable_parameters=0,full_model_calls_before_first_training_update=0,failed_final_checkpoint_loaded=False))
        save('runtime_v1.json',dict(torch=torch.__version__,CUDA=torch.version.cuda,device=torch.cuda.get_device_name(),FP32=True,TF32=False,threads=1,full_BPTT=True,initialization_order='Original adapter constructor separately twice; no checkpoint initialization',paired_execution_order='For each locked training row: C update then T update',matched_RNG='Restore identical CPU/CUDA states before each paired update; require identical post-update RNG states',source_reference='1ed1d773264b3f5f53bd1ba169e819709d33cbb2'))
        result['phases']['starting_equivalence']='PASS'
        for update,index in enumerate(order['indices'],1):
            shared=rng();after=None
            for arm in ['C','T']:
                current_arm=arm;set_rng(shared)
                r=train_update(models[arm],optimizers[arm],ledgers[arm],data['train'][index],update,index,logs[arm]);updates[arm].append(r)
                if arm=='C':after=rng()
                else:assert same_rng(after,rng()),'Intervention unexpectedly changed RNG consumption'
            if update%10==0:
                print('PAIRED_UPDATE',update,'C_CE',updates['C'][-1]['CE'],'T_CE',updates['T'][-1]['CE'],'C_acc',updates['C'][-1]['pointer_accuracy'],'T_acc',updates['T'][-1]['pointer_accuracy'],'elapsed',time.monotonic()-start,flush=True)
            save('paired_progress_v1.json',dict(completed_paired_updates=update,elapsed_seconds=time.monotonic()-start,C=ledgers['C'].summary(),T=ledgers['T'].summary(),RNG_post_states_equal=True))
        for arm in ['C','T']:
            current_arm=arm;l=ledgers[arm];assert l.optimizer_completed==l.backward_completed==1000 and frozen_identity(models[arm])==frozen[arm]
            logs[arm].close();m=updates[arm]
            save(arm+'_training_summary_v1.json',dict(updates=1000,mean_online_CE=sum(r['CE'] for r in m)/1000,mean_online_pointer_accuracy=sum(r['pointer_accuracy'] for r in m)/1000,mean_online_exact_array_accuracy=sum(r['exact_array_accuracy'] for r in m)/1000,final_model_identity=identity(models[arm]),note='Online pre-update training metrics; not generalization evidence'))
            path=OUT/(arm+'_update1000_v1.pt')
            with path.open('xb') as f:torch.save(dict(model_state_dict=models[arm].state_dict(),optimizer_state_dict=optimizers[arm].state_dict(),optimizer_update_count=1000,model_seed=6200,arm=arm,intervention='original' if arm=='C' else '(X @ (P+Q).T)*g; Q=I-ones/64; eight bank mixers only',intervention_sites=[] if arm=='C' else sites,training_order_sha256=sha(ENG/'training_results_v1/training_order_v1.json'),implementation_seal_sha256=sha(OUT/'implementation_seal_v1.json')),f)
            save(arm+'_checkpoint_identity_v1.json',dict(path=path.name,sha256=sha(path),size=path.stat().st_size,optimizer_updates=1000,selection='Predetermined update1000 only'))
        result['phases']['training']='BOTH_COMPLETED_1000_UPDATES'
        for phase,rows in [('training_subset',data['train'][:128]),('development',data['development']),('size_transfer',data['size_transfer'])]:
            summaries[phase]={}
            for arm in ['C','T']:
                current_arm=arm;records,summary,mechanism=evaluate(models[arm],rows,ledgers[arm],phase);summaries[phase][arm]=summary
                result['phases'][arm+'_'+phase]='COMPLETED'
        for arm,l in ledgers.items():
            assert l.attempted==l.completed==l.commits==41216 and l.failed==0 and l.bank_attempted==l.bank_completed==329728
            assert l.contextual_entries==288512 and l.optimizer_completed==l.backward_completed==1000 and frozen_identity(models[arm])==frozen[arm]
        save('evaluation_summaries_v1.json',summaries)
        result['status']='ALL_TRAINING_AND_PRESCRIBED_EVALUATIONS_COMPLETED'
    except BaseException as e:
        result.update(status='TREATMENT_NUMERICAL_OUTCOME_PAIRED_EVALUATION_INCOMPLETE' if current_arm=='T' and isinstance(e,NumericalFailure) else 'INCOMPLETE',failure_type=type(e).__name__,failure=str(e),execution_boundary=dict(arm=current_arm,phase=ledgers[current_arm].phase if current_arm in ledgers else 'construction'))
        save('failure_v1.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),boundary=result['execution_boundary'],treatment_numerical_outcome=current_arm=='T' and isinstance(e,NumericalFailure),no_retry=True))
        print('STOP',type(e).__name__,str(e),flush=True)
    finally:
        for f in logs.values():
            if not f.closed:f.close()
        for l in ledgers.values():l.trace.close();l.persist()
        result['elapsed_seconds_to_execution_boundary']=time.monotonic()-start
        result['deadline_met']=result['elapsed_seconds_to_execution_boundary']<=21600
        result['resource']={a:l.summary() for a,l in ledgers.items()}
        result['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated();result['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
        result['trace_verification']=reconcile(ledgers);result['integrity']=integrity()
        if not result['integrity']['success'] or not result['deadline_met']:result['status']='INCOMPLETE'
        save('run_result_v1.json',result);signal.setitimer(signal.ITIMER_REAL,0)
        print(json.dumps(dict(status=result['status'],elapsed=result['elapsed_seconds_to_execution_boundary'],resource=result['resource'],integrity=result['integrity']),indent=2),flush=True)
if __name__=='__main__':main()
