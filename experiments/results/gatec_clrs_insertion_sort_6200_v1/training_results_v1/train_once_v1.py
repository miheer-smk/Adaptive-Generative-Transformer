"""One authorized, non-resumable fixed-program CLRS training run."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import collections, datetime, gzip, hashlib, json, math, signal, sys, time, traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
from clrs_adapter_v1 import InsertionSortAdapter, pointer_loss, pointer_predictions
from training_instrumentation_v1 import Ledger, parameter_identity
from offline_support_v1 import OUT, ENGINEERING, save, sha, integrity, invalid_chain, metrics, controls

def frozen_identity(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode());h.update(bytes(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).tolist()))
    return h.hexdigest()

def record(row, output, loss):
    pred=pointer_predictions(output['pointer_logits'])[0]
    scores=output['pointer_logits'].detach().cpu()[0]
    return dict(id=row['id'],n=row['n'],original_input_sha256=row['original_input_sha256'],
        predictions=pred,truth=row['pred'],pointer_logits=scores.tolist(),
        pointer_accuracy=sum(a==b for a,b in zip(pred,row['pred']))/row['n'],
        exact_array_accuracy=pred==row['pred'],CE=float(loss.detach()),invalid_chain=invalid_chain(pred),
        final_tokens_committed=output['state'].tokens_committed,
        margins=(scores.topk(2,dim=-1).values[:,0]-scores.topk(2,dim=-1).values[:,1]).tolist())

def evaluate(model,rows,ledger,phase):
    ledger.phase=phase;records=[]
    stream=OUT/(phase+'_records_v1.jsonl')
    with stream.open('x') as f:
        for row in rows:
            ledger.check();ledger.array_attempted+=1
            try:
                begin=time.monotonic()
                with torch.no_grad():
                    out=model(row['key_fp32'],row['pos'],ledger,row['id'])
                    loss=pointer_loss(out['pointer_logits'],row['pred'])
                torch.cuda.synchronize()
                assert bool(torch.isfinite(loss))
                r=record(row,out,loss);r['elapsed_seconds']=time.monotonic()-begin
                f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
                records.append(r);ledger.array_completed+=1
                save('execution_ledger_v1.json',ledger.summary())
                print(phase,len(records),'/',len(rows),'CE',r['CE'],'pointer',r['pointer_accuracy'],flush=True)
            except BaseException:
                ledger.array_failed+=1;raise
    save(phase+'_summary_v1.json',metrics(records))
    return records

def reconcile(ledger):
    c=collections.Counter();phases={}
    with gzip.open(OUT/'token_traces_v1.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);c['attempted']+=1;c[r['status']]+=1
            c['blocks']+=len(r['blocks']);c['commits']+=r['commits']
            p=phases.setdefault(r['phase'],collections.Counter());p['attempted']+=1;p[r['status']]+=1;p['blocks']+=len(r['blocks']);p['commits']+=r['commits']
            if r['status']=='completed':
                assert r['blocks']==list(range(8)) and r['selected_actions']==list(range(8))
                assert r['P']==r['D']==8 and r['commits']==1
                assert r['parent_unchanged'] and r['returned_state_disjoint']
    checks=dict(attempted=c['attempted']==ledger.attempted,completed=c['completed']==ledger.completed,
                failed=c['failed']==ledger.failed,bank_completed=c['blocks']==ledger.bank_completed,
                commits=c['commits']==ledger.commits)
    result=dict(counts=dict(c),phases={k:dict(v) for k,v in phases.items()},checks=checks,success=all(checks.values()))
    save('complete_trace_verification_v1.json',result);assert result['success'];return result

def main():
    if (OUT/'training_started_v1.json').exists():raise SystemExit('STOP: no retry or resume')
    seal=json.loads((OUT/'training_implementation_seal_v1.json').read_text())
    for name,h in seal['files'].items():assert sha(OUT/name)==h,name
    e_seal=json.loads((ENGINEERING/'implementation_seal_v1.json').read_text())
    for name,h in e_seal['files'].items():assert sha(ENGINEERING/name)==h,name
    assert integrity()['success']
    assert json.loads((OUT/'runtime_reconciliation_v1.json').read_text())['remaining_projected_seconds']<=17280
    data={name:json.loads((ENGINEERING/file).read_text()) for name,file in [('train','train_pool_v1.json'),('development','development_pool_v1.json'),('size_transfer','size_transfer_pool_v1.json')]}
    order=json.loads((OUT/'training_order_v1.json').read_text())
    assert sorted(order['indices'])==list(range(1000))
    assert [data['train'][i]['id'] for i in order['indices']]==order['ids']
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    ledger=Ledger();model=None;optimizer=None;evals={};checkpoint=None
    result=dict(status='INCOMPLETE',phases={},gates_unverified=True)
    save('training_started_v1.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),no_retry_resume=True,allocation_seconds=21600))
    def deadline(*unused):raise RuntimeError('RESOURCE_STOP: six-hour hard deadline')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(21600)
    try:
        torch.cuda.init();torch.cuda.reset_peak_memory_stats()
        model=InsertionSortAdapter().to('cuda')
        initial=parameter_identity(model)
        assert initial==json.loads((ENGINEERING/'initial_model_identity_v1.json').read_text())['identities']['cuda']
        frozen=frozen_identity(model)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,
                                   weight_decay=0,betas=(.9,.999),eps=1e-8)
        save('initial_training_model_v1.json',dict(parameter_sha256=initial,frozen_parameter_sha256=frozen,
             fresh_seed=6200,trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],
             torch_version=torch.__version__,cuda_version=torch.version.cuda,device=torch.cuda.get_device_name(),
             initial_optimizer_state_entries=len(optimizer.state),TF32=False,FP32=True,full_BPTT=True))
        evals['initial_diagnostic']=evaluate(model,data['train'][:8],ledger,'initial_diagnostic')
        result['phases']['initial_diagnostic']='COMPLETED_DESCRIPTIVE_ONLY'
        ledger.phase='training';train_begin=time.monotonic();running=[]
        with (OUT/'training_metrics_v1.jsonl').open('x') as stream:
            for update,index in enumerate(order['indices'],1):
                ledger.check();row=data['train'][index];ledger.array_attempted+=1
                try:
                    begin=time.monotonic();optimizer.zero_grad(set_to_none=True)
                    out=model(row['key_fp32'],row['pos'],ledger,row['id'])
                    loss=pointer_loss(out['pointer_logits'],row['pred'])
                    assert bool(torch.isfinite(loss))
                    torch.cuda.synchronize();forward=time.monotonic()-begin
                    r=record(row,out,loss);back_start=time.monotonic();ledger.backward(loss)
                    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                    grads=[p.grad for p in model.parameters() if p.grad is not None]
                    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
                    group_grad={name:float(torch.stack([p.grad.detach().square().sum() for n,p in model.named_parameters() if n.startswith(name) and p.grad is not None]).sum().sqrt()) for name in ('body.','encoder.','query.','key.')}
                    gradnorm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                    torch.cuda.synchronize();backward=time.monotonic()-back_start
                    before=[p.detach().clone() for p in model.parameters() if p.requires_grad]
                    ledger.check();assert ledger.optimizer_attempted<1000
                    ledger.optimizer_attempted+=1
                    try:
                        optimizer.step();ledger.optimizer_completed+=1
                    except BaseException:
                        ledger.optimizer_failed+=1;raise
                    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
                    assert all(bool(torch.isfinite(v).all()) for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))
                    displacement=float(torch.stack([(p.detach()-b).square().sum() for p,b in zip([p for p in model.parameters() if p.requires_grad],before)]).sum().sqrt())
                    torch.cuda.synchronize()
                    r.update(update=update,canonical_pool_index=index,unclipped_gradient_norm=float(gradnorm),
                             gradient_norms_by_group=group_grad,all_gradients_finite=True,parameters_optimizer_finite=True,
                             step_parameter_displacement=displacement,forward_seconds=forward,backward_and_clip_seconds=backward,
                             elapsed_update_seconds=time.monotonic()-begin,total_elapsed_seconds=time.monotonic()-ledger.started)
                    stream.write(json.dumps(r,allow_nan=False)+'\n');stream.flush();running.append(r)
                    ledger.array_completed+=1;save('execution_ledger_v1.json',ledger.summary())
                    del out,loss,before
                    if update%10==0:
                        print('UPDATE',update,'CE',r['CE'],'pointer',r['pointer_accuracy'],'grad',r['unclipped_gradient_norm'],'elapsed',r['total_elapsed_seconds'],flush=True)
                    ledger.check()
                except BaseException:
                    ledger.array_failed+=1;raise
        assert ledger.optimizer_completed==ledger.backward_completed==1000
        assert frozen_identity(model)==frozen
        result['phases']['training']='COMPLETED_1000_UPDATES'
        save('training_summary_v1.json',dict(online_metrics=metrics(running),training_seconds=time.monotonic()-train_begin,
                update_count=1000,final_model_parameter_sha256=parameter_identity(model),frozen_parameters_unchanged=True,
                note='Online predictions are pre-update training observations, not held-out evaluation'))
        checkpoint=OUT/'model_update1000_v1.pt'
        with checkpoint.open('xb') as f:
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                            optimizer_update_count=1000,model_seed=6200,training_order_sha256=sha(OUT/'training_order_v1.json'),
                            engineering_seal_sha256=sha(ENGINEERING/'implementation_seal_v1.json'),
                            training_implementation_seal_sha256=sha(OUT/'training_implementation_seal_v1.json')),f)
        save('checkpoint_identity_v1.json',dict(path=checkpoint.name,sha256=sha(checkpoint),size=checkpoint.stat().st_size,
             optimizer_updates=1000,selection='Predetermined final update1000 only'))
        for name,rows in [('training_subset',data['train'][:128]),('development',data['development']),('size_transfer',data['size_transfer'])]:
            evals[name]=evaluate(model,rows,ledger,name);result['phases'][name]='COMPLETED'
        assert ledger.attempted==ledger.completed==41472 and not ledger.failed
        assert ledger.bank_completed==331776 and ledger.commits==41472
        assert ledger.backward_completed==ledger.optimizer_completed==1000
        result['status']='AWAITING_OFFLINE_VERIFICATION'
        save('analysis_v1.json',dict(metrics={n:metrics(rows,True) for n,rows in evals.items() if n!='initial_diagnostic'},
            controls={n:controls(rows) for n,rows in [('training_subset',data['train'][:128]),('development',data['development']),('size_transfer',data['size_transfer'])]},
            bootstrap=dict(replicates=20000,seed=6202,unit='complete array',quantiles='linear',per_split_rng_reset=True),
            size_transfer_not_pooled=True,no_checkpoint_selection=True))
        assert frozen_identity(model)==frozen
    except BaseException as exc:
        result.update(status='INCOMPLETE',failure_type=type(exc).__name__,failure=str(exc),execution_boundary=ledger.phase)
        save('failure_v1.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
                                  phase=ledger.phase,ledger=ledger.summary(),retry_authorized=False))
        print('STOP',type(exc).__name__,str(exc),flush=True)
    finally:
        ledger.trace.close();torch.cuda.synchronize() if torch.cuda.is_initialized() else None
        result['resource_ledger']=ledger.summary()
        save('execution_ledger_v1.json',result['resource_ledger'])
        reconcile(ledger)
        result['integrity']=integrity()
        result['final_checkpoint']=json.loads((OUT/'checkpoint_identity_v1.json').read_text()) if checkpoint and checkpoint.exists() and (OUT/'checkpoint_identity_v1.json').exists() else None
        result['allocation_seconds_at_gpu_completion']=time.monotonic()-ledger.started
        result['deadline_met']=result['allocation_seconds_at_gpu_completion']<21600
        if not result['integrity']['success'] or not result['deadline_met']:
            result['status']='INCOMPLETE'
        save('run_result_pending_official_metric_v1.json',result)
        signal.alarm(0)
        print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
