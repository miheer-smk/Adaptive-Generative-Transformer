"""One fresh bounded CPU FP64 refit, exact prior objective and line search."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,time,json,hashlib,signal,traceback
sys.path.append('/tmp/agt_clrs_cpu_6200_v1/lib/python3.12/site-packages')
import torch
from support_v1 import OUT,ROOT,CACHE,PRIOR,save,sha,check,integrity
torch.set_num_threads(1)
class Limit(Exception):pass
def raw(x):return bytes(x.detach().cpu().contiguous().flatten().view(torch.uint8).tolist())
def tsha(x):return hashlib.sha256(raw(x)).hexdigest()
def main():
    check();assert not (OUT/'fit_started_v1.json').exists(),'No second fit'
    seal=json.loads((OUT/'protocol_seal_v1.json').read_text())
    for n,h in seal['files'].items():assert sha(ROOT/n)==h,n
    before=integrity();assert before['success'];save('integrity_before_fit_v1.json',before)
    cache=torch.load(CACHE/'training_representation_cache_v1.pt',map_location='cpu',weights_only=True)
    pre=torch.load(PRIOR/'preconditioners_v1.pt',map_location='cpu',weights_only=True)
    C=cache['candidate_committed_s'];Q=cache['query_committed_s'];y=cache['predecessor_targets']
    assert C.shape==Q.shape==(128,16,64) and C.dtype==Q.dtype==torch.float32 and y.shape==(128,16) and y.dtype==torch.int64
    assert torch.equal(cache['row_indices'],torch.arange(128))
    assert torch.equal(cache['candidate_original_element_indices'],torch.arange(16)) and torch.equal(cache['query_original_element_indices'],torch.arange(16))
    assert torch.isfinite(C).all() and torch.isfinite(Q).all() and ((y>=0)&(y<16)).all()
    records=[json.loads(x) for x in (CACHE/'per_array_extraction_v1.jsonl').read_text().splitlines()]
    inputs=json.loads((CACHE/'input_manifest_v1.json').read_text())['arrays']
    assert len(records)==len(inputs)==128 and cache['array_ids']==[f'official/train/row{i:04d}' for i in range(128)]
    for i,(r,e) in enumerate(zip(records,inputs)):
        assert r['id']==e['id']==cache['array_ids'][i] and tsha(C[i])==r['candidate_tensor_sha256'] and tsha(Q[i])==r['query_tensor_sha256'] and tsha(y[i])==r['target_tensor_sha256']
        assert y[i].tolist()==e['targets'] and r['original_input_sha256']==e['original_input_sha256']==cache['original_input_sha256'][i]
    U=pre['query_whitener'];V=pre['candidate_whitener']
    assert pre['mean_subtracted'] is False and pre['all64_dimensions_retained'] is True and pre['training_arrays']==cache['array_ids']
    assert U.shape==V.shape==(64,64) and U.dtype==V.dtype==torch.float64
    Ut=Q.double()@U;Vt=C.double()@V
    assert torch.isfinite(Ut).all() and torch.isfinite(Vt).all()
    save('training_cache_verification_v1.json',dict(success=True,arrays=128,order_verified=True,all_per_array_hashes_verified=True,preconditioners_reused_without_recalculation=True,heldout_tensors_opened=False))
    B=torch.zeros((64,64),dtype=torch.float64,requires_grad=True)
    save('fit_started_v1.json',dict(initialization='B=0; fresh optimizer',optimizer='torch.optim.LBFGS',lr=1.,history_size=20,line_search_fn='strong_wolfe',c1=1e-4,c2=.9,tolerance_grad=0.,tolerance_change=0.,accepted_update_limit=512,actual_evaluation_limit=1024,wall_seconds=10.,clock_boundary='Immediately before optimizer construction; constructor included',gradient='Exact analytic full regularized objective gradient',no_AGT_imports=True,no_restart=True))
    events=(OUT/'fit_evaluations_v1.jsonl').open('x');steps=(OUT/'fit_iterations_v1.jsonl').open('x')
    attempted=completed=failed=accepted=step_calls=closure_calls=hits=0
    values={};accepted_B=B.detach().clone();accepted_info=None;stop=None;optimizer=None
    def alarm(signum,frame):raise Limit('time_limit')
    old=signal.signal(signal.SIGALRM,alarm)
    # Arm just before the clock, so even constructor time cannot overrun silently.
    signal.setitimer(signal.ITIMER_REAL,10.)
    start=time.monotonic()
    try:
        optimizer=torch.optim.LBFGS([B],lr=1.,max_iter=1,max_eval=1024,tolerance_grad=0.,tolerance_change=0.,history_size=20,line_search_fn='strong_wolfe')
        deadline=start+10.
        def evaluate():
            nonlocal attempted,completed,failed,hits
            if time.monotonic()>=deadline:raise Limit('time_limit')
            key=raw(B);digest=hashlib.sha256(key).hexdigest()
            if key in values:
                hits+=1
                events.write(json.dumps(dict(event='cache_hit',cache_hit=hits,B_sha256=digest,source_evaluation=values[key]['evaluation'],step_call=step_calls,accepted_so_far=accepted,elapsed=time.monotonic()-start))+'\n');events.flush()
                return values[key]
            if attempted>=1024:raise Limit('evaluation_limit')
            attempted+=1
            events.write(json.dumps(dict(event='evaluation_attempt',evaluation=attempted,B_sha256=digest,step_call=step_calls,accepted_so_far=accepted,elapsed=time.monotonic()-start))+'\n');events.flush()
            try:
                with torch.no_grad():
                    S=(Ut@B)@Vt.transpose(1,2)/8
                    shift=S-S.max(2,keepdim=True).values;logp=shift-torch.logsumexp(shift,dim=2,keepdim=True)
                    ce=-logp.gather(2,y[:,:,None]).mean();penalty=1e-6/2*B.square().sum();F=ce+penalty
                    D=logp.exp();D.scatter_add_(2,y[:,:,None],-torch.ones_like(y[:,:,None],dtype=torch.float64));D/=128*16
                    grad=(Ut.transpose(1,2)@D@Vt).sum(0)/8+1e-6*B
                    assert torch.isfinite(F) and torch.isfinite(grad).all()
                    bound=float(grad.square().sum()/(2e-6))
                value=dict(F=float(F),CE=float(ce),regularizer=float(penalty),gradient=grad.clone(),bound=bound,key=key,evaluation=attempted)
                values[key]=value;completed+=1
                events.write(json.dumps(dict(event='evaluation_completed',evaluation=attempted,accepted_so_far=accepted,step_call=step_calls,F=value['F'],CE=value['CE'],regularizer=value['regularizer'],gradient_norm=float(grad.norm()),certificate_bound=bound,B_sha256=digest,elapsed=time.monotonic()-start))+'\n');events.flush()
                return value
            except BaseException:
                failed+=1
                events.write(json.dumps(dict(event='evaluation_failed',evaluation=attempted,B_sha256=digest,elapsed=time.monotonic()-start))+'\n');events.flush()
                raise
        def closure():
            nonlocal closure_calls
            closure_calls+=1;v=evaluate();B.grad=v['gradient'].clone();return torch.tensor(v['F'],dtype=torch.float64)
        accepted_info=evaluate()
        while True:
            if accepted_info['bound']<=1e-6:stop='converged';break
            if accepted>=512:stop='accepted_iteration_limit';break
            if attempted>=1024:stop='evaluation_limit';break
            if time.monotonic()>=deadline:raise Limit('time_limit')
            previous=accepted_info;values={previous['key']:previous};step_calls+=1
            optimizer.param_groups[0]['max_eval']=max(2,1024-attempted+1)
            optimizer.step(closure)
            key=raw(B);assert key in values,'Accepted point must have matching full gradient'
            now=values[key];moved=not torch.equal(B.detach(),accepted_B)
            if moved:
                state=optimizer.state[B];direction=state['d'].reshape(64,64);step=float(state['t'])
                g0=float((previous['gradient']*direction).sum());g1=float((now['gradient']*direction).sum())
                armijo=now['F']<=previous['F']+1e-4*step*g0+64*torch.finfo(torch.float64).eps*max(1,abs(previous['F']))
                curvature=abs(g1)<=.9*abs(g0)+64*torch.finfo(torch.float64).eps
                assert armijo and curvature,'Strong-Wolfe acceptance violation'
                accepted+=1
            accepted_info=now;accepted_B=B.detach().clone()
            steps.write(json.dumps(dict(accepted_iteration=accepted,step_calls=step_calls,moved=moved,F=now['F'],CE=now['CE'],regularizer=now['regularizer'],gradient_norm=float(now['gradient'].norm()),certificate_bound=now['bound'],matching_evaluation=now['evaluation'],B_sha256=hashlib.sha256(key).hexdigest(),actual_evaluations=attempted,completed_evaluations=completed,cache_hits=hits,elapsed=time.monotonic()-start,strong_Wolfe_pass=True))+'\n');steps.flush()
            if not moved:stop='no_accepted_update';break
    except Limit as e:
        stop=str(e)
        with torch.no_grad():B.copy_(accepted_B)
    except BaseException as e:
        with torch.no_grad():B.copy_(accepted_B)
        save('fit_failure_v1.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),retry=False))
        raise
    finally:
        fit_seconds=time.monotonic()-start
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old);events.close();steps.close()
    assert attempted==completed+failed and accepted<=512 and attempted<=1024
    assert accepted_info is None or raw(B)==accepted_info['key']
    certified=bool(accepted_info is not None and accepted_info['bound']<=1e-6 and fit_seconds<=10. and stop=='converged')
    terminal=dict(B=B.detach().clone(),query_whitener=U,candidate_whitener=V,gradient=None if accepted_info is None else accepted_info['gradient'],initialization='zero',regularization=1e-6,training_cache_sha256=sha(CACHE/'training_representation_cache_v1.pt'),convergence_bound=None if accepted_info is None else accepted_info['bound'])
    with (OUT/'terminal_readout_v1.pt').open('xb') as f:torch.save(terminal,f)
    if optimizer is not None:
        with (OUT/'terminal_optimizer_state_v1.pt').open('xb') as f:torch.save(optimizer.state_dict(),f)
    result=dict(certified=certified,converged=certified,stop_reason=stop,accepted_iterations=accepted,step_calls=step_calls,closure_calls=closure_calls,evaluations_attempted=attempted,evaluations_completed=completed,evaluations_failed=failed,cache_hits=hits,elapsed_seconds=fit_seconds,
        objective=None if accepted_info is None else accepted_info['F'],CE=None if accepted_info is None else accepted_info['CE'],regularizer=None if accepted_info is None else accepted_info['regularizer'],gradient_norm=None if accepted_info is None else float(accepted_info['gradient'].norm()),certificate_bound=None if accepted_info is None else accepted_info['bound'],certificate_threshold=1e-6,heldout_opened=False,AGT_model_calls=0,AGT_backwards=0,AGT_optimizers=0,terminal_extra_evaluations=0,terminal_gradient_source='Matching retained full regularized-objective evaluation; no recomputation')
    save('fit_results_v1.json',result)
    names=['terminal_readout_v1.pt','terminal_optimizer_state_v1.pt','fit_results_v1.json','fit_evaluations_v1.jsonl','fit_iterations_v1.jsonl','protocol_v1.md','protocol_seal_v1.json','fit_readout_v1.py','training_cache_verification_v1.json']
    save('fit_seal_v1.json',dict(certified=certified,files={n:sha(OUT/n) for n in names if (OUT/n).exists()},sealed_before_heldout_evaluation=True))
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
