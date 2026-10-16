"""One deterministic CPU FP64 full-batch fit; training cache only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,time,json,hashlib,signal,traceback
sys.path.append('/tmp/agt_clrs_cpu_6200_v1/lib/python3.12/site-packages')
import numpy as np
import scipy.linalg
import torch
from support_v1 import OUT,CACHE,ROOT,save,sha,check,integrity
torch.set_num_threads(1)

class FitLimit(Exception):pass
def tsha(x):return hashlib.sha256(bytes(x.detach().cpu().contiguous().flatten().view(torch.uint8).tolist())).hexdigest()
def main():
    check();assert not (OUT/'fit_started_v1.json').exists(),'No restart authorized'
    base=integrity();assert base['success'];save('integrity_before_fit_v1.json',base)
    path=CACHE/'training_representation_cache_v1.pt';assert sha(path)=='83b8a4c1d19ba479901d5714c246e89385aadfc76eb971c463fe9ac115352f41'
    cm=json.loads((CACHE/'cache_manifest_v1.json').read_text());assert sha(path)==cm['sha256']
    cache=torch.load(path,map_location='cpu',weights_only=True)
    C=cache['candidate_committed_s'];Q=cache['query_committed_s'];y=cache['predecessor_targets']
    assert C.shape==Q.shape==(128,16,64) and C.dtype==Q.dtype==torch.float32 and y.shape==(128,16) and y.dtype==torch.int64
    assert torch.equal(cache['row_indices'],torch.arange(128)) and torch.equal(cache['candidate_original_element_indices'],torch.arange(16)) and torch.equal(cache['query_original_element_indices'],torch.arange(16))
    assert torch.isfinite(C).all() and torch.isfinite(Q).all() and ((y>=0)&(y<16)).all()
    records=[json.loads(x) for x in (CACHE/'per_array_extraction_v1.jsonl').read_text().splitlines()]
    inputs=json.loads((CACHE/'input_manifest_v1.json').read_text())['arrays']
    assert len(records)==128 and cache['array_ids']==[f'official/train/row{i:04d}' for i in range(128)]
    for i,(r,e) in enumerate(zip(records,inputs)):
        assert r['id']==e['id']==cache['array_ids'][i] and tsha(C[i])==r['candidate_tensor_sha256'] and tsha(Q[i])==r['query_tensor_sha256'] and tsha(y[i])==r['target_tensor_sha256']
        assert y[i].tolist()==e['targets'] and r['original_input_sha256']==e['original_input_sha256']==cache['original_input_sha256'][i]
    save('fit_cache_verification_v1.json',dict(success=True,arrays=128,identities_and_roles=True,source_manifest_sha256=sha(CACHE/'input_manifest_v1.json'),targets_sha256=tsha(y),heldout_representations_accessed=False))
    def whiten(X):
        x=X.double().reshape(-1,64);M=x.T@x/len(x)
        values,vectors=scipy.linalg.eigh(np.asarray(M.tolist(),dtype=np.float64),driver='evr',check_finite=True)
        e=torch.tensor(values.tolist(),dtype=torch.float64);V=torch.tensor(vectors.tolist(),dtype=torch.float64)
        floor=1e-8*float(e.max());assert floor>0
        clipped=e.clamp_min(floor);W=(V*clipped.rsqrt()[None,:])@V.T
        assert torch.isfinite(W).all() and torch.allclose(W,W.T,atol=1e-12,rtol=1e-12)
        return W,dict(moment=M,eigenvalues=e,floored_eigenvalues=clipped,eigenvectors=V,floor=floor,floored_dimensions=int((e<floor).sum()))
    U,Pq=whiten(Q);V,Pk=whiten(C)
    Ut=Q.double()@U;Vt=C.double()@V
    assert torch.isfinite(Ut).all() and torch.isfinite(Vt).all()
    pre=dict(query_whitener=U,candidate_whitener=V,query_moment=Pq,candidate_moment=Pk,mean_subtracted=False,all64_dimensions_retained=True,training_arrays=cache['array_ids'])
    with (OUT/'preconditioners_v1.pt').open('xb') as f:torch.save(pre,f)
    B=torch.zeros((64,64),dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([B],lr=1.,max_iter=1,max_eval=1024,tolerance_grad=0.,tolerance_change=0.,history_size=20,line_search_fn='strong_wolfe')
    start=time.monotonic();fit_deadline=min(start+300.,__import__('support_v1').deadline())
    save('fit_started_v1.json',dict(monotonic=start,deadline=fit_deadline,initial='B=0',optimizer='One persistent PyTorch LBFGS object; max_iter1 calls retain global history20',closure_gradient='Exact analytical gradient of full objective; no autograd backward or AGT import',unprescribed_defaults='lr1; PyTorch strong-Wolfe c1=1e-4,c2=.9',no_restart=True))
    evaluations=0;accepted=0;step_calls=0;closure_calls=0;cache_values={};log=(OUT/'fit_evaluations_v1.jsonl').open('x');iterations=(OUT/'fit_iterations_v1.jsonl').open('x')
    accepted_B=B.detach().clone();accepted_info=None;stop=None
    def expired(signum,frame):raise FitLimit('time_limit')
    old=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,max(.001,fit_deadline-time.monotonic()))
    def evaluate():
        nonlocal evaluations
        check()
        if time.monotonic()>=fit_deadline:raise FitLimit('time_limit')
        key=tsha(B)
        if key in cache_values:return cache_values[key]
        if evaluations>=1024:raise FitLimit('evaluation_limit')
        evaluations+=1
        with torch.no_grad():
            S=(Ut@B)@Vt.transpose(1,2)/8
            shift=S-S.max(2,keepdim=True).values;logp=shift-torch.logsumexp(shift,dim=2,keepdim=True)
            ce=-logp.gather(2,y[:,:,None]).mean();penalty=1e-6/2*B.square().sum();F=ce+penalty
            D=logp.exp();D.scatter_add_(2,y[:,:,None],-torch.ones_like(y[:,:,None],dtype=torch.float64));D/=128*16
            grad=(Ut.transpose(1,2)@D@Vt).sum(0)/8+1e-6*B
            assert torch.isfinite(F) and torch.isfinite(grad).all()
            bound=float(grad.square().sum()/(2e-6))
        value=dict(F=float(F),CE=float(ce),regularizer=float(penalty),gradient=grad.clone(),bound=bound,key=key)
        cache_values[key]=value
        log.write(json.dumps(dict(evaluation=evaluations,accepted_so_far=accepted,step_call=step_calls,F=value['F'],CE=value['CE'],regularizer=value['regularizer'],gradient_norm=float(grad.norm()),certificate_bound=bound,B_sha256=key,elapsed=time.monotonic()-start))+'\n');log.flush()
        return value
    def closure():
        nonlocal closure_calls
        closure_calls+=1;v=evaluate();B.grad=v['gradient'].clone();return torch.tensor(v['F'],dtype=torch.float64)
    try:
        accepted_info=evaluate()
        while True:
            if accepted_info['bound']<=1e-6:stop='converged';break
            if accepted>=256:stop='accepted_iteration_limit';break
            if evaluations>=1024:stop='evaluation_limit';break
            if time.monotonic()>=fit_deadline:stop='time_limit';break
            previous=accepted_info;accepted_B=B.detach().clone();cache_values={previous['key']:previous}
            step_calls+=1;optimizer.param_groups[0]['max_eval']=max(2,1024-evaluations+1)
            optimizer.step(closure)
            key=tsha(B);assert key in cache_values,'Accepted point lacks evaluated full gradient'
            now=cache_values[key]
            moved=not torch.equal(B.detach(),accepted_B)
            if moved:
                accepted+=1
                state=optimizer.state[B];direction=state['d'].reshape(64,64);step=float(state['t'])
                g0=float((previous['gradient']*direction).sum());g1=float((now['gradient']*direction).sum())
                armijo=now['F']<=previous['F']+1e-4*step*g0+64*torch.finfo(torch.float64).eps*max(1,abs(previous['F']))
                curvature=abs(g1)<=.9*abs(g0)+64*torch.finfo(torch.float64).eps
                assert armijo and curvature,'Strong-Wolfe acceptance numerical violation'
            accepted_info=now;accepted_B=B.detach().clone()
            iterations.write(json.dumps(dict(accepted_iteration=accepted,step_calls=step_calls,moved=moved,F=now['F'],CE=now['CE'],gradient_norm=float(now['gradient'].norm()),certificate_bound=now['bound'],evaluations=evaluations,elapsed=time.monotonic()-start))+'\n');iterations.flush()
    except FitLimit as e:
        stop=str(e)
        with torch.no_grad():B.copy_(accepted_B)
    except BaseException as e:
        with torch.no_grad():B.copy_(accepted_B)
        save('fit_failure_v1.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),retry=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old);log.close();iterations.close()
    assert accepted_info is not None and torch.isfinite(B).all() and accepted<=256 and evaluations<=1024
    assert tsha(B)==accepted_info['key']
    state=dict(B=B.detach().clone(),query_whitener=U,candidate_whitener=V,objective='mean query CE+1e-6/2*||B||_F²',regularization=1e-6,convergence_bound=accepted_info['bound'],training_cache_sha256=sha(path),initialization='zero',precision='FP64')
    with (OUT/'terminal_readout_v1.pt').open('xb') as f:torch.save(state,f)
    result=dict(status='FIT_SEALED',stop_reason=stop,converged=accepted_info['bound']<=1e-6,accepted_iterations=accepted,loss_gradient_evaluations=evaluations,closure_calls=closure_calls,step_calls=step_calls,
        objective=accepted_info['F'],CE=accepted_info['CE'],regularizer=accepted_info['regularizer'],gradient_norm=float(accepted_info['gradient'].norm()),certificate_bound=accepted_info['bound'],elapsed_seconds=time.monotonic()-start,
        parameters=4096,history_size=20,one_optimizer_for_readout_only=1,AGT_optimizer_objects=0,AGT_backwards=0,readout_autograd_backwards=0,gradient='Analytic full-batch exact objective gradient',query_floor=Pq['floor'],candidate_floor=Pk['floor'],query_floored_dimensions=Pq['floored_dimensions'],candidate_floored_dimensions=Pk['floored_dimensions'],heldout_data_used=False)
    save('fit_results_v1.json',result)
    files={n:sha(OUT/n) for n in ['terminal_readout_v1.pt','preconditioners_v1.pt','fit_results_v1.json','fit_evaluations_v1.jsonl','fit_iterations_v1.jsonl','scope_lock_v1.json','fit_readout_v1.py','support_v1.py']}
    save('fit_seal_v1.json',dict(files=files,sealed_before_heldout_extraction=True,terminal_iterate_only=True,training_cache_sha256=sha(path)))
    assert integrity()['success'];check();print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
