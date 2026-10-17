"""Terminal-readout tensor arithmetic only; no AGT import or fitting."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,math,random,hashlib,time
import torch
from support_v1 import OUT,CACHE,PRIOR,ENG,ROOT,save,sha,check
torch.set_num_threads(1);torch.set_grad_enabled(False)

def invalid(p):
    n=len(p)
    if any(not isinstance(v,int) or not 0<=v<n for v in p):return True
    roots=[i for i,v in enumerate(p) if i==v]
    if len(roots)!=1:return True
    successors=[[] for _ in range(n)]
    for i,v in enumerate(p):
        if i!=v:successors[v].append(i)
    if any(len(v)>1 for v in successors):return True
    seen=set();node=roots[0]
    while node not in seen:
        seen.add(node)
        if not successors[node]:return len(seen)!=n
        node=successors[node][0]
    return True

def metrics(scores,target,pred=None):
    n=len(target);y=target.tolist()
    if scores is None:
        assert pred is not None
        return dict(predictions=pred,predecessor_accuracy=sum(a==b for a,b in zip(pred,y))/n,exact_array_accuracy=int(pred==y),invalid_chain=int(invalid(pred)),
            CE=None,target_ranks=None,stable_index_ranks=None,target_margins=None,distinct_predicted_candidates=len(set(pred)),score_metrics='Unavailable for hard-only control; no probability smoothing')
    S=scores.cpu().double();assert S.shape==(n,n) and torch.isfinite(S).all()
    p=S.argmax(1).tolist();shift=S-S.max(1,keepdim=True).values;logp=shift-torch.logsumexp(shift,1,keepdim=True)
    ce=-logp[torch.arange(n),target].mean();ranks=[];stable=[];margins=[];ties=[]
    for i,t in enumerate(y):
        ranks.append(1+int((S[i]>S[i,t]).sum()));ties.append(int((S[i]==S[i,t]).sum()))
        stable.append(ranks[-1]+int(((S[i]==S[i,t])&(torch.arange(n)<t)).sum()))
        wrong=S[i].clone();wrong[t]=-torch.inf;margins.append(float(S[i,t]-wrong.max()))
    return dict(predictions=p,predecessor_accuracy=sum(a==b for a,b in zip(p,y))/n,exact_array_accuracy=int(p==y),invalid_chain=int(invalid(p)),CE=float(ce),
        target_ranks=ranks,stable_index_ranks=stable,target_ties=ties,target_margins=margins,mean_target_rank=sum(ranks)/n,
        mean_stable_index_rank=sum(stable)/n,mean_target_margin=sum(margins)/n,distinct_predicted_candidates=len(set(p)),mean_max_probability=float(logp.exp().max(1).values.mean()))

def aggregate(rows):
    names=rows[0]['conditions'].keys();out={}
    for name in names:
        values=[x['conditions'][name] for x in rows]
        keys=['predecessor_accuracy','exact_array_accuracy','invalid_chain','distinct_predicted_candidates','CE','mean_target_rank','mean_stable_index_rank','mean_target_margin','mean_max_probability']
        out[name]={k:sum(v[k] for v in values)/len(values) if all(v.get(k) is not None for v in values) else None for k in keys}
        out[name]['arrays']=len(values)
    return out

def compute(U,V,B,dtype):
    uq=(U.to('cuda',dtype=dtype)@B['query_whitener'].to('cuda',dtype=dtype))@B['B'].to('cuda',dtype=dtype)
    vk=V.to('cuda',dtype=dtype)@B['candidate_whitener'].to('cuda',dtype=dtype)
    S=(uq@vk.T)/8;Q=uq.cpu().double();K=vk.cpu().double();n=len(Q)
    Qc=Q-Q.mean(0,keepdim=True);Kc=K-K.mean(0,keepdim=True)
    b=Kc@Q.mean(0)/8;I=Qc@Kc.T/8;Salgebra=Q@K.T/8;row=Salgebra-Salgebra.mean(1,keepdim=True)
    error=float((row-b[None,:]-I).abs().max());bound=1024*torch.finfo(torch.float64).eps*(float(Salgebra.abs().max())+1)
    assert error<=bound
    actual=S.cpu();difference=float((actual.double()-Salgebra).abs().max())
    assert torch.allclose(actual.double(),Salgebra,atol=1e-6,rtol=1e-5),'Readout score arithmetic/decomposition mismatch'
    common=b[None,:].expand(n,-1).to(dtype=dtype).clone()
    return actual,common,I,b,dict(exact_algebra_identity_max_error=error,FP_score_vs_double_algebra_max_error=difference,common_matrix_norm=float((b[None,:].expand(n,-1)).norm()),interaction_norm=float(I.norm()),
        interaction_common_norm_ratio=float(I.norm()/(b.norm()*math.sqrt(n))) if b.norm()>0 else None)

def main():
    check();assert not (OUT/'evaluation_started_v1.json').exists(),'No evaluation retry'
    for n,h in json.loads((OUT/'fit_seal_v1.json').read_text())['files'].items():assert sha(OUT/n)==h
    assert json.loads((OUT/'fit_results_v1.json').read_text())['certified'], 'Certificate barrier'
    for n,h in json.loads((OUT/'protocol_seal_v1.json').read_text())['files'].items():assert sha(ROOT/n)==h,n
    ext=json.loads((PRIOR/'heldout_extraction_results_v1.json').read_text());assert ext['status']=='COMPLETED_VERIFIED_HELDOUT_CACHE'
    eq=json.loads((PRIOR/'heldout_equivalence_v1.json').read_text());assert eq['success'] and len(eq['comparisons'])==64
    assert all(x['scores_within_tolerance'] and x['predictions_exact'] for x in eq['comparisons'])
    assert sha(PRIOR/ext['cache_file'])==ext['cache_sha256']
    save('evaluation_started_v1.json',dict(monotonic=time.monotonic(),fit_sealed=True,all64_archived_original_heads_verified_before_new_metrics=True))
    B=torch.load(OUT/'terminal_readout_v1.pt',map_location='cpu',weights_only=True)
    train=torch.load(CACHE/'training_representation_cache_v1.pt',map_location='cpu',weights_only=True)
    fitting_refs=[json.loads(s) for s in (ENG/'training_results_v1/training_subset_records_v1.jsonl').read_text().splitlines()]
    sets={'fitting':[dict(id=identity,candidate_committed_s=train['candidate_committed_s'][i],query_committed_s=train['query_committed_s'][i],targets=train['predecessor_targets'][i],original_pointer_scores=torch.tensor(fitting_refs[i]['pointer_logits'],dtype=torch.float32)) for i,identity in enumerate(train['array_ids'])]}
    sets.update(torch.load(PRIOR/ext['cache_file'],map_location='cpu',weights_only=True))
    # Retained original outputs and labels must still agree with the archive.
    for split in ['development','size_transfer']:
        refs=[json.loads(s) for s in (ENG/'training_results_v1'/(split+'_records_v1.jsonl')).read_text().splitlines()]
        pools=json.loads((ENG/(split+'_pool_v1.json')).read_text())
        assert len(sets[split])==len(refs)==len(pools)==32
        for i,(a,r,p) in enumerate(zip(sets[split],refs,pools)):
            assert a['id']==r['id']==p['id']==f'official/{split}/row{i:04d}'
            assert a['targets'].tolist()==r['truth']==p['pred']
            assert a['input_sha256']==r['original_input_sha256']==p['original_input_sha256']
            assert torch.equal(a['original_pointer_scores'],torch.tensor(r['pointer_logits'],dtype=torch.float32))
            assert a['original_pointer_scores'].argmax(1).tolist()==r['predictions']
    save('retained_original_head_verification_v1.json',dict(success=True,arrays=64,bitwise_archived_scores=True,predictions_exact=True,targets_exact=True,new_AGT_calls=0))
    assert torch.cuda.is_available(), 'Prior operational CUDA tensor arithmetic required'
    save('evaluation_runtime_v1.json',dict(torch=torch.__version__,CUDA=torch.version.cuda,device=torch.cuda.get_device_name(),arithmetic='Prior FP32/FP64 readout tensor algebra only; no AGT construction or calls',new_AGT_calls=0))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    predictions={};summaries={};score_evidence={};interaction_records=[];sensitivity={}
    with torch.no_grad():
        for split,arrays in sets.items():
            check();records=[];score_evidence[split]=[];different=0;total=0;maxdiff=0.
            for array in arrays:
                check();Q=array['query_committed_s'];V=array['candidate_committed_s'];y=array['targets'];n=len(y)
                S32,C32,I32,b32,d32=compute(Q,V,B,torch.float32)
                S64,C64,I64,b64,d64=compute(Q,V,B,torch.float64)
                conditions={'original':metrics(array['original_pointer_scores'],y),'fitted_FP32':metrics(S32,y),'uniform':metrics(torch.zeros((n,n)),y),
                    'self_pointer':metrics(None,y,list(range(n))),'original_order':metrics(None,y,[0]+list(range(n-1))),
                    'common_FP32':metrics(C32,y),'fitted_FP64':metrics(S64,y),'common_FP64':metrics(C64,y)}
                interaction={}
                for mode,S,C,I,b,detail in [('FP32',S32,C32,I32,b32,d32),('FP64',S64,C64,I64,b64,d64)]:
                    positive=negative=common_positive=actual_positive=flipped_good=flipped_bad=0;margins=[]
                    for i,t in enumerate(y.tolist()):
                        for j in range(n):
                            if j==t:continue
                            cm=float(b[t]-b[j]);im=float(I[i,t]-I[i,j]);am=float(S[i,t]-S[i,j])
                            positive+=im>0;negative+=im<0;common_positive+=cm>0;actual_positive+=am>0
                            flipped_good+=cm<=0 and am>0;flipped_bad+=cm>0 and am<=0;margins.append(im)
                    interaction[mode]=dict(**detail,pair_comparisons=n*(n-1),interaction_improves=positive,interaction_worsens=negative,
                        common_positive=common_positive,actual_positive=actual_positive,flipped_toward_target=flipped_good,flipped_against_target=flipped_bad,
                        mean_interaction_target_margin=sum(margins)/len(margins),common_preferred_candidate=int(b.argmax()),
                        total_target_rank_improvement=sum(a-b for a,b in zip(conditions['common_'+mode]['target_ranks'],conditions['fitted_'+mode]['target_ranks'])))
                records.append(dict(id=array['id'],n=n,targets=y.tolist(),conditions=conditions,interaction=interaction,
                    uniform_expected_random=dict(predecessor_accuracy=1/n,exact_array_accuracy=n**(-n),interpretation='Analytic random-draw expectations only; primary uniform row reports deterministic argmax0 with all ties')))
                score_evidence[split].append(dict(id=array['id'],fitted_FP32=S32,common_FP32=C32,interaction_FP32=I32,fitted_FP64=S64,common_FP64=C64,interaction_FP64=I64))
                different+=int((S32.argmax(1)!=S64.argmax(1)).sum());total+=n;maxdiff=max(maxdiff,float((S32.double()-S64).abs().max()))
            predictions[split]=records;summaries[split]=aggregate(records);sensitivity[split]=dict(prediction_disagreements=different,total_queries=total,max_score_difference=maxdiff)
    save('per_array_predictions_v1.json',predictions);save('metrics_v1.json',summaries);save('precision_sensitivity_v1.json',sensitivity)
    with (OUT/'score_decomposition_tensors_v1.pt').open('xb') as f:torch.save(score_evidence,f)
    # One shared paired whole-development-array resampling bank for both contrasts.
    bank=torch.load(PRIOR/'bootstrap_bank_v1.pt',map_location='cpu',weights_only=True)
    assert bank.shape==(20000,32) and bank.dtype==torch.int16 and int(bank.min())>=0 and int(bank.max())<32
    with (OUT/'bootstrap_bank_v1.pt').open('xb') as f:torch.save(bank,f)
    contrasts={}
    for mode in ['FP32','FP64']:
        contrasts[mode]={}
        for baseline in ['original','common_'+mode]:
            differences=torch.tensor([r['conditions']['fitted_'+mode]['predecessor_accuracy']-r['conditions'][baseline]['predecessor_accuracy'] for r in predictions['development']],dtype=torch.float64)
            values=differences[bank.long()].mean(1)
            contrasts[mode][baseline]=dict(point_estimate=float(differences.mean()),lower_bound=float(torch.quantile(values,.025,interpolation='linear')),per_array_differences=differences.tolist(),positive_adjusted_lower_bound=float(torch.quantile(values,.025,interpolation='linear'))>0)
    save('paired_contrasts_v1.json',dict(primary=contrasts['FP32'],sensitivity=contrasts['FP64'],replicates=20000,seed=6300,unit='complete development array',family='two primary paired contrasts; one-sided lower quantile .025 each',bank_sha256=sha(OUT/'bootstrap_bank_v1.pt'),quantile_interpolation='linear',uncertainty='Input-sampling uncertainty conditional on one frozen core and one readout fit; previously exposed development panel, not untouched confirmation'))
    fit=json.loads((OUT/'fit_results_v1.json').read_text())
    def competence(split,mode):
        x=summaries[split]['fitted_'+mode];return x['predecessor_accuracy']>=(.90 if split=='size_transfer' else .95) and x['exact_array_accuracy']>=(.50 if split=='size_transfer' else .80)
    both32=all(x['positive_adjusted_lower_bound'] for x in contrasts['FP32'].values());both64=all(x['positive_adjusted_lower_bound'] for x in contrasts['FP64'].values())
    strong32=fit['converged'] and competence('development','FP32') and both32
    strong64=fit['converged'] and competence('development','FP64') and both64
    if competence('development','FP64') and both64 and not (competence('development','FP32') and both32):classification='NUMERICALLY_FRAGILE_FP64_ONLY_RECOVERY'
    elif strong32:classification='STRONG_BILINEAR_DECODABILITY'
    elif both32:classification='CERTIFIED_BELOW_COMPETENCE_THRESHOLDS'
    elif competence('fitting','FP32') and not competence('development','FP32'):classification='FITTING_WITHOUT_ESTABLISHED_TRANSFER'
    elif not fit['converged']:classification='INCONCLUSIVE_NONCONVERGED_OPTIMIZATION'
    else:classification='CERTIFIED_FIXED_OBJECTIVE_NON_SUCCESS'
    save('analysis_results_v1.json',dict(classification=classification,fit_converged=fit['converged'],development_competence_FP32=competence('development','FP32'),transfer_competence_FP32=competence('size_transfer','FP32'),development_competence_FP64=competence('development','FP64'),both_primary_contrasts_positive=both32,strong_positive_evidence=strong32,precision_sensitivity=sensitivity,
        mean_fitting_CE_FP64=summaries['fitting']['fitted_FP64']['CE'],sealed_fit_CE=fit['CE'],historical_training_result='COMPLETED_NON_SUCCESS',new_AGT_calls_during_evaluation=0,readout_fits=1))
    assert abs(summaries['fitting']['fitted_FP64']['CE']-fit['CE'])<1e-9
    check();print(json.dumps(dict(classification=classification,development=summaries['development']['fitted_FP32'],transfer=summaries['size_transfer']['fitted_FP32'],contrasts=contrasts['FP32']),indent=2),flush=True)

if __name__=='__main__':main()
