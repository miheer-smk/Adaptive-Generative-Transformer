"""CPU tensor algebra on retained states/weights only; no model imports/calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,csv,hashlib,math
from pathlib import Path
import torch
torch.set_grad_enabled(False);torch.set_num_threads(1)
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
LOCAL=OUT.parent/'gatec_clrs_pointer_localization_6200_v1'
CAP=OUT.parent/'gatec_clrs_mhc_capture_6200_v1'
ENG=OUT.parent/'gatec_clrs_insertion_sort_6200_v1'

def save(name,value):
    with (OUT/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def csvsave(name,rows):
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def energy(x):return float(x.square().sum())
def center(x):return x-x.mean(0,keepdim=True)
def ratio(a,b):return a/b if b else None
def distribution(x):
    x=x.double().flatten()
    q=torch.quantile(x,torch.tensor([0.,.05,.25,.5,.75,.95,1.],dtype=torch.float64),interpolation='linear')
    return dict(count=len(x),mean=float(x.mean()),standard_deviation=float(x.std(unbiased=False)),
        minimum=float(q[0]),q05=float(q[1]),q25=float(q[2]),median=float(q[3]),q75=float(q[4]),q95=float(q[5]),maximum=float(q[6]),
        positive=int((x>0).sum()),zero=int((x==0).sum()),negative=int((x<0).sum()))

def geometry(x,condition,stage,rows):
    x=x.double();c=center(x);r=x.norm(dim=1);mu=x.mean(0);rad=ang=total=0.;err=0.;cos=[];dist=[]
    for i in range(16):
        for j in range(i+1,16):
            sq=float((x[i]-x[j]).square().sum());d=math.sqrt(sq);dist.append(d)
            if r[i]>0 and r[j]>0:
                a=float((r[i]-r[j]).square());b=float(r[i]*r[j]*(x[i]/r[i]-x[j]/r[j]).square().sum())
                cosine=float(x[i]@x[j]/(r[i]*r[j]));cos.append(cosine);error=sq-a-b
                assert abs(error)<=512*torch.finfo(torch.float64).eps*(float(r[i].square()+r[j].square())+sq+1)
                rad+=a;ang+=b;err=max(err,abs(error))
            else:a=b=cosine=error=None
            total+=sq
            rows.append(dict(condition=condition,stage=stage,i=i,j=j,distance=d,squared_distance=sq,cosine=cosine,
                radial_squared=a,angular_squared=b,identity_error=error))
    return dict(shape=list(x.shape),row_norms=r.tolist(),pooled_RMS=float(x.square().mean().sqrt()),
        total_energy=energy(x),centered_energy=energy(c),common_energy=16*energy(mu),
        pairwise_distance=distribution(torch.tensor(dist,dtype=torch.float64)),
        pairwise_cosine=distribution(torch.tensor(cos,dtype=torch.float64)) if cos else None,
        radial_sum=rad,angular_sum=ang,total_pairwise_squared_distance=total,radial_fraction=ratio(rad,total),angular_fraction=ratio(ang,total),
        radial_identity_max_error=err,zero_vectors=int((r==0).sum()),finite=bool(torch.isfinite(x).all()))

def score_condition(label,Hq,Hk,Wq,Wk,target,marginrows,queryrows,geometryrows):
    Q=Hq@Wq.T;K=Hk@Wk.T;S=(Q@K.T)/8
    assert S.shape==(16,16) and torch.isfinite(S).all()
    lse=torch.logsumexp(S,dim=1);logp=S-lse[:,None];p=torch.exp(logp)
    ce=lse-S[torch.arange(16),target];pred=S.argmax(dim=1)
    Qc,Kc=center(Q),center(K);b=Kc@Q.mean(0)/8;I=Qc@Kc.T/8
    rowcenter=S-S.mean(1,keepdim=True);reconstructed=b[None,:]+I
    error=float((rowcenter-reconstructed).abs().max())
    bound=1024*torch.finfo(torch.float64).eps*(float(S.abs().max())+float(b.abs().max())+float(I.abs().max())+1)
    assert error<=bound
    commons=b[None,:].expand(16,-1);commonCE=torch.logsumexp(commons,dim=1)-commons[torch.arange(16),target]
    G=(p-torch.eye(16,dtype=torch.float64)[target])/16
    ranks=[];stable_ranks=[];targetmargins=[];all_margins=[];common_margins=[];interaction_margins=[]
    common_wrong_to_correct=common_correct_to_wrong=0
    for i,t in enumerate(target.tolist()):
        greater=int((S[i]>S[i,t]).sum());ties=int((S[i]==S[i,t]).sum())
        stable_rank=1+greater+int(((S[i]==S[i,t])&(torch.arange(16)<t)).sum())
        rank=1+greater;ranks.append(rank);stable_ranks.append(stable_rank)
        assert rank==1+int((rowcenter[i]>rowcenter[i,t]).sum())
        bestwrong=max(float(S[i,j]) for j in range(16) if j!=t);margin=float(S[i,t])-bestwrong;targetmargins.append(margin)
        queryrows.append(dict(condition=label,query=i,target=t,prediction=int(pred[i]),correct=int(pred[i])==t,
            correct_target_rank=rank,stable_index_rank=stable_rank,target_ties=ties,target_margin=margin,
            target_score=float(S[i,t]),maximum_score=float(S[i].max()),logit_span=float(S[i].max()-S[i].min()),
            maximum_probability=float(p[i].max()),CE=float(ce[i]),common_only_target_rank=1+int((b>b[t]).sum()),
            interaction_only_target_rank=1+int((I[i]>I[i,t]).sum())))
        for j in range(16):
            if j==t:continue
            a=float(S[i,t]-S[i,j]);c=float(b[t]-b[j]);v=float(I[i,t]-I[i,j]);e=a-c-v
            assert abs(e)<=2*bound
            common_wrong_to_correct+=int(c<=0 and a>0);common_correct_to_wrong+=int(c>0 and a<=0)
            all_margins.append(a);common_margins.append(c);interaction_margins.append(v)
            marginrows.append(dict(condition=label,query=i,target=t,incorrect_candidate=j,total_margin=a,
                common_margin=c,interaction_margin=v,interaction_improves_target_margin=v>0,common_favors_target=c>0,
                total_favors_target=a>0,decomposition_error=e))
    geometries={stage:geometry(x,label,stage,geometryrows) for stage,x in [('query_representations',Hq),('candidate_representations',Hk),('projected_queries',Q),('projected_keys',K)]}
    span=S.max(1).values-S.min(1).values
    summary=dict(condition=label,pointer_accuracy=float(pred.eq(target).double().mean()),correct_pointers=int(pred.eq(target).sum()),
        exact_array_accuracy=int(bool(pred.eq(target).all())),predictions=pred.tolist(),prediction_counts={str(j):int((pred==j).sum()) for j in range(16)},
        target_ranks=ranks,mean_target_rank=sum(ranks)/16,stable_target_ranks=stable_ranks,
        mean_CE=float(ce.mean()),CE_by_query=ce.tolist(),CE_common_only=float(commonCE.mean()),CE_effect_of_existing_interaction=float((ce-commonCE).mean()),
        GS_inner_I=float((G*I).sum()),score_scale=dict(raw_min=float(S.min()),raw_max=float(S.max()),raw_Frobenius=float(S.norm()),
            row_centered_Frobenius=float(rowcenter.norm()),within_query_span=distribution(span)),
        mean_maximum_probability=float(p.max(1).values.mean()),mean_entropy=float((-(p*logp).sum(1)).mean()),
        target_margin=distribution(torch.tensor(targetmargins,dtype=torch.float64)),
        all240_target_margins=distribution(torch.tensor(all_margins,dtype=torch.float64)),
        all240_common_margins=distribution(torch.tensor(common_margins,dtype=torch.float64)),
        all240_interaction_margins=distribution(torch.tensor(interaction_margins,dtype=torch.float64)),
        common_wrong_to_total_correct=common_wrong_to_correct,common_correct_to_total_wrong=common_correct_to_wrong,
        common_preferred_candidate=int(b.argmax()),common_preference_vector=b.tolist(),common_preferred_candidate_target_frequency=int((target==int(b.argmax())).sum()),
        common_matrix_Frobenius=float(commons.norm()),interaction_Frobenius=float(I.norm()),interaction_common_norm_ratio=ratio(float(I.norm()),float(commons.norm())),
        decomposition_max_error=error,decomposition_roundoff_bound=bound,representation_geometry=geometries,
        interpretation='Analytical readout only; counts of240 margins are dependent within one array. CE score-space derivative does not imply parameter realizability.')
    tensors=dict(query_representations=Hq,candidate_representations=Hk,Q=Q,K=K,scores=S,row_centered_scores=rowcenter,
        common=b,interaction=I,probabilities=p,log_probabilities=logp,score_CE_adjoint_algebra=G,target=target)
    return summary,tensors

def main():
    manifest=json.loads((OUT/'manifest_v1.json').read_text())
    for name,r in manifest['inputs'].items():assert sha(ROOT/name)==r['sha256'],name
    assert sha(ENG/'training_results_v1/model_update1000_v1.pt')=='2c0e3d6ad3711d8f0ee209eb3e3b0f8db8964e6eeaf2be77b7bd56120f29e6c8'
    definitions=json.loads((OUT/'analysis_definitions_v1.json').read_text());assert definitions['sealed_before_outcomes']
    for n,h in json.loads((OUT/'analysis_seal_v1.json').read_text())['files'].items():assert sha(OUT/n)==h,n
    row=json.loads((LOCAL/'locked_input_v1.json').read_text());assert row==json.loads((ENG/'train_pool_v1.json').read_text())[0]
    target=torch.tensor(row['pred'],dtype=torch.long)
    assert hashlib.sha256(json.dumps(row['pred'],separators=(',',':')).encode()).hexdigest()==manifest['target_sha256']
    saved=torch.load(LOCAL/'final_tensor_evidence_v1.pt',map_location='cpu',weights_only=True)
    captured=torch.load(CAP/'captured_tensors_v1.pt',map_location='cpu',weights_only=True)
    ckpt=torch.load(ENG/'training_results_v1/model_update1000_v1.pt',map_location='cpu',weights_only=True)
    weights=ckpt['model_state_dict'];assert ckpt['optimizer_update_count']==1000
    assert torch.equal(target,saved['targets']) and torch.equal(target,captured['targets'])
    H=saved['block_outputs']['7'];C=saved['all_committed_streams'];assert H.shape==C.shape==(32,64)
    assert torch.equal(H,captured['blocks']['7']['outputs']) and torch.equal(C,captured['committed_streams'])
    assert torch.equal(C[:16],saved['candidate_states']) and torch.equal(C[16:],saved['query_states'])
    gain=weights['body.commit.state_norm.weight'];Wq=weights['query.weight'];Wk=weights['key.weight']
    assert Wq.shape==Wk.shape==(64,64) and gain.shape==(64,)
    assert all(v.device.type=='cpu' for v in [H,C,Wq,Wk,gain,target])
    normH=H.double()/(H.double().square().mean(1,keepdim=True)+1e-6).sqrt()*gain.double()
    assert torch.allclose(normH,C.double(),atol=1e-6,rtol=1e-5)
    # Literal matrix algebra only; no call to the checkpoint's registered head.
    Q32=C[16:]@Wq.T;K32=C[:16]@Wk.T;S32=(Q32@K32.T)/8
    assert torch.allclose(Q32,saved['projected_queries'],atol=1e-6,rtol=1e-5)
    assert torch.allclose(K32,saved['projected_keys'],atol=1e-6,rtol=1e-5)
    assert torch.allclose(S32,saved['raw_scores'],atol=1e-6,rtol=1e-5)
    H=H.double();C=C.double();Wq=Wq.double();Wk=Wk.double()
    alpha=float(C.square().mean().sqrt()/H.square().mean().sqrt())
    assert alpha>0 and math.isfinite(alpha)
    Hr=H.norm(dim=1,keepdim=True);assert (Hr>0).all();directions=H/Hr
    scalar=dict(alpha=alpha,recipe=definitions['global_scalar_recipe'],raw_pooled_RMS=float(H.square().mean().sqrt()),
        committed_pooled_RMS=float(C.square().mean().sqrt()),globally_scaled_raw_pooled_RMS=float((alpha*H).square().mean().sqrt()),
        scalar_squared=alpha**2,targets_used=False,separate_per_group_or_per_token_scales=False)
    save('global_scalar_v1.json',scalar)
    marginrows=[];queryrows=[];geometryrows=[];summaries={};tensors={}
    for label,X in [('A',C),('B',H),('B_global',alpha*H),('C',directions)]:
        summaries[label],tensors[label]=score_condition(label,X[16:],X[:16],Wq,Wk,target,marginrows,queryrows,geometryrows)
    exp=json.loads((LOCAL/'final_results_v1.json').read_text());A=tensors['A']['scores']
    assert torch.allclose(A,saved['raw_scores'].double(),atol=1e-6,rtol=1e-5)
    assert math.isclose(summaries['A']['mean_CE'],exp['CE'],abs_tol=1e-6,rel_tol=1e-5)
    assert summaries['A']['predictions']==saved['raw_scores'].argmax(1).tolist()
    b,bg=tensors['B']['scores'],tensors['B_global']['scores']
    assert torch.equal(b.argmax(1),bg.argmax(1))
    assert torch.equal(b.argsort(dim=1,descending=True,stable=True),bg.argsort(dim=1,descending=True,stable=True))
    score_scaling_error=float((bg-b*(alpha**2)).abs().max())
    assert score_scaling_error<=1024*torch.finfo(torch.float64).eps*(float(bg.abs().max())+1)
    comparisons={}
    for label in ['B','B_global','C']:
        x=summaries[label];a=summaries['A'];r1=torch.tensor(a['target_ranks']);r2=torch.tensor(x['target_ranks'])
        comparisons[label]=dict(pointer_accuracy_change_vs_A=x['pointer_accuracy']-a['pointer_accuracy'],
            mean_rank_change_vs_A=x['mean_target_rank']-a['mean_target_rank'],queries_with_better_rank=int((r2<r1).sum()),
            queries_with_worse_rank=int((r2>r1).sum()),queries_with_equal_rank=int((r2==r1).sum()),CE_change_vs_A=x['mean_CE']-a['mean_CE'])
    assert summaries['A']['target_ranks']==[x['target_rank'] for x in exp['queries']]
    save('condition_summaries_v1.json',summaries);save('condition_comparisons_v1.json',comparisons)
    csvsave('all_target_margins_v1.csv',marginrows);csvsave('per_query_v1.csv',queryrows);csvsave('pairwise_geometry_v1.csv',geometryrows)
    csvsave('condition_summary_v1.csv',[dict(condition=k,pointer_accuracy=v['pointer_accuracy'],exact_array_accuracy=v['exact_array_accuracy'],
        mean_target_rank=v['mean_target_rank'],mean_target_margin=v['target_margin']['mean'],mean_CE=v['mean_CE'],
        mean_max_probability=v['mean_maximum_probability'],common_norm=v['common_matrix_Frobenius'],interaction_norm=v['interaction_Frobenius'],
        interaction_improves_240=v['all240_interaction_margins']['positive'],common_favors_240=v['all240_common_margins']['positive'],actual_favors_240=v['all240_target_margins']['positive'],common_preferred_candidate=v['common_preferred_candidate']) for k,v in summaries.items()])
    with (OUT/'analytical_readout_tensors_v1.pt').open('xb') as f:torch.save(tensors,f)
    save('evidence_verification_v1.json',dict(success=True,checkpoint_sha256=manifest['checkpoint_sha256'],target_sha256=manifest['target_sha256'],array=row['id'],
        actual_terminal_states='Saved block7 outputs; source AGT.step h_terminal=h after final block, terminal_commit final_norm(h_terminal); exact equality across localization/capture',
        raw_and_committed_evidence_agrees=True,terminal_norm_reconstruction_max_error=float((normH-C).abs().max()),
        Q32_max_error=float((Q32-saved['projected_queries']).abs().max()),K32_max_error=float((K32-saved['projected_keys']).abs().max()),
        score32_max_error=float((S32-saved['raw_scores']).abs().max()),score64_max_error=float((A-saved['raw_scores'].double()).abs().max()),
        CE_reconstructed=summaries['A']['mean_CE'],CE_retained=exp['CE'],absolute_CE_difference=abs(summaries['A']['mean_CE']-exp['CE']),
        baseline_predictions_and_ranks_agree=summaries['A']['target_ranks']==[x['target_rank'] for x in exp['queries']],
        global_scaling_preserves_argmax_and_all_rankings=True,global_score_scaling_max_error=score_scaling_error,
        conditions=4,per_query_rows=len(queryrows),target_margin_rows=len(marginrows),pairwise_rows=len(geometryrows),
        model_constructions=0,forwards=0,backwards=0,optimizer_objects=0,optimizer_updates=0,tests=0,profiling=0,GPU_operations=0,
        torch_version=torch.__version__,numeric_precision='CPU FP64; literal FP32 baseline separately checked',no_AGT_import=True))
    print(json.dumps(dict(scalar=scalar,conditions={k:{n:v[n] for n in ['pointer_accuracy','mean_target_rank','mean_CE','predictions','interaction_common_norm_ratio','GS_inner_I']} for k,v in summaries.items()},comparisons=comparisons),indent=2))

if __name__=='__main__':main()
