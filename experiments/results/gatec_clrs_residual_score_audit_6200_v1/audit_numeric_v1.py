"""Offline tensor algebra only. No AGT imports, modules, forwards or backwards."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import csv,hashlib,json,math
from pathlib import Path
import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False)
D=Path(__file__).resolve().parent
R=D.parents[2]
L=D.parent/'gatec_clrs_pointer_localization_6200_v1'
T=D.parent/'gatec_clrs_insertion_sort_6200_v1/training_results_v1'
EPS=torch.finfo(torch.float64).eps

def save(name,x):
    with (D/name).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def csvsave(name,rows):
    with (D/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def center(x):return x-x.mean(0,keepdim=True)
def energy(x):return float(x.square().sum())
def safe_ratio(x,y):return x/y if y else None
def correlation(x,y):
    x=center(x.double().reshape(-1,1)).flatten();y=center(y.double().reshape(-1,1)).flatten()
    n=float(x.norm()*y.norm());return float(x@y)/n if n else None
def stats(x):
    x=x.double();c=center(x);mu=x.mean(0);n,d=x.shape
    norms=x.norm(dim=1);dist=torch.pdist(x);cos=[]
    for i in range(n):
        for j in range(i+1,n):
            if norms[i]>0 and norms[j]>0:cos.append(float((x[i]@x[j])/(norms[i]*norms[j])))
    sv=torch.linalg.svdvals(c)
    return dict(rows=n,features=d,total_energy=energy(x),centered_energy=energy(c),common_energy=n*energy(mu),
        common_vector_norm=float(mu.norm()),mean_row_norm=float(norms.mean()),min_row_norm=float(norms.min()),max_row_norm=float(norms.max()),
        mean_pairwise_L2=float(dist.mean()),max_pairwise_L2=float(dist.max()),mean_pairwise_squared_L2=float(dist.square().mean()),
        pairwise_cosine_mean=sum(cos)/len(cos) if cos else None,pairwise_cosine_min=min(cos) if cos else None,
        centered_singular_values=sv.tolist(),centered_spectral_energy=float(sv.square().sum()),
        numerical_centered_rank=int((sv>max(n,d)*EPS*float(sv.max())).sum()),
        total_energy_identity_error=energy(x)-energy(c)-n*energy(mu))

def radial(x,model,positions,stage,rows):
    x=x.double();r=x.norm(dim=1);valid=r>0;rad=ang=total=0.;err=0.
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            if not (valid[i] and valid[j]):
                rows.append(dict(model=model,positions=positions,stage=stage,i=i,j=j,defined=False,total_squared_distance=float((x[i]-x[j]).square().sum()),radial_squared=None,angular_squared=None,identity_error=None));continue
            a=float((r[i]-r[j]).square());b=float(r[i]*r[j]*(x[i]/r[i]-x[j]/r[j]).square().sum());v=float((x[i]-x[j]).square().sum());e=v-a-b
            assert abs(e)<=512*EPS*(float(r[i].square()+r[j].square())+v+1)
            rad+=a;ang+=b;total+=v;err=max(err,abs(e))
            rows.append(dict(model=model,positions=positions,stage=stage,i=i,j=j,defined=True,total_squared_distance=v,radial_squared=a,angular_squared=b,identity_error=e))
    return dict(radial_sum=rad,angular_sum=ang,total_pairwise_squared_distance=total,radial_fraction=safe_ratio(rad,total),
                angular_fraction=safe_ratio(ang,total),maximum_identity_error=err,zero_vector_count=int((~valid).sum()),
                fraction_definition='Ratio of summed pair squared-distance components; not mean of per-pair fractions')

def mixer(logits,scale,initial):
    # Literal source operation in CPU FP32; this is reconstructed parameter
    # algebra, not a call to MHC.forward or any nn.Module.
    p=torch.exp(logits.float().clamp(-10,10))
    for _ in range(5):
        p=p/(p.sum(-1,keepdim=True)+1e-6)
        p=p/(p.sum(-2,keepdim=True)+1e-6)
    A=(scale.float()[:,None]*p).double();p=p.double();s=scale.double();d=len(s)
    u,sv,vh=torch.linalg.svd(A);psv=torch.linalg.svdvals(p)
    e=torch.ones(d,dtype=torch.float64)/math.sqrt(d);perp=torch.eye(d,dtype=torch.float64)-torch.outer(e,e)
    au=A@e;orth=A@perp;orth_sv=torch.linalg.svdvals(orth);orth_rms=float(orth.norm())/math.sqrt(d-1)
    rank=int((sv>d*EPS*float(sv[0])).sum())
    data=dict(source='Source-fixed logits=zeros/scale=ones; not an initial checkpoint' if initial else 'Direct final checkpoint parameters',
        initial_analytic_rank=1 if initial else None,logits_min=float(logits.min()),logits_max=float(logits.max()),
        scale_min=float(s.min()),scale_max=float(s.max()),scale_mean=float(s.mean()),scale_std=float(s.std(unbiased=False)),
        scale_negative_count=int((s<0).sum()),scale_zero_count=int((s==0).sum()),
        P_singular_values=psv.tolist(),A_singular_values=sv.tolist(),A_numerical_rank=rank,
        rank_tolerance=d*EPS*float(sv[0]),condition2=None if initial or rank<d else float(sv[0]/sv[-1]),
        condition_status='Analytically singular rank1 at initialization' if initial else 'Numerically singular at stated tolerance' if rank<d else 'Finite full-rank numeric estimate',
        singular_ratio_raw=float(sv[0]/sv[-1]) if sv[-1]>0 else None,
        uniform_input_gain=float(au.norm()),orthogonal_input_rms_gain=orth_rms,orthogonal_input_spectral_gain=float(orth_sv[0]),
        uniform_to_orthogonal_rms_ratio=None if initial else safe_ratio(float(au.norm()),orth_rms),
        initial_analytic_orthogonal_gain=0 if initial else None,
        P_row_sum_range=[float(p.sum(1).min()),float(p.sum(1).max())],P_column_sum_range=[float(p.sum(0).min()),float(p.sum(0).max())],
        leading_output_direction=u[:,0].tolist(),uniform_feature_direction=e.tolist())
    return data,p,A,u

def main():
    manifest=json.loads((D/'manifest_v1.json').read_text())
    for name,r in manifest['inputs'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==r['sha256'],name
    checkpoint=torch.load(T/'model_update1000_v1.pt',map_location='cpu',weights_only=True)
    weights=checkpoint['model_state_dict'];assert checkpoint['optimizer_update_count']==1000
    assert all(v.device.type=='cpu' for v in weights.values())
    row=json.loads((L/'locked_input_v1.json').read_text());target=torch.tensor(row['pred'],dtype=torch.long)
    residual=[];radials={};radialrows=[];norms={};mixers={};mix_align=[];scores={};marginrows=[];queryrows=[];derived={};projection={}
    for model in ['initial','final']:
        data=torch.load(L/(model+'_tensor_evidence_v1.pt'),map_location='cpu',weights_only=True)
        assert all(v.device.type=='cpu' for v in data.values() if torch.is_tensor(v))
        assert torch.equal(data['targets'],target)
        initial=model=='initial';gain=torch.ones(64) if initial else weights['body.commit.state_norm.weight']
        mix={};mixers[model]={};derived[model]={}
        for name in sorted(n[:-7] for n in weights if n.endswith('mhc.logits')):
            logits=torch.zeros_like(weights[name+'.logits']) if initial else weights[name+'.logits']
            scale=torch.ones_like(weights[name+'.scale']) if initial else weights[name+'.scale']
            md,p,A,u=mixer(logits,scale,initial);mixers[model][name]=md;mix[name]=(A,u)
            derived[model]['mixer/'+name]=dict(P=p,A=A,scale=scale.double())
        for group,sl in [('candidate',slice(0,16)),('query',slice(16,32))]:
            for b in range(8):
                X=(data['h0'] if b==0 else data['block_outputs'][str(b-1)])[sl].double()
                Y=data['block_outputs'][str(b)][sl].double();delta=Y-X
                xc,yc,dc=center(X),center(Y),center(delta)
                coordinate_error=float((yc-xc-dc).abs().max());ex,ed,ey=energy(xc),energy(dc),energy(yc);cross=2*float((xc*dc).sum());er=ey-ex-ed-cross
                assert abs(er)<=512*EPS*(ex+ed+abs(cross)+ey+1)
                assert coordinate_error<=512*EPS*(float(X.abs().max()+Y.abs().max()+delta.abs().max())+1)
                xs,ys=stats(X),stats(Y);mn='body.blocks.blocks.'+str(b)+('.core.mhc' if b>=6 else '.mhc')
                A,u=mix[mn];mu=delta.mean(0);pdir=u[:,0]
                alignment=abs(float(mu@pdir))/float(mu.norm()) if mu.norm()>0 else None
                leadfrac=float((dc@pdir).square().sum())/ed if ed else None
                item=dict(model=model,positions=group,block=b,species=('SWA','SWA','GQA','GQA','GDN','GDN','FFN','FFN')[b],
                    input_source='retained h0' if b==0 else 'previous retained block output; exact source-backed input',
                    X=xs,Y=ys,delta=stats(delta),centered_input_energy=ex,centered_increment_energy=ed,
                    twice_centered_inner_product=cross,centered_output_energy=ey,centered_energy_change=ey-ex,
                    identity_error=er,coordinate_identity_max_error=coordinate_error,
                    common_energy_change=ys['common_energy']-xs['common_energy'],common_vector_increment_norm=float(mu.norm()),
                    common_vector_inner_product=float(X.mean(0)@mu),
                    increment_input_centered_cosine=float((xc*dc).sum()/(xc.norm()*dc.norm())) if xc.norm()*dc.norm()>0 else None,
                    cancellation_fraction_of_positive_terms=-cross/(ex+ed) if cross<0 and ex+ed else 0.,
                    centered_change_classification='reduction' if ey<ex-512*EPS*(ex+ey+1) else 'growth' if ey>ex+512*EPS*(ex+ey+1) else 'unchanged within numeric precision',
                    negative_cross_term=cross<0,
                    task_alignment='Unresolved for block increments; target-conditioned readout analyzed separately')
                residual.append(item)
                mix_align.append(dict(model=model,positions=group,block=b,mixer=mn,common_increment_abs_cosine_dominant_output=alignment,
                    centered_increment_energy_fraction_in_dominant_output=leadfrac,
                    all_increment_energy_fraction_top_output=float((delta@pdir).square().sum())/energy(delta) if energy(delta) else None,
                    centered_increment_energy_fraction_top5=float((dc@u[:,:5]).square().sum())/ed if ed else None,
                    causal_status='Susceptibility/output alignment only; pre-mHC core outputs unavailable; no inversion performed'))
                derived[model][group+'/block'+str(b)]=dict(X_centered=xc,Y_centered=yc,delta_centered=dc)
            H=data['block_outputs']['7'][sl].double();C=data['all_committed_streams'][sl].double();g=gain.double()
            pre=H/torch.sqrt(H.square().mean(-1,keepdim=True)+1e-6);weighted=pre*g
            err=float((weighted-C).abs().max());assert torch.allclose(weighted,C,atol=1e-6,rtol=1e-5)
            # Audit literal CPU FP32 reconstruction separately from double geometry.
            hf=data['block_outputs']['7'][sl];source32=hf*torch.rsqrt(hf.square().mean(-1,keepdim=True)+1e-6)*gain
            sourceerr=float((source32.double()-C).abs().max())
            assert torch.allclose(source32.double(),C,atol=1e-6,rtol=1e-5)
            key=model+'/'+group;radials[key]={}
            for stage,x in [('precommit',H),('pre_gain_normalized',pre),('gain_weighted',weighted),('retained_committed',C)]:
                radials[key][stage]=radial(x,model,group,stage,radialrows)
            pairs=torch.stack([pre[i]-pre[j] for i in range(16) for j in range(i+1,16)])
            ratios=(pairs*g).norm(dim=1)/pairs.norm(dim=1)
            G=data['gradients']['candidate_states' if group=='candidate' else 'query_states'].double()
            norms[key]=dict(precommit=stats(H),pre_gain=stats(pre),gain_weighted=stats(weighted),retained_committed=stats(C),
                gains=dict(source='Deterministic source ones' if initial else 'Direct checkpoint body.commit.state_norm.weight',
                    values=g.tolist(),min=float(g.min()),max=float(g.max()),mean=float(g.mean()),negative=int((g<0).sum())),
                gain_pairwise_distance_ratio=dict(min=float(ratios.min()),max=float(ratios.max()),mean=float(ratios.mean())),
                gain_centered_energy_ratio=energy(center(weighted))/energy(center(pre)),
                task_adjoint_weighted_gain_rms=float((G*g).norm()/G.norm()),
                reconstructed_committed_max_error=err,source_FP32_committed_max_error=sourceerr,
                radial_magnitude_correlations=dict(original_keys=correlation(H.norm(dim=1),torch.tensor(row['key_original'])),
                    original_positions=correlation(H.norm(dim=1),torch.tensor(row['pos'])),numeric_predecessor_indices=correlation(H.norm(dim=1),target)),
                correlation_limit='Single-array descriptive correlation; arbitrary pointer-index labels; no usability or causality inference')
            derived[model][group+'/normalization']=dict(pre_gain=pre,gain_weighted=weighted,gains=g)
        Q=data['projected_queries'].double();K=data['projected_keys'].double();S=data['raw_scores'].double()
        qbar=Q.mean(0);kbar=K.mean(0);Qc=center(Q);Kc=center(K)
        b=Kc@qbar/8;I=Qc@Kc.T/8;Srow=S-S.mean(1,keepdim=True);reconstruction=b[None,:]+I
        matrixerror=float((S-Q@K.T/8).abs().max());decomperror=float((Srow-reconstruction).abs().max())
        assert torch.allclose(S,Q@K.T/8,atol=1e-6,rtol=1e-5)
        assert torch.allclose(Srow,reconstruction,atol=1e-6,rtol=1e-5)
        probs=data['probabilities'].double();G=data['gradients'].get('raw_scores')
        analytic=(probs-torch.nn.functional.one_hot(target,16))/16
        adjoint_source='Retained original CE backward score adjoint' if G is not None else 'Algebraic mean CE derivative from saved probabilities and sealed labels'
        G=analytic if G is None else G.double();assert torch.allclose(G,analytic,atol=1e-6,rtol=1e-5)
        interaction_margins=[];common_margins=[];total_margins=[];margin_error=0.
        for i,y in enumerate(target.tolist()):
            rank=lambda v:1+int((v>v[y]).sum())
            wrong=[j for j in range(16) if j!=y]
            qr=dict(model=model,query=i,target=y,prediction=int(S[i].argmax()),raw_rank=rank(S[i]),row_centered_rank=rank(Srow[i]),
                    common_rank=rank(b),interaction_only_rank=rank(I[i]),raw_worst_target_margin=float(S[i,y]-S[i,wrong].max()),
                    centered_worst_target_margin=float(Srow[i,y]-Srow[i,wrong].max()),
                    common_worst_target_margin=float(b[y]-b[wrong].max()),interaction_mean_target_margin=float((I[i,y]-I[i,wrong]).mean()))
            assert qr['raw_rank']==qr['row_centered_rank']
            queryrows.append(qr)
            for j in wrong:
                common=float(b[y]-b[j]);interaction=float(I[i,y]-I[i,j]);actual=float(S[i,y]-S[i,j]);res=actual-common-interaction
                margin_error=max(margin_error,abs(res));assert abs(res)<=2e-6
                interaction_margins.append(interaction);common_margins.append(common);total_margins.append(actual)
                marginrows.append(dict(model=model,query=i,target=y,incorrect_candidate=j,actual_target_margin=actual,
                    common_target_margin=common,interaction_target_margin=interaction,margin_identity_error=res,
                    interaction_direction='improves target margin' if interaction>0 else 'worsens target margin' if interaction<0 else 'zero',
                    interaction_corrects_common_negative=common<0 and common+interaction>0,
                    interaction_breaks_common_positive=common>0 and common+interaction<0))
        ce=lambda x:float((torch.logsumexp(x,dim=1)-x[torch.arange(16),target]).mean())
        gsI=float((G*I).sum());alI=float((analytic*I).sum());gn=float(G.norm());inorm=float(I.norm());cnorm=float(b.norm()*4)
        scores[model]=dict(Q_shape=list(Q.shape),K_shape=list(K.shape),score_shape=list(S.shape),means_Q=qbar.tolist(),means_K=kbar.tolist(),
            common_preference=b.tolist(),interaction=I.tolist(),retained_row_centered_scores=Srow.tolist(),
            raw_score_reconstruction_max_error=matrixerror,row_centering_identity_max_error=decomperror,margin_identity_max_error=margin_error,
            common_matrix_Frobenius_norm=cnorm,interaction_Frobenius_norm=inorm,interaction_to_common_norm_ratio=safe_ratio(inorm,cnorm),
            common_candidate_span=float(b.max()-b.min()),interaction_max_abs=float(I.abs().max()),
            common_top_candidate=int(b.argmax()),final_argmaxes=S.argmax(1).tolist(),common_predictions_correct=int((target==int(b.argmax())).sum()),
            interaction_margin_positive=sum(v>0 for v in interaction_margins),interaction_margin_negative=sum(v<0 for v in interaction_margins),
            interaction_margin_zero=sum(v==0 for v in interaction_margins),mean_interaction_target_margin=sum(interaction_margins)/240,
            common_pairwise_margin_correct=sum(v>0 for v in common_margins),actual_pairwise_margin_correct=sum(v>0 for v in total_margins),
            CE_retained_score=ce(S),CE_common_only_score_algebra=ce(b[None,:].expand(16,-1)),CE_reconstructed_score_algebra=ce(reconstruction),
            CE_effect_of_existing_interaction=ce(reconstruction)-ce(b[None,:].expand(16,-1)),
            score_adjoint_source=adjoint_source,score_adjoint_agreement_max_error=float((G-analytic).abs().max()),
            GS_inner_I=gsI,algebraic_GS_inner_I=alI,inner_product_error_bound_from_saved_G=float((G-analytic).norm()*I.norm()),
            local_score_space_direction='Infinitesimal increase in existing interaction reduces CE' if gsI<0 else 'Infinitesimal increase in existing interaction increases CE' if gsI>0 else 'Zero directional derivative',
            gradient_norm=gn,cosine_GS_I=safe_ratio(gsI,gn*inorm),interaction_singular_values=torch.linalg.svdvals(I).tolist(),
            interpretation_limit='Fixed score-space direction only; no implication of parameter realizability, parity, deployed policies or historical optimizer effect')
        projection[model]={}
        for label,x,y in [('query',data['query_states'].double(),Q),('candidate',data['candidate_states'].double(),K)]:
            xs,ys=stats(x),stats(y)
            projection[model][label]=dict(input=xs,output=ys,centered_energy_ratio=ys['centered_energy']/xs['centered_energy'],
                relative_pairwise_variation_ratio=(ys['mean_pairwise_L2']/ys['mean_row_norm'])/(xs['mean_pairwise_L2']/xs['mean_row_norm']),
                initial_parameter_matrix_available=False if initial else True)
        derived[model]['score']=dict(Qc=Qc,Kc=Kc,b=b,I=I,score_adjoint=G,retained_raw_scores=S)
    save('residual_energy_v1.json',residual);save('radial_summary_v1.json',radials);save('normalization_v1.json',norms)
    save('mhc_operators_v1.json',mixers);save('mhc_increment_alignment_v1.json',mix_align);save('target_score_decomposition_v1.json',scores)
    save('projection_geometry_v1.json',projection);csvsave('target_margins_v1.csv',marginrows);csvsave('query_target_ranks_v1.csv',queryrows);csvsave('radial_pairs_v1.csv',radialrows)
    csvsave('residual_summary_v1.csv',[{k:r[k] for k in ['model','positions','block','species','centered_input_energy','centered_increment_energy','twice_centered_inner_product','centered_output_energy','centered_energy_change','common_energy_change','negative_cross_term','centered_change_classification','identity_error']} for r in residual])
    with (D/'derived_algebra_v1.pt').open('xb') as f:torch.save(derived,f)
    save('evidence_limits_v1.json',dict(direct=['Stored h0 and all block outputs for32 positions per model','Committed and projected candidate/query tensors, scores, probabilities and original CE adjoints','Final checkpoint mHC parameters and RMSNorm gains','Sealed input keys, original positions and official predecessor targets'],
        algebraic=['Block input from h0/previous output; Delta=Y-X','Centered residual energy, radial/angular identities','Pre-gain RMS normalization and gain application','Effective mHC A from checkpoint/source equations','Fixed initial mHC and RMSNorm constants from source; no initial model built','Common and interaction score components and target margins'],
        unavailable=['Core activations immediately before mHC','Full initial parameter checkpoint, including initial query/key weight matrices','Intermediate training checkpoints and optimizer history beyond retained scalar records','Multi-input or independently trained-seed mechanistic replication','Causal intervention on residuals, RMSNorm, gains or mixers'],
        prohibited_not_performed=['No inverse or pseudoinverse of mixers','No reconstructed pre-mHC core outputs','No model construction, AGT import, forward, backward, tests, profiling, optimizer or GPU execution'],
        initial_constants_provenance='src/mhc.py:38-46 and src/norms.py:27-28; AGTv3/adapter source inspected for overrides; fresh recorded identity verified by prior diagnostic'))
    save('numeric_verification_v1.json',dict(success=True,residual_decompositions=len(residual),radial_pairs=len(radialrows),target_margin_rows=len(marginrows),
        max_residual_energy_error=max(abs(r['identity_error']) for r in residual),max_residual_coordinate_error=max(r['coordinate_identity_max_error'] for r in residual),
        score_max_errors={k:v['row_centering_identity_max_error'] for k,v in scores.items()},
        normalization_max_errors={k:v['reconstructed_committed_max_error'] for k,v in norms.items()},
        model_constructions=0,model_calls=0,backward_calls=0,optimizer_operations=0,tests=0,profiling=0,GPU_execution=0,
        numeric_precision='CPU float64 audit; source-equivalent FP32 mHC/RMSNorm reconstructions identified separately'))
    print(json.dumps(dict(scores={k:{n:v[n] for n in ['common_matrix_Frobenius_norm','interaction_Frobenius_norm','interaction_to_common_norm_ratio','interaction_margin_positive','interaction_margin_negative','GS_inner_I','CE_effect_of_existing_interaction']} for k,v in scores.items()},
        normalization={k:{'pre_centered_energy':v['precommit']['centered_energy'],'normalized_centered_energy':v['pre_gain']['centered_energy'],'committed_centered_energy':v['retained_committed']['centered_energy'],'gain_ratio':v['gain_centered_energy_ratio']} for k,v in norms.items()}),indent=2))

if __name__=='__main__':main()
