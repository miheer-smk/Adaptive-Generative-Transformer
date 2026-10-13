"""Locked case-clustered acceptance analysis. Runs only after required execution."""
import json,time
import torch
from statistical_rules_v3 import make_bank,stratified_means,quantile,ablation_rule,latency_rule


def analyze(confirmation,ablated,profiling,panel,eligible,out):
    start=time.perf_counter();strata=torch.tensor([2*r['length']+r['label'] for r in panel]);bank=make_bank(strata)
    n=len(panel);assert n==512
    cases={a:{r['id']:r for r in confirmation[a]} for a in ('T','C')}
    S=torch.zeros(n,2,2,2,dtype=torch.float64);A=torch.zeros_like(S);current=torch.zeros_like(S);ce=torch.zeros(n,2,2,2,4,dtype=torch.float64)
    mean_ce=torch.zeros(n,2,2,2,dtype=torch.float64)
    for i,row in enumerate(panel):
        for ai,arm in enumerate(('T','C')):
            for h in (0,1):
                for b in (0,1):
                    cell=cases[arm][row['id']]['cells'][f'{h}{b}']
                    S[i,ai,h,b]=cell['robust'];A[i,ai,h,b]=cell['alternating_robust'];current[i,ai,h,b]=cell['current']['correct']
                    ce[i,ai,h,b,0]=cell['current']['CE']
                    mean_ce[i,ai,h,b]=(cell['current']['CE']+sum(v['CE'] for z in cell['futures']['boundary'] for v in z))/17
                    for hi,horizon in enumerate((1,4,8),1):
                        ce[i,ai,h,b,hi]=sum(v[horizon-1]['CE'] for v in cell['futures']['boundary'])/2
    D=.5*((S[:,:,0,0]-S[:,:,1,0])+(S[:,:,1,1]-S[:,:,0,1]))
    delta,draws=stratified_means(D[:,1]-D[:,0],bank);interval=quantile(draws,torch.tensor([.025,.975],dtype=torch.float64))
    lengths=torch.tensor([r['length'] for r in panel]);native={};comparable=True;competent=True;compatible=True
    for p in (0,1):
        for length in (16,24):
            ids=lengths==length;m=int(ids.sum());values=current[ids,0,p,p]-current[ids,1,p,p]
            subbank=make_bank(torch.tensor([r['label'] for r in panel if r['length']==length]))
            point,boots=stratified_means(values,subbank)
            ci=quantile(boots,torch.tensor([.00625,.99375],dtype=torch.float64))
            zero_bound=1-.0125**(1/m) if not bool(values.ne(0).any()) else None
            lower=float(ci[0]);upper=float(ci[1])
            if zero_bound is not None:lower=min(lower,-zero_bound);upper=max(upper,zero_bound)
            native_ce_diff=(ce[ids,0,p,p]-ce[ids,1,p,p]).mean(0)
            means=ce[ids,:,p,p].mean(0)
            all_mean=mean_ce[ids,:,p,p].mean(0)
            for ai in (0,1):
                competent=competent and float(current[ids,ai,p,p].mean())>=.99 and float(S[ids,ai,p,p].mean())>=.98 and float(all_mean[ai])<=.05 and bool((means[ai]<=.05).all())
            comparable=comparable and lower>=-.02 and upper<=.02 and abs(float(all_mean[0]-all_mean[1]))<=.01
            h=1-p
            deterioration=(ce[ids,0,h,p]-ce[ids,0,p,p]).mean(0)
            compatible=compatible and float(current[ids,0,h,p].mean())>=.99 and float(S[ids,0,h,p].mean())>=.98 and bool((deterioration<=.02).all())
            native[f'P{p}/L{length}']=dict(current_accuracy=[float(v) for v in current[ids,:,p,p].mean(0)],
                  robust_accuracy=[float(v) for v in S[ids,:,p,p].mean(0)],horizon_CE=means.tolist(),
                  native_mean_CE=all_mean.tolist(),native_mean_CE_difference=float(all_mean[0]-all_mean[1]),
                  paired_native_accuracy_difference=float(point),adjusted_interval=[lower,upper],zero_discordance_bound=zero_bound,
                  paired_CE_differences=native_ce_diff.tolist(),switched_T_accuracy=float(current[ids,0,h,p].mean()),
                  switched_T_robust=float(S[ids,0,h,p].mean()),cross_program_CE_deterioration=deterioration.tolist())
    alternating={}
    for length in (16,24):
        ids=lengths==length
        for h in (0,1):
            for b in (0,1):
                rate=float(A[ids,0,h,b].mean());alternating[f'L{length}/{h}{b}']=rate;compatible=compatible and rate>=.98
    u=torch.stack([S[:,ai,p,p] for ai in (0,1) for p in (0,1)],dim=1)
    aa=torch.zeros_like(u)
    for ai,arm in enumerate(('T','C')):
        bykey={(r['id'],r['program'].split('/')[-1]):r for r in ablated[arm]}
        for i,row in enumerate(panel):
            for p in (0,1):aa[i,ai*2+p]=bykey[(row['id'],f'P{p}')]['robust']
    ar=ablation_rule(u,aa,bank)
    selected_ids=list(dict.fromkeys(j['case'] for j in profiling));metadata={r['id']:r for r in panel}
    pl=torch.tensor([metadata[i]['length'] for i in selected_ids]);pp=torch.tensor([metadata[i]['label'] for i in selected_ids])
    timing=torch.full((32,2,2,5),float('nan'),dtype=torch.float64)
    for r in profiling:
        if r['warmup'] or r['path'].startswith('F/'):continue
        arm,program=r['path'].split('/');i=selected_ids.index(r['case']);ai=0 if arm=='T' else 1
        slot=r['repetition']-1
        if torch.isfinite(timing[i,ai,int(program[1]),slot]):raise RuntimeError('Duplicate prescribed profiling measurement')
        timing[i,ai,int(program[1]),slot]=r['seconds']
    profile_bank=make_bank(pl*2+pp);lr=latency_rule(timing,pl,pp,profile_bank)
    patterns={a:{} for a in ('T','C')};current_patterns={a:{} for a in ('T','C')}
    for ai,arm in enumerate(('T','C')):
        for pattern in range(16):
            bits=torch.tensor([(pattern>>shift)&1 for shift in (3,2,1,0)],dtype=torch.float64)
            patterns[arm][format(pattern,'04b')]=int((S[:,ai].reshape(n,4)==bits).all(1).sum())
            current_patterns[arm][format(pattern,'04b')]=int((current[:,ai].reshape(n,4)==bits).all(1).sum())
    continuation_metrics={}
    for arm in ('T','C'):
        for length in (16,24):
            rows=[v for v in confirmation[arm] if v['length']==length]
            for h in (0,1):
                for b in (0,1):
                    for schedule in ('boundary','alternating'):
                        group=[r['cells'][f'{h}{b}']['futures'][schedule] for r in rows]
                        continuation_metrics[f'{arm}/L{length}/{h}{b}/{schedule}']={str(q):dict(
                            accuracy=sum(z[q-1]['correct'] for futures in group for z in futures)/(2*len(rows)),
                            CE=sum(z[q-1]['CE'] for futures in group for z in futures)/(2*len(rows))) for q in (1,4,8)}
    bylength={}
    for length in (16,24):
        ids=lengths==length;lb=make_bank(torch.tensor([r['label'] for r in panel if r['length']==length]))
        point,rep=stratified_means((D[:,1]-D[:,0])[ids],lb)
        bylength[str(length)]=dict(point=float(point),interval=quantile(rep,torch.tensor([.025,.975],dtype=torch.float64)).tolist())
    result=dict(primary_delta=float(delta),primary_interval=interval.tolist(),primary_pass=float(interval[0])>=.05,
        length_specific=bylength,native=native,native_competence=competent,native_comparability=comparable,
        treatment_compatibility=compatible,alternating_robust=alternating,robust_patterns=patterns,current_patterns=current_patterns,
        continuation_metrics=continuation_metrics,
        ablation={k:v.tolist() if torch.is_tensor(v) else v for k,v in ar.items()},
        latency={k:v.tolist() if torch.is_tensor(v) else v for k,v in lr.items()},
        native_only_checkpoint_eligible=eligible,conditional_input_sampling_uncertainty=True,
        analysis_seconds=time.perf_counter()-start,bootstrap_replicates=20000,bootstrap_seed=6010,
        separate_ablation_latency_families=True,confirmation_bank=bank.identity,profiling_bank=profile_bank.identity)
    result['full_success']=eligible and competent and comparable and compatible and result['primary_pass'] and bool(ar['passes'].all()) and bool(lr['passes'].all())
    if result['analysis_seconds']>7200:raise RuntimeError('RESOURCE STOP: offline analysis exceeds two CPU hours')
    (out/'final_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    torch.save(dict(S=S,alternating=A,current=current,CE=ce,D=D,confirmation_indices=bank.indices,
                    profiling_indices=profile_bank.indices),out/'analysis_case_arrays.pt')
    return result
