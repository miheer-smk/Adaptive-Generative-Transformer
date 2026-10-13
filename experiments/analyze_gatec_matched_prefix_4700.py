"""Sealed record-only matched B/C prefix analysis; no models or autograd."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import gzip,hashlib,json,math,subprocess,traceback
from collections import Counter,defaultdict
from datetime import datetime,timezone
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'experiments/results/gatec_st_action_value_4700_v1'
TEMPORAL=ROOT/'experiments/results/gatec_temporal_vjp_4700_update1040_v1'
OUT=ROOT/'experiments/results/gatec_matched_prefix_analysis_4700_v1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as h:json.dump(value,h,indent=2,allow_nan=False);h.write('\n')
def vec(values):return torch.as_tensor(values,dtype=torch.float64)
def finite(x):
    if torch.is_tensor(x):assert torch.isfinite(x).all()
    elif isinstance(x,float):assert math.isfinite(x)
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,(list,tuple)):
        for v in x:finite(v)

def summarize_tapes(values,bootstrap_indices):
    """Scalar paired-tape contrast; no new resampling indices generated."""
    x=vec(values);boot=x[bootstrap_indices].mean(1)
    interval=torch.quantile(boot,vec([.025,.975])).tolist()
    halves=[float(x[:4].mean()),float(x[4:].mean())]
    sign='positive' if min(halves)>0 and interval[0]>0 else 'negative' if max(halves)<0 and interval[1]<0 else 'unresolved'
    return dict(mean=float(x.mean()),tape_values=x.tolist(),half_means=halves,bootstrap_95=interval,
        resolved_sign=sign,degenerate_interval=interval[0]==interval[1],sample_SD=float(x.std()))

def summarize_product(point,halves,bootstrap):
    interval=torch.quantile(bootstrap,vec([.025,.975])).tolist()
    sign='positive' if min(halves)>0 and interval[0]>0 else 'negative' if max(halves)<0 and interval[1]<0 else 'unresolved'
    return dict(value=float(point),half_values=halves,bootstrap_95=interval,resolved_sign=sign,degenerate_interval=interval[0]==interval[1])

def rare_group(probabilities):
    order=sorted(range(8),key=lambda a:(float(probabilities[a]),a))
    chosen=[];mass=0.
    for a in order:
        following=mass+float(probabilities[a])
        if following>.01:break
        mass=following;chosen.append(a)
    return chosen,order,mass

def local_credit(pi,R,Q,gradient):
    centered=R-R[0] # Equivalent categorical derivative; constant values are exactly zero.
    bpar=pi*(centered-pi@centered);gce=pi*(Q-pi@Q)
    return bpar,-(bpar@gradient),gce@gradient

def hard_and_relaxed(logits,uniforms,mask,tau):
    """Reconstruct original FP32 sampler arithmetic on CPU; no sampling."""
    raw=torch.tensor(logits,dtype=torch.float32);mask=torch.tensor(mask,dtype=torch.bool)
    assert raw.shape==(11,) and mask.shape==(10,) and mask[:8].all() and not mask[8:].any()
    legal=raw[:10].masked_fill(~mask,-torch.inf)
    eps=torch.finfo(torch.float32).eps
    g=-torch.log(-torch.log(uniforms.clamp(eps,1-eps)))
    relaxed=F.softmax((F.log_softmax(legal,-1)+g)/tau,-1).masked_fill(~mask,0.)
    log_prob=F.log_softmax(raw[:8].double(),-1);prob=log_prob.exp()
    assert prob.gt(0).all()
    return prob,log_prob,relaxed,int(relaxed.argmax())

def state_table(previous,noise,root_rows,cf_runs,st_indices,cf_indices):
    spec=previous['probe'];raw=previous['raw_logits_both_checkpoints'][str(spec['checkpoint_update'])]
    relaxed=[];gradients=[];st_records=[]
    for k in range(8):
        run_id=f'{spec["probe_id"]}/ST/tape{k}/actionNone';record=root_rows[run_id]
        assert record['token']==spec['token_index'] and record['decision']==0
        assert record['noise_coordinate']==[4801,spec['sequence_id'],spec['token_index'],0]
        assert record['legal_successor_mask']==[True]*8+[False]*2
        assert record['legal_action_mask']==[True]*8+[False]*2+[True]
        assert record['execution_before']==0 and record['pointer_before']==0 and not record['just_restarted_before']
        torch.testing.assert_close(vec(record['raw_controller_logits'][0]),vec(raw),atol=1e-6,rtol=1e-5)
        pi,lp,z,selected=hard_and_relaxed(record['raw_controller_logits'][0],noise['tapes'][k][spec['sequence_id']][spec['token_index'],0],record['legal_successor_mask'],2.)
        assert selected==record['selected_action'] and record['actual_executed_block']==selected
        torch.testing.assert_close(pi,vec(record['probabilities'][:8]),atol=1e-6,rtol=1e-5)
        relaxed.append(z[:8].double());grad=previous['ST_runs'][k]['probe_logit_gradient']
        assert grad is not None,'Required retained full gradient unavailable/disconnected; cannot fabricate it'
        assert len(grad)==11 and grad[8:]==[0,0,0]
        gradients.append(grad[:8]);st_records.append(dict(tape=k,noise_coordinate=record['noise_coordinate'],
            uniform=noise['tapes'][k][spec['sequence_id']][spec['token_index'],0].tolist(),
            selected_action=selected,full_logit_gradient=grad,relaxed_weights_FP32_CPU=z.tolist(),raw_logits=record['raw_controller_logits'][0]))
    # Root logits are fixed by incoming state, not by the unforced first draw.
    pi,lp=vec(raw[:8]).softmax(-1),vec(raw[:8]).log_softmax(-1)
    q=vec(previous['CF_final_ce']);correct=vec(previous['CF_correct']);gs=vec(gradients);z=torch.stack(relaxed)
    assert q.shape==correct.shape==(8,8) and gs.shape==z.shape==(8,8)
    for k in range(8):
        for a in range(8):
            run=cf_runs[f'{spec["probe_id"]}/CF/tape{k}/action{a}']
            assert run['forced_action']==a and run['bank']=='CF' and run['tape']==k
            assert run['final_ce']==float(q[k,a]) and bool(run['correct'])==bool(correct[k,a])
    mean_g=gs.mean(0);R=correct.mean(0);Q=q.mean(0);vpar=pi@R
    bpar,_,_=local_credit(pi,R,Q,mean_g);gce=pi*(Q-pi@Q)
    torch.testing.assert_close(gce,vec(previous['g_CF']),atol=1e-12,rtol=1e-10)
    torch.testing.assert_close(gce@mean_g,vec(previous['inner_product']),atol=1e-12,rtol=1e-10)
    qb=q[cf_indices].mean(1);rb=correct[cf_indices].mean(1);gb=gs[st_indices].mean(1)
    centered_rb=rb-rb[:,:1]
    bpar_boot=pi*(centered_rb-(centered_rb*pi).sum(-1,keepdim=True));gce_boot=pi*(qb-(qb*pi).sum(-1,keepdim=True))
    gamma_boot=-(bpar_boot*gb).sum(-1);ce_boot=(gce_boot*gb).sum(-1)
    par_halves=[];ce_halves=[]
    for sl in [slice(0,4),slice(4,8)]:
        hh=correct[sl].mean(0);qq=q[sl].mean(0);gg=gs[sl].mean(0)
        centered_hh=hh-hh[0]
        par_halves.append(float(-(pi*(centered_hh-pi@centered_hh))@gg));ce_halves.append(float((pi*(qq-pi@qq))@gg))
    group,order,mass=rare_group(pi)
    actions=[]
    for a in range(8):
        actions.append(dict(action=a,label=f'ADVANCE({a})',probability=float(pi[a]),log_probability=float(lp[a]),
            mean_CE=float(Q[a]),mean_correctness=float(R[a]),CE_by_CF_tape=q[:,a].tolist(),correct_by_CF_tape=correct[:,a].tolist(),
            full_gradient_by_ST_tape=gs[:,a].tolist(),mean_full_gradient=float(mean_g[a]),relaxed_weight_by_ST_tape=z[:,a].tolist(),
            constant_binary_outcomes=bool(correct[:,a].eq(correct[0,a]).all()),zero_of_eight=bool(correct[:,a].sum()==0),eight_of_eight=bool(correct[:,a].sum()==8)))
    output=dict(probe=spec,actions=actions,root_logits=raw,legal_successor_actions=list(range(8)),ST_records=st_records,
        sampler='nominal hard softmax(raw legal logits); tau2 affects relaxation; original FP32 clamped-uniform convention preserved',
        relaxed_provenance='reconstructed from retained root logits/masks/noise with original FP32 formula on CPU; argmax verified; original CUDA relaxed tensors were not retained, so bitwise soft-weight equality cannot be certified',
        V_par=float(vpar),b_par=bpar.tolist(),Gamma_par=summarize_product(-(bpar@mean_g),par_halves,gamma_boot),
        CE_alignment=summarize_product(gce@mean_g,ce_halves,ce_boot),existing_CE_alignment=dict(value=previous['inner_product'],
            direction=previous['direction'],bootstrap_95=previous['inner_product_bootstrap_95'],half_values=previous['half_bank_inner_products']),
        predicted_CE_derivative_under_descent=float(-(gce@mean_g)),
        observed_action_correctness_means_flat=bool(R.eq(R[0]).all()),all_binary_outcomes_constant=bool(correct.eq(correct[0,0]).all()),
        observed_action_correctness_flat_on_each_tape=bool(correct.eq(correct[:,:1]).all()),
        low_probability_group=dict(actions=group,sort_order=order,hard_mass=mass,threshold=.01),
        gradient_mean=mean_g.tolist(),hard_probability_entropy=float(-(pi*lp).sum()))
    finite(output)
    return output,dict(pi=pi,q=q,R=correct,gs=gs,z=z)

def pair_contrasts(B,C,bi,ci):
    # Both value tables share the same CF tape labels/action columns.
    piB,piC=B['pi'],C['pi'];uniform=torch.full((8,),.125,dtype=torch.float64)
    contrasts={};expectations={}
    for name,w in [('pi_B',piB),('pi_C',piC),('uniform',uniform)]:
        expectations[name]={}
        for arm,state in [('B',B),('C',C)]:
            expectations[name][arm]=dict(parity=summarize_tapes(state['R']@w,ci),CE=summarize_tapes(state['q']@w,ci))
        contrasts[f'state_parity_C_minus_B_under_{name}']=summarize_tapes((C['R']-B['R'])@w,ci)
        contrasts[f'state_CE_C_minus_B_under_{name}']=summarize_tapes((C['q']-B['q'])@w,ci)
    for name,w in [('pi_C',piC),('uniform',uniform)]:
        contrasts[f'within_B_parity_{name}_minus_pi_B']=summarize_tapes((B['R']-B['R'][:,:1])@(w-piB),ci)
        contrasts[f'within_B_CE_{name}_minus_pi_B']=summarize_tapes((B['q']-B['q'][:,:1])@(w-piB),ci)
    allocationC_par=(C['R']-C['R'][:,:1])@(piC-piB)
    allocationC_ce=(C['q']-C['q'][:,:1])@(piC-piB)
    natural_par=(C['R']-B['R'])@piB+allocationC_par
    natural_ce=(C['q']-B['q'])@piB+allocationC_ce
    contrasts['natural_root_and_state_parity_C_minus_B']=summarize_tapes(natural_par,ci)
    contrasts['natural_root_and_state_CE_C_minus_B']=summarize_tapes(natural_ce,ci)
    torch.testing.assert_close(natural_par,C['R']@piC-B['R']@piB,atol=1e-12,rtol=1e-10)
    torch.testing.assert_close(natural_ce,C['q']@piC-B['q']@piB,atol=1e-12,rtol=1e-10)
    group,order,mass=rare_group(piB)
    rare=dict(actions=group,sort_order=order,hard_mass=mass,threshold=.01,empty=not group)
    if group:
        w=torch.zeros(8,dtype=torch.float64);w[group]=1/len(group)
        for arm,state in [('B',B),('C',C)]:
            rare[arm]=dict(parity=summarize_tapes(state['R']@w,ci),CE=summarize_tapes(state['q']@w,ci),
                parity_minus_current_pi_B=summarize_tapes(state['R']@(w-piB),ci),CE_minus_current_pi_B=summarize_tapes(state['q']@(w-piB),ci))
        relaxed_mass=B['z'][:,group].sum(1)
        rare['relaxed_mass']=summarize_tapes(relaxed_mass,bi)
        indicator=w.gt(0).double();bmass=piB*(indicator-mass)
        grad=-B['gs']@bmass
        rare['mass_change_under_mean_full_gradient_descent']=summarize_tapes(grad,bi)
        rare['mass_derivative_wrt_logits']=bmass.tolist()
    discord=(B['R']!=C['R']);n=int(discord.sum())
    return dict(expectations=expectations,contrasts=contrasts,low_probability_control=rare,
        binary_discordance=dict(total=n,out_of=64,per_action=discord.sum(0).tolist(),
            C_correct_B_wrong=int(((C['R']==1)&(B['R']==0)).sum()),B_correct_C_wrong=int(((B['R']==1)&(C['R']==0)).sum()),
            small_discordance_flag=n<=4,zero_discordance_flag=n==0),
        semantics='C-minus-B state contrasts fix the root weighting; CE negative favors C; within-B parity positive/CE negative favors the changed root weighting; all are analytic retained-branch mixtures, never executed policies')

def forbid_execution():
    def forbidden(*args,**kwargs):raise RuntimeError('Forbidden forward/backward/RNG execution in record-only analysis')
    torch.autograd.grad=forbidden;torch.autograd.backward=forbidden;torch.Tensor.backward=forbidden
    torch.nn.Module._call_impl=forbidden
    for name in ['rand','rand_like','randn','randn_like','randint','randperm','multinomial']:setattr(torch,name,forbidden)

def main():
    if OUT.exists():raise RuntimeError('Refusing to overwrite/repeat matched-prefix analysis')
    pm=load(PARENT/'manifest.json');previous=load(PARENT/'results.json')
    protected={}
    for directory in [PARENT,TEMPORAL]:
        index=load(directory/'artifact_index.json')
        for name,item in index['files'].items():
            assert sha(directory/name)==item['sha256'],str(directory/name)
            protected[str(directory/name)]=item['sha256']
        protected[str(directory/'artifact_index.json')]=sha(directory/'artifact_index.json')
    for path,digest in pm['source_sha256'].items():assert sha(ROOT/path)==digest;protected[str(ROOT/path)]=digest
    for ck in [pm['backbone'],*pm['checkpoints'].values()]:assert sha(ROOT/ck['path'])==ck['sha256'];protected[str(ROOT/ck['path'])]=ck['sha256']
    protected.update({str(p):sha(p) for p in ROOT.glob('src/**/*') if p.is_file()})
    assert sha(PARENT/'manifest.json')==previous['manifest_sha256']
    diff=hashlib.sha256(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT)).hexdigest()
    assert diff==pm['git']['tracked_diff_sha256']
    specs=[dict(original_index=i,probe=p['probe']) for i,p in enumerate(previous['probes']) if p['probe']['condition'] in ['B','C']]
    assert len(specs)==24
    first_B=next(x for x in specs if x['probe']['condition']=='B');anchor=first_B['original_index'];assert anchor==1
    OUT.mkdir(exist_ok=False)
    current_git=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),tracked_diff_sha256=diff)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol='record-only matched-prefix primary1040/context0-v1',
        script_sha256=sha(__file__),parent_manifest_sha256=sha(PARENT/'manifest.json'),temporal_manifest_sha256=sha(TEMPORAL/'manifest.json'),
        git=current_git,tracked_diff_sha256=diff,sources=protected,six_sequences=pm['six_sequences'],states=specs,
        prefix_definitions=pm['approved_clarifications'],prefix_noise_mapping=pm['prefix_noise'],controller_checkpoints=pm['checkpoints'],backbone=pm['backbone'],
        primary='six equally weighted snapshot1040 B/C pairs, separately stratified by data source; snapshot0 is historical context, not independent replication',
        hard_probabilities='softmax(raw legal ADVANCE logits) without tau; log_softmax retained; no rounding to zero; nominal Gumbel law with original FP32 uniform clamp',
        relaxed='reconstruct original FP32 formula on CPU from retained root logits/masks/ST-bank4801 coordinates, verify selected argmax; no model call; soft CUDA tensors were not directly retained',
        action_tables='all eight legal ADVANCE actions; CF-bank4800 per-tape CE/correctness; retained full ST gradients and reconstructed relaxed weights',
        contrasts=dict(primary_parity='sum_a pi_B(a)*(R_C(a)-R_B(a))',CE_state='sum_a pi_B(a)*(Q_C(a)-Q_B(a)); negative favors C',
            alternative_weights=['pi_C','uniform8'],within_B='sum_a (alternative_weight-pi_B)*R_B or Q_B',
            pi_C='same checkpoint at canonical-prefix state, not canonical teacher actions',analytic_only=True),
        low_mass=dict(threshold=.01,rule='sort increasing pi_B, tie by actionID; largest initial segment with cumulative mass<=.01; chosen without outcomes',
            contrasts='uniform within group for B/C value tables versus current pi_B weighting',
            ST_access='retained draw relaxed mass, no hard-probability starvation inference',
            descent='-dot(pi_B*(group_indicator-group_mass), retained mean full_ST_gradient); independent local-logit perturbation'),
        credit=dict(V_par='dot(pi,R)',b_par='pi*(R-V_par)',Gamma_par='-dot(b_par,mean_full_ST_gradient)',
            CE_alignment='dot(pi*(Q-dot(pi,Q)),mean_full_ST_gradient); positive predicts CE reduction',
            restriction='incoming state/downstream policy fixed; not shared-parameter optimizer prediction; no new loss/estimator'),
        numerical_stability='Center R by its first action before categorical derivative, and value rows before zero-sum weighting differences; mathematically equivalent, avoids floating baseline residues for constant outcomes. Probabilities/log probabilities remain unrounded.',
        focused_validation=dict(passed=7,model_calls=0,backward_calls=0,experimental_noise=0,
            preserved_pre_analysis_failure=dict(test='test_constant_parity_values_have_exactly_zero_credit_and_allocation_effect',
                counts='6 passed,1 failed before final7 passed',cause='synthetic fixture mixed Float32 value tensors with Float64 probabilities',
                correction='fixture dtype made consistently Float64; no retained-outcome calculation had run')),
        uncertainty=dict(seed=4802,replicates=2000,no_new_randomness=True,
            anchor_original_index=anchor,anchor_probe=first_B['probe'],CF_indices='existing bootstrap_indices.pt CF[1]',ST_indices='existing ST[1]',
            coupling='resample full tape coordinates jointly across B/C, actions, sequences and checkpoints; share resample row for fixed-panel summaries; CF and ST banks independent',
            reason='first locked B by manifest order, not outcomes; preserves every retained tape association including gradient/noise; previous per-state intervals retained separately as context',
            halves='first four versus last four original tapes; same-sign halves AND95% interval excluding0',
            interpretation='approximate conditional Monte Carlo uncertainty for fixed inputs/prefixes; no population/seed inference; unadjusted intervals',
            binary_flags='0/8,8/8,identical outcomes,degenerate intervals; small prefix discordance defined as <=4 of64 paired binary entries'),
        training_state_exposure_gate=dict(status='not explicitly defined in supplied protocol or existing records; clarification requested before derived analysis',
            no_substituted_gate=True,report='training-input fixed-pi contrasts and resolved-direction rule; cannot certify an unspecified additional gate'),
        historical_exceptions=dict(bytecode=pm['historical_exception'],temporal_failures='preserved pre-execution toy-fixture failure and offline_verification_failure_1.json; no model failure or rerun'),
        budgets=dict(new_model_forwards=0,new_backwards=0,new_counterfactuals=0,new_noise=0,training_updates=0),
        runtime='CPU array arithmetic only, no_grad; autograd, Module calls and random tensor draws disabled during analysis')
    write(OUT/'manifest.json',manifest)
    print('MANIFEST SEALED '+sha(OUT/'manifest.json')+' before new derived results',flush=True)
    torch.set_grad_enabled(False);forbid_execution();torch.set_num_threads(4)
    noise=torch.load(PARENT/'ST_probe_noise.pt',map_location='cpu',weights_only=True)
    bs=torch.load(PARENT/'bootstrap_indices.pt',map_location='cpu',weights_only=True)
    assert noise['seed']==4801 and bs['seed']==4802
    ci,bi=bs['CF'][anchor],bs['ST'][anchor]
    selected_ids={x['probe']['probe_id'] for x in specs};roots={};cf_runs={}
    with gzip.open(PARENT/'runs.jsonl.gz','rt') as h:
        for line in h:
            row=json.loads(line)
            if row['probe']['probe_id'] in selected_ids and row['bank']=='CF':cf_runs[row['run_id']]=row
    with gzip.open(PARENT/'decisions.jsonl.gz','rt') as h:
        for line in h:
            row=json.loads(line)
            if row['phase']=='ST' and row['decision']==0 and row['sequence_id'].endswith('/actionNone'):
                pid=row['sequence_id'].rsplit('/ST/',1)[0]
                if pid in selected_ids and row['token']==next(x['probe']['token_index'] for x in specs if x['probe']['probe_id']==pid):roots[row['sequence_id']]=row
    if len(cf_runs)!=1536 or len(roots)!=192:raise RuntimeError(f'Required raw records incomplete: CF={len(cf_runs)}/1536, STroots={len(roots)}/192')
    states=[];arrays={}
    for x in specs:
        previous_state=previous['probes'][x['original_index']]
        assert previous_state['probe']['token_index']==next(s['length']-1 for s in pm['six_sequences'] if s['sequence_id']==x['probe']['sequence_id'])
        out,data=state_table(previous_state,noise,roots,cf_runs,bi,ci)
        states.append(out);arrays[x['probe']['probe_id']]=data
    pairs=[]
    for cp in [0,1040]:
        for s in pm['six_sequences']:
            B=arrays[f'update{cp:04d}/{s["sequence_id"]}/B'];C=arrays[f'update{cp:04d}/{s["sequence_id"]}/C']
            paired=pair_contrasts(B,C,bi,ci)
            pairs.append(dict(checkpoint=cp,sequence_id=s['sequence_id'],source=s['source'],label=s['label'],
                token_index=s['length']-1,human_token=s['length'],**paired))
    summaries={}
    for cp in [0,1040]:
        for source in ['all','training','length16_eval','length32_ood']:
            panel=[p for p in pairs if p['checkpoint']==cp and (source=='all' or p['source']==source)]
            fields=panel[0]['contrasts']
            # Fixed-sequence arithmetic means; no resampling sequence identities.
            summary={key:summarize_tapes(vec([p['contrasts'][key]['tape_values'] for p in panel]).mean(0),ci) for key in fields}
            summaries[f'{cp}/{source}']=dict(pair_count=len(panel),equal_pair_weight=1/len(panel),contrasts=summary)
    output=dict(manifest_sha256=sha(OUT/'manifest.json'),counts=dict(states=24,pairs=12,retained_CF_runs=1536,retained_ST_roots=192),
        states=states,pairs=pairs,summaries=summaries,zero_model_execution=True,zero_backward=True,zero_new_noise=True)
    finite(output)
    for path,digest in protected.items():assert sha(path)==digest,path
    assert sha(__file__)==manifest['script_sha256']
    assert hashlib.sha256(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT)).hexdigest()==diff
    assert subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True).returncode==0
    write(OUT/'results.json',output)
    print('ANALYSIS COMPLETE; 12 pairs/24 states; zero model calls/backwards/noise',flush=True)

if __name__=='__main__':
    existed=OUT.exists()
    try:main()
    except Exception as error:
        if not existed:
            OUT.mkdir(exist_ok=True)
            if not (OUT/'failure.json').exists():write(OUT/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),new_model_calls=0,new_backward_calls=0,new_noise=0,stop='No rerun of models'))
        raise
