"""Offline verification/reporting of retained temporal VJPs. No model calls."""
import sys
sys.dont_write_bytecode=True
import gzip,json,hashlib,math,statistics,subprocess
from collections import Counter,defaultdict
from pathlib import Path
import torch
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
PARENT=ROOT/'experiments/results/gatec_st_action_value_4700_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def records(name):
    with gzip.open(OUT/(name+'.jsonl.gz'),'rt') as h:
        for line in h:yield json.loads(line)
def write(name,data):
    with (OUT/name).open('x') as h:json.dump(data,h,indent=2,allow_nan=False);h.write('\n')
def finite(x):
    if isinstance(x,float):assert math.isfinite(x)
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,list):
        for v in x:finite(v)
def variance(x):return ((x-x.mean(-2,keepdim=True))**2).sum(-1).sum(-1)/(x.shape[-2]-1)
def CI(x):return torch.quantile(x,torch.tensor([.025,.975],dtype=torch.float64)).tolist()
def vec(value):return torch.tensor(value,dtype=torch.float64)

def main():
    assert not (OUT/'failure.json').exists()
    m=load(OUT/'manifest.json');r=load(OUT/'results.json');old=load(PARENT/'results.json')
    assert sha(OUT/'manifest.json')==r['manifest_sha256']
    assert r['counts']==dict(states=18,forward_replays=144,VJP_evaluations=384,new_CF_continuations=0)
    for name,digest in m['protected_files_sha256'].items():assert sha(name)==digest,name
    assert sha(ROOT/'experiments/diagnose_gatec_temporal_vjp_4700.py')==m['script_sha256']
    for item in [m['controller'],m['backbone']]:assert sha(ROOT/item['path'])==item['sha256']
    finite(r)
    bs=torch.load(PARENT/'bootstrap_indices.pt',map_location='cpu',weights_only=True)
    runs={x['run_id']:x for x in records('runs')};assert len(runs)==144
    calls={};task_vjps=0;all_probes=[];adjoint_counts=Counter();logit_status=Counter()
    for p in r['probes']:
        prior=old['probes'][p['original_index']];assert prior['probe']==p['probe']
        assert p['probe']['checkpoint_update']==1040
        assert p['CF_final_ce']==prior['CF_final_ce'] and p['CF_correct']==prior['CF_correct'] and p['g_CF']==prior['g_CF']
        assert len(p['replays'])==8
        for replay in p['replays']:
            finite(replay);assert replay['forward_matches_previous']
            assert replay['P']==[8]*len(replay['P'])
            assert replay['commits']==len(replay['P'])
            task_vjps+=replay['VJP_evaluations']
            calls[replay['run_id']]=replay['controller_calls']
            cumul=replay['cumulative_gradients'];parts=replay['components']
            assert parts is not None
            torch.testing.assert_close(sum(vec(v) for v in parts.values()),vec(cumul['full']),atol=1e-6,rtol=1e-5)
            original=prior['ST_runs'][replay['tape']]['probe_logit_gradient']
            torch.testing.assert_close(vec(cumul['full']),vec(original),atol=1e-6,rtol=1e-5)
            for value in cumul.values():assert value[8:]==[0,0,0]
            if p['probe']['condition'] in ['B','C']:
                assert cumul['token']==cumul['next']==cumul['full'] and replay['VJP_evaluations']==2
                assert all(v==0 for k in ['next','distant'] for v in parts[k])
            else:assert replay['VJP_evaluations']==4
        components=torch.tensor([[x['components'][k][:8] for k in ['exec','within','next','distant']] for x in p['replays']],dtype=torch.float64)
        means=components.mean(0);centered=components-means
        covariance=torch.einsum('tca,tda->cd',centered,centered)/7
        st_indices=bs['ST'][p['original_index']];cf_indices=bs['CF'][p['original_index']]
        boot=components[st_indices];bc=boot-boot.mean(1,keepdim=True)
        covboot=torch.einsum('btca,btda->bcd',bc,bc)/7
        # Use the original cumulative vectors, as the primary calculation did.
        # Summing serialized FP32 differences in float64 changes their rounding.
        full=vec([x['cumulative_gradients']['full'][:8] for x in p['replays']])
        exe=vec([x['cumulative_gradients']['exec'][:8] for x in p['replays']]);rest=full-exe
        vdiff=variance(full[st_indices])-variance(exe[st_indices])
        torch.testing.assert_close(vec(CI(vdiff)),vec(p['variance']['paired_difference_bootstrap_95']),atol=1e-12,rtol=1e-10)
        gcf=vec(p['g_CF']);pi=vec(p['pi']);q=vec(p['CF_final_ce'])
        qb=q[cf_indices].mean(1);gb=pi*(qb-(qb*pi).sum(-1,keepdim=True))
        rest_projection=(rest[st_indices].mean(1)*gb).sum(-1)
        torch.testing.assert_close(vec(CI(rest_projection)),vec(p['cancellation']['rest_projection_bootstrap_95']),atol=1e-12,rtol=1e-10)
        snr={}
        for mode in ['exec','token','next','full']:
            gs=vec([x['cumulative_gradients'][mode][:8] for x in p['replays']]);projected=gs@gcf
            noise=float(projected.std());signal=float(projected.mean())
            snr[mode]=dict(projection_mean=signal,projection_SD=noise,mean_over_SD=signal/noise if noise>0 else None)
        all_probes.append(dict(probe=p['probe'],classification=p['classification'],components_order=['exec','within','next','distant'],
            trace_covariance_matrix=covariance.tolist(),trace_covariance_bootstrap_95={f'{i}/{j}':CI(covboot[:,i,j]) for i in range(4) for j in range(4)},
            component_covariance_sum_minus_original_full_variance=float(covariance.sum()-variance(full)),
            projection_signal_and_noise=snr,variance_ratio_full_to_execution=p['variance']['full']/p['variance']['execution'] if p['variance']['execution']>0 else None,
            conditional_correctness=p['conditional_correctness'],parity_correctness_varies=p['eligibility']['parity_correctness_varies']))
    assert task_vjps==384
    observed=Counter();decision_lookup={}
    for d in records('decisions'):
        key=(d['sequence_id'],d['token'],d['decision']);decision_lookup[key]=d
        observed[d['sequence_id']]+=1
        assert d['execution_after']==d['execution_before']+int(d['selected_action']!=9)
        assert d['actual_executed_block']==(d['selected_action'] if d['selected_action']<8 else d['pointer_before'] if d['selected_action']==8 else None)
        assert d['candidate_block_calls']==list(range(8))+[d['pointer_before']]
    assert observed==Counter(calls)
    for a in records('adjoints'):
        finite(a);d=decision_lookup[(a['run_id'],a['token'],a['decision'])]
        assert a['execution_before']==d['execution_before'] and a['execution_after']==d['execution_after']
        assert a['decision_count_before']==a['decision'] and a['decision_count_after']==a['decision']+1
        g=a['original_logit_gradient'];logit_status[a['gradient_status']]+=1
        if g is not None:
            assert len(g)==11 and g[10]==0
            assert all(value==0 for value,legal in zip(g[:10],d['legal_successor_mask']) if not legal)
        for key,item in a['execution_state_adjoints'].items():adjoint_counts[f'{key}/{item["status"]}']+=1
    assert sum(logit_status.values())==sum(calls.values())
    strata=defaultdict(Counter)
    for p in r['probes']:strata[f'{p["probe"]["source"]}/{p["probe"]["condition"]}'][p['classification']]+=1
    useful=sum(p['eligibility']['resolved_aligned_execution_CE_credit'] for p in r['probes'])
    supported=r['classifications'].get('supports_temporal_interference_CE_at_state',0)
    cancellation=sum(p['cancellation']['resolved_opposing_remaining_projection'] for p in r['probes'])
    variance_added=sum(p['variance']['resolved_added_vector_variance'] for p in r['probes'])
    projection_variance_added=sum(p['variance']['resolved_added_projection_variance'] for p in r['probes'])
    reduction=sum(max(p['variance']['paired_difference_halves'])<0 and p['variance']['paired_difference_bootstrap_95'][1]<0 for p in r['probes'])
    B_C=[]
    for s in m['six_sequences']:
        b=next(p for p in r['probes'] if p['probe']['sequence_id']==s['sequence_id'] and p['probe']['condition']=='B')
        c=next(p for p in r['probes'] if p['probe']['sequence_id']==s['sequence_id'] and p['probe']['condition']=='C')
        B_C.append(dict(sequence_id=s['sequence_id'],source=s['source'],remaining_tokens=0,
            B_state=b['incoming_state_sha256'],C_state=c['incoming_state_sha256'],
            B_classification=b['classification'],C_classification=c['classification'],
            B_exec_direction=b['cumulative_alignment']['exec']['direction'],C_exec_direction=c['cumulative_alignment']['exec']['direction'],
            B_full_direction=b['cumulative_alignment']['full']['direction'],C_full_direction=c['cumulative_alignment']['full']['direction']))
    verification=dict(counts=r['counts'],original_gradient_and_forward_verified_all_144=True,other_controller_adjoints=sum(calls.values()),
        full_controller_gradient_statuses=dict(logit_status),execution_state_adjoint_statuses=dict(adjoint_counts),
        no_new_model_calls_or_VJPs_in_offline_analysis=True,parameter_source_and_historical_hashes_verified=True,
        component_sum_and_variance_covariance_verified=True,joint_ST_and_independent_CF_bootstrap_recomputed=True,
        no_new_noise=True,optimizer_steps=0,production_changes=0,
        algebraic_categorical_identity_autograd_checks=216,
        budget_scope='384 VJPs of the original AGT final task loss; existing comparison helper also performs small algebraic categorical expected-loss identity checks, never additional AGT forward/backward probes')
    conclusion=('The hypothesis has local CE-based support at the explicitly identified states, with unresolved cases elsewhere.' if supported else
        'The full unreliability hypothesis is unresolved at the tested states and is not supported by a resolved execution-credit-to-full-credit failure. Opposing components, if resolved, are separate from loss of resolved full alignment.')
    interpretation=dict(conclusion=conclusion,classifications=r['classifications'],strata={k:dict(v) for k,v in strata.items()},
        resolved_aligned_execution_CE_states=useful,resolved_remaining_opposition_states=cancellation,
        resolved_added_vector_variance_states=variance_added,resolved_added_projection_variance_states=projection_variance_added,
        resolved_reduced_vector_variance_states=reduction,strong_support_states=supported,
        matched_horizon_prefix_comparisons=B_C,probes=all_probes,verification=verification)
    finite(interpretation);write('interpretation.json',interpretation);write('verification.json',verification)
    # Human report with all states, not an outcome-selected subset.
    fmt=lambda x:'None' if x is None else f'{x:.6g}'
    text=['# Seed4700 snapshot1040: temporal controller-output VJP diagnostic','',conclusion,'',
        f'Completed **144 identical forward replays and 384 original-task-loss VJPs** at all 18 predeclared states. No new counterfactual continuations or noise were generated. Resolved aligned execution-only CE credit: {useful}/18. Resolved opposing remaining projection: {cancellation}/18. Resolved added vector variance: {variance_added}/18; resolved added projection variance: {projection_variance_added}/18. Resolved reduced vector variance: {reduction}/18. Strong state-level support under the sealed criterion: {supported}/18.',
        '',
        '## Identity, controls and graph definition','',
        f"Manifest SHA-256 `{r['manifest_sha256']}`; HEAD `{m['git']['head']}`; tracked-diff SHA-256 `{m['git']['tracked_diff_sha256']}`. Snapshot1040 `{m['controller']['path']}`, SHA-256 `{m['controller']['sha256']}`. Frozen backbone `{m['backbone']['path']}`, SHA-256 `{m['backbone']['sha256']}`.",
        '',
        'All completed action-value artifact hashes, selected weights, input states and historical parent identities were checked. The historical bytecode-only exception remains disclosed and preserved. Bytecode writing was disabled. No optimizer, training, parameter/buffer update, checkpoint overwrite, production edit or commit occurred.',
        '',
        'A starts at human ordinal1/internal0, with zero prior commits. B/C start at human ordinal L/internal L−1. Incoming states were restored from the full retained states, including memory, all caches, GDN recurrence/convolution buffers, versions and commit counters. B preserves its sealed checkpoint-owned sampled prefix; C its canonical prefix. No prefix was rerun, and C forcing does not extend into the suffix. All eight ST tapes from seed4801 and all retained CF outcomes from seed4800 were reused. Snapshot0 supplied recorded context only; it was never replayed.',
        '',
        'Every forward uses the existing hard-forward exhaustive-candidate ST path, tau2, FP32/batch1, zero budget/presence, P8, original masks, frozen in-flight state and one terminal commit/token. Incoming state is fixed conditioning; the historical full-sequence BPTT policy is preserved within the suffix. Model/runtime flags and exact state/noise sources are in manifest.json.',
        '',
        'A tensor hook on each original unified controller-logit output gates only its VJP. It never changes a forward value or an action. The probe output is always allowed. Execution-only excludes every later controller-output path. Token includes later calls in the probe token. Next additionally includes calls in the immediately following token. Full suppresses no output VJP. Direct execution, candidate/state, memory and commit derivatives remain available in every mode. Existing ST relaxed.detach and hard action/pointer/count/mask boundaries are unchanged; none was reconnected.',
        '',
        'Each A graph is replayed once with four task-loss VJPs. Each B/C graph is replayed once with execution-only and full VJPs: its suffix has no later token, so token=next=full was verified structurally and reused without redundant derivatives. Full VJPs also retain original gradients at every controller call and available live-state/anchor/read/slot-mean adjoints. These nonprobe calls receive no new action values or action-value alignment labels.',
        '',
        'Execution-state adjoints refer to the actual graph tensors. Shared anchors, memory reads and slot means are deduplicated VJP targets; their repeated call records contain the total adjoint of the shared tensor, not separate per-use gradient attributions.',
        '',
        'Components are c_exec=g_exec, c_within=g_token−g_exec, c_next=g_next−g_token and c_distant=g_full−g_next. This is an exact first-order telescoping path partition within numerical tolerance, not an additive decomposition of task losses or historical optimization effects. Logit-hook boundary indices are recorded per tape.',
        '',
        '## Previous classifications','',
        '| Checkpoint | Source | Probe | Previous classifications (two parity inputs) |','|---|---|---|---|']
    for key,values in sorted(m['previous_classifications'].items()):
        cp,source,condition=key.split('/');text.append(f'| {cp} | {source} | {condition} | {values} |')
    text.extend(['','Full prior per-input classifications are retained in the manifest; aggregate alignment counts were not used to select states.',
        '',
        '## All 18 states','',
        'Directions below use the original four-tape-half and 95% interval rule. Variance ratio is descriptive; the paired variance-difference interval is the inferential quantity. Conditional correctness range uses the retained hard action continuations.',
        '',
        '| Sequence / probe | Exec / token / next / full | Remaining CE projection [95% CI] | Full/exec variance | Variance difference [95% CI] | Correctness range | Classification |',
        '|---|---|---|---:|---|---|---|'])
    for p in r['probes']:
        d=p['cumulative_alignment'];c=p['cancellation'];v=p['variance'];e=p['conditional_correctness']
        ratio=v['full']/v['execution'] if v['execution']>0 else None
        text.append(f"| {p['probe']['sequence_id']} / {p['probe']['condition']} | {' / '.join(d[k]['direction'] for k in ['exec','token','next','full'])} | {fmt(c['rest_projection'])} {c['rest_projection_bootstrap_95']} | {fmt(ratio)} | {fmt(v['paired_full_minus_execution'])} {v['paired_difference_bootstrap_95']} | {fmt(min(e))}…{fmt(max(e))} | {p['classification']} |")
    text.extend(['','## Component projections','',
        '| Sequence / probe | Exec | Within token | Next token | Distant tokens | Full |','|---|---:|---:|---:|---:|---:|'])
    for p in r['probes']:
        cs=p['components'];text.append(f"| {p['probe']['sequence_id']} / {p['probe']['condition']} | "+' | '.join(fmt(cs[k]['projection']) for k in ['exec','within','next','distant'])+f" | {fmt(p['cumulative_alignment']['full']['inner_product'])} |")
    text.extend(['','## Stratification and matched horizons','',
        '| Source / probe | State classifications |','|---|---|'])
    for key,values in sorted(strata.items()):text.append(f'| {key} | {dict(values)} |')
    text.extend(['','| Sequence | B exec/full | C exec/full | B class | C class |','|---|---|---|---|---|'])
    for pair in B_C:text.append(f"| {pair['sequence_id']} | {pair['B_exec_direction']}/{pair['B_full_direction']} | {pair['C_exec_direction']}/{pair['C_full_direction']} | {pair['B_classification']} | {pair['C_classification']} |")
    text.extend(['',
        'B versus C holds the remaining one-token horizon fixed while changing prefix-conditioned incoming state. These are state-conditioned comparisons, not random-assignment causal prefix interventions. A versus B additionally changes horizon, token input and incoming state; no causal horizon claim is inferred from that comparison.',
        '',
        '## Uncertainty, cancellation and variance','',
        'The original 2,000 bootstrap index rows (seed4802, original36-state indexing) were reused. All components of an ST tape are resampled together. CF action outcomes retain their pairing, and the CF and ST banks are independently resampled. Projections use the retained g_CF, with its CF uncertainty propagated during bootstrap. Positive alignment concerns descent of expected final CE with incoming state/downstream policy fixed.',
        '',
        'A resolved cancellation claim requires a negative remaining projection in both halves and a95% upper bound below zero. A resolved added-variance claim requires positive paired differences in both halves and a95% lower bound above zero. Norm shrinkage alone is not cancellation evidence. Variance/covariance is sample trace covariance, ddof1: Var(full)=Var(exec)+Var(rest)+2Cov(exec,rest). Both vector and g_CF-projection decompositions, covariance matrices and intervals are retained.',
        '',
        'Strong hypothesis support requires informative values and resolved aligned execution-only CE credit, followed by resolved cancellation or added variance and loss of resolved full alignment. An adverse component with still-resolved full alignment is reported separately. Weak/unresolved execution-only credit is not attributed to later-controller cancellation. All intervals are conditional Monte Carlo intervals from eight tapes per bank, unadjusted for multiplicity; they are not training-seed/population estimates.',
        '',
        '## Parity and detach evidence','',
        f"Conditional correctness differs across first-action branches at {sum(p['eligibility']['parity_correctness_varies'] for p in r['probes'])}/18 states. The other {sum(not p['eligibility']['parity_correctness_varies'] for p in r['probes'])}/18 show only CE/calibration differences in this bank. No policy update was tested, and CE-gradient alignment is not labeled useful parity learning.",
        '',
        f'Original full routing-gradient statuses across all calls: `{dict(logit_status)}`. Available execution-state adjoint statuses: `{dict(adjoint_counts)}`. Every record includes actual token ID, within-token decision index/count, execution count and remaining-token/execution horizon. Fixed incoming tensors are labeled separately from disconnected gradients, connected exact zeros and connected small derivatives.',
        '',
        'Existing detach boundaries are the ST relaxed stop-gradient and hard routing metadata. The incoming state is deliberately conditioned as fixed. No extra suffix detach was added. Magnitude versus position alone is not treated as temporal causation; causal path exclusions occur only at the explicitly recorded controller-output VJP boundaries.',
        '',
        '## Validation and limits','',
        'All144 forward trajectories reproduce retained incoming identities, sealed noise/actions, selected blocks, logits/CE and exact per-token incoming/outgoing commit-state hashes. Full probe gradients reproduce their retained values at atol1e−6/rtol1e−5. Every component sum reconstructs full; B/C next/distant components are exact zeros. Accounting, legality, candidate/GDN/anchor purity, cache versions and one seven-entry commit/token passed. Parameters, production source and historical files remained hash-identical.',
        '',
        'Focused tests:32 passed. A pre-execution synthetic zero-loss fixture failure was corrected and is preserved in manifest.json; no AGT replay occurred during that failed test. Model derivative budget:144 forwards/384 task-loss VJPs. The reused comparator additionally performed216 small categorical-formula autograd identity checks; these never execute AGT or measure another trajectory. Offline analysis performs no model call or VJP. No full regression or training was launched.',
        '',
        'An initial offline bootstrap check used float64 sums of serialized FP32 component differences instead of the original cumulative vectors. Its remaining-projection interval differed by2.45e−11. The offline input selection was corrected without changing gradients, tolerances or measurements. The failure is preserved in offline_verification_failure_1.json. Component covariance matrices retain the FP32 subtraction rounding; their small reconstruction residuals are reported explicitly.',
        '',conclusion,'',
        'This isolates first-order derivative paths at18 selected states of one checkpoint and one controller-training seed. It does not establish the historical cause of training failure, a replacement estimator, task-loss route discovery, general trainability or parity improvement. No isolated component is used for training or recommended as a replacement. Sampling was not expanded.',
        '',
        '## Artifacts','',
        '- manifest.json: sealed graph boundaries, identities, prior classifications and budget.',
        '- results.json and probe18.json…probe35.json: complete per-tape cumulative/component vectors, alignments, conditional outcomes and uncertainty.',
        '- runs/decisions/tokens.jsonl.gz: replay outcomes, logits/actions/masks/noise coordinates, state summaries and commit identities.',
        '- adjoints.jsonl.gz: original gradients at every controller call and available execution-state adjoints.',
        '- interpretation.json: covariance matrices/intervals, signal-to-noise summaries, all strata and matched B/C comparisons.',
        '- verification.json: independent offline checks; no model execution.',
    ])
    with (OUT/'report.md').open('x') as h:h.write('\n'.join(text)+'\n')
    assert hashlib.sha256(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT)).hexdigest()==m['git']['tracked_diff_sha256']
    assert subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True).returncode==0
    write('artifact_index.json',dict(files={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(conclusion=conclusion,classes=r['classifications'],useful_exec_CE=useful,cancellation=cancellation,
        added_variance=variance_added,reduced_variance=reduction,verification=verification),indent=2))

if __name__=='__main__':main()
