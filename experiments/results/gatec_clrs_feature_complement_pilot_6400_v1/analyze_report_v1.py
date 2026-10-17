"""Locked primary analysis and closure from retained prescribed results only."""
import os,json,csv,time
import torch
from support_v1 import OUT,ROOT,ENG,save,sha,integrity
torch.set_num_threads(1);torch.set_grad_enabled(False)

def main():
    assert not (OUT/'final_results_v1.json').exists(),'No analysis restart/overwrite'
    run=json.loads((OUT/'run_result_v1.json').read_text());complete=run['status']=='ALL_TRAINING_AND_PRESCRIBED_EVALUATIONS_COMPLETED'
    seal=json.loads((OUT/'implementation_seal_v1.json').read_text())
    for n,h in seal['files'].items():assert sha(ROOT/n)==h,n
    before=integrity();assert before['success']
    classification='INCOMPLETE_ENGINEERING_OR_RESOURCE';metrics={};gates={};contrasts={};geometry={}
    if run['status']=='TREATMENT_NUMERICAL_OUTCOME_PAIRED_EVALUATION_INCOMPLETE':classification=run['status']
    if complete:
        started=json.loads((OUT/'execution_started_v1.json').read_text());assert time.monotonic()<started['deadline_monotonic'],'Analysis exceeded shared six-hour ceiling'
        metrics=json.loads((OUT/'evaluation_summaries_v1.json').read_text())
        dev={arm:[json.loads(s) for s in (OUT/(arm+'_development_records_v1.jsonl')).read_text().splitlines()] for arm in ['C','T']}
        assert len(dev['C'])==len(dev['T'])==32 and [r['id'] for r in dev['C']]==[r['id'] for r in dev['T']]
        assert [r['original_input_sha256'] for r in dev['C']]==[r['original_input_sha256'] for r in dev['T']]
        bankpath=OUT/'bootstrap_bank_v1.bin';raw=bytearray(bankpath.read_bytes());assert len(raw)==20000*32
        bank=torch.frombuffer(raw,dtype=torch.uint8).reshape(20000,32).long();assert int(bank.min())>=0 and int(bank.max())<32
        differences={
            'T_minus_C':[t['conditions']['native']['predecessor_accuracy']-c['conditions']['native']['predecessor_accuracy'] for t,c in zip(dev['T'],dev['C'])],
            'T_minus_own_query_independent':[t['conditions']['native']['predecessor_accuracy']-t['conditions']['query_independent']['predecessor_accuracy'] for t in dev['T']]}
        for name,values in differences.items():
            x=torch.tensor(values,dtype=torch.float64);boot=x[bank].mean(1);lower=float(torch.quantile(boot,.025,interpolation='linear'))
            contrasts[name]=dict(point_estimate=float(x.mean()),adjusted_one_sided_lower_bound=lower,resolved_positive=lower>0,per_array_differences=values)
        save('paired_bootstrap_v1.json',dict(contrasts=contrasts,replicates=20000,seed=6400,unit='complete size16 development array',quantile=.025,interpolation='linear',family='two contrasts, Bonferroni-adjusted',bank_sha256=sha(bankpath),conditional_on_one_paired_seed=True,previously_examined_panels=True))
        for arm in ['C','T']:
            gates[arm]={}
            for split,pointer,exact in [('development',.95,.80),('size_transfer',.90,.50)]:
                v=metrics[split][arm]['native'];gates[arm][split]=dict(predecessor_accuracy=v['predecessor_accuracy'],predecessor_threshold=pointer,predecessor_pass=v['predecessor_accuracy']>=pointer,exact_array_accuracy=v['exact_array_accuracy'],exact_threshold=exact,exact_pass=v['exact_array_accuracy']>=exact)
            gates[arm]['all_competence_pass']=all(gates[arm][s]['predecessor_pass'] and gates[arm][s]['exact_pass'] for s in ['development','size_transfer'])
        T=gates['T']['all_competence_pass'];C=gates['C']['all_competence_pass'];both=all(v['resolved_positive'] for v in contrasts.values())
        if T and both and not C:classification='CREDIBLE_RESEARCH_CANDIDATE_EXPLORATORY'
        elif T and C:classification='BOTH_COMPETENT_NO_REQUIRED_INTERVENTION_ATTRIBUTION'
        elif not T and contrasts['T_minus_C']['resolved_positive']:classification='PARTIAL_EFFECT_TASK_NON_SUCCESS'
        elif T:classification='COMPETENT_TREATMENT_WITHOUT_REQUIRED_PAIRED_EVIDENCE'
        else:classification='COMPLETED_NON_SUCCESS_INTERVENTION_INSUFFICIENT'
        for arm in ['C','T']:
            geometry[arm]={}
            for split in ['training_subset','development','size_transfer']:
                rows=[json.loads(s) for s in (OUT/(arm+'_'+split+'_mechanistic_v1.jsonl')).read_text().splitlines()]
                geometry[arm][split]={}
                for role in ['candidates','queries']:
                    g={}
                    for i in range(8):
                        v=[r['roles'][role]['mixers'][str(i)] for r in rows]
                        g[str(i)]=dict(mean_input_centered_energy=sum(x['input']['centered_energy'] for x in v)/len(v),mean_output_centered_energy=sum(x['output']['centered_energy'] for x in v)/len(v),mean_input_pair_difference_squared=sum(x['mean_input_difference_squared'] for x in v)/len(v),mean_output_pair_difference_squared=sum(x['mean_transported_difference_squared'] for x in v)/len(v))
                    norms=[r['roles'][role]['terminal_normalization'] for r in rows]
                    geometry[arm][split][role]=dict(mixers=g,terminal=dict(mean_input_centered_energy=sum(x['input']['centered_energy'] for x in norms)/len(norms),mean_output_centered_energy=sum(x['output']['centered_energy'] for x in norms)/len(norms),mean_input_norm=sum(x['input']['mean_norm'] for x in norms)/len(norms),mean_output_norm=sum(x['output']['mean_norm'] for x in norms)/len(norms),mean_input_angular_dispersion=sum(x['input']['mean_angular_dispersion'] for x in norms)/len(norms),mean_output_angular_dispersion=sum(x['output']['mean_angular_dispersion'] for x in norms)/len(norms)))
        save('mechanistic_summary_v1.json',geometry);save('competence_gates_v1.json',gates)
        with (OUT/'metric_table_v1.csv').open('x',newline='') as f:
            fields=['arm','panel','condition','predecessor_accuracy','exact_array_accuracy','CE','invalid_chain','mean_target_rank','mean_target_margin','distinct_predicted_candidates'];w=csv.DictWriter(f,fields);w.writeheader()
            for panel,arms in metrics.items():
                for arm,conditions in arms.items():
                    for condition,m in conditions.items():w.writerow(dict(arm=arm,panel=panel,condition=condition,**{k:m[k] for k in fields[3:]}))
        if time.monotonic()>=started['deadline_monotonic']:classification='INCOMPLETE_ENGINEERING_OR_RESOURCE';complete=False
    total_keys=['attempted_tokens','completed_tokens','failed_tokens','selected_bank_attempted','selected_bank_completed','commits','contextual_commit_entries','backward_attempted','backward_completed','backward_failed','optimizer_attempted','optimizer_updates','optimizer_failed']
    total={k:sum(v[k] for v in run['resource'].values()) for k in total_keys}
    if complete:assert total['attempted_tokens']==total['completed_tokens']==total['commits']==82432 and total['selected_bank_attempted']==total['selected_bank_completed']==659456 and total['optimizer_updates']==total['backward_completed']==2000 and not total['failed_tokens']
    final_integrity=integrity();assert final_integrity['success'];save('integrity_final_v1.json',final_integrity)
    result=dict(status='COMPLETED_PAIRED_EXPLORATORY_PILOT' if complete else 'INCOMPLETE',classification=classification,gates=gates,contrasts=contrasts,resource_totals=total,per_arm_resource=run['resource'],run_execution_seconds=run['elapsed_seconds_to_execution_boundary'],analysis_finished_seconds_since_clock=time.monotonic()-json.loads((OUT/'execution_started_v1.json').read_text())['start_monotonic'],peak_cuda_allocated_bytes=run['peak_cuda_allocated_bytes'],peak_cuda_reserved_bytes=run['peak_cuda_reserved_bytes'],integrity=final_integrity,historical_statuses_unchanged=True,learned_adaptive_routing_established=False)
    save('final_results_v1.json',result)
    branch=json.loads((OUT/'branch_creation_v1.json').read_text())
    protocol=ROOT/'AGT_v3_feature_complement_causal_pilot_locked_protocol_6400_v1.md'
    report=['# AGT-v3 feature-complement causal pilot','',f"**{classification}**. Status: {result['status']}. This is one paired exploratory intervention under the locked insertion-sort task/recipe, not learned routing or broad AGT superiority.",'',
        '## Protocol, preservation and exact intervention','',f"Protocol `{protocol.relative_to(ROOT)}` SHA256 `{sha(protocol)}`. Reviewed source commit `1ed1d773264b3f5f53bd1ba169e819709d33cbb2`; local source HEAD `{branch['HEAD']}` with the preserved historical diff. All41 checked local source files match the reviewed remote content. Branch `{branch['branch']}`. Production and historical scientific artifacts are unchanged; source-level diffs and seals document research-only code.",'',
        'Control uses `(X @ P.T)*g`. Treatment uses exactly `(X @ (P+Q).T)*g`, with `Q=I64-ones(64,64)/64`, at six outer bank mixers0–5 and FFN-core mixers6/7. No prelude-mixer change, coefficient, renormalization, moved gain, parameter-capacity increase or residual/commit/normalization change. Active-site calls and the unexecuted prelude mixer are checked on every token.','',
        'Operator-only checks passed6 CPU cases and3 genuinely executed CUDA cases, with no full AGT construction/calls/backwards. Starting equality and independent ownership of the two actual arms are recorded in `starting_equivalence_v1.json` when that gate ran. Exact original initial parameter/buffer identity: `c28e83425e505734e1b6a39f4c6d5ecf414f1b1fb28a77e01ddc31bbfa92280e`. No failed-final-checkpoint initialization.','',
        '## Locked training, data and execution','',
        'Original sealed1000 size16 training arrays, order seed6201, fresh initialization seed6200. Original native mean pointer CE, AdamW LR1e-3 WD0 betas(.9,.999) eps1e-8, clip global norm1, FP32 TF32off, full BPTT, original eval-mode flags, frozen controller/vocabulary head. C then T on each identical row; CPU/CUDA RNG streams matched without intervention draws. Exactly1000 prescribed updates/arm at completion; no initial diagnostic, warm-up, hints, selector, whitening, extra loss/readout fit, checkpoint selection or retry.','',
        'Training pool/target/source/order hashes and original configuration seals are in `preflight_source_configuration_v1.json`. Development/size64 pools are reused and historically examined, not untouched confirmation. Training subset is the original first128 rows; development32 size16 and transfer32 size64 rows retain sealed order and targets. Native final heads are evaluated once per prescribed row. No task error filters a case.','',
        '## Task results','']
    if complete:
        report+=['|Panel|Arm/condition|Predecessor accuracy|Exact-array accuracy|CE|Invalid chains|','|---|---|---:|---:|---:|---:|']
        for panel,arms in metrics.items():
            for arm,conditions in arms.items():
                for name,v in conditions.items():report.append(f"|{panel}|{arm}/{name}|{100*v['predecessor_accuracy']:.9g}%|{100*v['exact_array_accuracy']:.9g}%|{'unavailable' if v['CE'] is None else format(v['CE'],'.10g')}|{100*v['invalid_chain']:.9g}%|")
        report+=['','Uniform probability metrics use established deterministic lowest-index ties, with analytic independent-random expectations separately recorded. Competition rank1 on tied scores is not discrimination. Self-pointer and original-order hard controls have no invented CE, score margin or target rank. Full scores, targets, predictions, ranks, ties and target margins are retained per array. Invalid chains are task errors, not executor failures.','',
            '## Prospective gates and paired uncertainty','',json.dumps(gates,indent=2),'','|Size16 development contrast|Point estimate|Adjusted one-sided lower bound|Resolved positive|','|---|---:|---:|---|']
        for name,c in contrasts.items():report.append(f"|{name}|{100*c['point_estimate']:.9g} pp|{100*c['adjusted_one_sided_lower_bound']:.9g} pp|{c['resolved_positive']}|")
        report+=['','Exactly20000 paired complete-array bootstrap replicates, seed6400, shared bank, linear empirical quantile.025 for each of two contrasts. The bank was sealed before training. All queries within an array stay together. Intervals condition on one paired initialization/recipe and previously examined panels; they do not establish seed reproducibility or untouched population confirmation. Strongest positive classification additionally requires treatment to pass all competence gates while control fails overall competence.','',
            '## Secondary mechanistic measurements','',
            'Each prescribed evaluation captures every active mixer input/output, whole-block input/output, pre-terminal-normalization and committed vectors, native query/key projections and scores. Source labels distinguish core outputs from later FFN transforms. No mixer inversion or extra forward is used. Per-array candidate/query roles remain separate. Recorded centered energy and common components, actual corresponding-pair difference transport, norms, angular/radial decomposition and terminal normalization summaries are in each arm/panel mechanistic JSONL; raw tensors are retained. Every captured whole-block residual identity is checked exactly.','',
            'The native score matrix is decomposed into repeated query-independent candidate preference plus query-dependent interaction; target/incorrect margins, component norms, rank changes and positive/negative interaction counts are retained. Geometry is descriptive only: no new binary collapse threshold, independent-pair inference, component attribution or task-success claim is introduced. A change in variation cannot substitute for competence or the paired contrasts.']
    else:report+=['The paired experiment is incomplete. Only actually executed checks/results in the run record and partial logs are evidence. Unexecuted final gates/contrasts/mechanistic panels remain unverified. No scientific non-success is inferred from an engineering/resource stop.',json.dumps(run.get('execution_boundary',{}),indent=2),str(run.get('failure',''))]
    report+=['','## Resource and integrity accounting','',json.dumps(total,indent=2),'',f"Execution boundary elapsed {run['elapsed_seconds_to_execution_boundary']:.6f}s under the21600s paired ceiling. Peak allocated/reserved CUDA bytes: {run['peak_cuda_allocated_bytes']}/{run['peak_cuda_reserved_bytes']}. Per-arm phase and attempted/completed/failed counters, actual selected calls, contextual commit entries, backwards and optimizer steps are retained. Complete token-trace reconciliation is in `complete_trace_verification_v1.json`. No additional model trajectories, diagnostics, profiling or warm-ups were run.",'',
        f"All{final_integrity['protected_files']} pre-existing file identities are preserved except the explicitly authorized Git HEAD switch/log append for the new branch; original Git log bytes remain a verified prefix. Original branch ref, HEAD commit, index and tracked diff are preserved. All14 historical tracked modifications and2093 original untracked paths remain. The supplied protocol adds one new root file; pilot evidence is isolated. No reset, clean, stash, source overwrite or deletion occurred.",'',
        'Final checkpoints and optimizer/update metadata, when completed, are named C_update1000_v1.pt and T_update1000_v1.pt; their individual identity records contain SHA256. All new file hashes, source amendment diff and configuration seals accompany publication. The original training remains COMPLETED_NON_SUCCESS; original readout remains INCONCLUSIVE_NONCONVERGED_OPTIMIZATION; converged refit remains CERTIFIED_FIXED_OBJECTIVE_NON_SUCCESS.','',
        '## Interpretation and stop','',
        f"Outcome: **{classification}**. Any positive effect concerns the complete prescribed intervention under this task/recipe; it does not uniquely identify historical collapse causation. No claim of internally learned insertion sort, seed replication, untouched confirmation, learned adaptive routing or general architectural superiority is established. No variant, seed, extra update, coefficient, gain/normalization change, task, fitted readout, selector or follow-up is launched. Stop after verified publication and report; await a new research decision.",'']
    with (OUT/'report_v1.md').open('x') as f:f.write('\n'.join(report))
    save('artifact_inventory_v1.json',dict(files={p.name:dict(size=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file()},self_excluded=True))
    print(json.dumps(dict(classification=classification,metrics=metrics,contrasts=contrasts,resource_totals=total,integrity=final_integrity),indent=2),flush=True)
if __name__=='__main__':main()
