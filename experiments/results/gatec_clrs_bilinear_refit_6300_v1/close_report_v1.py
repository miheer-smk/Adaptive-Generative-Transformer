"""Close the one refit using recorded results only; no tensors/model/optimizer."""
import json,csv,hashlib
from support_v1 import OUT,ROOT,PRIOR,CACHE,save,sha,integrity
def main():
    assert not (OUT/'final_results_v1.json').exists(),'No closure overwrite'
    fit=json.loads((OUT/'fit_results_v1.json').read_text());assert fit['certified']
    steps=[json.loads(s) for s in (OUT/'fit_iterations_v1.jsonl').read_text().splitlines()]
    events=[json.loads(s) for s in (OUT/'fit_evaluations_v1.jsonl').read_text().splitlines()]
    completed=[x for x in events if x['event']=='evaluation_completed'];attempts=[x for x in events if x['event']=='evaluation_attempt'];hits=[x for x in events if x['event']=='cache_hit']
    assert len(completed)==len(attempts)==360==fit['evaluations_completed'] and len(hits)==fit['cache_hits']==341
    assert len(steps)==341==fit['accepted_iterations'] and fit['step_calls']==341 and fit['closure_calls']==700
    assert completed[0]['certificate_bound']>1e-6 and completed[0]['F']==2.772588722239781
    assert completed[0]['B_sha256']==hashlib.sha256(bytes(64*64*8)).hexdigest()
    assert all(s['certificate_bound']>1e-6 for s in steps[:-1]) and steps[-1]['certificate_bound']<=1e-6
    assert all(s['moved'] and s['strong_Wolfe_pass'] for s in steps)
    assert steps[-1]['certificate_bound']==fit['certificate_bound'] and fit['elapsed_seconds']<=10
    assert steps[-1]['matching_evaluation']==completed[-1]['evaluation']
    for n,h in json.loads((OUT/'fit_seal_v1.json').read_text())['files'].items():assert sha(OUT/n)==h,n
    for n,h in json.loads((OUT/'protocol_seal_v1.json').read_text())['files'].items():assert sha(ROOT/n)==h,n
    assert sha(OUT/'evaluate_readout_v1.py')==json.loads((OUT/'evaluation_source_seal_v1.json').read_text())['source_sha256']
    original=json.loads((OUT/'retained_original_head_verification_v1.json').read_text());assert original['success']
    assert json.loads((PRIOR/'final_results_v1.json').read_text())['classification']=='INCONCLUSIVE_NONCONVERGED_OPTIMIZATION'
    trace=dict(success=True,first_passing_accepted_iteration=341,zero_initialization_verified=True,fresh_optimizer=True,continuous_history=20,accepted_updates=341,actual_evaluations=360,cache_hits=341,rejected_trial_evaluations=18,matching_terminal_full_gradient=True,extra_terminal_full_evaluations=0,accepted_strong_Wolfe_checks_all_pass=True,no_new_model_calls=True)
    save('trace_verification_v1.json',trace)
    metrics=json.loads((OUT/'metrics_v1.json').read_text());pairs=json.loads((OUT/'paired_contrasts_v1.json').read_text());analysis=json.loads((OUT/'analysis_results_v1.json').read_text())
    pred=json.loads((OUT/'per_array_predictions_v1.json').read_text())
    interactions={}
    for split,rows in pred.items():
        interactions[split]={}
        for mode in ['FP32','FP64']:
            keys=['pair_comparisons','interaction_improves','interaction_worsens','common_positive','actual_positive','flipped_toward_target','flipped_against_target','total_target_rank_improvement']
            interactions[split][mode]={k:sum(r['interaction'][mode][k] for r in rows) for k in keys}
            interactions[split][mode]['mean_common_norm']=sum(r['interaction'][mode]['common_matrix_norm'] for r in rows)/len(rows)
            interactions[split][mode]['mean_interaction_norm']=sum(r['interaction'][mode]['interaction_norm'] for r in rows)/len(rows)
    save('interaction_summary_v1.json',interactions)
    fields=['split','condition','predecessor_accuracy','exact_array_accuracy','CE','invalid_chain','mean_target_rank','mean_target_margin','distinct_predicted_candidates']
    with (OUT/'metrics_table_v1.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for split,conditions in metrics.items():
            for condition,v in conditions.items():writer.writerow(dict(split=split,condition=condition,**{k:v.get(k) for k in fields[2:]}))
    result=integrity();assert result['success'];save('integrity_final_v1.json',result)
    resources=dict(AGT_model_constructions=0,model_tokens_attempted=0,model_tokens_completed=0,model_tokens_failed=0,selected_bank_calls=0,terminal_commits=0,AGT_backwards=0,AGT_optimizers=0,AGT_optimizer_updates=0,readout_fits=1,readout_optimizer_objects=1,readout_accepted_updates=fit['accepted_iterations'],actual_full_loss_gradient_evaluations_attempted=fit['evaluations_attempted'],actual_full_loss_gradient_evaluations_completed=fit['evaluations_completed'],actual_full_loss_gradient_evaluations_failed=fit['evaluations_failed'],exact_B_cache_hits=fit['cache_hits'],closure_calls=fit['closure_calls'],readout_autograd_backwards=0,fit_wall_seconds_including_optimizer_construction=fit['elapsed_seconds'],extra_terminal_full_loss_gradient_evaluations=0,heldout_extraction_calls=0,evaluation='One terminal iterate; fixed FP32 primary and FP64 sensitivity tensor algebra only',tests=0,retries=0)
    save('resource_ledger_v1.json',resources)
    sources={p.name:sha(p) for p in OUT.glob('*.py')}
    save('implementation_seal_v1.json',dict(files=sources,prior_optimizer_source=json.loads((PRIOR/'source_input_provenance_v1.json').read_text())['optimizer_source'],prior_evaluation_source_sha256=sha(PRIOR/'evaluate_readout_v1.py')))
    report=['# Exploratory bounded bilinear-readout refit','',f"**{analysis['classification']}**. The fresh zero-initialized fit reached the first accepted-iterate certificate within every limit, but the approximately optimized fixed objective did not demonstrate competent query-conditioned decoding. The original frozen-readout study remains **INCONCLUSIVE_NONCONVERGED_OPTIMIZATION** and the historical AGT result remains **COMPLETED_NON_SUCCESS**.",'',
        '## Provenance and scientific boundary','',
        f"This separately sealed exploratory attempt follows the prior nonconvergence. Protocol SHA256 `{sha(OUT/'protocol_v1.md')}`; input/source protocol seal `{sha(OUT/'protocol_seal_v1.json')}`. Training cache `{sha(CACHE/'training_representation_cache_v1.pt')}`; frozen checkpoint `2c0e3d6ad3711d8f0ee209eb3e3b0f8db8964e6eeaf2be77b7bd56120f29e6c8`. The original terminal matrix was identity-verified but not loaded as initialization. Exact training-only preconditioners are reused from the prior file (SHA256 `{sha(PRIOR/'preconditioners_v1.pt')}`). Held-out representation SHA256 `{sha(PRIOR/'heldout_representations_v1.pt')}`.",'',
        'All128 fitting arrays were previously seen during original AGT training. Development32 size16 and transfer32 size64 arrays are held out from readout fitting, but examined historically; neither is untouched confirmation. No data or state extraction was repeated.','',
        '## Fit and certificate','',
        f"One4096-parameter FP64 B starts at exactly0, with one fresh optimizer and continuous L-BFGS history20, strong-Wolfe, lr1, c1=1e-4/c2=.9, other stopping tolerances disabled. Whiteners, full-batch meanCE averaging, division8 and regularization1e-6 are unchanged. The exact analytic full-gradient formula is the prior implementation, with no autograd backward. The timer includes optimizer construction and the initial evaluation. Stop reason: **{fit['stop_reason']}**.",'',
        f"Accepted updates **{fit['accepted_iterations']}** /512; wrapper calls341; actual objective/gradient evaluations attempted/completed/failed **360/360/0** /1024; cache hits341; closure calls700. Initial evaluation plus341 accepted points plus18 rejected trial points accounts for360 actual evaluations. No duplicate full recomputations or extra terminal full evaluation occurred. Cache keys contain the exact FP64 B bytes; values/gradients are reused only for the identical matrix and fixed objective.",'',
        f"Fitting wall time **{fit['elapsed_seconds']:.12g}s** /10. Terminal F={fit['objective']:.17g}, CE={fit['CE']:.17g}, regularizer={fit['regularizer']:.17g}; gradient norm={fit['gradient_norm']:.17g}. Certificate **{fit['certificate_bound']:.17g} <=1e-6**. The initial point and all340 preceding accepted points fail the certificate; every341 accepted update passes strong-Wolfe checks. Terminal objective/gradient uses its matching retained full evaluation. Trials alone never authorize selection.",'',
        f"Terminal matrix SHA256 `{sha(OUT/'terminal_readout_v1.pt')}`. Full terminal optimizer state, evaluation/accepted-update/cache trace and source identities are retained. `fit_seal_v1.json` SHA256 `{sha(OUT/'fit_seal_v1.json')}` was verified before held-out metrics. Only the first certificate-passing iterate is evaluated. No restart, old history, regularizer sweep or matrix selection occurred.",'',
        '## Prescribed operational metrics','',
        'Primary FP32 CUDA tensor arithmetic and fixed FP64 sensitivity use the prior evaluation arithmetic unchanged. Original archived pointer scores, predictions, targets and input identities match the retained cache exactly for all64 held-out arrays before new-head results. No model is constructed.','','|Split|Condition|Pointer accuracy|Exact arrays|CE|Invalid chains|Mean rank|Mean target margin|Distinct candidates/array|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for split,conditions in metrics.items():
        for condition,v in conditions.items():
            if condition in ['fitted_FP64','common_FP64']:continue
            def num(k):return 'unavailable' if v.get(k) is None else f"{v[k]:.10g}"
            report.append(f"|{split}|{condition}|{100*v['predecessor_accuracy']:.8g}%|{100*v['exact_array_accuracy']:.8g}%|{num('CE')}|{100*v['invalid_chain']:.8g}%|{num('mean_target_rank')}|{num('mean_target_margin')}|{num('distinct_predicted_candidates')}|")
    report+=['','Uniform score rows use deterministic lowest-index argmax, with analytic independent uniform probabilities reported separately (expected pointer accuracy1/n, exact-array n^(-n)). Competition rank1 for all-tied uniform rows is not discrimination; stable index ranks/ties are retained. Self-pointer/original-order controls are hard-only, with CE/ranks/margins unavailable and no invented smoothing. Invalid chains are task errors, not executor failures. Complete per-array/query records are retained.','',
        '## Paired contrasts and competence','',
        '|Development FP32 contrast|Point difference|Adjusted lower bound|Positive lower bound|','|---|---:|---:|---|']
    for name,v in pairs['primary'].items():report.append(f"|Fitted minus {name}|{100*v['point_estimate']:.8g} pp|{100*v['lower_bound']:.8g} pp|{v['positive_adjusted_lower_bound']}|")
    report+=['','The exact previously retained20000 paired whole-development-array bootstrap bank is reused (seed6300); both contrasts use empirical one-sided quantile.025, linear interpolation. Pointers are not independent resampling units. Uncertainty is input-sampling uncertainty conditional on one frozen core and this exploratory fitted matrix. No selection or fitting uses evaluation targets.','',
        'Size16 predecessor>=95% and exact>=80%: **fail**. Size64 predecessor>=90% and exact>=50%: **fail**, separately reported. The original-head contrast resolves positively, but fitted-minus-own-common does not; therefore no required query-conditioned advantage or strong competent recovery is demonstrated.','',
        '## Score decomposition and numerical sensitivity','',
        'The actual projected vectors verify Srow=1*b.T+I. Both matrix components, target-versus-incorrect margins and all target ranks are retained. Components are not separately fitted heads. Pair counts are descriptive dependent comparisons, not independent observations.']
    for split,v in interactions.items():
        x=v['FP32'];report.append(f"\n{split}: {x['pair_comparisons']} target/incorrect comparisons; interaction improves{x['interaction_improves']} and worsens{x['interaction_worsens']}. Common-positive{x['common_positive']}, actual-positive{x['actual_positive']}; flips toward/against target{x['flipped_toward_target']}/{x['flipped_against_target']}; total target-rank improvement{x['total_target_rank_improvement']}. Mean common/interaction Frobenius norms{x['mean_common_norm']:.10g}/{x['mean_interaction_norm']:.10g}.")
    report+=['','|Split|FP32 accuracy|FP64 accuracy|FP32 CE|FP64 CE|Prediction disagreements|','|---|---:|---:|---:|---:|---:|']
    for split,v in metrics.items():
        s=analysis['precision_sensitivity'][split];report.append(f"|{split}|{100*v['fitted_FP32']['predecessor_accuracy']:.8g}%|{100*v['fitted_FP64']['predecessor_accuracy']:.8g}%|{v['fitted_FP32']['CE']:.12g}|{v['fitted_FP64']['CE']:.12g}|{s['prediction_disagreements']}/{s['total_queries']}|")
    report+=['','FP64 does not restore competence. Its fitting CE matches the sealed objective CE within floating-point precision; this is prescribed metric evaluation, not another full loss/gradient evaluation. Neither the readout nor precision is selected using these results.','',
        '## Resources, preservation and boundary','',
        f"Zero AGT constructions, model tokens, selected bank calls, terminal commits, core backward passes, AGT optimizers/updates, extraction calls, tests, warm-ups or retries. Exactly one new readout fit. Computational work outside the1.553-second fit comprises identity checks, prescribed terminal readout matrix arithmetic/analysis and publication; it adds no optimizer iterations or objective/gradient evaluations. All{result['protected_files']} pre-existing files (including the prior study and local Git administration), HEAD, tracked diff and historical bytecode exception are unchanged. Numerical warnings from the unchanged Torch environment are not silently suppressed; actual arithmetic completed and all required checks passed.",'',
        'The approximately optimized solution of this fixed regularized bilinear objective does not demonstrate required competence. This is not proof that every bilinear readout or representation fails, does not establish historical causation or internal insertion-sort learning, and says nothing positive about learned routing. Existing nonconverged and negative historical classifications remain unchanged.','',
        'Publication uses a unique branch with the previously verified study commit as parent; all raw content hashes and remote tree entries are verified separately. Remote commit identity is reported in terminal output and the publication receipt. No local Git staging/ref change, history rewrite or deletion is used. **Stop: no third fit, extended limit, alternate readout, AGT training, extra seed/input or automatic follow-up.**','']
    with (OUT/'report_v1.md').open('x') as f:f.write('\n'.join(report))
    save('final_results_v1.json',dict(status='COMPLETED_EXPLORATORY_REFIT',classification=analysis['classification'],fit=fit,analysis=analysis,resource=resources,integrity=result,original_study_classification='INCONCLUSIVE_NONCONVERGED_OPTIMIZATION',historical_AGT_training='COMPLETED_NON_SUCCESS',publication_status='Pending independently verified publication receipt'))
    save('artifact_inventory_v1.json',dict(files={p.name:dict(size=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file()},self_excluded=True))
    print(json.dumps(dict(status='COMPLETED_EXPLORATORY_REFIT',classification=analysis['classification'],protected_files=result['protected_files'],new_AGT_calls=0),indent=2))
if __name__=='__main__':main()
