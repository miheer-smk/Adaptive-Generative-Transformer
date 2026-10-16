"""Offline record/trace reconciliation and report; no AGT or tensor imports."""
import json,gzip,time,csv,collections,math
from support_v1 import OUT,CACHE,ROOT,ENG,sha,save,check,integrity

def main():
    check();fit=json.loads((OUT/'fit_results_v1.json').read_text());ext=json.loads((OUT/'heldout_extraction_results_v1.json').read_text())
    analysis=json.loads((OUT/'analysis_results_v1.json').read_text());metrics=json.loads((OUT/'metrics_v1.json').read_text());contrasts=json.loads((OUT/'paired_contrasts_v1.json').read_text())
    pred=json.loads((OUT/'per_array_predictions_v1.json').read_text());seal=json.loads((OUT/'fit_seal_v1.json').read_text())
    for n,h in seal['files'].items():assert sha(OUT/n)==h,n
    assert ext['status']=='COMPLETED_VERIFIED_HELDOUT_CACHE'
    trials=[json.loads(x) for x in (OUT/'fit_evaluations_v1.jsonl').read_text().splitlines()];iterations=[json.loads(x) for x in (OUT/'fit_iterations_v1.jsonl').read_text().splitlines()]
    assert len(trials)==fit['loss_gradient_evaluations']==271 and len(iterations)==fit['accepted_iterations']==256
    assert min(r['certificate_bound'] for r in trials)>1e-6 and all(r['moved'] for r in iterations)
    assert fit['stop_reason']=='accepted_iteration_limit' and not fit['converged']
    hs=json.loads((OUT/'heldout_started_v1.json').read_text());fs=json.loads((OUT/'fit_started_v1.json').read_text());es=json.loads((OUT/'evaluation_started_v1.json').read_text())
    assert hs['monotonic']>fs['monotonic']+fit['elapsed_seconds'] and es['monotonic']>hs['monotonic']+ext['ledger']['elapsed_seconds']-.1
    assert sha(OUT/'fit_seal_v1.json')==json.loads((OUT/'heldout_input_manifest_v1.json').read_text())['fit_seal_sha256']
    cases=collections.defaultdict(list);counts=collections.Counter()
    with gzip.open(OUT/'token_traces_v1.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);t=r['token']
            assert r['status']=='completed' and r['blocks']==r['selected_actions']==list(range(8)) and r['D']==r['P']==8
            assert r['commits']==1 and r['controller_calls']==0 and r['parent_unchanged'] and r['returned_state_disjoint']
            assert r['incoming_metadata']['/tokens_committed']==t and r['returned_metadata']['/tokens_committed']==t+1
            c=r['commit'];assert c['entry_count']==7 and c['token_index']==t and all(v==t and c['versions_after'][k]==t+1 for k,v in c['versions_before'].items())
            cases[r['case']].append(t);counts['tokens']+=1;counts['selected']+=8;counts['commits']+=1;counts['contextual']+=7
    expected=[(f'official/development/row{i:04d}',32) for i in range(32)]+[(f'official/size_transfer/row{i:04d}',128) for i in range(32)]
    assert list(cases)==[x[0] for x in expected] and all(cases[n]==list(range(length)) for n,length in expected)
    assert counts['tokens']==counts['commits']==5120 and counts['selected']==40960
    eq=json.loads((OUT/'heldout_equivalence_v1.json').read_text());assert all(x['bitwise_scores'] for x in eq['comparisons'])
    interaction={}
    for split,rows in pred.items():
        interaction[split]={k:sum(r['interaction']['FP32'][k] for r in rows) for k in ['pair_comparisons','interaction_improves','interaction_worsens','common_positive','actual_positive','flipped_toward_target','flipped_against_target','total_target_rank_improvement']}
    save('interaction_summary_v1.json',interaction)
    verification=dict(success=True,fit_seal_intact=True,fit_completed_before_heldout_extraction=True,all_original_head_comparisons_before_new_metrics=True,
        trial_evaluations=len(trials),accepted_steps=len(iterations),no_trial_met_convergence=True,full_trace_rows=counts['tokens'],arrays=64,selected=counts['selected'],commits=counts['commits'],contextual_entries=counts['contextual'],native_original_scores_all_bitwise=True,
        no_AGT_backwards_or_optimizers=True,training_cache_not_reextracted=True,bootstrap_replicates=20000,bootstrap_unit='complete array')
    save('complete_record_verification_v1.json',verification)
    integ=integrity();assert integ['success'];save('integrity_final_v1.json',integ)
    clock=json.loads((OUT/'stage_clock_v1.json').read_text());elapsed=time.monotonic()-clock['measured_start_monotonic']
    resource=dict(prior_cache_extraction_seconds=31.302,measured_remaining_stage_elapsed_at_report=elapsed,conservative_pre_clock_reserve_seconds=600,
        cumulative_budget_charge_at_report=31.302+600+elapsed,remaining_stage_cap=2668,cumulative_cap=2700,
        remaining_model_tokens_attempted=5120,remaining_model_tokens_completed=5120,remaining_model_tokens_failed=0,selected_attempted=40960,selected_completed=40960,new_commits=5120,
        prior_cache_tokens=4064,cumulative_new_study_tokens=9184,cumulative_selected_calls=73472,cumulative_commits=9184,AGT_backwards=0,AGT_optimizer_objects=0,AGT_optimizer_updates=0,
        readout_optimizer_objects=1,readout_accepted_updates=256,readout_loss_gradient_evaluations=271,readout_autograd_backwards=0,readout_gradient_method='Analytic exact full-batch gradient',LBFGS_seconds=fit['elapsed_seconds'],heldout_extraction_seconds=ext['ledger']['elapsed_seconds'],extraction_peak_CUDA_allocated=ext['ledger']['peak_CUDA_allocated'],extraction_peak_CUDA_reserved=ext['ledger']['peak_CUDA_reserved'],warmups=0,retries=0,
        publication_timing='Still included in the live remaining-stage guard; final terminal verification reports final elapsed/charged time')
    save('resource_ledger_v1.json',resource)
    tables=['|Split|Condition|Pointer accuracy|Exact arrays|Mean CE|Invalid chains|Mean target rank|Mean target margin|Distinct candidates/array|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    csvrows=[]
    for split,d in metrics.items():
        for condition,v in d.items():
            if condition in ['fitted_FP64','common_FP64']:continue
            fmt=lambda k:'unavailable' if v.get(k) is None else f'{v[k]:.10g}'
            tables.append(f"|{split}|{condition}|{100*v['predecessor_accuracy']:.8g}%|{100*v['exact_array_accuracy']:.8g}%|{fmt('CE')}|{100*v['invalid_chain']:.8g}%|{fmt('mean_target_rank')}|{fmt('mean_target_margin')}|{fmt('distinct_predicted_candidates')}|")
            csvrows.append(dict(split=split,condition=condition,**v))
    with (OUT/'metrics_table_v1.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    sens=['|Split|FP32 pointer accuracy|FP64 pointer accuracy|FP32 CE|FP64 CE|Different predictions|','|---|---:|---:|---:|---:|---:|']
    for split,d in metrics.items():
        s=analysis['precision_sensitivity'][split];sens.append(f"|{split}|{100*d['fitted_FP32']['predecessor_accuracy']:.8g}%|{100*d['fitted_FP64']['predecessor_accuracy']:.8g}%|{d['fitted_FP32']['CE']:.12g}|{d['fitted_FP64']['CE']:.12g}|{s['prediction_disagreements']}/{s['total_queries']}|")
    contrast_table=['|Development paired contrast|Point difference|Adjusted one-sided lower bound|Positive lower bound|','|---|---:|---:|---|']
    for k,v in contrasts['primary'].items():contrast_table.append(f"|fitted FP32 − {k}|{100*v['point_estimate']:.8g} pp|{100*v['lower_bound']:.8g} pp|{v['positive_adjusted_lower_bound']}|")
    text=f'''# Frozen-core bilinear-readout decodability study

## Result and boundary

**{analysis['classification']}**. All scheduled fitting, extraction and evaluation stages completed within their numerical/resource limits. The fit reached256 accepted iterations without the declared convergence certificate; it was not restarted or extended. The terminal readout improves over the original head on development, but is worse than its own query-independent component and far below competence thresholds. Thus it does not demonstrate competent query-conditioned bilinear decoding. The historical AGT training result remains **COMPLETED_NON_SUCCESS**.

This is one frozen-core readout study, not core training, algorithm-discovery evidence, a learned-routing result or an architectural-superiority comparison. The128 fitting arrays were seen during original AGT training. Development/transfer arrays are held out from **readout fitting** but were examined previously; they are not untouched confirmation data.

## Fit, whitening and sealing

The original training cache SHA-256 is `83b8a4c1d19ba479901d5714c246e89385aadfc76eb971c463fe9ac115352f41`. All per-array hashes, original-index associations, targets, shapes and dtypes were verified before fitting. Its128 trajectories were not repeated.

Separate64×64 uncentered query/candidate moments used all2048 fitting states per role. Symmetric inverse-square-root whiteners retain all64 dimensions. Eigenvalue floors are query{fit['query_floor']:.15g} and candidate{fit['candidate_floor']:.15g}, exactly1e-8 times each largest eigenvalue. Respectively62/59 eigenvalues are floored, not removed. No mean subtraction, per-vector normalization, evaluation-data moment or extra feature was introduced. Matrices/eigensystems are retained in `preconditioners_v1.pt`.

One unrestricted4096-parameter B starts at0. The objective is `meanCE+1e-6/2*||B||²`; the full-batch gradient is `sum_r U_r.T@(P_r-Y_r)@V_r/(8*2048)+1e-6*B`. This exact analytic gradient is supplied to one continuous CPU FP64 PyTorch L-BFGS optimizer with history20 and strong-Wolfe search. The default initial step scale is1; algorithm defaults c1=1e-4/c2=.9 are recorded. Default alternative stopping tolerances are disabled. Caching identical deterministic B evaluations does not change the objective and is disclosed:526 closure calls correspond to271 actual loss/gradient evaluations, including the initial point; all calls/steps are logged.

Stop reason: **accepted_iteration_limit**. Accepted steps256; actual evaluations271; fit duration{fit['elapsed_seconds']:.9f}s. Final F={fit['objective']:.15g}, CE={fit['CE']:.15g}, regularizer={fit['regularizer']:.15g}, gradient norm={fit['gradient_norm']:.15g}. Certificate `||gradF||²/(2e-6)` is **{fit['certificate_bound']:.15g}**, exceeding1e-6. Neither an accepted iterate nor a line-search trial met that threshold. Nonconvergence is reported exactly, without rounding into a pass. No alternative iterate was evaluated or selected.

`terminal_readout_v1.pt` and the objective/optimizer records were hashed in `fit_seal_v1.json` **before** opening held-out pools for extraction. That seal remains unchanged after evaluation. FP64 fitting CE recomputed during the prescribed terminal evaluation agrees with sealed CE to floating-point precision. The readout is separate from the archived model; no model checkpoint is replaced.

## Frozen extraction and original-head equivalence

The frozen checkpoint SHA-256 remains `2c0e3d6ad3711d8f0ee209eb3e3b0f8db8964e6eeaf2be77b7bd56120f29e6c8`. Parameter/buffer identity before/after is `7ab9239da5b1a4957cde546bb95e4c2961800b337f8bf677f8202e662e94f1a0`. Existing source/adapter behavior was reused unchanged, with namespaced observer import paths and bounded counters.

All32 development n16 and32 transfer n64 arrays ran once in original order, using fresh initial states, FP32 canonical `[0,1,2,3,4,5,6,7]`, normal RMSNorm/memory/cache/GDN semantics and terminal commits. Encoder/core/normalization/original pointer parameters are frozen. Read-only hooks verify cached candidates and queries equal the original head's inputs. No fitting-array re-extraction, warm-up or retry occurred.

All64 ordinary original-head score matrices and predictions match the archive **bit-for-bit**, before any new-head task metrics were calculated. Both representation roles, exact targets and original input identities are retained in `heldout_representations_v1.pt` (SHA-256 `{ext['cache_sha256']}`). Complete traces verify5120 tokens,40960 actual selected bank calls,5120 commits and35840 contextual commit entries. Each token has D=P=8, the correct selected order, unchanged parent/in-flight state and valid cache versions. Invalid predicted pointer chains below are **task-output errors**, not executor/mechanical failures.

## Primary operational metrics

FP32 CUDA arithmetic explicitly computes `states@whitener`, then `transformed_queries@B`, then candidate dot products divided by8. FP64 sensitivity uses precisely the same sealed matrix and preconditioners, cast to64, with no fitting or selection. CE is calculated stably from retained scores.

{chr(10).join(tables)}

Competition ranks are1+number of strictly greater scores; stable candidate-index ranks and all ties are also retained. Uniform scores have ties everywhere, so their competition rank1 does **not** indicate discrimination. The table reports deterministic lowest-index argmax for uniform probabilities; analytic independent-uniform expected accuracy is1/n and exact-array probability is n^(-n), separately recorded without random draws. Hard self-pointer/original-order controls have genuine hard-decision metrics; CE, ranks and score margins are unavailable, with no smoothing invented. Full per-query predictions, ranks, ties, target margins and per-array distinct-candidate counts are retained in `per_array_predictions_v1.json`.

## Primary contrasts and competence decisions

{chr(10).join(contrast_table)}

These use20,000 paired complete-development-array bootstrap replicates, seed6300, sharing one bank across both contrasts. The two-contrast family uses one-sided lower empirical quantile.025 with linear interpolation. Pointers within an array are not resampled independently. `bootstrap_bank_v1.pt` and `paired_contrasts_v1.json` preserve the bank/hash and results. Intervals describe input-sampling uncertainty conditional on this frozen core/readout and the historically exposed panel, not seed reproducibility.

Size16 requirements (95% predecessor,80% exact): **both fail**. Size64 requirements (90% predecessor,50% exact): **both fail**, reported separately. The original-head contrast has a positive adjusted lower bound; the common-component contrast does not. Strong positive evidence therefore fails in addition to the convergence failure. The modest original-head gain cannot be promoted to useful query-conditioned recovery.

## Target-conditioned decomposition

For the actual transformed/projected query and candidate vectors, verify `Srow=1*b.T+I`, with `b=Kc@mean(Q)/8` and `I=Qc@Kc.T/8`. The common component is an analytical repeated candidate preference, not a second fitted head or a causal deployed policy. Algebraic identities are checked in FP64 from each precision's actual vectors; actual FP32 score arithmetic agrees under the original atol1e-6/rtol1e-5 comparison standard.

On development, interaction improves3805/7680 target-versus-incorrect margins and worsens3875. It flips567 comparisons toward the target and671 away, and worsens total target ranks by104 positions. Fitted CE2.75718348 is worse than common-only2.75264954. Although predictions vary across3.65625 candidates/array rather than1, that variability does not establish useful task credit.

Fitting interaction counts are16706 improved/14014 worsened (30720 dependent comparisons); transfer counts65133/63891 (129024 comparisons). These pair counts are descriptive, not independent statistical observations. `interaction_summary_v1.json`, per-array records and score tensors retain complete component evidence. No target label is used to construct the common component itself.

## Numerical sensitivity

{chr(10).join(sens)}

FP64 does not recover competence. Maximum FP32/64 score differences are recorded, as are4 fitting,1 development and10 transfer prediction disagreements. No precision, iterate, regularizer, eigenvalue floor or threshold was changed after these results. The primary result remains FP32. No fitted-readout data is pooled across lengths.

## Resource and integrity accounting

Remaining-stage new work:5120 attempted/completed model-token calls,0 failures;40960 attempted/completed selected calls;5120 terminal commits;0 AGT backward passes;0 AGT optimizer objects/updates. There is exactly1 readout optimizer and256 accepted readout updates, with271 analytic full-batch loss/gradient evaluations and0 readout autograd backwards. No extra head, seed, sample, warm-up or retry occurred. Extraction took{ext['ledger']['elapsed_seconds']:.9f}s; peak allocated/reserved CUDA memory during extraction was{ext['ledger']['peak_CUDA_allocated']:,}/{ext['ledger']['peak_CUDA_reserved']:,} bytes.

Prior cache extraction is charged31.302s and4064 tokens separately. Full staged extraction totals are9184 newly executed tokens,73472 selected calls and9184 commits; reused historical row0000 adds0 new calls. At report closure, measured elapsed time since the remaining-stage clock marker is{elapsed:.3f}s. A conservative600s opening-inspection reserve is additionally charged (not claimed as measured runtime), giving a cumulative budget charge of{31.302+600+elapsed:.3f}s, below2700s. A live guard includes subsequent publication; final terminal verification reports its inclusive charged time. No ceiling is extended.

All{integ['protected_files']} pre-existing protected files, original cache/checkpoint/source, local HEAD and tracked diff pass final integrity verification. Historical bytecode exceptions are preserved without additions. `resource_ledger_v1.json`, `complete_record_verification_v1.json`, hashes, input manifests, readout/preconditioners, scores and versioned research-only code accompany this report. Publication adds only this fresh subdirectory on a unique branch and verifies contents independently.

## Interpretation and stop

Under the approved interpretation, poor results without the declared convergence certificate are **inconclusive**. They do not establish universal representation impossibility, a historical cause of the original failure, learned routing, internal sorting-algorithm acquisition or superiority over CLRS learners. A small gain over a weak original head, without competent exact arrays or an advantage over query-independent allocation, is insufficient.

**The authorized one-fit study is complete. No optimizer restart, extra iterations, new seed, core training, alternate readout, selector, task change or follow-up experiment is launched.**
'''
    with (OUT/'report_v1.md').open('x') as f:f.write(text)
    save('final_results_v1.json',dict(status='COMPLETED_FROZEN_READOUT_STUDY',classification=analysis['classification'],historical_training_status='COMPLETED_NON_SUCCESS',fit=fit,analysis=analysis,resource=resource,integrity=integ,verification=verification,report='report_v1.md',publication_status_at_closure='Pending separate verified receipt'))
    check();print(json.dumps(dict(classification=analysis['classification'],integrity=integ,resource=resource),indent=2))

if __name__=='__main__':main()
