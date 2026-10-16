"""Report retained analytical outcomes; no tensor or model imports."""
import csv,json,collections
from pathlib import Path
OUT=Path(__file__).resolve().parent
def write(name,x):
    with (OUT/name).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def main():
    d=json.loads((OUT/'condition_summaries_v1.json').read_text());scalar=json.loads((OUT/'global_scalar_v1.json').read_text())
    verification=json.loads((OUT/'evidence_verification_v1.json').read_text());manifest=json.loads((OUT/'manifest_v1.json').read_text())
    queries=list(csv.DictReader((OUT/'per_query_v1.csv').open()));targets=manifest['target_indices']
    target_count=collections.Counter(targets);assert target_count[2]==2 and target_count[13]==0 and all(target_count[j]==1 for j in range(16) if j not in [2,13])
    table=['|Condition|Accuracy / exact array|Mean target rank|Mean target margin|CE|Mean max probability|',
        '|---|---|---:|---:|---:|---:|']
    norms=['|Condition|Common preference norm|Interaction norm|Interaction/common|Interaction improves / worsens margins|Common / combined positive margins|CE effect of interaction|',
        '|---|---:|---:|---:|---|---|---:|']
    geom=['|Condition/stage|Centered energy|Radial fraction|Mean pair cosine|Mean pair distance|',
        '|---|---:|---:|---:|---:|']
    audit={}
    for label,v in d.items():
        b=v['common_preference_vector'];order=sorted(range(16),key=lambda j:(-b[j],j));ranks={j:1+order.index(j) for j in range(16)}
        actual=[q for q in queries if q['condition']==label]
        same=all(q['correct_target_rank']==q['common_only_target_rank'] for q in actual)
        assert same
        formula=(136-ranks[13]+ranks[2])/16;assert formula==v['mean_target_rank']
        audit[label]=dict(common_candidate_order=order,minimum_index2_rank=ranks[2],maximum_index13_rank=ranks[13],
            mean_target_rank_from_fixed_common_order=formula,all16_target_ranks_equal_common_only=same,
            target_pair_sign_changes_due_to_interaction=v['common_wrong_to_total_correct']+v['common_correct_to_total_wrong'],
            predicted_candidate=v['common_preferred_candidate'],its_target_frequency=v['common_preferred_candidate_target_frequency'])
        table.append(f"|{label}|{100*v['pointer_accuracy']:.2f}% / {v['exact_array_accuracy']}%|{v['mean_target_rank']:.6g}|{v['target_margin']['mean']:.10g}|{v['mean_CE']:.12g}|{v['mean_maximum_probability']:.12g}|")
        norms.append(f"|{label}|{v['common_matrix_Frobenius']:.10g}|{v['interaction_Frobenius']:.10g}|{v['interaction_common_norm_ratio']:.10g}|{v['all240_interaction_margins']['positive']} / {v['all240_interaction_margins']['negative']}|{v['all240_common_margins']['positive']} / {v['all240_target_margins']['positive']}|{v['CE_effect_of_existing_interaction']:.10g}|")
        for stage,g in v['representation_geometry'].items():
            geom.append(f"|{label}/{stage}|{g['centered_energy']:.10g}|{g['radial_fraction']:.10g}|{g['pairwise_cosine']['mean']:.12g}|{g['pairwise_distance']['mean']:.10g}|")
    write('query_independence_accounting_v1.json',dict(target_counts={str(i):target_count[i] for i in range(16)},formula='For a fixed candidate ordering, mean rank=(sum(1..16)-rank(13)+rank(2))/16; this array has target2 twice and target13 absent.',conditions=audit))
    perq=['|Query|Official target|A rank / prediction|B rank / prediction|B_global rank / prediction|C rank / prediction|',
        '|---:|---:|---|---|---|---|']
    for i in range(16):
        cells=[]
        for label in ['A','B','B_global','C']:
            q=next(q for q in queries if q['condition']==label and int(q['query'])==i)
            cells.append(q['correct_target_rank']+' / '+q['prediction'])
        perq.append('|'+str(i)+'|'+str(targets[i])+'|'+'|'.join(cells)+'|')
    text=f'''# Counterfactual pointer readout: retained raw, committed and direction-only states

Status: **COMPLETED_OFFLINE_COUNTERFACTUAL_READOUT_ANALYSIS**. Historical CLRS training remains **COMPLETED_NON_SUCCESS**. This is CPU tensor algebra on one retained training array and one checkpoint, not a trained model, AGT rollout, benchmark evaluation or intervention on persistent-state evolution.

## Main diagnosis

**Restoring raw radii does not recover query-conditioned predecessor selection under the existing weights on this array.** The raw readout improves from0/16 to1/16 correct pointers and mean target rank9.4375→7.75, but predicts candidate14 for every query. The apparent gain is completely reproduced by its query-independent candidate preference. Its query-dependent interaction improves86/240 margins, worsens154/240, changes no target-pair sign or target rank, and increases CE. Thus this result matches the predeclared outcome **radial variation survives but remains dominated by a global preference and target-misaligned interaction**, rather than useful query-dependent recovery.

Raw CE is1284.6773849 nats with near-certain wrong predictions. The fixed global scalar reduces it to2.77109080045 while preserving every ranking. The mean-rank gain over A remains because the candidate preference changed, not because scaling created information. Direction-only C has baseline-identical predictions and target ranks; its slightly lower CE than A is a calibration effect. Its much lower CE than unscaled B does not demonstrate better target discrimination.

## Evidence identity and observation boundary

- Checkpoint SHA-256: `2c0e3d6ad3711d8f0ee209eb3e3b0f8db8964e6eeaf2be77b7bd56120f29e6c8`.
- Array: `official/train/row0000`, exactly the prior locked localization input and first retained sealed training row. Original input SHA-256: `e147c8817d320036834f4318fe40485a7fa37e9fa243d2ef500b1a08017f1b7c`.
- Official target indices: `{targets}`. Compact-JSON target SHA-256: `{manifest['target_sha256']}`. No prediction was used as a target.
- The mHC directory named in the request is absent. The actual retained directory is `experiments/results/gatec_clrs_mhc_capture_6200_v1/`. All35 files match their Git blob identities in the supplied capture commit `d9d5eb0061b860b796f64fbc559420caff863509`. No directory or historical artifact was renamed.
- Raw H is the retained last-block live output. `src/agt.py:278-299` assigns `h_terminal=h` and passes it to terminal commit; `src/terminal_commit.py:215-218` applies `final_norm(h_terminal)` before memory write; `src/commit.py:28-47` calls `state_norm`; `src/norms.py:30-36` computes `h*rsqrt(mean(h²)+1e-6)*gain`. Thus these are actual pre-RMSNorm live tensors, not inferred core activations.
- The localization raw block7 outputs and committed streams are bitwise equal to the latest capture. Stored candidate rows are the first16 committed streams; query rows are the next16. Candidate/query index j refers to original input element j. Source-backed alignment: `clrs_adapter_v1.py:73-108`, and the retained localization capture.
- All checkpoint/input/tensor/source/adapter identities are in `manifest_v1.json` and checked before analysis. The norm reconstruction agrees with retained committed states with max error{verification['terminal_norm_reconstruction_max_error']:.10g}.

## Fixed analytical definitions

A uses exact saved committed representations. B uses actual raw pre-RMSNorm states. C uses unit-L2 versions of B. B_global multiplies **every** raw query/candidate coordinate by the same positive scalar. All four use the unchanged checkpoint matrices `query.weight` and `key.weight`, without bias or fitting: `Q=Xq Wqᵀ`, `K=Xk Wkᵀ`, `S=QKᵀ/8`.

`alpha = pooled_RMS(A) / pooled_RMS(B) = {scalar['alpha']:.17g}`. Both groups and all32×64 coordinates are pooled equally. Raw RMS is{scalar['raw_pooled_RMS']:.14g}; committed RMS is{scalar['committed_pooled_RMS']:.14g}; scaled raw RMS is{scalar['globally_scaled_raw_pooled_RMS']:.14g}. No targets or predictions choose the scalar. Its square is{scalar['scalar_squared']:.17g}; `S_Bglobal=alpha² S_B` within max numerical error{verification['global_score_scaling_max_error']:.10g}. All argmaxes and full candidate rankings are unchanged.

Definitions and code were sealed before new outcomes. A is reconstructed in CPU FP64, with a separate literal FP32 matrix comparison to saved projections/scores. Maximum baseline score error is{verification['score64_max_error']:.10g} in FP64 and{verification['score32_max_error']:.10g} in FP32, within the original atol1e-6/rtol1e-5 requirements. Baseline CE is{verification['CE_reconstructed']:.15g}, versus retained{verification['CE_retained']:.15g}; predictions and all target ranks agree exactly.

**Scope caveats:** B and C omit the learned coordinate gains present in A. B versus C isolates radial removal under the same unchanged projection weights; A comparisons additionally involve those gains and RMS epsilon. All stored raw states were produced under the original normal committed-state trajectory. This analysis does **not** simulate bypassing normalization during future state carry or memory/cache updates.

## Readout metrics

{chr(10).join(table)}

Exact-array accuracy is0 for every condition. Target margin means target score minus the strongest incorrect candidate. `all_target_margins_v1.csv` retains all960 condition/query/incorrect-candidate margins; distributions including quantiles, min/max and signs are in `condition_summaries_v1.json`. Raw scores range −65258.4050952 to −56770.021243, with mean within-query span7934.48269947. CE uses finite float64 log-sum-exp, without exponentiating raw logits directly. B_global score range is −1.03305843 to −0.898684988; C is near uniform in confidence.

{chr(10).join(perq)}

B and B_global improve ranks on9 queries and worsen them on7 relative to A. There is no uniform target-ranking improvement across queries. All query/candidate indices and ties are retained in `per_query_v1.csv`.

## Target-conditioned common/interaction decomposition

For each condition, `b=Kc mean(Q)/8`, `I=Qc Kcᵀ/8` and `Srow=1 bᵀ+I`. Every one of the240 target-versus-incorrect comparisons is decomposed into its common and interaction contributions. Common norm is the Frobenius norm of the replicated matrix `1 bᵀ`, not merely the vector b. No comparison treats the240 dependent pairs as independent observations.

{chr(10).join(norms)}

Positive interaction margins improve the existing common-only target margin; negative ones worsen it. The raw interaction mean margin is−1.53236360429 (globally scaled−2.42577356315e-5), whereas A is+1.53292341056e-7. No interaction crosses a target-pair sign in **any** condition; every query's target rank equals its common-only rank.

The raw interaction/common norm ratio is0.00284446 (about0.2844%), versus{d['A']['interaction_common_norm_ratio']:.9g} for A. More interaction magnitude does not imply better alignment. Raw CE increases by1.43659087903 when adding its interaction to common-only scores; B_global increases by2.27449491521e-5. A decreases CE by1.43711438250e-7. Algebraic CE score adjoints are also retained; these are local score-space diagnostics, not gradients obtained by backward or predictions of realizable parameter updates.

**Why mean rank and1/16 accuracy can improve without useful query conditioning:** the sealed target vector contains index2 twice, index13 zero times and every other index once. For any single fixed candidate ordering, the mean target rank is `(136−rank(13)+rank(2))/16`. A/C rank index13 first and index2 last, giving9.4375. B/B_global rank index13 fourteenth and index2 second, giving7.75. These formulae exactly reproduce the observed means without a query-dependent decision. Raw's preferred candidate14 is the target of only query5, accounting for its entire1/16 accuracy. `query_independence_accounting_v1.json` records the fixed orders and arithmetic. This is descriptive accounting of retained outcomes, not a fitted baseline or an oracle.

## Radial and angular geometry

{chr(10).join(geom)}

Raw query/candidate squared-distance variation is98.46497%/99.80996% radial. Global scaling preserves those fractions and all angles. Unit-vector C has effectively zero radial variation (nonzero residuals of order1e-26 arise from floating-point normalization). A's surviving pairwise variation is predominantly angular but very small in absolute energy. Normalization does discard large magnitude differences; these magnitudes nevertheless fail to yield useful query-conditioned predecessor discrimination under the existing readout here.

Full row norms, cosine/distance distributions and all1920 pairwise representation/projection records are retained. No inference is made from a difference of norms as if it were a tensor distance. `analytical_readout_tensors_v1.pt` stores the actual analytical Q/K, score, probability, common and interaction arrays, and is **not a model checkpoint**.

## Evidence classification, limitations and recommendation

Direct observations are the saved raw/committed tensors, official targets and checkpoint weights. Analytical reconstructions are the four fixed readouts, scalar, geometries, scores and decompositions. No omitted activation is manufactured. No model or AGT execution module is imported. No development/transfer inputs or targets are used for fitting or selection.

This does not prove that radial information is intrinsically useless, that another trained readout could never exploit it, or that RMSNorm was harmless throughout training. Neither an alternative persistent trajectory nor parameter optimization is evaluated. One array/checkpoint provides no generalization estimate, independent replication or historical causal chronology. A small raw rank gain through a different global preference is not the predeclared favorable outcome of task-aligned query-dependent recovery. Direction-only's lower CE is not an accuracy gain.

**One recommended next step:** decline a raw-state bypass on this evidence; require demonstrated query-dependent target discrimination as the prerequisite for any separately authorized upstream representation/readout study. Do not restore raw activations merely because their norms are larger. No next experiment or architecture change is initiated here.

## Preservation and publication

The analysis runs on CPU only with CUDA hidden and autograd disabled. Execution counters are all0: model constructions, forwards, backwards, optimizer objects/updates, tests, profiling and GPU operations. The checkpoint is deserialized as CPU parameter tensors only, never instantiated as a model.

`integrity_initial_v1.json` fingerprints2516 pre-existing files, including the completed negative training result and every prior capture/audit. Final integrity and publication receipts are separate files. Original HEAD, tracked diff, source, checkpoint and historical bytecode exceptions are preserved. All new artifacts remain confined to this new analysis directory.
'''
    with (OUT/'report_v1.md').open('x') as f:f.write(text)
    write('unavailable_evidence_v1.json',dict(not_available_or_not_evaluated=['Raw-state recurrent trajectory if normalization were removed','Fitted readout exploiting radius','Other arrays, independently trained models or intermediate checkpoints','Task-specific causal intervention or historical onset'],no_missing_tensor_substituted=True,one_array_one_checkpoint=True,interpretation='No recommendation to bypass RMSNorm from this analysis.'))
    print('Report written from retained mathematical results only.')

if __name__=='__main__':main()
