"""Descriptive presentation of already captured evidence; no model/tensor execution."""
from pathlib import Path
import json,math,hashlib
OUT=Path(__file__).resolve().parent
ENG=OUT.parent/'gatec_clrs_insertion_sort_6200_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (OUT/'mechanistic_interpretation_v1.md').exists()
    geometry=json.loads((OUT/'mechanistic_summary_v1.json').read_text());metrics=json.loads((OUT/'evaluation_summaries_v1.json').read_text())
    old=json.loads((ENG/'training_results_v1/training_summary_v1.json').read_text())['final_model_parameter_sha256'];control=json.loads((OUT/'C_training_summary_v1.json').read_text())['final_model_identity']
    records={};inputs={};out=[]
    for arm in ['C','T']:
        records[arm]={}
        for panel in ['training_subset','development','size_transfer']:
            p=OUT/(arm+'_'+panel+'_records_v1.jsonl');inputs[p.name]=sha(p);rows=[json.loads(x) for x in p.read_text().splitlines()]
            keys=['pair_comparisons','interaction_improves','interaction_worsens','common_positive','actual_positive','flips_toward_target','flips_against_target']
            stats={k:sum(r['interaction'][k] for r in rows) for k in keys}
            stats.update(arrays=len(rows),native_common_predictions_identical_arrays=sum(r['conditions']['native']['predictions']==r['conditions']['query_independent']['predictions'] for r in rows),native_single_candidate_arrays=sum(r['conditions']['native']['distinct_predicted_candidates']==1 for r in rows),mean_common_component_norm=sum(r['interaction']['common_component_norm'] for r in rows)/len(rows),mean_interaction_norm=sum(r['interaction']['interaction_norm'] for r in rows)/len(rows))
            records[arm][panel]=stats
    data=dict(description_only=True,no_new_statistics_or_acceptance_criteria=True,control_historical_parameter_buffer_identity_match=control==old,control_identity=control,historical_identity=old,score_summaries=records,input_hashes=inputs,classification_unchanged=json.loads((OUT/'final_results_v1.json').read_text())['classification'])
    with (OUT/'descriptive_verification_v1.json').open('x') as f:json.dump(data,f,indent=2);f.write('\n')
    out=['# Retained-evidence interpretation: feature-complement pilot','',
        'These are descriptive summaries of the already retained prescribed evaluations. No model, tensor forward/backward, fit, test, sample or bootstrap is run here. No acceptance threshold or classification is changed.','',
        f"The control final parameter/buffer identity is `{control}`, exactly matching the historical failed training identity `{old}`. All three control-panel native metrics also reproduce the archived reported values. This is a final control reconstruction check, not initialization from the failed checkpoint. The new serialized checkpoint SHA differs because its metadata and archive names differ.",'',
        '## Main task finding','',
        'The locked result is PARTIAL_EFFECT_TASK_NON_SUCCESS. Treatment development accuracy10.9375% versus control3.125% gives+7.8125 percentage points, with adjusted lower bound+6.4404296875 points. Treatment minus its own query-independent component has point0 and lower bound0: no resolved query-conditioned advantage. Both arms have0% exact-array accuracy on all panels and fail all four competence thresholds. A zero-width observed bound is not a universal equivalence guarantee.','',
        '|Panel|Arm|Native single-candidate arrays|Native/common identical-prediction arrays|Mean common score norm|Mean interaction norm|','|---|---|---:|---:|---:|---:|']
    for panel in ['training_subset','development','size_transfer']:
        for arm in ['C','T']:
            s=records[arm][panel];out.append(f"|{panel}|{arm}|{s['native_single_candidate_arrays']}/{s['arrays']}|{s['native_common_predictions_identical_arrays']}/{s['arrays']}|{s['mean_common_component_norm']:.10g}|{s['mean_interaction_norm']:.10g}|")
    out+=['','The native predictions remain concentrated on one candidate per array. Interaction can change individual scores/margins without changing the selected candidate. Native/common accuracy equality is an observed finite-panel result; it does not imply tensor/state equality or absence of information in every persistent channel.','',
        '## Terminal geometry by role','',
        '|Panel|Role|Arm|Pre-norm centered energy|Committed centered energy|Pre-norm mean norm|Committed mean norm|Pre-norm angular dispersion|Committed angular dispersion|','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for panel in ['training_subset','development','size_transfer']:
        for role in ['candidates','queries']:
            for arm in ['C','T']:
                v=geometry[arm][panel][role]['terminal'];out.append(f"|{panel}|{role}|{arm}|{v['mean_input_centered_energy']:.10g}|{v['mean_output_centered_energy']:.10g}|{v['mean_input_norm']:.10g}|{v['mean_output_norm']:.10g}|{v['mean_input_angular_dispersion']:.10g}|{v['mean_output_angular_dispersion']:.10g}|")
    out+=['','On development, treatment committed candidate centered energy is0.4481071202 versus control0.001813061384; query energy is0.06147524211 versus0.0001104346787. Committed mean angular dispersion is4.656011570e-4 versus2.036306428e-6 for candidates, and6.384009605e-5 versus1.240311372e-7 for queries. The terminal representations show substantially more absolute positional variation and directional dispersion in treatment on this panel. This did not establish useful query-conditioned task computation or competence. No binary collapse cutoff is invented.','',
        '## Observed difference transport at each development mixer','',
        'The entries below are RMS amplification of actual corresponding position differences: sqrt(mean output pair-distance-squared / mean input pair-distance-squared), with equal weighting of development arrays within each role. This is not a worst-case singular value and not the mean of individual pair ratios. Absolute input/output centered energies are retained in mechanistic_summary_v1.json, and each array retains full transport, cosine, radial/angular and whole-block residual summaries.','',
        '|Block|Role|C observed RMS amplification|T observed RMS amplification|','|---|---|---:|---:|']
    for i in range(8):
        for role in ['candidates','queries']:
            values=[]
            for arm in ['C','T']:
                v=geometry[arm]['development'][role]['mixers'][str(i)];values.append(math.sqrt(v['mean_output_pair_difference_squared']/v['mean_input_pair_difference_squared']))
            out.append(f"|{i}|{role}|{values[0]:.10g}|{values[1]:.10g}|")
    out+=['','This compares two separately learned final arms. The mixer inputs, gains, constrained matrices and other learned parameters differ after training, so it is not a fixed-weight, same-input attribution to one module. Some treatment mixers preserve/amplify actual differences and others reduce them; the complete raw tensor records permit stage-accurate reconstruction. Whole-block residual identities were checked exactly. There is no inversion, omitted-activation reconstruction or claim that one mixer uniquely caused historical failure.','',
        'The task-level conclusion remains non-success despite a supported partial accuracy effect and changed geometry. Under the locked stop rule this supplies no authorization to strengthen the complement, alter normalization/interface, extend training or run another experiment. Await a new research decision.','']
    with (OUT/'mechanistic_interpretation_v1.md').open('x') as f:f.write('\n'.join(out))
    print(json.dumps(dict(classification=data['classification_unchanged'],historical_control_identity_match=control==old,prediction_equivalence={a:{s:v['native_common_predictions_identical_arrays'] for s,v in b.items()} for a,b in records.items()},no_model_calls=True),indent=2))
if __name__=='__main__':main()
