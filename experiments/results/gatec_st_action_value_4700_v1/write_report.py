"""Report completed artifacts only; no model calls or diagnostic resampling."""
import sys
sys.dont_write_bytecode=True
import json, hashlib, statistics
from collections import Counter,defaultdict
from pathlib import Path
OUT=Path(__file__).resolve().parent

def main():
    m=json.loads((OUT/'manifest.json').read_text());r=json.loads((OUT/'results.json').read_text())
    v=json.loads((OUT/'verification.json').read_text());ps=r['probes']
    assert not (OUT/'failure.json').exists()
    classes=Counter(p['classification'] for p in ps);directions=Counter(p['direction'] for p in ps)
    correct_flat=sum(all(len(set(row))==1 for row in p['CF_correct']) for p in ps)
    means_flat=sum(len(set(p['conditional_correctness']))==1 for p in ps)
    resolved_pairs=sum(sum(x['resolved'] for x in p['paired_action_loss_differences']) for p in ps)
    opposed=classes['informative_opposed_ST'];aligned=classes['informative_aligned_ST']
    if opposed==0:
        verdict='The proposed directional-failure mechanism is not supported at the tested states. Resolved comparisons favor alignment; the remaining comparisons are unresolved. This contradicts a blanket claim of consistently opposed local ST credit at these probes, while leaving a broader or rarer failure mechanism unresolved.'
    elif aligned:
        verdict='Mixed local evidence: resolved opposition supports the hypothesis at the identified probes, while resolved alignment contradicts a uniform directional-failure mechanism. The experiment does not establish that opposition caused the historical training outcome.'
    else:
        verdict='Resolved opposition supports the hypothesis locally at the identified probes. Unresolved probes and the restricted state sample prevent a general mechanism or historical causal claim.'
    groups=defaultdict(list)
    for p in ps:
        s=p['probe'];groups[(s['checkpoint_update'],s['source'],s['condition'])].append(p)
    shift=Counter()
    for p in ps:
        lo,hi=p['checkpoint_probability_shift']['bootstrap_95']
        shift['lower_Q' if hi<0 else 'higher_Q' if lo>0 else 'unresolved']+=1
    earlylate={}
    for cp in [0,1040]:
        for condition in ['A','B','C']:
            sub=[p for p in ps if p['probe']['checkpoint_update']==cp and p['probe']['condition']==condition]
            norms=[p['ST_norm'] for p in sub if p.get('ST_norm') is not None]
            earlylate[f'{cp}/{condition}']=dict(ST_norm_min=min(norms),ST_norm_median=statistics.median(norms),ST_norm_max=max(norms),
                CF_norm_min=min(p['g_CF_norm'] for p in sub),CF_norm_max=max(p['g_CF_norm'] for p in sub))
    prefix=[]
    for cp in [0,1040]:
        for sid in [s['sequence_id'] for s in m['six_sequences']]:
            b=next(p for p in ps if p['probe']['checkpoint_update']==cp and p['probe']['sequence_id']==sid and p['probe']['condition']=='B')
            c=next(p for p in ps if p['probe']['checkpoint_update']==cp and p['probe']['sequence_id']==sid and p['probe']['condition']=='C')
            prefix.append(dict(checkpoint=cp,sequence_id=sid,same_incoming_state=b['incoming_state']['sha256']==c['incoming_state']['sha256'],
                B_class=b['classification'],C_class=c['classification'],B_ST_norm=b.get('ST_norm'),C_ST_norm=c.get('ST_norm'),
                B_Q_range=[min(b['Q']),max(b['Q'])],C_Q_range=[min(c['Q']),max(c['Q'])]))
    status=f'unresolved across all tested states; {aligned} resolved comparisons contradict local directional opposition, {opposed} support, {36-aligned-opposed} unresolved'
    summary=dict(hypothesis_verdict=verdict,hypothesis_status=status,classifications=dict(classes),directions=dict(directions),
        informative_probes=sum(p['informative_action_values'] for p in ps),resolved_paired_action_differences=resolved_pairs,
        all_action_conditional_accuracy_means_equal_probes=means_flat,all_action_correctness_equal_on_every_paired_tape_probes=correct_flat,
        gradient_norm_by_checkpoint_and_condition=earlylate,identical_state_probability_shift_intervals=dict(shift),
        prefix_condition_comparisons=prefix,mechanical_verification=v,
        interpretation_limit='Local final-CE gradient benchmark with fixed incoming state and downstream policy; not a population estimate, full optimizer-gradient decomposition, or proof of the cause of task-loss collapse.')
    with (OUT/'interpretation.json').open('x') as f:json.dump(summary,f,indent=2,allow_nan=False);f.write('\n')
    def fmt(x): return 'disconnected' if x is None else f'{x:.6g}'
    lines=[
        '# AGT-v3 seed4700: local ST credit versus hard closed-loop action values',
        '',verdict,'',
        f'Overall hypothesis verdict: **unresolved across the full tested set**. {aligned} resolved comparisons contradict local directional opposition; {opposed} support it. The other {36-aligned-opposed} states cannot resolve that hypothesis under the sealed uncertainty rule.','',
        f'Completed the sealed budget exactly: **36 states, 2,304 counterfactual continuations, 288 unforced ST gradient probes**. {aligned} informative probes have resolved alignment and {opposed} have resolved opposition. Classifications: `{dict(classes)}`.',
        '',
        '## Protocol and identity',
        '',
        f"Git HEAD: `{m['git']['head']}`; branch `{m['git']['branch']}`. Manifest SHA-256: `{r['manifest_sha256']}`. Tracked-diff SHA-256: `{m['git']['tracked_diff_sha256']}`. The manifest was saved before any counterfactual or ST probe.",
        '',
        f"Backbone: `{m['backbone']['path']}`, SHA-256 `{m['backbone']['sha256']}`. Only the controller state from the two fixed snapshots was loaded; all non-controller weights remained identical and frozen.",
        '',
        '| Snapshot | Checkpoint | SHA-256 |','|---|---|---|',
    ]
    for cp,c in m['checkpoints'].items():lines.append(f"| {cp} | {c['path']} | {c['sha256']} |")
    lines.extend(['','No optimizer was constructed. No parameters or buffers changed. No training, checkpoint overwrite, production edit, additional samples or commit occurred. Bytecode writing was disabled.',
        '',
        'The parent experiment preserved a bytecode-only file-identity exception. Its original failure record remains intact. This diagnostic verified the recorded post-exception cache/diff identities and all experiment-relevant sources, inputs, noise, backbone and selected checkpoints. This exception was disclosed, not repaired or hidden.',
        '',
        '| Locked sequence | Source | Parity | Bits |','|---|---|---|---|'])
    for s in m['six_sequences']:lines.append(f"| {s['sequence_id']} | {s['source']} | {s['label']} | {''.join(map(str,s['bits']))} |")
    lines.extend(['',
        'These sequences were selected by manifest order, not outcomes. Training inputs were used during the historical ten-pass run. The four evaluation inputs exclude exact training membership but were previously evaluated. Stage-1 training membership is unknown; this is not an untouched generalization assessment. Sequence/label hashes and row identities are in the manifest.',
        '',
        '## Prefixes and sealed randomness',
        '',
        'A is human token ordinal 1, internal index 0, with zero preceding commits. B is the final token (internal L−1) after the checkpoint’s own sampled prefix. C is the final token after the fixed canonical prefix; canonical forcing ends at that boundary. Every probe starts after normal memory read/Prelude initialization, before decision zero, with P=D=0 and exactly A0…A7 legal successor choices. The source mask still contains its HALT slot; fixed-depth successor routing excludes it unchanged. C continuations follow the checkpoint, not a canonical suffix.',
        '',
        'Historical tape 0 was reconstructed from CPU generator seed 924088217, in its original 96-sequence order. Every retained tensor and sequence-coordinate association was bitwise identical. The same generator was then continued to append train/L16/row0 and train/L16/row1 in training-manifest order. The extension is separate and supplies training B-prefix randomness only. Historical tensors were neither modified nor relabeled. All eight available historical evaluation B-prefix incoming hashes matched exactly.',
        '',
        'The CF bank has eight tapes from seed 4800; the independent ST bank has eight tapes from seed 4801. Both were precomputed and sealed before measurement. Keys are bank/tape/raw-sequence/token/decision/action; checkpoint, branch, condition and outcomes are excluded. Differing route lengths therefore do not shift unrelated coordinates. Bootstrap indices (2,000 replicates/state, seed 4802) were also sealed. Tensor hashes and exact generation order are in manifest.json.',
        '',
        '## Computation and comparison',
        '',
        'Each incoming state was independently cloned for every branch. The CF arm forces only its first legal ADVANCE successor after the unchanged exhaustive candidate construction. Every subsequent action is sampled autonomously under the owner checkpoint until the original final-sequence parity CE. The ST arm is entirely unforced and retains the original probe logits for autograd.grad of that same final CE.',
        '',
        'Runtime configuration: FP32, batch 1, CUDA GB10, tau=2, zero budget and zero presence, eight actual executions/token, HALT/convergence disabled. Persistent states evolve normally across suffix tokens with one commit/token. Frozen computations remain differentiable in ST. The fixed incoming prefix state is deliberate conditioning; there is no additional suffix detach. The historical full-sequence BPTT policy, existing ST stop-gradient and hard metadata transitions are unchanged. GQA/SWA/GDN context horizons are those recorded in the parent manifest.',
        '',
        'The hard categorical law is pi=softmax(raw legal logits); tau affects the relaxed derivatives, not the argmax categorical law. The implementation uses finite-precision, epsilon-clamped uniforms; pi is the nominal Gumbel categorical law. Q(a) is mean final CE across the eight actual hard continuations. The independent local-logit benchmark is g_CF[a]=pi[a]*(Q(a)−sum_b pi[b]Q(b)). It was checked against differentiation of the explicit categorical expected loss with Q held constant at every state.',
        '',
        'A positive inner product means the negative ST gradient locally reduces the benchmark expected CE; negative means opposition. Resolution requires both four-tape halves to share the sign and the independently resampled full-bank 95% interval to exclude zero, with meaningful norms >1e−8. CF action pairing is retained. ST and CF banks are independently resampled. Intervals describe conditional Monte Carlo uncertainty, not controller-seed or population uncertainty. Pairwise action intervals and per-probe directional intervals are unadjusted for multiplicity; they are not simultaneous confidence bands across all probes.',
        '',
        '## All 36 probes',
        '',
        'Accuracy range below is min/max conditional correctness across eight action means. All eight Q values, 28 paired action differences, per-tape outcomes, 11-component ST vectors, component variability, pi, entropy, half-bank signs and probability-shift estimates are retained in probeNN.json and results.json.',
        '',
        '| Update / sequence / prefix | Q min…max | Correctness min…max | g_CF L2 | Mean ST L2 | Dot | Dot 95% interval | Direction |',
        '|---|---:|---:|---:|---:|---:|---|---|'])
    for p in ps:
        s=p['probe'];ci=p['inner_product_bootstrap_95']
        lines.append(f"| {s['checkpoint_update']} / {s['sequence_id']} / {s['condition']} | {fmt(min(p['Q']))}…{fmt(max(p['Q']))} | {fmt(min(p['conditional_correctness']))}…{fmt(max(p['conditional_correctness']))} | {fmt(p['g_CF_norm'])} | {fmt(p.get('ST_norm'))} | {fmt(p['inner_product'])} | {ci} | {p['direction']} |")
    lines.extend(['','## Stratified results','','Each stratum contains the two preselected parity inputs. No stratum was selected after outcomes.','',
        '| Update | Source | Prefix | Aligned | Opposed | Other classifications |','|---|---|---|---:|---:|---|'])
    for (cp,source,condition),items in sorted(groups.items()):
        c=Counter(p['classification'] for p in items)
        other={k:n for k,n in c.items() if k not in ['informative_aligned_ST','informative_opposed_ST']}
        lines.append(f"| {cp} | {source} | {condition} | {c['informative_aligned_ST']} | {c['informative_opposed_ST']} | {other} |")
    lines.extend(['','## Temporal credit, prefix sensitivity and task meaning','',
        f"All {v['connected_gradient_probes']} ST probes were connected; {v['disconnected_gradient_probes']} were disconnected, {v['exact_zero_gradient_probes']} were exactly zero, and {v['small_connected_gradient_probes']} had connected L2≤1e−8. HALT and the initially illegal REPEAT/RESTART slots were experimentally zero in every connected probe gradient.",
        '',
        '| Update / prefix | Minimum ST norm | Median ST norm | Maximum ST norm |','|---|---:|---:|---:|'])
    for key,stats in earlylate.items():lines.append(f"| {key} | {fmt(stats['ST_norm_min'])} | {fmt(stats['ST_norm_median'])} | {fmt(stats['ST_norm_max'])} |")
    lines.extend(['',
        'Early and final-token gradients differ in magnitude, but connectivity alone cannot establish useful credit or explain optimizer failure. No early/late difference is labeled a truncation failure: the measured paths remain connected and the saved suffix graph has no additional token-state detach.',
        '',
        f"{sum(p['informative_action_values'] for p in ps)}/36 states have at least one resolved paired CE difference ({resolved_pairs}/1008 paired comparisons). {means_flat}/36 have identical action-conditional accuracy means; {correct_flat}/36 have identical correctness across all eight actions on every paired tape. Thus measurable loss sensitivity can primarily reflect calibration, rather than a change in parity correctness. Such CE effects are not evidence of useful task discovery.",
        '',
        f"Probe entropy ranges {fmt(min(p['entropy'] for p in ps))}…{fmt(max(p['entropy'] for p in ps))}; maximum action probability ranges {fmt(min(p['max_probability'] for p in ps))}…{fmt(max(p['max_probability'] for p in ps))}. Saturation flags (max pi≥0.99): {sum(p['saturated_probability'] for p in ps)}/36.",
        '',
        'B-versus-C comparisons use different incoming histories. Their local values and gradients describe prefix-conditioned sensitivity; they are not a decomposition of historical optimizer effects. The complete paired comparisons and incoming-state identities are in interpretation.json.',
        '',
        f"At each identical frozen state, sum_a(pi1040−pi0)Q_owner was evaluated with downstream policy fixed to the indicated owner checkpoint. Conditional intervals classify {dict(shift)}. Negative means the local probability shift lowers that owner-policy Q, positive raises it. This omits changes in upstream state, downstream policy and shared parameters, so it cannot identify the complete historical update effect.",
        '',
        '## Invariants and validation','',
        f"Runtime assertions and independent offline trace checks verified {v['counts']['completed_continuation_tokens']} completed suffix tokens, P=8, D=8+R∈[8,15], legal actions, selected-successor equality, zero-compute RESTART, all recorded continuations, state/cache versions and one seven-entry commit/token. There were {v['counts']['continuation_decisions']} continuation decisions, {v['counts']['forced_decisions']} single CF interventions and {v['counts']['unforced_noise_reconstructions']} independently reconstructed unforced decisions.",
        '',
        f"The original exhaustive training path makes nine physical speculative bank calls per decision (eight ADVANCE candidates plus the current-block REPEAT candidate). These total {v['physical_candidate_bank_calls']}; they are separate from eight selected executions/token. GDN purity checks: {v['GDN_purity_checks']}. Shared candidate inputs, committed GDN state/convolution buffers, memory/context and the Prelude anchor remained unchanged in flight; terminal updates were checked against the commit plan. No mechanical or nonfinite violation occurred.",
        '',
        'Focused validation: **25 passed** (new artifact-independent diagnostic helper tests, paired-noise tests, routing tests and successor-mixing tests). Compilation wrote bytecode only to /tmp. No new full-regression claim is made. Offline checks reran no model: they verified all counts, states, hashes, logical noise/action reconstructions, categorical benchmark and sealed bootstrap calculations. git diff --check is clean; the pre-existing tracked diff is unchanged.',
        '',
        '## Limits and conclusion','',verdict,'',
        'The experiment conditions on six previously selected inputs, two snapshots of one training seed, first-decision states and eight tapes/bank. It does not cover later within-token decisions or all reachable states. Unresolved values are not proof that useful gradients exist nowhere. Resolved alignment does not guarantee optimizer success; resolved opposition, if present, is local evidence rather than proof of the cause of historical collapse. No canonical action was used as an action-value correctness label.',
        '',
        'This does not establish task-loss route discovery, useful topology/diversity, adaptive depth, learned HALT, general trainability or superiority. No fix or training run is proposed or launched. The diagnostic stops at measurement.',
        '',
        '## Artifacts','',
        '- manifest.json: sealed definitions, identities, six inputs and randomness recipes.',
        '- results.json and probe00.json…probe35.json: complete action-value and ST comparisons.',
        '- decisions/tokens/runs/prefixes.jsonl.gz: routes, logits, masks, noise keys, state summaries, gradients and outcome records.',
        '- states/probe00.pt…probe35.pt: full frozen incoming states and initialized controller features.',
        '- prefix_noise_extension.pt, continuation_noise.pt, ST_probe_noise.pt, bootstrap_indices.pt: separately sealed randomness.',
        '- verification.json and interpretation.json: independent offline checks and stratified interpretation.',
        '- analyze_artifacts.py and write_report.py: offline processing only; no model execution.',
    ])
    with (OUT/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    index={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(OUT.rglob('*')) if p.is_file()}
    with (OUT/'artifact_index.json').open('x') as f:json.dump(dict(files=index),f,indent=2);f.write('\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
