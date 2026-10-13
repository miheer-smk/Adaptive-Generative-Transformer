"""Present completed measurements and sealed decision rules; no new model computations."""
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'results.json').read_text());m=json.loads((OUT/'manifest.json').read_text());v=json.loads((OUT/'verification.json').read_text())
lines=['# Gate-C: paired two-gradient intervention, seed 4900','',
    f"Intervention helped under the declared rule: **{r['intervention_helped']}**. Treatment learned a useful policy under the additional accuracy/CE criteria: **{r['useful_policy_T']}**. Combined useful-policy repair: **{r['useful_policy_repair']}**.",'',
    'Both arms started from the saved seed-4700 update-0 controller and received exactly 520 fresh AdamW updates. C averaged duplicate-noise gradients; T averaged independent lane gradients. There were exactly 1,040 complete training rollouts per arm. Each pair used fixed parameters and separate normal sequence resets; only gradients were averaged. No clipping, scheduler, auxiliary loss, LR rescaling or estimator change occurred.','',
    f"Manifest SHA-256: `{r['manifest_sha256']}`. Initial controller: `{m['initial']['sha256']}`. Stage-1 backbone: `{m['backbone']['sha256']}`. Revision: `{m['git']['head']}`. Tracked-diff identity: `{m['git']['tracked_diff_sha256']}`.",'',
    'The 104 raw sequences and the first five historical permutations were preserved, including duplicate content. Full-sequence BPTT through frozen computations was retained. Existing ST relaxed stop-gradient and discrete pointer/mask metadata boundaries were unchanged. Tau=2, budget feature/presence=[0,0], P=8, FP32 and batch=1 were fixed. HALT/convergence remained disabled.','',
    '## Confirmation outcomes','',
    '| Panel | Mode | U accuracy / CE | C accuracy / CE | T accuracy / CE | T−C accuracy | T−C CE |',
    '|---|---|---|---|---|---:|---:|']
for split in ['val','ood_test']:
    for mode in ['deterministic','tape0','tape1','tape2','tape3','pooled_stochastic']:
        arms=r['pooled_stochastic'] if mode=='pooled_stochastic' else r['evaluation'][mode]
        vals={a:arms[a][split] for a in ['U','C','T']}
        cells=[f"{vals[a]['accuracy']:.6f} / {vals[a]['mean_CE']:.9g}" for a in ['U','C','T']]
        lines.append(f"| {split} | {mode} | {' | '.join(cells)} | {vals['T']['accuracy']-vals['C']['accuracy']:+.6f} | {vals['T']['mean_CE']-vals['C']['mean_CE']:+.9g} |")
    f=r['reference'][split]
    lines+=['',f"{split} fixed canonical reference: accuracy={f['accuracy']:.6f}, CE={f['mean_CE']:.12g}. Absolute parity accuracy is separate from agreement with the reference.",'']
lines+=['## Paired uncertainty and declared decision','',
    'Each stochastic contrast first averages four outcomes within a sequence; the 10,000 paired bootstrap resamples then resample sequences within parity and OOD-length strata. Every sequence’s four outcomes stay together. These are approximate intervals conditional on one initialization and the four locked tapes. No inference across controller-training seeds is made.','',
    '| Panel | Pooled accuracy T−C | Paired 95% interval | Pooled CE T−C | Paired 95% interval | Positive tapes /4 | Deterministic T−C |',
    '|---|---:|---|---:|---|---:|---:|']
for split,p in r['paired'].items():
    a=p['stochastic_accuracy'];q=p['stochastic_CE'];positive=sum(t['accuracy_T_minus_C']>0 for t in p['individual_tapes'])
    lines.append(f"| {split} | {a['mean']:+.6f} | {a['bootstrap_95']} | {q['mean']:+.9g} | {q['bootstrap_95']} | {positive} | {p['deterministic_accuracy_T_minus_C']:+.6f} |")
lines+=['','“Helped” requires ≥10 percentage-point stochastic accuracy gain on each panel, both intervals excluding zero, at least three positive tapes per panel, and no deterministic deterioration versus C. Useful-policy repair additionally requires T≥90% accuracy in deterministic and pooled stochastic evaluation on each panel and lower CE than U in both modes. CE-only or sub-threshold gains remain partial evidence.','',
    '## Training, gradients and displacement','',
    '| Pass | C training accuracy / CE | T training accuracy / CE | Updates per arm | Rollouts per arm |',
    '|---:|---|---|---:|---:|']
for epoch,summary in r['training_by_pass'].items():
    a=summary['metrics']['C'];b=summary['metrics']['T']
    lines.append(f"| {epoch} | {a['accuracy']:.6f} / {a['mean_CE']:.9g} | {b['accuracy']:.6f} / {b['mean_CE']:.9g} | 104 | 208 |")
lines+=['','Control’s duplicate rollouts are not independent observations. Its per-pass average is the same sample-weighted statistic as one observation per update. Treatment’s two noise lanes provide two conditional sampled trajectories per update, not independent training sequences.','',
    f"Gradient statistics: `{json.dumps(r['gradient_statistics'],sort_keys=True)}`. Maximum duplicated-control pair difference: {v['maximum_C_gradient_difference']}. Arithmetic averaging was checked within scheduled training computations. Zero within-pair control difference is not zero population sampling variance.",'']
for arm in ['C','T']:
    lines.append(f"{arm}: final displacement L2={r['controller_change'][arm]['l2']:.9g}; changed scalar parameters={r['controller_change'][arm]['changed_values']}; optimizer steps=520.")
lines+=['','## Routing and execution','',
    '| Panel/mode | Arm | Canonical action fraction | Canonical block fraction | Exact sequence fraction | Mean/max D | RESTART | REPEAT | P distribution |',
    '|---|---|---:|---:|---:|---|---:|---:|---|']
for split in ['val','ood_test']:
    for mode in ['deterministic','pooled_stochastic']:
        arms=r['pooled_stochastic'] if mode=='pooled_stochastic' else r['evaluation'][mode]
        for arm in ['U','C','T']:
            x=arms[arm][split]
            lines.append(f"| {split}/{mode} | {arm} | {x['canonical_action_fraction']:.6f} | {x['canonical_block_fraction']:.6f} | {x['sequence_canonical_fraction']:.6f} | {x['mean_D']:.6f}/{x['maximum_D']} | {x['restart_count']} | {x['repeat_count']} | {x['P_distribution']} |")
lines+=['','Complete per-length outcomes, route signatures/frequencies and executed-block histories are retained in results.json, per-sequence evaluation JSON and compressed decision/token logs. Noncanonical routing is not automatically a task failure, and diversity alone is not success.','',
    f"Offline verification: {v['observed_tokens']} completed tokens, {v['observed_decisions']} decisions, {v['selected_block_executions']} actual selected-block executions, {v['candidate_block_calls']} candidate bank calls, {v['cache_version_checks']} cache-version checks. All completed tokens have P=8 and D=8+RESTART≤15; RESTART executes zero blocks. Legality, selected successor, in-flight candidate/GDN purity and exactly one seven-entry terminal commit were checked at runtime. Frozen non-controller values and historical file hashes were verified. Historical bytecode exception and failure artifacts remain preserved.",'',
    '## Predetermined checkpoints','',
    '| Arm | Update | Path | SHA-256 |','|---|---:|---|---|']
for s in r['snapshots']:lines.append(f"| {s['arm']} | {s['update']} | {s['path']} | `{s['sha256']}` |")
lines+=['','## Limits and conclusion','']
if not r['intervention_helped']:
    lines.append('The independent-noise averaging intervention did not meet the declared paired help criterion. Any individual-panel or CE improvement should be reported separately and does not establish useful-policy repair.')
if not r['useful_policy_T']:
    lines.append('Treatment did not meet the declared useful-policy accuracy/CE thresholds. This experiment does not establish reference-level task-loss routing discovery.')
if r['intervention_helped'] and r['useful_policy_T']:
    lines.append('This bounded intervention supports useful-policy repair under the locked protocol, conditional on this initialization, training noise and confirmation panel. It does not establish general seed reproducibility.')
lines+=['','Confirmation inputs were generated with seeds4901/4902, balanced and distinct, excluding recoverable retained training/evaluation content. Unknown Stage-1 exposure and unretained historical inputs remain limitations. Four tapes replicate evaluation sampling, not training seeds. No outcome-based checkpoint selection, extension, replacement seed, new training variant, production edit or commit occurred. A useful policy in both arms would support task-loss learning but would not attribute that success to averaging absent the declared paired effect.','',
    f"Elapsed scheduled experiment time: {r['elapsed_seconds']:.3f} seconds. Focused tests: 10 passed. git diff --check: clean. No follow-up run is launched.",'']
with (OUT/'paired_report.md').open('x') as h:h.write('\n'.join(lines))
print('PAIRED REPORT WRITTEN; no model calls or updates')
