"""Verify retained-record arithmetic and write this analysis's report; no model/RNG/autograd."""
import sys
sys.dont_write_bytecode = True
import csv, hashlib, json, math, subprocess
from collections import Counter
from pathlib import Path
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, obj):
    with Path(path).open('x') as h: json.dump(obj, h, indent=2, allow_nan=False); h.write('\n')
def tensor(x): return torch.tensor(x, dtype=torch.float64)
def close(a, b): torch.testing.assert_close(tensor(a), tensor(b), atol=1e-12, rtol=1e-10)
def forbidden(*args, **kwargs): raise RuntimeError('Record analysis cannot execute a model, backward, or new RNG')
torch.set_grad_enabled(False)
torch.autograd.grad = torch.autograd.backward = torch.Tensor.backward = forbidden
torch.nn.Module._call_impl = forbidden
for name in ['rand', 'rand_like', 'randn', 'randn_like', 'randint', 'randperm', 'multinomial']:
    setattr(torch, name, forbidden)
torch.set_num_threads(4)
m = read(OUT/'manifest.json'); r = read(OUT/'results.json')
assert digest(OUT/'manifest.json') == r['manifest_sha256']
assert digest(ROOT/'experiments/analyze_gatec_matched_prefix_4700.py') == m['script_sha256']
for path, expected in m['sources'].items(): assert digest(path) == expected, path
assert hashlib.sha256(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT)).hexdigest() == m['tracked_diff_sha256']
assert subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True).returncode == 0
bs = torch.load(ROOT/'experiments/results/gatec_st_action_value_4700_v1/bootstrap_indices.pt', weights_only=True, map_location='cpu')
assert bs['seed'] == 4802
ci, si = bs['CF'][1], bs['ST'][1]
assert ci.shape == si.shape == (2000, 8)
states = {}; arrays = {}
def check_summary(x, saved):
    x = tensor(x); close(x.mean(), saved['mean']); close(x, saved['tape_values'])
    halves = [x[:4].mean().item(), x[4:].mean().item()]
    interval = torch.quantile(x[ci].mean(1), tensor([.025, .975])).tolist()
    close(halves, saved['half_means']); close(interval, saved['bootstrap_95'])
    sign = 'positive' if min(halves)>0 and interval[0]>0 else 'negative' if max(halves)<0 and interval[1]<0 else 'unresolved'
    assert sign == saved['resolved_sign']
def check_credit(value, halves, boot, saved):
    close(value, saved['value']); close(halves, saved['half_values'])
    interval = torch.quantile(boot, tensor([.025, .975])).tolist()
    close(interval, saved['bootstrap_95'])
    sign = 'positive' if min(halves)>0 and interval[0]>0 else 'negative' if max(halves)<0 and interval[1]<0 else 'unresolved'
    assert sign == saved['resolved_sign']

for s in r['states']:
    spec = s['probe']; key = (spec['checkpoint_update'], spec['sequence_id'], spec['condition'])
    assert spec['condition'] in ['B', 'C'] and spec['token_index'] in [15, 31]
    assert s['legal_successor_actions'] == list(range(8)) and len(s['actions']) == 8
    pi = tensor([a['probability'] for a in s['actions']]); lp = tensor([a['log_probability'] for a in s['actions']])
    close(pi, tensor(s['root_logits'][:8]).softmax(-1)); close(lp.exp(), pi)
    assert pi.gt(0).all(); close(pi.sum(), 1)
    Q = tensor([a['CE_by_CF_tape'] for a in s['actions']]).T
    R = tensor([a['correct_by_CF_tape'] for a in s['actions']]).T
    G = tensor([a['full_gradient_by_ST_tape'] for a in s['actions']]).T
    Z = tensor([a['relaxed_weight_by_ST_tape'] for a in s['actions']]).T
    assert Q.shape == R.shape == G.shape == Z.shape == (8, 8)
    assert ((R == 0) | (R == 1)).all()
    close(Q.mean(0), [a['mean_CE'] for a in s['actions']]); close(R.mean(0), [a['mean_correctness'] for a in s['actions']])
    close(G.mean(0), s['gradient_mean'])
    # FP32 softmax sums may differ from one by several ulps; use sampler tolerance here.
    torch.testing.assert_close(Z.sum(1), torch.ones(8,dtype=torch.float64), atol=1e-6, rtol=1e-5)
    for k, row in enumerate(s['ST_records']):
        assert row['tape'] == k and row['noise_coordinate'] == [4801, spec['sequence_id'], spec['token_index'], 0]
        close(G[k], row['full_logit_gradient'][:8]); close(Z[k], row['relaxed_weights_FP32_CPU'][:8])
        assert row['full_logit_gradient'][8:] == [0, 0, 0]
    centered = R.mean(0)-R.mean(0)[0]; b = pi*(centered-pi@centered)
    close(b, s['b_par']); close(pi@R.mean(0), s['V_par'])
    rb = R[ci].mean(1); qb = Q[ci].mean(1); gb = G[si].mean(1)
    rb = rb-rb[:, :1]
    pb = pi*(rb-(rb*pi).sum(1, keepdim=True)); cb = pi*(qb-(qb*pi).sum(1, keepdim=True))
    par_halves=[]; ce_halves=[]
    for sl in [slice(0,4), slice(4,8)]:
        hh=R[sl].mean(0); hh=hh-hh[0]; qq=Q[sl].mean(0); gg=G[sl].mean(0)
        par_halves.append(float(-(pi*(hh-pi@hh))@gg)); ce_halves.append(float((pi*(qq-pi@qq))@gg))
    check_credit(-(b@G.mean(0)), par_halves, -(pb*gb).sum(1), s['Gamma_par'])
    check_credit((pi*(Q.mean(0)-pi@Q.mean(0)))@G.mean(0), ce_halves, (cb*gb).sum(1), s['CE_alignment'])
    states[key]=s; arrays[key]=(pi,Q,R)

for p in r['pairs']:
    cp, seq=p['checkpoint'],p['sequence_id']; piB,QB,RB=arrays[(cp,seq,'B')]; piC,QC,RC=arrays[(cp,seq,'C')]
    assert min(piB).item()>.01 and p['low_probability_control']['empty']
    for name, w in [('pi_B',piB),('pi_C',piC),('uniform',torch.full((8,),.125,dtype=torch.float64))]:
        for arm,Q,R in [('B',QB,RB),('C',QC,RC)]:
            check_summary(R@w,p['expectations'][name][arm]['parity']); check_summary(Q@w,p['expectations'][name][arm]['CE'])
        check_summary((RC-RB)@w,p['contrasts'][f'state_parity_C_minus_B_under_{name}'])
        check_summary((QC-QB)@w,p['contrasts'][f'state_CE_C_minus_B_under_{name}'])
        if name!='pi_B':
            check_summary((RB-RB[:,:1])@(w-piB),p['contrasts'][f'within_B_parity_{name}_minus_pi_B'])
            check_summary((QB-QB[:,:1])@(w-piB),p['contrasts'][f'within_B_CE_{name}_minus_pi_B'])
    check_summary((RC-RB)@piB+(RC-RC[:,:1])@(piC-piB),p['contrasts']['natural_root_and_state_parity_C_minus_B'])
    check_summary((QC-QB)@piB+(QC-QC[:,:1])@(piC-piB),p['contrasts']['natural_root_and_state_CE_C_minus_B'])
    assert int((RB!=RC).sum()) == p['binary_discordance']['total']
for key, summary in r['summaries'].items():
    cp,source=key.split('/'); panel=[p for p in r['pairs'] if p['checkpoint']==int(cp) and (source=='all' or p['source']==source)]
    for field, saved in summary['contrasts'].items():
        check_summary(tensor([p['contrasts'][field]['tape_values'] for p in panel]).mean(0),saved)
assert r['counts'] == dict(states=24,pairs=12,retained_CF_runs=1536,retained_ST_roots=192)

# Complete long-form tables retain precision; report rounding affects presentation only.
with (OUT/'action_tables.csv').open('x',newline='') as h:
    fields=['checkpoint','sequence_id','source','prefix','action','hard_probability','log_probability','mean_CE','mean_correctness','CF_tape_CE','CF_tape_correctness','ST_tape_gradient','ST_tape_relaxed_weight']
    writer=csv.DictWriter(h,fieldnames=fields); writer.writeheader()
    for s in r['states']:
        p=s['probe']
        for a in s['actions']:
            writer.writerow(dict(checkpoint=p['checkpoint_update'],sequence_id=p['sequence_id'],source=p['source'],prefix=p['condition'],action=a['action'],hard_probability=repr(a['probability']),log_probability=repr(a['log_probability']),mean_CE=repr(a['mean_CE']),mean_correctness=repr(a['mean_correctness']),CF_tape_CE=json.dumps(a['CE_by_CF_tape']),CF_tape_correctness=json.dumps(a['correct_by_CF_tape']),ST_tape_gradient=json.dumps(a['full_gradient_by_ST_tape']),ST_tape_relaxed_weight=json.dumps(a['relaxed_weight_by_ST_tape'])))
with (OUT/'paired_contrasts.csv').open('x',newline='') as h:
    fields=['checkpoint','sequence_id','source','contrast','mean','half_means','bootstrap_95','resolved_sign','tape_values']
    writer=csv.DictWriter(h,fieldnames=fields); writer.writeheader()
    for p in r['pairs']:
        for name,x in p['contrasts'].items():
            writer.writerow(dict(checkpoint=p['checkpoint'],sequence_id=p['sequence_id'],source=p['source'],contrast=name,mean=repr(x['mean']),half_means=json.dumps(x['half_means']),bootstrap_95=json.dumps(x['bootstrap_95']),resolved_sign=x['resolved_sign'],tape_values=json.dumps(x['tape_values'])))

def ci(x): return f"[{x['bootstrap_95'][0]:.6g}, {x['bootstrap_95'][1]:.6g}]"
lines=['# Gate-C matched final-token prefix analysis', '',
'The primary hypothesis is **UNRESOLVED** across the six snapshot-1040 pairs. Holding the checkpoint’s B-state root probabilities fixed, canonical prefixes increase estimated parity correctness by **0.0165821 (1.65821 percentage points)**; the approximate conditional 95% bootstrap interval is **[-0.00466511, 0.0394748]**. Both fixed halves are positive (0.0261934 and 0.00697084), but the interval includes zero. One length-16 evaluation pair supports the hypothesis; the remaining five pairs do not resolve it. This is a fixed-panel, saved-prefix comparison, not a causal estimate over training or all prefix draws.', '',
'The two training-input pairs do not establish reduced parity correctness from controller-generated prefixes: their equal-weight contrast is **-0.0110899**, interval **[-0.0644520, 0.0391206]**. No explicit training-state exposure gate was found in the supplied protocol or repository records; clarification was requested. The requested gate therefore cannot be certified, and these training contrasts do not meet the existing resolved-positive direction rule. No exposure experiment is designed or run.', '',
'## Locked inputs and definitions', '',
f"Manifest SHA-256: `{r['manifest_sha256']}`. Revision: `{m['git']['head']}`; branch: `{m['git']['branch']}`; tracked-diff SHA-256: `{m['tracked_diff_sha256']}`. The manifest was written and hash-sealed before deriving new results.", '',
f"Parent action-value manifest: `{m['parent_manifest_sha256']}`. Temporal diagnostic manifest: `{m['temporal_manifest_sha256']}`. Every indexed historical artifact, relevant source/checkpoint and recorded post-exception identity was verified before and after arithmetic. The historical bytecode-only identity exception, temporal offline verification failure and pre-execution fixture failures remain preserved.", '',
'Only B/C at human token L / internal index L-1 are included: 24 states, 12 pairs, 1,536 retained action/tape outcomes and 192 retained ST roots. B uses each checkpoint’s sealed sampled prefix; C uses the sealed canonical prefix. In both cases the same checkpoint controls stochastic continuation. C is not a fully canonical trajectory. Incoming prefix states are fixed, not replayed here.', '',
'Hard probabilities are nominal softmax of the eight legal raw ADVANCE logits without temperature division. Log probabilities and tiny probabilities are retained at full precision. HALT/REPEAT/RESTART are excluded at these first-decision successor roots; all B/C roots have exactly ADVANCE(0..7). Tau=2 applies to relaxed weights, reconstructed with original FP32 clamped-uniform sampler arithmetic on CPU. Selected actions match retained records. Original CUDA soft weights were not retained; bitwise CUDA relaxed-weight identity is unavailable.', '',
'Eight CF tapes (4800) and eight independent ST tapes (4801) are reused. No model, autograd or random-generation calls were allowed in this analysis. Analytic value mixtures do not execute reweighted policies. Canonical actions are not used as action-value labels.', '',
'The six sequence selections are unchanged:', '',
'| Sequence ID | Length | Label | Input/label SHA-256 | Exposure |',
'|---|---:|---:|---|---|']
for s in m['six_sequences']:
    lines.append(f"| {s['sequence_id']} | {s['length']} | {s['label']} | `{s['sequence_label_hash']}` | {s['prior_exposure']} |")
lines += ['', 'The two training inputs are recorded training members; four evaluation inputs are outside the materialized 104 inputs but were previously evaluated. Stage-1 training membership remains unknown. Snapshot 0 is historical context and is not pooled as an independent replication.', '',
'## Fixed-probability prefix-state effects', '',
'Delta parity = pi_B dot (R_C-R_B); positive favors C. Delta CE = pi_B dot (Q_C-Q_B); negative favors C. Probabilities are held fixed in each contrast. All parity values below are proportions, not percentages.', '',
'| Snapshot | Sequence | B parity under pi_B | C parity under pi_B | Delta parity | 95% interval | Direction | Delta CE | CE direction | B/C binary discordance /64 |',
'|---:|---|---:|---:|---:|---|---|---:|---|---:|']
for p in r['pairs']:
    x=p['contrasts']['state_parity_C_minus_B_under_pi_B']; q=p['contrasts']['state_CE_C_minus_B_under_pi_B']
    lines.append(f"| {p['checkpoint']} | {p['sequence_id']} | {p['expectations']['pi_B']['B']['parity']['mean']:.6f} | {p['expectations']['pi_B']['C']['parity']['mean']:.6f} | {x['mean']:.6f} | {ci(x)} | {x['resolved_sign']} | {q['mean']:.6g} | {q['resolved_sign']} | {p['binary_discordance']['total']} |")
lines += ['', 'Equal-pair summaries (two pairs per source; six in each snapshot summary):', '',
'| Snapshot/source | Delta parity | 95% interval | Direction | Delta CE | 95% interval | Direction |',
'|---|---:|---|---|---:|---|---|']
for name,s in r['summaries'].items():
    x=s['contrasts']['state_parity_C_minus_B_under_pi_B'];q=s['contrasts']['state_CE_C_minus_B_under_pi_B']
    lines.append(f"| {name} | {x['mean']:.6g} | {ci(x)} | {x['resolved_sign']} | {q['mean']:.6g} | {ci(q)} | {q['resolved_sign']} |")
lines += ['', 'Snapshot1040 val/L16/row2 supports the hypothesis (parity delta +0.171996; interval [0.0844860,0.276314]). The training parity-0 and evaluation parity-0 pairs have negative point estimates but unresolved directions. The other three pairs have identical observed B/C binary outcomes, yielding zero contrasts and degenerate intervals. This is not proof of zero population effect. Snapshot0’s six-pair contrast is +0.165970, conditionally resolved; its different states/checkpoint do not establish an optimizer causal effect.', '',
'## Root probability allocation', '',
'Within-B contrasts change only root weights; B values and the downstream policy stay fixed. pi_C is the checkpoint distribution at the C state, not teacher forcing. Complete B/C expectations under pi_B, pi_C and uniform, every pair/contrast and all uncertainty are in results.json and paired_contrasts.csv.', '',
'| Snapshot/source | Parity: pi_C minus pi_B within B | Direction | Parity: uniform minus pi_B within B | Direction | CE: pi_C minus pi_B within B | Direction |',
'|---|---:|---|---:|---|---:|---|']
for name,s in r['summaries'].items():
    a=s['contrasts']['within_B_parity_pi_C_minus_pi_B'];b=s['contrasts']['within_B_parity_uniform_minus_pi_B'];c=s['contrasts']['within_B_CE_pi_C_minus_pi_B']
    lines.append(f"| {name} | {a['mean']:.6g} | {a['resolved_sign']} | {b['mean']:.6g} | {b['resolved_sign']} | {c['mean']:.6g} | {c['resolved_sign']} |")
lines += ['', 'At snapshot1040, replacing pi_B with pi_C on B values reduces parity by 0.00297329 (interval [-0.00484036,-0.000857538]) and increases CE by 0.0216468 (interval [0.0105696,0.0322147]), both resolved. Uniform weighting changes parity by +0.000616040, unresolved. Fixed-probability state contrasts remain unresolved under pi_C and uniform as well. Root allocation and prefix-state effects are distinct; changing allocation cannot be inferred to repair the entire trajectory.', '',
'## Predefined low-hard-mass coverage control', '',
'All 12 B groups are empty: each legal action individually exceeds the 1% threshold. Across all 24 B/C states, individual hard probabilities range from 0.0879200 to 0.161846. Thus no nonempty rare-action group exists under the declared rule, and group-uniform opportunity, group relaxed mass and group descent predictions are not applicable. No group was widened after observing results. All action-level hard/relaxed weights remain in the complete tables.', '',
'This offers no evidence for 1%-rare root-action undercoverage at these measured states. It does not estimate training-wide visits or later-route coverage. Exhaustive successor construction and relaxed backward paths preclude inferring gradient starvation from hard probability alone.', '',
'## CE versus parity credit', '',
'Gamma_par = -[pi*(R-pi dot R)] dot mean full ST gradient. Positive predicts improved parity under an independent local-logit descent direction; positive CE alignment predicts reduced CE. Neither predicts the shared-parameter optimizer step. CE and parity action values come from the independent CF bank, gradients from the ST bank.', '',
'| Snapshot | Sequence/prefix | Gamma_par | 95% interval | Direction | CE alignment | 95% interval | Direction | Binary table constant |',
'|---:|---|---:|---|---|---:|---|---|---|']
for s in r['states']:
    p=s['probe'];g=s['Gamma_par'];c=s['CE_alignment']
    lines.append(f"| {p['checkpoint_update']} | {p['sequence_id']}/{p['condition']} | {g['value']:.6g} | {ci(g)} | {g['resolved_sign']} | {c['value']:.6g} | {ci(c)} | {c['resolved_sign']} | {s['all_binary_outcomes_constant']} |")
lines += ['', 'Snapshot1040: four states resolve positive parity credit, eight are unresolved; none resolves negative parity credit. Eight states resolve positive CE alignment. Three of those have constant observed correctness (both B OOD states and C ood_test/L32/row0), so their measured credit concerns CE/calibration alone. C val/L16/row2 has resolved CE alignment with unresolved parity credit. These observations do not establish opposed CE/parity credit as the failure mechanism. Constant outcomes can conceal opportunities outside the eight sampled tapes.', '',
'Snapshot0: five states resolve positive parity credit and seven are unresolved; eight resolve positive CE alignment. No snapshot is treated as another independent controller-training seed.', '',
'## Uncertainty, limitations and integrity', '',
'Intervals are approximate conditional Monte Carlo intervals, unadjusted for multiple comparisons. The first-four and last-four fixed halves must agree in sign and the full 95% interval must exclude zero to resolve a direction. Existing 2,000-replicate bootstrap indices from seed4802 are reused: CF[1] and ST[1], the first locked B in manifest order, chosen before new outcomes. Whole tape coordinates are jointly resampled across actions, B/C and fixed-panel sequences; retained gradient/noise associations travel together. CF and ST banks are independently resampled. No sequence identities or controller seeds are resampled. Original per-state CE intervals are retained separately; the newly paired intervals use the sealed joint coupling and can differ.', '',
'Small discordance (at most four of 64 paired binary entries), zero discordance, 0/8, 8/8 and constant per-action binary outcomes are explicitly flagged in results.json. The snapshot1040 evaluation parity-0 pair has only 3/64 discordances. Degenerate zero intervals do not establish no population opportunity. B/C values depend on the frozen checkpoint’s stochastic continuation, not a fully canonical suffix. One saved prefix realization per condition and eight continuation tapes limit conclusions.', '',
'Historical bytecode-only identity exception and prior failure records remain unchanged. The array-only focused suite passed 7 tests; a pre-analysis mixed-dtype synthetic fixture failure was corrected and preserved in the manifest. Independent offline checks verified every per-action mean, nominal probability, original projection identity, paired mixture, bootstrap interval, fixed-half decision and equal-pair summary. No full model test suite was run because this task forbids new model executions.', '',
'No model forwards, backwards, counterfactual continuations, experimental noise, optimizer steps, parameter updates or production edits occurred. No training-state exposure protocol or new training run is proposed. Checkpoints, historical artifacts and protected sources remained byte-identical, and git diff --check is clean.', '',
'Artifacts: manifest.json; results.json (complete states, root/noise/gradient records, per-tape outcomes, mixtures and uncertainty); action_tables.csv; paired_contrasts.csv; verification.json; offline_verification.py; artifact_index.json. All files were created under a new analysis directory without overwriting historical artifacts.']
with (OUT/'report.md').open('x') as h: h.write('\n'.join(lines)+'\n')
verification=dict(states=24,pairs=12,complete_action_rows=192,CF_outcomes=1536,ST_roots=192,
    probability_and_projection_identities_verified=True,mixtures_halves_intervals_summaries_verified=True,
    prior_identities_verified=len(m['sources']),no_model_calls=True,no_backwards=True,no_new_noise=True,
    historical_exceptions_preserved=m['historical_exceptions'],focused_tests='7 passed; array-only',
    git_diff_check='clean',training_state_exposure_gate='not specified; cannot certify; training-only contrast unresolved',
    primary_verdict='UNRESOLVED at snapshot1040 six-pair panel; one evaluation pair supports',
    coverage_verdict='all predefined 1% B groups empty',CE_parity_verdict='no resolved opposing parity credit; some CE-only credit with constant observed binary outcomes')
for path, expected in m['sources'].items(): assert digest(path) == expected, path
write(OUT/'verification.json', verification)
files={str(p.relative_to(OUT)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file()}
write(OUT/'artifact_index.json',dict(files=files,manifest_sha256=r['manifest_sha256'],self_excluded=True))
print('OFFLINE VERIFIED: 24 states, 12 pairs, all contrasts and bootstrap summaries; historical identities unchanged')
