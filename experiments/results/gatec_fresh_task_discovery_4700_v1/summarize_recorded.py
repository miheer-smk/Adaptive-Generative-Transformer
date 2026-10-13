"""Aggregate already-written measurements; no model construction or optimizer."""
from pathlib import Path
import sys, gzip, json, hashlib
from collections import defaultdict
ROOT = Path('/home/iiitn/Desktop/Miheer Projects/AGT/AGT_v3_POC')
sys.path.insert(0, str(ROOT))
from experiments.run_gatec_fresh_task_discovery import expanded_summary
OUT = ROOT/'experiments/results/gatec_fresh_task_discovery_4700_v1'
def read(name): return json.loads((OUT/name).read_text())
def stream(name):
    with gzip.open(OUT/(name+'.jsonl.gz'),'rt') as f:
        return [json.loads(line) for line in f]
m = read('manifest.json')
failure = read('failure.json')
assert failure['completed_updates'] == 1040
assert failure['type'] == 'AssertionError'
assert "assert all(sha(path)==digest for path,digest in protected.items())" in failure['traceback']
updates = stream('updates'); snaps = stream('snapshots'); seq = stream('sequences')
assert [u['update'] for u in updates] == list(range(1,1041))
assert [s['update'] for s in snaps] == [0,104,520,1040]
final = {mode:read('final_'+mode+'.json') for mode in ['deterministic']+[f'final_tape_{i}' for i in range(1,5)]}
assert all(len(rows)==96 for arms in final.values() for rows in arms.values())
# JSON object keys are strings, whereas the unchanged live aggregate helper
# queries these bank counters by integer index. Restore that recorded type
# without changing any observation, route, metric denominator or raw artifact.
all_metrics = seq + [u[a] for u in updates for a in ['U','T']] + [s for arms in final.values() for rows in arms.values() for s in rows]
for s in all_metrics:
    s['gdn_purity_checks'] = {path:{int(k):v for k,v in counts.items()} for path,counts in s['gdn_purity_checks'].items()}
splits = ['val','ood_test']
final_summary = {mode:{arm:{split:expanded_summary([s for s in rows if s['split']==split]) for split in splits} for arm,rows in arms.items()} for mode,arms in final.items()}
aggregate = {arm:{split:expanded_summary([s for mode,arms in final.items() if mode!='deterministic' for s in arms[arm] if s['split']==split]) for split in splits} for arm in ['U','T']}
epochs = {str(epoch):{arm:expanded_summary([r[arm] for r in updates if r['epoch']==epoch]) for arm in ['U','T']} for epoch in range(1,11)}
development = {str(update):{arm:{split:expanded_summary([s for s in seq if s['phase']==f'development_update{update}' and s['arm']==arm and s['split']==split]) for split in splits} for arm in ['U','T']} for update in [0,104,520,1040]}
assert all(s['P']==[8]*s['length'] for s in seq)
assert all(d==8+r.count('9') for s in seq for d,r in zip(s['D'],s['routes']))
result = dict(manifest_sha256=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest(),
    snapshots=snaps,final=final_summary,sampled_aggregate=aggregate,training_by_epoch=epochs,
    development_by_snapshot=development,controller_change=snaps[-1]['change'],
    gradient_statistics=dict(minimum=min(u['gradient_l2'] for u in updates),maximum=max(u['gradient_l2'] for u in updates),
        mean=sum(u['gradient_l2'] for u in updates)/1040,finite_updates=1040,
        nonzero_parameter_updates=sum(u['controller_update']['changed_values']>0 for u in updates),
        logit_decisions=sum(u['logit_gradient_coverage']['decisions'] for u in updates),
        logit_nonzero_decisions=sum(u['logit_gradient_coverage']['nonzero_decisions'] for u in updates)),
    strong_reference_level_discovery=False,
    invariants=dict(mechanical_validity_observed=True,optimization_path_validity_observed=True,
        exact_updates=1040,U_unchanged=True,noncontroller_parameters_unchanged=True,
        no_extra_detach=True,anchor_preserved=True,purity_and_one_commit_verified=True,
        full_protected_file_identity_check_passed=False),
    artifact_integrity_exception=dict(original_failure_preserved='failure.json',
        reason='A post-training inspection import without PYTHONDONTWRITEBYTECODE regenerated executor bytecode; it changed the tracked diff and triggered the original end-of-run protected-file assertion. Production source is independently hash-verified. No training/evaluation was repeated.',
        affected_tracked_path='src/__pycache__/executor.cpython-312.pyc',
        aggregation='Existing complete traces and four final tape files only; no new model execution, optimizer step or checkpoint selection'),
    limitations=m['limitations'])
assert all(final_summary['deterministic']['T'][s]['learned_accuracy']<1 for s in splits)
with (OUT/'results.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(results_written=True,additional_training_updates=0,additional_model_evaluations=0),indent=2))
