"""Offline closure, timing projection and trace reconciliation; no model imports."""
from pathlib import Path
import json,gzip,hashlib,statistics,shutil,datetime
OUT=Path(__file__).resolve().parent

def main():
    r=json.loads((OUT/'smoke_results_v1.json').read_text());ledger=r['ledger'];rows=[]
    with gzip.open(OUT/'token_traces_v1.jsonl.gz','rt') as f:
        for l in f:rows.append(json.loads(l))
    completed=[x for x in rows if x['status']=='completed'];failed=[x for x in rows if x['status']=='failed']
    checks=dict(trace_attempts=len(rows)==ledger['attempted_tokens'],completed=len(completed)==ledger['completed_tokens'],
        failed=len(failed)==ledger['failed_tokens'],selected_completions=sum(len(x['blocks']) for x in rows)==ledger['selected_bank_completed'],
        commits=sum(x['commits'] for x in rows)==ledger['commits'],
        completed_paths=all(x['blocks']==list(range(8)) and x['P']==8 and x['commits']==1 and x['parent_unchanged'] and x['returned_state_disjoint'] for x in completed))
    calibration=[x for x in r['records'] if x['phase']=='CUDA_calibration'];timed=[x for x in calibration if not x['warmup']]
    forecast={'assessable':False,'training_authorized':False}
    if r['status']=='SMOKE_PASSED' and len(timed)==3:
        gpu=[x for x in r['records'] if x.get('device')=='cuda' and 'synchronized_forward_seconds' in x]
        f16=next(x['synchronized_forward_seconds'] for x in gpu if x['n']==16)
        f64=next(x['synchronized_forward_seconds'] for x in gpu if x['n']==64)
        max_update=max(x['total_seconds'] for x in timed)
        training=1000*(1.25*max_update+1.)
        initial=8*1.25*f16
        final_train=128*1.25*f16;development=32*1.25*f16;transfer=32*1.25*f64
        reserve=600.;total=ledger['elapsed_seconds']+training+initial+final_train+development+transfer+reserve
        forecast=dict(assessable=True,training_authorized=False,
            measured_three_timed_updates_seconds=[x['total_seconds'] for x in timed],
            measured_forward_seconds=[x['forward_seconds'] for x in timed],
            measured_backward_seconds=[x['backward_seconds'] for x in timed],
            measured_warmup_seconds=calibration[0]['total_seconds'],
            measured_CUDA_N16_fixture_forward_seconds=f16,
            N16_fixture_autograd_enabled=True,
            measured_CUDA_N64_fixture_inference_seconds=f64,
            measured_smoke_elapsed_seconds=ledger['elapsed_seconds'],
            margin_multiplier=1.25,unmeasured_optimizer_clipping_logging_allowance_per_update=1.,
            setup_checkpoint_analysis_reporting_reserve_seconds=reserve,
            estimated_training_seconds=training,estimated_initial_diagnostic_seconds=initial,
            estimated_final_training_subset_seconds=final_train,estimated_development_seconds=development,
            estimated_size_transfer_seconds=transfer,conservative_complete_seconds=total,
            ceiling_seconds=21600,eligibility_limit_seconds=17280,
            resource_gate_pass=total<=17280,
            limitations=['Three synthetic calibration runs are not a runtime guarantee.',
                        'Native N16 fixture timing includes autograd; used as conservative inference proxy.',
                        'Optimizer steps and checkpoint/analysis times were not executed.',
                        'No official train/development/size-transfer arrays were model-evaluated.',
                        'Common filesystem/accelerator contention can change runtime.'])
    integrity=r.get('integrity',{})
    metric_path=OUT/'smoke_official_metric_verification_v1.json'
    official_metric=json.loads(metric_path.read_text()) if metric_path.exists() else {'success':False,'unverified':True}
    ok=r['status']=='SMOKE_PASSED' and all(checks.values()) and integrity.get('success') and official_metric.get('success')
    final=dict(engineering_pass=bool(ok),status='ENGINEERING_PASSED' if ok else 'STOPPED',
        ledger=ledger,trace_checks=checks,runtime=forecast,official_metric_verified=official_metric.get('success',False),
        official_samples={'train':1000,'development':32,'size_transfer':32},
        no_optimizer_or_weight_update=True,no_study_checkpoint=True,
        recommendation='Separate training authorization may be considered; not granted by this milestone.' if ok and forecast.get('resource_gate_pass') else 'Do not authorize training; prerequisite or resource gate unresolved/failed.',
        integrity=integrity,disk_free=shutil.disk_usage(OUT).free)
    (OUT/'runtime_projection_v1.json').write_text(json.dumps(forecast,indent=2)+'\n')
    (OUT/'complete_trace_verification_v1.json').write_text(json.dumps(checks,indent=2)+'\n')
    (OUT/'final_results_v1.json').write_text(json.dumps(final,indent=2)+'\n')
    report=['# CLRS insertion-sort engineering milestone',f'\nStatus: **{final["status"]}**.',
        '\nOfficial source: `6d5728b9efadf8648413c8c473929a733caacced`. Unchanged official sampler, algorithm, probing and evaluator; isolated package namespaces avoid unrelated convenience imports. TensorFlow is required by probing; TFDS was not installed.',
        '\nPools: 1,000 size-16 train, 32 size-16 development, 32 size-64 transfer. Source-generated official pools; no byte-equivalence claim for released TFRecords. Stable ties, pointer bounds, source targets, FP32 ordering and split overlap checked before AGT execution.',
        '\nAdapter: fresh seed-6200 D64/16-slot core, canonical blocks 0–7, trainable 4→64 encoder and size-independent query/key readout. Controller and old two-class head frozen/unused. Readout candidate representations remain autograd-connected. This is serialized output-only AGT, not the official GNN.',
        '\n## Scope and accounting','\n```json',json.dumps(ledger,indent=2),'```',
        '\nNative CPU/CUDA fixtures, canonical executor comparison and gradient records: `smoke_results_v1.json`. Complete retained fixture states: device-specific `.pt` files. Trace reconciliation: '+str(checks),
        '\n## Runtime forecast','\n```json',json.dumps(forecast,indent=2),'```',
        '\n'+final['recommendation'],
        '\nNo training, eight-array diagnostic, full development, held-out model evaluation, profiling study, optimizer creation/step, parameter update, checkpoint creation, retry or follow-up occurred. Synthetic size-64 fixtures are engineering evidence, not larger-problem generalization results.',
        '\nProtected integrity: '+str(integrity)]
    (OUT/'engineering_report_v1.md').write_text('\n'.join(report)+'\n')
    manifest={}
    for f in sorted(OUT.rglob('*')):
        if f.is_file() and f.name!='artifact_index_final_v1.json':
            manifest[f.relative_to(OUT).as_posix()]=dict(size=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    (OUT/'artifact_index_final_v1.json').write_text(json.dumps(dict(files=manifest,self_excluded=True,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
    print(json.dumps(final,indent=2))

if __name__=='__main__':main()
