"""Offline closure using retained predictions and the pinned official evaluator."""
import os,sys,json,gzip,collections,datetime
from pathlib import Path
OUT=Path(__file__).resolve().parent
E=OUT.parent
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['KERAS_HOME']=str(OUT/'cpu_runtime_home/keras')
os.environ['XDG_CACHE_HOME']=str(OUT/'cpu_runtime_home/cache')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.insert(0,str(E))
from offline_support_v1 import save,sha,integrity,metrics,invalid_chain

def main():
    result=json.loads((OUT/'run_result_pending_official_metric_v1.json').read_text())
    verification=dict(model_calls=0,checked_arrays=0,source_revision='6d5728b9efadf8648413c8c473929a733caacced',records=[])
    seal=json.loads((OUT/'training_implementation_seal_v1.json').read_text())
    seal_ok=all(sha(OUT/n)==h for n,h in seal['files'].items())
    assert seal_ok,'New training implementation seal changed'
    if result['status']=='AWAITING_OFFLINE_VERIFICATION':
        import numpy as np
        from official_loader_v1 import load_official
        _,specs,probing,evaluation,_=load_official()
        analyses=json.loads((OUT/'analysis_v1.json').read_text())
        specs_to_check=[('initial_diagnostic','train',8),('training_subset','train',128),('development','development',32),('size_transfer','size_transfer',32)]
        for name,pool_name,count in specs_to_check:
            rows=[json.loads(line) for line in (OUT/(name+'_records_v1.jsonl')).read_text().splitlines()]
            pool=json.loads((E/(pool_name+'_pool_v1.json')).read_text())[:count]
            assert len(rows)==len(pool)==count
            for r,p in zip(rows,pool):
                assert r['id']==p['id'] and r['truth']==p['pred'] and r['original_input_sha256']==p['original_input_sha256']
                assert r['n']==p['n'] and len(r['predictions'])==p['n']
                assert r['final_tokens_committed']==2*p['n']
                assert r['invalid_chain']==invalid_chain(r['predictions'])
            truth=probing.DataPoint(name='pred',location=specs.Location.NODE,type_=specs.Type.POINTER,data=np.array([r['truth'] for r in rows]))
            pred=probing.DataPoint(name='pred',location=specs.Location.NODE,type_=specs.Type.POINTER,data=np.array([r['predictions'] for r in rows]))
            score=float(evaluation.evaluate((truth,),{'pred':pred})['score'])
            local=metrics(rows)['pointer_accuracy']
            assert score==local,(name,score,local)
            verification['checked_arrays']+=len(rows)
            verification['records'].append(dict(split=name,arrays=count,official_score=score,converted_score=local,exact_match=True))
        order=json.loads((OUT/'training_order_v1.json').read_text())
        train=[json.loads(line) for line in (OUT/'training_metrics_v1.jsonl').read_text().splitlines()]
        assert len(train)==1000 and [r['id'] for r in train]==order['ids']
        assert [r['update'] for r in train]==list(range(1,1001))
        pool=json.loads((E/'train_pool_v1.json').read_text())
        assert all(r['truth']==pool[index]['pred'] for r,index in zip(train,order['indices']))
        verification['training_order_and_target_alignment']=True
        gates={}
        for split,k,threshold in [('development','pointer_accuracy',.95),('development','exact_array_accuracy',.8),('size_transfer','pointer_accuracy',.9),('size_transfer','exact_array_accuracy',.5)]:
            value=analyses['metrics'][split][k]
            gates[split+'_'+k]=dict(value=value,threshold=threshold,passed=value>=threshold)
        gates['mechanical_numerical_integrity']=dict(passed=result['integrity']['success'] and result['deadline_met'] and not result['resource_ledger']['failed_tokens'] and not result['resource_ledger']['optimizer_failed'] and seal_ok)
        result['gates']=gates;result['gates_unverified']=False
        result['metrics']=analyses['metrics']
        result['status']='COMPLETED_SUCCESS' if all(v['passed'] for v in gates.values()) else 'COMPLETED_NON_SUCCESS'
        result['primary_interpretation']='Fixed-program AGT plus serialized task adapter on official source-generated insertion-sort pools; no routing, operator-necessity or CLRS-suite claim'
    verification['success']=result['status']!='INCOMPLETE'
    save('official_metric_verification_v1.json',verification)
    result['integrity']=integrity()
    result['training_seal_unchanged']=seal_ok
    if not result['integrity']['success']:result['status']='INCOMPLETE'
    result['offline_closure_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    save('final_results_v1.json',result)
    lines=['# CLRS insertion-sort fixed-program learning run',f'\nStatus: **{result["status"]}**.',
        '\nExactly one fresh seed6200 model, fixed blocks0–7, full BPTT, one seed6201 shuffled pass. No hints, schedule tuning, intermediate selection or retry.',
        '\nThe engineering milestone and its 75 published files remain separate, unchanged evidence. The approved remaining runtime forecast used a 2× margin under observed accelerator contention; no engineering model calls were repeated.',
        '\n## Evaluation']
    if 'metrics' in result:
        lines.extend(['','|Split|n|Arrays|Predecessor accuracy|Exact array accuracy|Mean CE|Invalid chain frequency|','|---|---:|---:|---:|---:|---:|---:|'])
        for name,m in result['metrics'].items():lines.append(f'|{name}|{m["n"]}|{m["arrays"]}|{m["pointer_accuracy"]:.9f}|{m["exact_array_accuracy"]:.9f}|{m["CE"]:.9f}|{m["invalid_chain"]:.9f}|')
        lines.extend(['\nComplete-array bootstrap intervals and target-only controls: `analysis_v1.json`. Official evaluator cross-checks: `official_metric_verification_v1.json`.','\n## Locked gates','```json',json.dumps(result['gates'],indent=2),'```'])
    else:lines.append('\nFinal evaluation/gates are incomplete; no partial result is substituted for the prescribed endpoint.')
    lines.extend(['\n## Resources','```json',json.dumps(result['resource_ledger'],indent=2),'```',
        '\nThe 512-token/6-backward engineering work is included only in combined totals; it is not pooled into task outcomes. Contextual entries describe actual commit-plan entries, not unsupported FLOP estimates.',
        '\n## Checkpoint and integrity','```json',json.dumps({'checkpoint':result.get('final_checkpoint'),'integrity':result['integrity'],'new_implementation_seal_unchanged':seal_ok},indent=2),'```',
        '\nNo learned selector, route discovery, compatibility objective, new task, new seed, Enwik8 experiment or follow-up was executed. Publication will use a new remote branch with the verified engineering commit as parent; local Git state remains untouched.'])
    if 'failure' in result:lines.append('\nExact stop boundary: '+result['execution_boundary']+'; '+result['failure'])
    (OUT/'final_report_v1.md').write_text('\n'.join(lines)+'\n')
    paths=[p for p in OUT.rglob('*') if p.is_file()]
    save('artifact_index_final_v1.json',dict(self_excluded=True,files={str(p.relative_to(OUT)):{'size':p.stat().st_size,'sha256':sha(p)} for p in paths},publication_receipts_added_later=True))
    print(json.dumps({'status':result['status'],'metrics':result.get('metrics'),'gates':result.get('gates'),'integrity':result['integrity']},indent=2))

if __name__=='__main__':main()
