"""Post-boundary accounting/reporting only. Standard library; zero model execution."""
import json,gzip,hashlib,collections,datetime,math,os
from pathlib import Path
OUT=Path(__file__).resolve().parent

def read(name):return json.loads((OUT/name).read_text())
def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def mean(rows,key):return sum(r[key] for r in rows)/len(rows) if rows else None

def main():
    if not (OUT/'final_results.json').exists():raise SystemExit('STOP: pilot has not reached its stopping boundary')
    result=read('final_results.json');ledger=read('execution_ledger.json');phases=collections.Counter();blocks=collections.Counter();n=completed=failed=commits=0;violations=[];truncation=False;last=None
    expected={'B':[0,1,2,3,4,5,6,7],'A':[0,1,2,3,3,5,6,7],'R':[0,1,2,4,4,5,6,7]}
    with gzip.open(OUT/'token_traces.jsonl.gz','rt') as f:
        try:
            for line in f:
                row=json.loads(line);last=row;n+=1;phases[row['phase']+'/'+row['status']]+=1
                p=row['program'].rsplit('/',1)[1];executed=row.get('executed_blocks',[])
                blocks.update(map(str,executed));commits+=row.get('commits',0)
                if row['status']=='completed':
                    completed+=1
                    tests={'D8_P8':row.get('D')==8 and row.get('P')==8,'bank_sequence':executed==expected[p],'one_commit':row.get('commits')==1,'eight_decisions':len(row.get('decisions',[]))==8,
                           'legal_actions':all(d['legal_mask'][d['selected']] and d['selected']==d['action'] for d in row['decisions']),
                           'commit_versions':len(row['commit']['versions_before'])==7 and row['commit']['versions_before'].keys()==row['commit']['versions_after'].keys() and all(v==row['token'] and row['commit']['versions_after'][k]==v+1 for k,v in row['commit']['versions_before'].items()),
                           'token_count':row['returned_metadata']['/tokens_committed']==row['token']+1,
                           'consumption_metadata':row['incoming_metadata']['/_consumed'] is False and row['returned_metadata']['/_consumed'] is False}
                    for name,ok in tests.items():
                        if not ok:violations.append({'attempt':row['attempt'],'test':name})
                elif row['status']=='failed':failed+=1
        except (EOFError,json.JSONDecodeError):truncation=True
    checks={'trace_attempts_vs_ledger':n==ledger['attempted_tokens'],'completed_vs_ledger':completed==ledger['completed_tokens'],'failed_vs_ledger':failed==ledger['failed_tokens'],'verified_bank_completions_vs_ledger':sum(blocks.values())==ledger['selected_bank_completed'],'commits_vs_ledger':commits==ledger['commits']}
    verification=dict(trace_rows=n,completed_tokens=completed,failed_tokens=failed,commits=commits,verified_bank_completions=sum(blocks.values()),by_phase=dict(phases),by_block=dict(blocks),counter_checks=checks,completed_call_violations=violations,truncated_trace=truncation,last_record=last,
                      reference_D_note='Fixed-reference legacy D=8 is the observed canonical bank-visit count; controller_decisions=0 is retained explicitly. Program-adapter D=8 counts script-provider decisions.',
                      no_new_model_execution=True,success=all(checks.values()) and not violations and not truncation)
    save('complete_trace_verification.json',verification)
    updates=[json.loads(line) for line in (OUT/'training_updates.jsonl').read_text().splitlines()] if (OUT/'training_updates.jsonl').exists() else []
    online=[r for u in updates for r in u['routes']]
    training={str(L):{p:{'episodes':len(sub),'complete_accuracy':mean(sub,'S'),'query_accuracy':mean(sub,'query_accuracy'),'mean_CE':mean(sub,'CE')} for p in ('B','A','R') for sub in [[r for r in online if r['length']==L and r['program']==p]]} for L in (64,96)}
    initial=read('initial_diagnostic_results.json') if (OUT/'initial_diagnostic_results.json').exists() else []
    initial_stats={str(L):{p:{'episodes':len(sub),'complete_accuracy':mean(sub,'S'),'query_accuracy':mean(sub,'query_accuracy'),'mean_CE':mean(sub,'CE')} for p in ('B','A','R') for sub in [[r for r in initial if r['length']==L and r['program']==p]]} for L in (64,96)}
    interruption=read('time_ceiling_enforcement_event.json') if (OUT/'time_ceiling_enforcement_event.json').exists() else None
    tb=result.get('failure',{}).get('traceback','')
    incomplete_backward=bool('loss/3).backward()' in tb or '_engine_run_backward' in tb)
    closure=dict(classification=result['classification'],protocol_sha256=hashlib.sha256((OUT/'protocol_approved_verbatim.md').read_bytes()).hexdigest(),training_completed_optimizer_updates=len(updates),training_online_metrics=training,training_metrics_note='Completed-update online rollouts under evolving parameters, not final-checkpoint evaluation. An interrupted batch may have extra retained token outputs outside this table.',initial_diagnostic_metrics=initial_stats,final_development_executed=(OUT/'development_results.json').exists(),preliminary_ablation_executed=(OUT/'preliminary_ablation_results.json').exists(),qualification_analysis_executed=(OUT/'qualification_analysis.json').exists(),final_checkpoint_created=(OUT/'checkpoint_update0128.pt').exists(),interrupted_backward_attempt=incomplete_backward,resource_interruption=interruption,ledger=ledger,trace_verification_success=verification['success'],integrity=read('integrity_final.json'),disk_free_after=os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize,remaining_phases_unverified=not (OUT/'development_results.json').exists(),no_retry_or_followup=True)
    save('closure_results_v1.json',closure)
    lines=['# Qualification pilot closure', '\n**Classification: '+result['classification']+'.**', '\nProtocol SHA-256: `'+closure['protocol_sha256']+'`.', '\nReceiver-anchored study remains COMPLETED_NON_SUCCESS; no prior scientific outcome was changed.', '\n## Execution boundary', '\n'+('Resource-time ceiling enforced; training incomplete. Final development, solvability, headroom, complementarity and preliminary ablations remain unverified.' if interruption else 'See final_results.json for the precise locked boundary.'), '\n## Training metrics', '\nThese are online results from completed optimizer updates, not final-model training/development accuracy.', '\n| Length | Program | Episodes | Complete accuracy | Query accuracy | Mean CE |','|---|---|---:|---:|---:|---:|']
    for L,ps in training.items():
        for p,v in ps.items():lines.append('| '+L+' | '+p+' | '+str(v['episodes'])+' | '+str(v['complete_accuracy'])+' | '+str(v['query_accuracy'])+' | '+str(v['mean_CE'])+' |')
    lines+=['\n## Gates','\n- Symbolic, encoding, counterfactual, separation and shortcut checks: see sealed validation records.','- CPU and genuinely executed CUDA fixture menus: passed; canonical equivalence and connected native gradients recorded.','- Training/final development: '+str(len(updates))+'/128 completed updates; final development executed: '+str(closure['final_development_executed'])+'.','- Qualification statistics executed: '+str(closure['qualification_analysis_executed'])+'.','- Preliminary ablation executed: '+str(closure['preliminary_ablation_executed'])+'. Any such results are descriptive, never inferential.','\n## Accounting','\n```json',json.dumps(ledger,indent=2),'```','\nTrace reconciliation: '+str(verification['success'])+'. Protected-file verification: '+str(closure['integrity']['success'])+'.','\nNo eligible final checkpoint was created unless the predetermined update 128 completed. No partial checkpoint is selected or substituted. No selector, compatibility training, profiling, transfer confirmation, retry or extension occurred.','\nResource-limited noncompletion does not establish task unsolvability, absence of program complementarity, or useful learned selection.']
    (OUT/'closure_report_v1.md').write_text('\n'.join(lines)+'\n')
    artifacts={}
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='artifact_index_final.json':
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(1048576),b''):h.update(b)
            artifacts[p.relative_to(OUT).as_posix()]={'size':p.stat().st_size,'sha256':h.hexdigest()}
    save('artifact_index_final.json',{'files':artifacts,'self_excluded':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:v for k,v in closure.items() if k not in ('training_online_metrics','initial_diagnostic_metrics','ledger','integrity')},indent=2))

if __name__=='__main__':main()
