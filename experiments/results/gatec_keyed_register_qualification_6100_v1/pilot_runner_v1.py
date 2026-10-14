"""One sealed 128-update qualification attempt; no resume or automatic retry."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,time,gzip,hashlib,traceback,shutil,collections,subprocess,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from program_adapter_v1 import PROGRAMS,P0,P1,PR,load_research_model,CompleteProgramModule,fields,snapshot,assert_equal,assert_disjoint,fixture_encoding,sha
from task_data_v1 import seal_inputs,encode,interpret,digest
from analysis_v1 import qualify,ablation_summary,metric

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def tensor_bytes(x):
    a=x.detach().cpu().contiguous()
    return bytes(a.reshape(-1).view(torch.uint8).tolist())

def model_hash(model,controller_only=False):
    h=hashlib.sha256()
    items=list(model.named_parameters())+list(model.named_buffers())
    for name,value in sorted(items):
        if controller_only and not name.startswith('controller.'):continue
        h.update(name.encode());h.update(str((value.shape,value.dtype)).encode());h.update(tensor_bytes(value))
    return h.hexdigest()

def cpu_fields(state):
    return {k:v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in fields(state).items()}

def state_metadata(state):
    return {k:({'shape':list(v.shape),'dtype':str(v.dtype)} if torch.is_tensor(v) else v) for k,v in fields(state).items()}

class Ledger:
    def __init__(self):
        self.start=time.monotonic();self.phase='fixtures_cpu';self.attempted=self.completed=self.failed=0
        self.blocks=self.block_completed=self.commits=self.commit_attempts=0
        self.block_ceiling=800000;self.optimizer_updates=self.backward_evaluations=0
        self.phase_counts=collections.Counter();self.block_counts=collections.Counter();self.context_counts=collections.Counter()
        self.pending=None;self.logged=0;self.clean_close=False
        self.trace=gzip.open(OUT/'token_traces.jsonl.gz','wt',compresslevel=6)
    def guard(self):
        if time.monotonic()-self.start>=7200:raise RuntimeError('RESOURCE_STOP: accelerator time ceiling')
        if shutil.disk_usage(OUT).free<128*1024**2:raise RuntimeError('RESOURCE_STOP: disk reserve')
    def begin(self,case,program,device,token):
        self.guard()
        if self.attempted>=100000 or self.blocks+8>800000:raise RuntimeError('RESOURCE_STOP: token/block ceiling')
        if self.pending is not None:raise RuntimeError('Unflushed/overlapping execution')
        self.attempted+=1
        row=dict(attempt=self.attempted,phase=self.phase,case=case,program=program.name,device=str(device),token=token,status='attempted',decisions=[],commits=0)
        self.pending=row;return row
    def persist(self):
        if self.pending is None:return
        row=self.pending
        self.phase_counts[(row['phase'],row['status'])]+=1
        for b in row.get('executed_blocks',[]):self.block_counts[str(b)]+=1
        if row['commits']:
            for kind,n in [('SWA_bank',2),('GQA_bank',2),('GDN_bank',2),('SWA_prelude',1)]:self.context_counts[kind]+=n*row['commits']
        if row['case'].get('reference'):
            row['controller_decisions']=0;row['fixed_reference_bank_visits']=8
        self.trace.write(json.dumps(row,allow_nan=False)+'\n');self.trace.flush();self.logged+=1;self.pending=None
        if self.logged%256==0:self.flush_summary()
    def summary(self):
        return dict(attempted_tokens=self.attempted,completed_tokens=self.completed,failed_tokens=self.failed,selected_bank_attempts=self.blocks,selected_bank_completed=self.block_completed,commit_attempts=self.commit_attempts,commits=self.commits,backward_VJP_evaluations=self.backward_evaluations,optimizer_updates=self.optimizer_updates,phase_counts={f'{p}/{s}':n for (p,s),n in self.phase_counts.items()},selected_block_counts=dict(self.block_counts),contextual_commit_payload_calls=dict(self.context_counts),elapsed_accelerator_seconds=time.monotonic()-self.start,peak_memory_allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,peak_memory_reserved=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,ceilings={'attempted_tokens':100000,'bank_calls':800000,'accelerator_seconds':7200},planned_max_tokens=99954,no_warmups_or_profiling=True)
    def flush_summary(self):save('execution_ledger.json',self.summary())
    def close(self):
        if not self.clean_close:self.trace.close();self.clean_close=True
        self.flush_summary()

def check_seal():
    d=json.loads((OUT/'implementation_seal.json').read_text())
    for name,value in d['files'].items():
        if sha(OUT/name)!=value:raise RuntimeError('INTEGRITY_STOP: implementation seal '+name)

def verify_integrity():
    b=json.loads((OUT/'integrity_initial.json').read_text());changed=[]
    for rel,v in b['files'].items():
        p=ROOT/rel
        if v['kind']=='symlink':ok=p.is_symlink() and os.readlink(p)==v['target']
        else:ok=p.is_file() and not p.is_symlink() and p.stat().st_size==v['size'] and sha(p)==v['sha256'] and p.stat().st_mode&0o777==v['mode']
        if not ok:changed.append(rel)
    env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,env=env).decode().strip()
    diff=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=ROOT,env=env)).hexdigest()
    extras=[]
    for base,ds,fs in os.walk(ROOT,followlinks=False):
        if Path(base)==OUT:ds[:]=[];continue
        for n in fs:
            rel=(Path(base)/n).relative_to(ROOT).as_posix()
            if rel not in b['files']:extras.append(rel)
    r=dict(protected_count=len(b['files']),changed_or_missing=changed,unexpected_outside_run=extras,HEAD=head,HEAD_unchanged=head==b['HEAD'],tracked_diff_sha256=diff,tracked_diff_unchanged=diff==b['diff_sha256'],historical_exception=b['historical_exception'],no_new_exception=not changed and not extras,success=not changed and not extras and head==b['HEAD'] and diff==b['diff_sha256'])
    save('integrity_final.json',r)
    if not r['success']:raise RuntimeError('INTEGRITY_STOP: protected identities')
    return r

def fixtures(device,ledger):
    ledger.phase='fixtures_'+device
    model=load_research_model(device);wrapper=CompleteProgramModule(model,ledger);before=model_hash(model)
    retained={}; comparisons=[]
    control=json.loads((ROOT/'experiments/results/gatec_execution_accounting_v1/gatec_canonical_replay.json').read_text())
    x,labels=fixture_encoding(control['bits'][0],device);a=wrapper.initial(x);b=wrapper.initial(x)
    with torch.no_grad():
        for j in range(8):
            ra=wrapper(x[:,j],a,P0,dict(id='canonical',reference=True));rb=wrapper(x[:,j],b,P0,dict(id='canonical',reference=False))
            assert_equal(snapshot(ra['state']),snapshot(rb['state']))
            for key in ('h0','h_terminal','logits','memory_read','memory_attention'):assert torch.allclose(ra[key],rb[key],atol=1e-6,rtol=1e-5),key
            retained[f'canonical/{j}/reference']=cpu_fields(ra['state']);retained[f'canonical/{j}/adapter']=cpu_fields(rb['state'])
            comparisons.append(dict(token=j,state_and_metadata_pass=True,logits_max_abs=float((ra['logits']-rb['logits']).abs().max())))
            a,b=ra['state'],rb['state']
        ce=float(F.cross_entropy(rb['logits'],labels[-1:]));
        if device=='cuda':assert abs(ce-.0014546061865985394)<=1e-6
        for p in (P1,PR):wrapper(x[:,0],wrapper.initial(x),p,dict(id='one_token_legality'))
    commands=[['SET',k,None,k%2] for k in range(8)]+[['XOR',0,None,1],['COPY',1,0,None],['QUERY',0,None,None],['QUERY',1,None,None]]
    fx=encode(commands).to(device);gradients=[]
    for p in PROGRAMS:
        model.zero_grad(set_to_none=True);state=wrapper.initial(fx);losses=[];incoming=None;sibling=None
        for j in range(12):
            if j==11:
                incoming=state;parent_snapshot=snapshot(incoming)
                left=incoming.clone();right=incoming.clone();assert_disjoint(incoming,left);assert_disjoint(left,right)
                retained[p.name+'/incoming']=cpu_fields(incoming);sibling=right
                supplied=left
            else:supplied=state
            out=wrapper(fx[:,j],supplied,p,dict(id='task_fixture',program=p.name));state=out['state']
            if j>=10:losses.append(F.cross_entropy(out['logits'],torch.tensor([1],device=device)))
        out2=wrapper(fx[:,11],sibling,p,dict(id='task_fixture_duplicate',program=p.name))
        assert_equal(parent_snapshot,snapshot(incoming),exact=True);assert_equal(snapshot(state),snapshot(out2['state']))
        assert_disjoint(state,out2['state']);retained[p.name+'/left']=cpu_fields(state);retained[p.name+'/right']=cpu_fields(out2['state'])
        components=[(k,t) for k,t in fields(incoming).items() if torch.is_tensor(t) and t.requires_grad]
        loss=torch.stack(losses).mean();assert torch.isfinite(loss)
        adj=torch.autograd.grad(loss,[t for k,t in components],allow_unused=True,retain_graph=True);ledger.backward_evaluations+=1
        assert any(k=='/s' and g is not None for (k,t),g in zip(components,adj)), 'Stream temporal path disconnected'
        loss.backward();ledger.backward_evaluations+=1
        norms=[v.grad.norm() for v in model.parameters() if v.grad is not None]
        assert norms and all(torch.isfinite(n) for n in norms)
        norm=float(torch.stack(norms).square().sum().sqrt());assert norm>0
        gradients.append(dict(program=p.name,loss=float(loss.detach()),parameter_gradient_norm=norm,state_adjoints={k:({'norm':float(g.norm()),'finite':bool(torch.isfinite(g).all()),'exact_zero':bool(g.eq(0).all())} if g is not None else {'disconnected':True}) for (k,t),g in zip(components,adj)}))
    assert before==model_hash(model),'Fixture parameters/buffers changed'
    torch.save(retained,OUT/f'fixture_complete_states_{device}.pt')
    r=dict(device=device,passed=True,completed_tokens=57,backward_VJP_evaluations=6,canonical_CE=ce,canonical_comparisons=comparisons,gradients=gradients,parameters_unchanged=True)
    save(f'fixtures_{device}.json',r);return r

def rollout(wrapper,row,x,ablate=False):
    wrapper.ablate_fifth=ablate
    state=wrapper.initial(x);losses=[];pred=[];logits=[];margins=[];query_index=0
    p=wrapper.active_program
    case={k:row[k] for k in ('id','family','variant','length')};case.update(ablation=ablate)
    for j in range(row['length']):
        out=wrapper(x[:,j],state,p,case);state=out['state']
        if j in row['query_positions']:
            log=out['logits'];assert log.shape==(1,2) and bool(torch.isfinite(log).all())
            label=row['labels'][query_index];query_index+=1
            losses.append(F.cross_entropy(log,torch.tensor([label],device=x.device)))
            pred.append(int(log.argmax(-1)));logits.append(log.detach().cpu()[0].tolist())
            margins.append(float((log[0,1]-log[0,0]).detach())*(2*label-1))
    assert query_index==8 and state.tokens_committed==row['length']
    loss=torch.stack(losses).mean();assert bool(torch.isfinite(loss))
    rec={k:row[k] for k in ('id','family','variant','length')}
    rec.update(program=p.name.rsplit('/',1)[1],S=int(pred==row['labels']),query_accuracy=sum(a==b for a,b in zip(pred,row['labels']))/8,CE=float(loss.detach()),margin=sum(margins)/8,predictions=pred,labels=row['labels'],logits=logits,query_positions=row['query_positions'],final_state_metadata=state_metadata(state),ablation=ablate)
    return loss,rec

def evaluate(wrapper,rows,encoded,phase,ledger,ablate=False):
    ledger.phase=phase;results=[]
    with torch.no_grad():
        for i,row in enumerate(rows):
            x=encoded[row['id']].to('cuda')
            for p in PROGRAMS:
                wrapper.active_program=p;_,record=rollout(wrapper,row,x,ablate);results.append(record)
            if (i+1)%16==0:print(f'{phase}: {i+1}/{len(rows)} episodes; tokens={ledger.completed}',flush=True)
    save(phase+'_results.json',results);return results

def train(wrapper,rows,encoded,order,ledger):
    ledger.phase='training';params=[p for p in wrapper.model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=0,betas=(.9,.999),eps=1e-8)
    assert len(order)==128 and set(order)==set(range(128))
    with (OUT/'training_updates.jsonl').open('w') as log:
        for update,index in enumerate(order,1):
            ledger.guard();optimizer.zero_grad(set_to_none=True);row=rows[index];x=encoded[row['id']].to('cuda');records=[]
            for p in PROGRAMS:
                wrapper.active_program=p;loss,rec=rollout(wrapper,row,x)
                (loss/3).backward();ledger.backward_evaluations+=1;records.append(rec)
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            before=[p.detach().clone() for p in params]
            optimizer.step();ledger.optimizer_updates+=1
            assert ledger.optimizer_updates==update<=128
            assert all(bool(torch.isfinite(p).all()) for p in params)
            delta=float(torch.stack([(p.detach()-old).square().sum() for p,old in zip(params,before)]).sum().sqrt());assert delta>0
            log.write(json.dumps(dict(update=update,episode_id=row['id'],family=row['family'],length=row['length'],routes=records,objective=sum(r['CE'] for r in records)/3,gradient_norm_before_clipping=float(norm),displacement_L2=delta,finite=True))+'\n');log.flush()
            ledger.flush_summary()
            if update%8==0:print(f'training: {update}/128; objective={sum(r["CE"] for r in records)/3:.6f}; tokens={ledger.completed}',flush=True)
    checkpoint=dict(model_state_dict={k:v.detach().cpu().clone() for k,v in wrapper.model.state_dict().items()},buffers={k:v.detach().cpu().clone() for k,v in wrapper.model.named_buffers()},optimizer_state_dict=optimizer.state_dict(),update=128,selection='predetermined_final_only',model_hash=model_hash(wrapper.model))
    torch.save(checkpoint,OUT/'checkpoint_update0128.pt')
    online=[rec for line in (OUT/'training_updates.jsonl').read_text().splitlines() for rec in json.loads(line)['routes']]
    online_summary={str(L):{p:metric([r for r in online if r['length']==L and r['program']==p]) for p in ('B','A','R')} for L in (64,96)}
    save('training_online_summary.json',dict(by_length_program=online_summary,interpretation='Online current-parameter rollouts, not a final-training-set evaluation'))
    return {'online_by_length_program':online_summary,'updates':128,'checkpoint':'checkpoint_update0128.pt','sha256':sha(OUT/'checkpoint_update0128.pt'),'model_hash':checkpoint['model_hash']}

def write_report(results,ledger):
    lines=['# Keyed-register qualification pilot v1',f'\n**Final classification: {results["classification"]}.**',f'\nProtocol SHA-256: `{sha(OUT/"protocol_approved_verbatim.md")}`.', '\nReceiver-anchored study remains COMPLETED_NON_SUCCESS; no historical outcome changed.', '\n## Phase status', '']
    for k,v in results['phases'].items():lines.append(f'- {k}: {v}')
    if 'qualification' in results:
        q=results['qualification'];lines += ['\n## Development results','\n| Length | Program | Complete accuracy | Query accuracy | Mean CE |','|---|---|---:|---:|---:|']
        for L,pg in q['by_length_program'].items():
            for p,v in pg.items():lines.append(f'| {L} | {p} | {v["complete_accuracy"]:.6f} | {v["query_accuracy"]:.6f} | {v["mean_CE"]:.6f} |')
        lines.append('\nSolvability/headroom/complementarity: '+str((q['solvability_pass'],q['headroom_pass'],q['complementarity_pass'])))
        lines.append('\nComplementarity: `'+json.dumps(q['complementarity'])+'`')
    lines += ['\n## Accounting', '\n```json',json.dumps(ledger.summary(),indent=2),'```','\n## Limits','\nOne initialization and one shared model. No learned selector, compatibility objective, length-transfer confirmation, profiling or model-selection sweep. Empirical oracle is clairvoyant and not deployable. Preliminary ablation, if executed, is descriptive only. No retry. Historical integrity exception preserved.']
    if 'failure' in results:lines.append('\nFailure: '+results['failure']['message'])
    (OUT/'final_report.md').write_text('\n'.join(lines)+'\n')

def main():
    if (OUT/'execution_started.json').exists():raise SystemExit('STOP: no resume/retry')
    check_seal();save('execution_started.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_resume':True})
    ledger=Ledger();results={'classification':'engineering/integrity/resource stop','phases':{}};wrapper=None
    try:
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU study substitution')
        torch.cuda.init();torch.cuda.reset_peak_memory_stats()
        protected=json.loads((OUT/'integrity_initial.json').read_text())['files']
        panels,encoded,order,diagnostic,shortcut_pass=seal_inputs(OUT,protected)
        input_names=['training_inputs.json','development_inputs.json','input_manifest.json','training_order.json','diagnostic_selection.json','exposure_inventory.json','generation_rejections.json','shortcut_records.json','shortcut_screen.json','symbolic_validation.json','sealed_encoded_inputs.pt']
        save('input_seal.json',{'files':{n:sha(OUT/n) for n in input_names},'sealed_before_model_outcomes':True})
        results['phases'].update(symbolic='PASS',encoding='PASS',counterfactual='PASS',shortcut='PASS' if shortcut_pass else 'FAIL')
        if not shortcut_pass:results['classification']='shortcut failure';return
        fixtures('cpu',ledger);results['phases']['cpu_fixtures']='PASS'
        fixtures('cuda',ledger);results['phases']['cuda_fixtures']='PASS'
        model=load_research_model('cuda');wrapper=CompleteProgramModule(model,ledger)
        initial=model_hash(model);controller=model_hash(model,True)
        save('initial_model_identity.json',dict(model_hash=initial,controller_hash=controller,parameter_count=sum(p.numel() for p in model.parameters()),trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),encoding_nontrainable=True))
        diag=[r for r in panels['development'] if r['id'] in diagnostic]
        initial_rows=evaluate(wrapper,diag,encoded,'initial_diagnostic',ledger);assert initial==model_hash(model)
        results['initial_diagnostic']={str(L):{p:metric([r for r in initial_rows if r['length']==L and r['program']==p]) for p in ('B','A','R')} for L in (64,96)}
        results['phases']['initial_diagnostic']='COMPLETE'
        results['training']=train(wrapper,panels['training'],encoded,order,ledger);results['phases']['training']='COMPLETE_128_UPDATES'
        final=evaluate(wrapper,panels['development'],encoded,'development',ledger);results['phases']['development']='COMPLETE_ALL_256_EPISODES_THREE_PROGRAMS'
        q=qualify(final);results['qualification']=q;save('qualification_analysis.json',q);results['phases']['qualification']=q['status']
        if q['status']!='CURRENT_GATES_PASS':results['classification']=q['status'];results['phases']['preliminary_ablation']='NOT_EXECUTED_FAILED_PREREQUISITE';return
        ablated=evaluate(wrapper,diag,encoded,'preliminary_ablation',ledger,ablate=True)
        a=ablation_summary(final,ablated);save('preliminary_ablation_analysis.json',a);results['ablation']=a
        results['phases']['preliminary_ablation']='COMPLETE_DESCRIPTIVE_ONLY';results['classification']=a['status']
    except BaseException as e:
        message=f'{type(e).__name__}: {e}'
        if message.startswith('RuntimeError: GENERATION_BLOCKED'):results['classification']='generation blocked'
        results['failure']={'message':message,'traceback':traceback.format_exc(),'phase':ledger.phase,'ledger':ledger.summary()};save('failure.json',results['failure']);print(message,flush=True)
    finally:
        try:
            if wrapper is not None:
                assert controller==model_hash(wrapper.model,True),'Frozen controller changed'
            check_seal()
            seal_path=OUT/'input_seal.json'
            if seal_path.exists():
                for n,h in json.loads(seal_path.read_text())['files'].items():assert sha(OUT/n)==h, 'Sealed input changed '+n
            ledger.guard();results['integrity']=verify_integrity()
        except BaseException as e:
            results['classification']='engineering/integrity/resource stop'
            results['termination_verification_failure']=f'{type(e).__name__}: {e}'
            save('termination_verification_failure.json',{'error':results['termination_verification_failure'],'traceback':traceback.format_exc()})
        ledger.close();results['ledger']=ledger.summary();save('final_results.json',results);write_report(results,ledger)
        print('FINAL_CLASSIFICATION: '+results['classification'],flush=True)

if __name__=='__main__':main()
