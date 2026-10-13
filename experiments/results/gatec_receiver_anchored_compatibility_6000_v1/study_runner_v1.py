"""One locked receiver-anchored study. No retries, additional variants or selector."""
from __future__ import annotations
import os,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
PRE=ROOT/'experiments/results/gatec_receiver_anchored_prerequisites_6000_v1'
sys.dont_write_bytecode=True
sys.path[:0]=[str(OUT),str(PRE),str(ROOT)]
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUDA_CACHE_PATH']=str(OUT/'runtime_cache')
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(OUT/'runtime_cache/inductor')
import collections,copy,ctypes,gzip,hashlib,json,math,random,subprocess,time,traceback
import torch
import torch.nn.functional as F
from study_adapter_v1 import (P0,P1,CompleteProgramModule,DetachedReceiver,load_research_model,
    rollout,episode,fields,snapshot,assert_equal,assert_disjoint)
from study_data_v1 import seal_data,file_sha
from statistical_rules_v3 import make_bank,stratified_means,quantile,ablation_rule,latency_rule

PROGRAMS=[P0,P1]
PHASE_LIMITS={'calibration':2064,'common_training':40960,'T_training':57344,'C_training':57344,
 'development':9216,'selection':73728,'confirmation':174080,'reference':18432,'ablation':73728,'profiling':26880}


def save_json(name,value):
    path=OUT/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def state_cpu(state):
    return {k:v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in fields(state).items()}


def model_hash(model):
    h=hashlib.sha256()
    for name,value in list(model.named_parameters())+list(model.named_buffers()):
        v=value.detach().cpu().contiguous()
        h.update(str((name,str(v.dtype),tuple(v.shape))).encode())
        h.update(ctypes.string_at(v.data_ptr(),v.numel()*v.element_size()))
    return h.hexdigest()


def integrity():
    original=json.loads((OUT/'integrity_before.json').read_text());bad=[]
    for name,expected in original['protected_files'].items():
        f=ROOT/name
        if not f.is_file() or file_sha(f)!=expected:bad.append(name)
    diff=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=ROOT)).hexdigest()
    extra=[str(f.relative_to(ROOT)) for f in ROOT.rglob('*') if f.is_file() and '.git' not in f.relative_to(ROOT).parts
           and not f.is_relative_to(OUT) and str(f.relative_to(ROOT)) not in original['protected_files']]
    result=dict(protected_count=len(original['protected_files']),changed_or_missing=bad,unexpected_outside=extra,
         tracked_diff_sha256=diff,tracked_diff_unchanged=diff==original['tracked_diff_sha256'],
         historical_exception=original['historical_exception'],success=not bad and not extra and diff==original['tracked_diff_sha256'])
    if not result['success']:raise RuntimeError('INTEGRITY FAILURE '+str(result))
    return result


class StudyLedger:
    def __init__(self,prior_seconds):
        self.carried=json.loads((PRE/'resource_ledger.json').read_text())
        self.block_ceiling=4274304
        self.blocks=self.carried['selected_bank_calls'];self.block_completed=self.blocks
        self.commits=0;self.commit_attempts=0;self.completed=self.attempted=self.failed=0
        self.optimizer_updates=0;self.backward_evaluations=0;self.warmups=0
        self.phase='calibration';self.phases=collections.Counter();self.phase_attempts=collections.Counter()
        self.rows=[];self.contextual=collections.Counter();self.selected=collections.Counter()
        self.start=time.perf_counter();self.prior_seconds=prior_seconds
        self.trace=gzip.open(OUT/'token_traces.jsonl.gz','wt',compresslevel=1)
        self.log=(OUT/'training_updates.jsonl').open('w')
        self.full_state_file_count=0

    def time_guard(self):
        if self.elapsed()+self.prior_seconds>=43200:raise RuntimeError('RESOURCE STOP: 12-hour ceiling')

    def elapsed(self):return time.perf_counter()-self.start

    def begin(self,case,program,device,token):
        self.time_guard()
        if self.attempted>=533776 or self.phase_attempts[self.phase]>=PHASE_LIMITS[self.phase]:
            raise RuntimeError('RESOURCE STOP: prescribed phase/study token allocation exhausted')
        if self.completed+self.carried['completed_tokens']>=534288 or self.blocks+8>self.block_ceiling:
            raise RuntimeError('RESOURCE STOP: overall execution ceiling')
        self.attempted+=1;self.phase_attempts[self.phase]+=1
        row=dict(attempt=self.attempted,phase=self.phase,case=case,program=program.name,device=str(device),token=token,
                 status='attempted',decisions=[],commits=0)
        self.rows=[row];return row

    def persist(self):
        if self.rows:
            row=self.rows[-1]
            if not row.get('_written'):
                public={k:v for k,v in row.items() if k!='_written'}
                self.trace.write(json.dumps(public)+'\n');row['_written']=True
                if row['status']=='completed':
                    self.phases[row['phase']]+=1
                    self.selected.update(row['executed_blocks'])
                elif row['status']=='failed':self.trace.flush()
        if self.attempted%128==0 or self.failed:
            self.trace.flush();self.record()

    def record(self):
        save_json('execution_ledger.json',dict(study_attempted_tokens=self.attempted,study_completed_tokens=self.completed,
          study_failed_tokens=self.failed,study_selected_calls=self.blocks-self.carried['selected_bank_calls'],
          overall_completed_tokens=self.completed+self.carried['completed_tokens'],
          overall_attempted_tokens=self.attempted+self.carried['attempted_tokens'],overall_selected_calls=self.blocks,
          study_commit_attempts=self.commit_attempts,study_commits=self.commits,backward_evaluations=self.backward_evaluations,
          optimizer_updates=self.optimizer_updates,phase_completed_tokens=dict(self.phases),phase_attempted_tokens=dict(self.phase_attempts),
          contextual_commit_calls=dict(self.contextual),selected_block_calls=dict(self.selected),
          profiling_warmup_tokens=self.warmups,elapsed_accelerator_window_seconds=self.elapsed(),
          carried_prerequisite_seconds=self.prior_seconds,carried_prerequisite_ledger=self.carried,
          peak_memory_allocated=torch.cuda.max_memory_allocated(),peak_memory_reserved=torch.cuda.max_memory_reserved(),
          full_confirmation_state_files=self.full_state_file_count))

    def close(self):
        self.persist();self.record();self.trace.close();self.log.close()


class StudyWrapper(CompleteProgramModule):
    def forward(self,*args,**kwargs):
        restored=[]
        def observer(original,kind):
            def call(*a,**k):
                self.ledger.contextual[kind]+=1
                return original(*a,**k)
            return call
        for block in list(self.model.blocks.blocks)+[self.model.prelude_block]:
            method='gdn_commit_entry' if block.species=='GDN' else 'attn_commit_entry' if block.species in ('SWA','GQA') else None
            if method:
                old=getattr(block,method);restored.append((block,method,old))
                setattr(block,method,observer(old,block.species+('/prelude' if block is self.model.prelude_block else '/bank')))
        try:
            out=super().forward(*args,**kwargs)
            state=out['state'];n=state.tokens_committed
            for c in state.ctx.kv.values():assert c.version==c.committed_length==c.next_position==n
            for c in state.ctx.swa.values():
                assert c.version==c.next_position==c.num_committed_tokens==n
                assert c.committed_length==min(n,c.window_size-1)
            for c in state.ctx.gdn.values():assert c.version==n
            if not torch.isfinite(out['logits']).all():raise RuntimeError('NONFINITE forward logits')
            return out
        finally:
            for block,method,old in restored:setattr(block,method,old)


def train_updates(wrapper,optimizer,schedule,encoded,arm,ledger,snapshots=None):
    ledger.phase='common_training' if arm=='common' else arm+'_training'
    params=[p for p in wrapper.model.parameters() if p.requires_grad]
    frozen_controller=model_hash(wrapper.model.controller)
    for update,pair in enumerate(schedule,1):
        ledger.time_guard();optimizer.zero_grad(set_to_none=True);losses=[];started=time.perf_counter()
        for case_id in pair:
            data=encoded[case_id];x=data['x'].cuda();labels=data['labels'].cuda()
            if arm=='common':
                n0,_,_=rollout(wrapper,x,labels,P0,f'{arm}/update{update}/{case_id}/P0')
                n1,_,_=rollout(wrapper,x,labels,P1,f'{arm}/update{update}/{case_id}/P1')
                loss=(n0+n1)/2
            else:
                loss,_=episode(wrapper,x,labels,arm,f'{arm}/update{update}/{case_id}')
            if not torch.isfinite(loss):raise RuntimeError('NONFINITE objective')
            losses.append(float(loss.detach()));(loss/2).backward();ledger.backward_evaluations+=1
        assert all(p.grad is None for p in wrapper.model.controller.parameters())
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        optimizer.step();ledger.optimizer_updates+=1
        if ledger.optimizer_updates>1536:raise RuntimeError('Optimizer ceiling')
        if not all(torch.isfinite(p).all() for p in params):raise RuntimeError('NONFINITE updated parameter')
        ledger.log.write(json.dumps(dict(arm=arm,update=update,case_ids=pair,episode_CE=losses,effective_batch_CE=sum(losses)/2,
               gradient_norm_before_clipping=float(norm),finite=True,elapsed_seconds=time.perf_counter()-started,
               cumulative_optimizer_steps=ledger.optimizer_updates))+'\n')
        if update%16==0:
            ledger.log.flush();ledger.record()
            print(f'{arm} update {update}/512 meanCE={sum(losses)/2:.6g} grad={float(norm):.6g} tokens={ledger.completed}',flush=True)
        if snapshots is not None and update in (256,512):
            snapshots[update]=save_checkpoint(wrapper.model,optimizer,f'{arm}_update{update:04}.pt',update,arm)
    assert model_hash(wrapper.model.controller)==frozen_controller


def save_checkpoint(model,optimizer,name,update,arm):
    path=OUT/'checkpoints'/name;path.parent.mkdir(exist_ok=True)
    if path.exists():raise RuntimeError('Refusing checkpoint overwrite')
    torch.save(dict(model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
             buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()},
             optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),update=update,arm=arm,
             protocol_sha256=file_sha(ROOT/'receiver_anchored_compatibility_protocol_v1.txt'),
             backbone_sha256='53b56e7e90f7eb7a6570107b6dfe3a297caac48d637fe27e5f00fbe94c6f09c3',
             input_manifest_sha256=file_sha(OUT/'input_manifest.json'),parameter_buffer_hash=model_hash(model)),path)
    return dict(path=str(path.relative_to(OUT)),sha256=file_sha(path),update=update,arm=arm)


def observed_sequence(wrapper,x,labels,program,case,initial=None,alternate=False):
    state=wrapper.initial(x) if initial is None else initial
    outputs=[]
    for j in range(x.shape[1]):
        chosen=PROGRAMS[1-PROGRAMS.index(program)] if alternate and j%2==0 else program
        out=wrapper(x[:,j],state,chosen,case);state=out['state']
        logits=out['logits'].detach().cpu()[0]
        outputs.append(dict(position=j,logits=logits.tolist(),prediction=int(logits.argmax()),label=int(labels[j]),
                       correct=int(logits.argmax())==int(labels[j]),CE=float(F.cross_entropy(logits.unsqueeze(0),labels[j:j+1].cpu())),
                       margin=float(logits[int(labels[j])]-logits[1-int(labels[j])])))
    return state,outputs


def native_case(wrapper,row,data,program,case):
    x=data['x'].cuda();labels=data['labels'].cuda()
    state,base=observed_sequence(wrapper,x,labels,program,case+'/base')
    suffixes=[]
    for zi,z in enumerate(row['suffixes']):
        zlabels=torch.tensor([(row['label']+sum(map(int,z[:j+1])))%2 for j in range(8)],device='cuda')
        _,outputs=observed_sequence(wrapper,data['suffixes'][zi].cuda(),zlabels,program,case+f'/z{zi}',initial=state.clone())
        suffixes.append(outputs)
    return dict(id=row['id'],length=row['length'],label=row['label'],program=program.name,
                base=base,current=base[-1],suffixes=suffixes,
                robust=base[-1]['correct'] and all(v['correct'] for values in suffixes for v in values),
                mean_CE=(base[-1]['CE']+sum(v['CE'] for values in suffixes for v in values))/17,
                native_prefix_mean_CE=sum(v['CE'] for v in base)/len(base))


def native_summary(results):
    summary={};passed=True
    for program in PROGRAMS:
        for length in (16,24):
            group=[r for r in results if r['program']==program.name and r['length']==length]
            n=len(group);assert n
            ce={0:sum(r['current']['CE'] for r in group)/n}
            ce.update({h:sum(r['suffixes'][z][h-1]['CE'] for r in group for z in (0,1))/(2*n) for h in (1,4,8)})
            current=sum(r['current']['correct'] for r in group)/n;robust=sum(r['robust'] for r in group)/n
            mean=sum(r['mean_CE'] for r in group)/n
            ok=current>=.99 and robust>=.98 and mean<=.05 and all(c<=.05 for c in ce.values())
            summary[program.name+f'/L{length}']=dict(n=n,current_accuracy=current,robust_accuracy=robust,
                   mean_CE=mean,horizon_CE=ce,native_prefix_mean_CE=sum(r['native_prefix_mean_CE'] for r in group)/n,passes=ok)
            passed=passed and ok
    return dict(passes=passed,strata=summary)


def evaluate_native(wrapper,panel,encoded,phase,name):
    wrapper.ledger.phase=phase;before=model_hash(wrapper.model);results=[]
    with torch.no_grad():
        for index,row in enumerate(panel):
            for program in PROGRAMS:results.append(native_case(wrapper,row,encoded[row['id']],program,name+'/'+row['id']+'/'+program.name))
            if (index+1)%16==0:print(f'{name} {index+1}/{len(panel)} cases',flush=True)
    assert model_hash(wrapper.model)==before
    save_json(name+'_native_results.json',results)
    summary=native_summary(results);save_json(name+'_native_summary.json',summary)
    return results,summary


def verify_common_fork(common,optimizer,ledger):
    models={arm:copy.deepcopy(common) for arm in ('T','C')};opts={}
    for arm,model in models.items():
        assert model.executor.block_bank is model.blocks
        assert_equal(snapshot(dict(common.named_parameters())),snapshot(dict(model.named_parameters())),exact=True)
        assert_equal(snapshot(dict(common.named_buffers())),snapshot(dict(model.named_buffers())),exact=True)
        for (_,a),(_,b) in zip(common.named_parameters(),model.named_parameters()):assert a.data_ptr()!=b.data_ptr()
        opts[arm]=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=0,betas=(.9,.999),eps=1e-8)
        opts[arm].load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for (_,a),(_,b) in zip(models['T'].named_buffers(),models['C'].named_buffers()):assert a.data_ptr()!=b.data_ptr()
    states=[optimizer]+list(opts.values())
    for triple in zip(*[[p for group in opt.param_groups for p in group['params']] for opt in states]):
        values=[opt.state[p] for opt,p in zip(states,triple)]
        assert values[0].keys()==values[1].keys()==values[2].keys()
        for key in values[0]:
            v=[s[key] for s in values]
            if torch.is_tensor(v[0]):
                assert torch.equal(v[0],v[1]) and torch.equal(v[1],v[2])
                assert len({t.data_ptr() for t in v})==3
    save_json('common_fork_verification.json',dict(parameter_buffer_hashes={k:model_hash(v) for k,v in models.items()},
            optimizer_state_equal_independently_owned=True,nonpersistent_buffers_copied=True))
    return {k:StudyWrapper(v,ledger) for k,v in models.items()},opts


def load_selected(wrapper,checkpoint):
    path=OUT/checkpoint['path'];assert file_sha(path)==checkpoint['sha256']
    ck=torch.load(path,map_location='cuda',weights_only=True)
    wrapper.model.load_state_dict(ck['model_state_dict'],strict=True)
    for name,buf in wrapper.model.named_buffers():buf.copy_(ck['buffers'][name])
    assert model_hash(wrapper.model)==ck['parameter_buffer_hash']


def calibration(wrapper,ledger):
    ledger.phase='calibration'
    control=json.loads((ROOT/'experiments/results/gatec_execution_accounting_v1/gatec_canonical_replay.json').read_text())
    from study_adapter_v1 import fixture_encoding
    x,labels=fixture_encoding(control['bits'][0],'cuda');a=wrapper.initial(x);b=wrapper.initial(x)
    comparisons=[]
    with torch.no_grad():
        for j in range(8):
            ra=wrapper(x[:,j],a,P0,'calibration/reference',reference=True)
            rb=wrapper(x[:,j],b,P0,'calibration/adapter')
            assert_equal(snapshot(ra['state']),snapshot(rb['state']))
            for key in ('h0','h_terminal','logits','memory_read','memory_attention'):assert torch.allclose(ra[key],rb[key],atol=1e-6,rtol=1e-5)
            comparisons.append(dict(token=j,complete_state_equal=True,logit_max_abs=float((ra['logits']-rb['logits']).abs().max())))
            a,b=ra['state'],rb['state']
        ce=float(F.cross_entropy(rb['logits'],labels[-1:]));assert abs(ce-.0014546061865985394)<=1e-6
    save_json('calibration.json',dict(comparisons=comparisons,final_CE=ce,passes=True))


def main():
    if (OUT/'execution_started.json').exists():raise SystemExit('STOP: study already started; no resume/retry')
    results=dict(status='PREPARATION',phases={});ledger=None
    try:
        integrity()
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        if not torch.cuda.is_available():raise RuntimeError('CUDA environment unavailable; no CPU substitution')
        panels,schedules,encoded,profile_order=seal_data(ROOT,OUT)
        # All scientific definitions and input identities are sealed before model construction.
        save_json('analysis_definitions.json',dict(native_mean_CE='current boundary plus both suffixes, 17 outputs per case; prefix readouts descriptive',
                 reported_horizon_CE='h0 current boundary; h1/h4/h8 endpoint means over both suffixes',
                 native_gate='current>=.99, robust>=.98, mean_CE<=.05 and each reported horizon_CE<=.05',
                 effective_batch='arithmetic mean of the two per-episode losses; physical B1; clip after both backward calls',
                 primary='equal length/parity weighted E[D_control-D_treatment]',
                 profiling='z0 only; 32 predetermined cases; one warmup+five timed; no reused prefixes',
                 source_of_constants='protocol_original_copy.txt plus clarifications_copy.md; latest user authorization enables full study'))
        hashes={p.name:file_sha(p) for p in OUT.glob('*.py')}
        save_json('implementation_seal.json',dict(source_hashes=hashes,validated_adapter_sha256=file_sha(PRE/'receiver_anchored_v2.py'),
                 adapter_difference='two fixture-only block assertions now use authorized ledger block_ceiling; exact diff retained',
                 protocol_sha256=file_sha(ROOT/'receiver_anchored_compatibility_protocol_v1.txt'),
                 clarification_sha256=file_sha(PRE/'protocol_clarifications_v1.md'),input_manifest_sha256=file_sha(OUT/'input_manifest.json'),
                 statistical_rules_sha256=file_sha(PRE/'statistical_rules_v3.py'),runtime=dict(torch=torch.__version__,device=torch.cuda.get_device_name(0),
                   dtype='float32',batch=1,TF32_matmul=False,TF32_cudnn=True,CPU_threads=1)))
        for p in OUT.glob('*.py'):compile(p.read_text(),str(p),'exec')
        prior_seconds=sum(json.loads(p.read_text()).get('elapsed_seconds',0) for p in PRE.glob('final_manifest*.json'))
        prior_seconds+=max(0,(PRE/'integrity_after_supplemental.json').stat().st_mtime-(PRE/'supplemental_started.json').stat().st_mtime)
        save_json('execution_started.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),no_restart=True,prior_prerequisite_seconds_charged=prior_seconds))
        ledger=StudyLedger(prior_seconds);model=load_research_model('cuda');wrapper=StudyWrapper(model,ledger)
        results['initial_parameter_count']=sum(p.numel() for p in model.parameters())
        results['trainable_parameter_count']=sum(p.numel() for p in model.parameters() if p.requires_grad)
        results['initial_model_hash']=model_hash(model)
        calibration(wrapper,ledger);results['phases']['calibration']='PASSED'
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=0,betas=(.9,.999),eps=1e-8)
        train_updates(wrapper,optimizer,schedules['common'],encoded,'common',ledger)
        results['common_checkpoint']=save_checkpoint(model,optimizer,'common_update0512.pt',512,'common')
        _,development=evaluate_native(wrapper,panels['development'],encoded,'development','development_common')
        results['development']=development
        results['phases']['common_acquisition']='PASSED' if development['passes'] else 'FAILED_NATIVE_COMPETENCE'
        integrity()
        if not development['passes']:
            results['status']='STOPPED_AT_COMMON_NATIVE_GATE';return
        arms,opts=verify_common_fork(model,optimizer,ledger);snapshots={a:{} for a in ('T','C')}
        # Same complete predetermined order; each arm starts from the exact common fork.
        # Run one matched update per arm at each position, no control/treatment tuning.
        for update,pair in enumerate(schedules['paired'],1):
            for arm in ('T','C'):
                train_one_paired_update(arms[arm],opts[arm],pair,encoded,arm,update,ledger)
                if update in (256,512):snapshots[arm][update]=save_checkpoint(arms[arm].model,opts[arm],f'{arm}_update{update:04}.pt',update,arm)
        results['phases']['paired_acquisition']='COMPLETED';results['snapshots']=snapshots
        evaluations={}
        for update in (256,512):
            evaluations[update]={}
            for arm in ('T','C'):
                load_selected(arms[arm],snapshots[arm][update])
                _,summary=evaluate_native(arms[arm],panels['selection'],encoded,'selection',f'selection_{arm}_{update}')
                evaluations[update][arm]=summary
        eligible=[u for u in (256,512) if all(evaluations[u][a]['passes'] for a in ('T','C'))]
        chosen=min(eligible,key=lambda u:(max(s['mean_CE'] for a in ('T','C') for s in evaluations[u][a]['strata'].values()),-u)) if eligible else 512
        results['selection']=dict(update=chosen,eligible=eligible,native_only=True,summaries=evaluations,
                                  acquisition_successful=bool(eligible))
        for a in ('T','C'):load_selected(arms[a],snapshots[a][chosen])
        confirmation=confirmation_all(arms,panels['confirmation'],encoded,ledger)
        frozen=StudyWrapper(load_research_model('cuda'),ledger)
        for p in frozen.model.parameters():p.requires_grad_(False)
        archival=evaluate_frozen(frozen,panels['confirmation'],encoded,ledger)
        ablated=ablations_all(arms,panels['confirmation'],encoded,ledger)
        profiling=profile_all(arms,frozen,panels['confirmation'],encoded,profile_order,ledger)
        from study_analysis_v1 import analyze
        results['analysis']=analyze(confirmation,ablated,profiling,panels['confirmation'],bool(eligible),OUT)
        results['phases'].update(confirmation='COMPLETED',reference='COMPLETED',ablation='COMPLETED',profiling='COMPLETED',analysis='COMPLETED')
        results['status']='FULL_SUCCESS' if results['analysis']['full_success'] else 'COMPLETED_NON_SUCCESS'
        for phase,maximum in PHASE_LIMITS.items():assert ledger.phases[phase]==maximum,(phase,ledger.phases[phase],maximum)
    except BaseException as error:
        results['status']='STOPPED_ENGINEERING_OR_RESOURCE';results['failure']=dict(error=str(error),traceback=traceback.format_exc())
        save_json('failure.json',results['failure'])
        print('STOP:',str(error),flush=True)
    finally:
        if ledger is not None:ledger.close()
        try:results['integrity']=integrity()
        except BaseException as error:results['integrity']=dict(success=False,error=str(error));results['status']='STOPPED_INTEGRITY'
        save_json('final_results.json',results);write_report(results)


def train_one_paired_update(wrapper,opt,pair,encoded,arm,update,ledger):
    ledger.phase=arm+'_training';opt.zero_grad(set_to_none=True);losses=[]
    params=[p for p in wrapper.model.parameters() if p.requires_grad]
    for cid in pair:
        d=encoded[cid];loss,_=episode(wrapper,d['x'].cuda(),d['labels'].cuda(),arm,f'{arm}/update{update}/{cid}')
        if not torch.isfinite(loss):raise RuntimeError('NONFINITE paired objective')
        losses.append(float(loss.detach()));(loss/2).backward();ledger.backward_evaluations+=1
    assert all(p.grad is None for p in wrapper.model.controller.parameters())
    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    opt.step();ledger.optimizer_updates+=1
    if ledger.optimizer_updates>1536 or not all(torch.isfinite(p).all() for p in params):raise RuntimeError('Optimizer/nonfinite stop')
    ledger.log.write(json.dumps(dict(arm=arm,update=update,case_ids=pair,episode_CE=losses,effective_batch_CE=sum(losses)/2,
                     gradient_norm_before_clipping=float(norm),finite=True,cumulative_optimizer_steps=ledger.optimizer_updates))+'\n')
    if update%16==0:
        ledger.log.flush();ledger.record();print(f'{arm} update{update}/512 meanCE={sum(losses)/2:.6g}',flush=True)


def write_report(results):
    ledger=json.loads((OUT/'execution_ledger.json').read_text()) if (OUT/'execution_ledger.json').exists() else {}
    lines=['# Receiver-anchored compatibility study', '', 'Status: **'+results['status']+'**.', '',
           'The original protocol and final prospective clarifications are preserved as separate files. New P1 is distinct from rejected v4 P1.',
           '', '## Phase decisions', '',json.dumps(results.get('phases',{}),indent=2), '', '## Development/native gate', '',
           '```json',json.dumps(results.get('development',{}),indent=2),'```','',
           '## Selection and final acceptance', '', '```json',json.dumps(results.get('selection',{}),indent=2),'```',
           '```json',json.dumps(results.get('analysis',{}),indent=2),'```', '',
           '## Actual resources', '', '```json',json.dumps(ledger,indent=2),'```', '',
           '## Integrity', '', '```json',json.dumps(results.get('integrity',{}),indent=2),'```','',
           'Task errors do not remove cases. Unexecuted phases have no scientific verdict. No results from prior attempts or prerequisites are pooled into outcomes.',
           'Unknown Stage1 exposure remains a limitation; held-out means disjoint from documented retained acquisition/selection inputs.',
           'Unsupported FLOPs are not estimated. Statistical intervals are conditional on these models, not training-seed replication.',
           'No production edits, prior artifact overwrite, selector, additional variant, replacement seed or Git commit is permitted.',
           'Stop boundary: this single study. No automatic retry or follow-up is authorized.']
    if results.get('status')=='STOPPED_AT_COMMON_NATIVE_GATE':
        lines += ['', 'Common native acquisition did not meet the locked development requirements. Paired acquisition, selection, switching confirmation, reference panel, ablation, profiling and final inferential analysis were NOT RUN. Compatibility training remains untested; this acquisition formulation closes on parity under its declared budget.']
    if 'failure' in results:lines+=['','## Failure','', '```json',json.dumps(results['failure'],indent=2),'```']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


# Further stage definitions are below; all are sealed before Phase A starts.

def confirmation_all(arms,panel,encoded,ledger):
    collected={a:[] for a in ('T','C')}
    (OUT/'confirmation_states').mkdir()
    with torch.no_grad():
        for arm,wrapper in arms.items():
            before=model_hash(wrapper.model)
            for index,row in enumerate(panel):
                data=encoded[row['id']];x=data['x'].cuda();labels=data['labels'].cuda()
                incoming={};prefixes={};states={};cells={}
                ledger.phase='confirmation'
                for h in (0,1):
                    incoming[h],prefixes[h]=observed_sequence(wrapper,x[:,:-1],labels[:-1],PROGRAMS[h],f'{arm}/{row["id"]}/history{h}')
                assert_disjoint(incoming[0],incoming[1])
                for h in (0,1):
                    parent_before=snapshot(incoming[h])
                    for b in (0,1):
                        ledger.phase='confirmation'
                        out=wrapper(x[:,-1],incoming[h].clone(),PROGRAMS[b],f'{arm}/{row["id"]}/boundary{h}{b}')
                        states[f'{h}{b}']=out['state']
                        logits=out['logits'].detach().cpu()[0]
                        current=dict(logits=logits.tolist(),prediction=int(logits.argmax()),label=row['label'],
                              correct=int(logits.argmax())==row['label'],CE=float(F.cross_entropy(logits[None],labels[-1:].cpu())))
                        comparison=None
                        if b==0:
                            ledger.phase='calibration'
                            ref=wrapper(x[:,-1],incoming[h].clone(),P0,f'{arm}/{row["id"]}/supplied-reference{h}',reference=True)
                            assert_equal(snapshot(ref['state']),snapshot(out['state']))
                            for key in ('logits','h0','h_terminal','memory_read','memory_attention'):
                                assert torch.allclose(ref[key],out[key],atol=1e-6,rtol=1e-5)
                            comparison=dict(full_returned_state_equal=True,logit_max_abs=float((ref['logits']-out['logits']).abs().max()),
                                            target='same research parameters and exact incoming history state')
                        futures={}
                        for schedule in ('boundary','alternating'):
                            values=[]
                            for zi,z in enumerate(row['suffixes']):
                                zlabels=torch.tensor([(row['label']+sum(map(int,z[:j+1])))%2 for j in range(8)],device='cuda')
                                ledger.phase='confirmation'
                                _,outputs=observed_sequence(wrapper,data['suffixes'][zi].cuda(),zlabels,PROGRAMS[b],
                                      f'{arm}/{row["id"]}/cell{h}{b}/{schedule}/z{zi}',initial=out['state'].clone(),alternate=schedule=='alternating')
                                values.append(outputs)
                            futures[schedule]=values
                        cells[f'{h}{b}']=dict(current=current,futures=futures,reference_comparison=comparison,
                              robust=current['correct'] and all(v['correct'] for outputs in futures['boundary'] for v in outputs),
                              alternating_robust=current['correct'] and all(v['correct'] for outputs in futures['alternating'] for v in outputs))
                    assert_equal(parent_before,snapshot(incoming[h]),exact=True)
                    assert_disjoint(states[f'{h}0'],states[f'{h}1'])
                path=OUT/'confirmation_states'/f'{arm}_L{row["length"]}_row{index:03}.pt'
                assert not path.exists()
                torch.save(dict(id=row['id'],arm=arm,input=row,incoming={h:state_cpu(s) for h,s in incoming.items()},
                       boundary={k:state_cpu(s) for k,s in states.items()},model_parameter_buffer_hash=before),path)
                ledger.full_state_file_count+=1
                collected[arm].append(dict(id=row['id'],length=row['length'],label=row['label'],cells=cells,prefix_readouts=prefixes,
                                          state_path=str(path.relative_to(OUT))))
                if (index+1)%16==0:
                    save_json(f'confirmation_{arm}.json',collected[arm]);ledger.record()
                    print(f'confirmation {arm} {index+1}/{len(panel)} cases',flush=True)
            assert model_hash(wrapper.model)==before
            save_json(f'confirmation_{arm}.json',collected[arm])
    return collected


def evaluate_frozen(wrapper,panel,encoded,ledger):
    ledger.phase='reference';records=[]
    with torch.no_grad():
        for row in panel:records.append(native_case(wrapper,row,encoded[row['id']],P0,'archival/'+row['id']))
    save_json('frozen_reference_results.json',records)
    return records


def ablations_all(arms,panel,encoded,ledger):
    ledger.phase='ablation';result={a:[] for a in ('T','C')}
    # Execute the normal operation and suppress only its returned live-hidden output.
    # The existing complete-program observer counts and validates that original call.
    with torch.no_grad():
        for arm,wrapper in arms.items():
            before=model_hash(wrapper.model)
            for row in panel:
                for p,program in enumerate(PROGRAMS):
                    ex=wrapper.model.executor;original=ex._run_block;position={'value':0}
                    def substituted(block_index,h,h0,memory,context,**kwargs):
                        out=original(block_index,h,h0,memory,context,**kwargs)
                        d=position['value']%8;position['value']+=1
                        if d==4:
                            assert block_index==(4 if p==0 else 3)
                            return h
                        return out
                    ex._run_block=substituted
                    try:record=native_case(wrapper,row,encoded[row['id']],program,f'ablation/{arm}/{row["id"]}/P{p}')
                    finally:ex._run_block=original
                    record['counterfactual']='normal fifth selected operation executed; returned live output replaced with its input at every token'
                    result[arm].append(record)
            assert model_hash(wrapper.model)==before
            save_json(f'ablation_{arm}.json',result[arm])
    return result


def profile_all(arms,frozen,panel,encoded,order,ledger):
    """Clean deployment path; no forensic tensor cloning or branch orchestration."""
    ledger.phase='profiling';models={'T':arms['T'].model,'C':arms['C'].model,'F':frozen.model}
    byid={r['id']:r for r in panel};records=[];hashes={k:model_hash(v) for k,v in models.items()}
    events={}
    def metadata(state):
        return {k:(tuple(v.shape),str(v.dtype),v._version) if torch.is_tensor(v) else v for k,v in fields(state).items()}
    with torch.no_grad():
        for job in order:
            arm,pt=job['path'].split('/');program=PROGRAMS[int(pt[1])];model=models[arm];row=byid[job['case']]
            base=encoded[row['id']]['x'];z=encoded[row['id']]['suffixes'][0];x=torch.cat([base,z],dim=1).cuda()
            old=(model.executor.controller,model.executor._run_block,model.memory.read,model.prelude.forward)
            import src.agt as am
            import src.terminal_commit as cm
            saved_commit=am.terminal_commit;saved_plan=cm.build_commit_plan;saved_apply=cm.apply_commit_plan
            event_lists=collections.defaultdict(list);block_ids=[];commits={'value':0};current={'d':0};commit_methods=[]
            def timed(original,kind):
                def invoke(*a,**k):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record();out=original(*a,**k);end.record();event_lists[kind].append((start,end));return out
                return invoke
            def provider(**features):
                d=current['d'];assert d<8
                current['d']+=1
                v=features['h'].new_full((1,11),-100);v[:,program.actions[d]]=100
                return {'logits':v}
            def run_block(block_index,h,h0,memory,context,**kw):
                d=len(block_ids)%8;assert block_index==program.validate()[d]['block']
                ledger.blocks+=1;ledger.block_completed+=1;ledger.selected[block_index]+=1
                block_ids.append(block_index)
                return timed(old[1],'selected_bank')(block_index,h,h0,memory,context,**kw)
            def commit(*a,**k):
                commits['value']+=1;ledger.commit_attempts+=1;out=saved_commit(*a,**k);ledger.commits+=1
                return out
            try:
                model.executor.controller=provider;model.executor._run_block=run_block
                model.memory.read=timed(old[2],'memory_read');model.prelude.forward=timed(old[3],'prelude')
                cm.build_commit_plan=timed(saved_plan,'terminal_plan');cm.apply_commit_plan=timed(saved_apply,'commit_apply_cache_updates')
                am.terminal_commit=commit
                for block in list(model.blocks.blocks)+[model.prelude_block]:
                    method='gdn_commit_entry' if block.species=='GDN' else 'attn_commit_entry' if block.species in ('SWA','GQA') else None
                    if method:
                        original=getattr(block,method);kind=block.species+('/prelude' if block is model.prelude_block else '/bank')
                        def counted(*a,_original=original,_kind=kind,**k):
                            ledger.contextual[_kind]+=1;return _original(*a,**k)
                        commit_methods.append((block,method,original));setattr(block,method,counted)
                torch.cuda.synchronize();started=time.perf_counter()
                state=model.initial_persistent_state(1,'cuda',torch.float32);outputs=[];completed_rows=[]
                for j in range(x.shape[1]):
                    item=ledger.begin(f'profiling/{job["case"]}/{job["path"]}/rep{job["repetition"]}',program,'cuda',j)
                    prior=metadata(state);current['d']=0
                    out=model.adaptive_differentiable_step(x[:,j],state,x.new_zeros(1,64),budget=x.new_zeros(1,2),training=False)
                    assert metadata(state)==prior and current['d']==8 and out['execution_count']==8
                    state=out['state'];assert state.tokens_committed==j+1
                    for c in state.ctx.kv.values():assert c.version==c.committed_length==c.next_position==j+1
                    for c in state.ctx.swa.values():
                        assert c.version==c.next_position==c.num_committed_tokens==j+1
                        assert c.committed_length==min(j+1,c.window_size-1)
                    for c in state.ctx.gdn.values():assert c.version==j+1
                    item.update(status='completed',D=8,P=8,commits=1,executed_blocks=block_ids[-8:],
                                termination='execution_budget_reached',warmup=job['warmup'])
                    completed_rows.append(item);ledger.completed+=1;ledger.phases['profiling']+=1
                    if job['warmup']:ledger.warmups+=1
                    outputs.append(out['logits'])
                torch.cuda.synchronize();seconds=time.perf_counter()-started
                assert commits['value']==x.shape[1]
                component={k:sum(a.elapsed_time(b) for a,b in values)/1000 for k,values in event_lists.items()}
                # Forensic persistence happens AFTER timed trajectory. All calls remain in ledger.
                for item,output in zip(completed_rows,outputs):
                    item['logits']=output.cpu().tolist();item['_written']=True
                    ledger.trace.write(json.dumps({k:v for k,v in item.items() if k!='_written'})+'\n')
                records.append(dict(**job,length=row['length'],label=row['label'],seconds=seconds,component_seconds=component,
                                    tokens=x.shape[1],selected_blocks=len(block_ids),commits=commits['value'],z0=row['suffixes'][0]))
            finally:
                model.executor.controller,model.executor._run_block,model.memory.read,model.prelude.forward=old
                am.terminal_commit=saved_commit;cm.build_commit_plan=saved_plan;cm.apply_commit_plan=saved_apply
                for block,method,original in commit_methods:setattr(block,method,original)
            if len(records)%32==0:save_json('profiling_results.json',records);ledger.record()
    assert all(model_hash(models[a])==h for a,h in hashes.items())
    save_json('profiling_results.json',records)
    return records


if __name__=='__main__':main()
