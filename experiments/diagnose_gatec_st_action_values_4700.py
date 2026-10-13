"""Locked, no-update local ST versus hard-continuation diagnostic.

Calls the existing model and routing primitives. Only CF decision zero is
intervened on; ST probes return the real routing result unchanged.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from collections import Counter
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import gzip, json, math, subprocess, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from data.task_a import TaskAParity
from src.masks import build_execution_mask
from experiments.run_gatec_paired_task_maintenance import (
    IndexedNoise, StochasticPath, gpu_snapshot, versions, assert_same, graph_state, trace_path, refs,
)
from experiments.run_gatec_fresh_task_discovery import anchor_guard
from experiments.replicate_gatec_locked_imitation import make_model
from experiments.replay_gatec_learned_policy import (
    ObservedPath, sha, copy, snapshot, unchanged, state_record, git_identity, encode,
)
from experiments.evaluate_gatec_locked_policy import write_json, record

PARENT=ROOT/'experiments/results/gatec_fresh_task_discovery_4700_v1'
PRECHECK=ROOT/'experiments/results/gatec_st_action_value_4700_identity_precheck_v1.json'
RESOLUTION=ROOT/'experiments/results/gatec_st_action_value_4700_protocol_resolution_v1.json'
OUT=ROOT/'experiments/results/gatec_st_action_value_4700_v1'
NORM_EPS=1e-8

def banks(rows,seed,count=8):
    """Precompute ordered tapes; no RNG stream advances during execution."""
    generator=torch.Generator(device='cpu').manual_seed(seed)
    return [{s['sequence_id']:torch.rand(s['length'],15,10,generator=generator,dtype=torch.float32)
             for s in rows} for _ in range(count)]

def extend_prefix_noise(rows,training,historical,seed):
    generator=torch.Generator(device='cpu').manual_seed(seed)
    assert list(historical)==[s['sequence_id'] for s in rows]
    for s in rows:
        recreated=torch.rand(s['length'],15,10,generator=generator,dtype=torch.float32)
        assert torch.equal(recreated,historical[s['sequence_id']]),'Historical tape reconstruction failed'
    assert all(s['sequence_id'] not in historical for s in training)
    return {s['sequence_id']:torch.rand(s['length'],15,10,generator=generator,dtype=torch.float32) for s in training}

def tape(draws,seed):
    result=IndexedNoise.__new__(IndexedNoise)
    result.draws,result.seed=draws,seed
    return result

def categorical_gradient(logits,q):
    """Nominal hard categorical law; tau affects relaxation, not argmax law."""
    logits=torch.as_tensor(logits,dtype=torch.float64)
    q=torch.as_tensor(q,dtype=torch.float64)
    pi=logits.softmax(-1)
    benchmark=pi*(q-(pi*q).sum())
    independent=logits.clone().requires_grad_(True)
    check=torch.autograd.grad((independent.softmax(-1)*q.detach()).sum(),independent)[0]
    torch.testing.assert_close(benchmark,check,atol=1e-12,rtol=1e-10)
    return pi,benchmark

def compare_banks(logits,q,gradients,correct,cf_indices,st_indices,pi_other):
    q=torch.tensor(q,dtype=torch.float64) # [tape, action], action pairing retained
    correct=torch.tensor(correct,dtype=torch.float64)
    pi,gcf=categorical_gradient(logits,q.mean(0))
    sampled_q=q[cf_indices].mean(1)
    pairwise=[]
    for a in range(8):
        for b in range(a+1,8):
            deltas=q[:,a]-q[:,b];boot=sampled_q[:,a]-sampled_q[:,b]
            interval=torch.quantile(boot,torch.tensor([.025,.975],dtype=torch.float64)).tolist()
            halves=[float(deltas[:4].mean()),float(deltas[4:].mean())]
            resolved=(halves[0]*halves[1]>0 and (interval[0]>0 or interval[1]<0))
            pairwise.append(dict(actions=[a,b],definition='Q(a)-Q(b)',paired_tape_deltas=deltas.tolist(),mean=float(deltas.mean()),
                half_bank_means=halves,bootstrap_95=interval,resolved=resolved))
    result=dict(Q=q.mean(0).tolist(),Q_tape_sd=q.std(0).tolist(),conditional_correctness=correct.mean(0).tolist(),
        pi=pi.tolist(),entropy=float(-(pi*pi.log()).sum()),max_probability=float(pi.max()),
        saturated_probability=bool(pi.max()>=.99),g_CF=gcf.tolist(),g_CF_norm=float(gcf.norm()),
        paired_action_loss_differences=pairwise,informative_action_values=any(p['resolved'] for p in pairwise),
        flat_action_values=bool((q-q[:,:1]).abs().max()<=NORM_EPS),
        bootstrap_uncertainty='conditional Monte Carlo; eight paired CF tapes and eight independent ST tapes resampled separately; pair intervals unadjusted for multiplicity')
    delta_pi=torch.tensor(pi_other[1],dtype=torch.float64)-torch.tensor(pi_other[0],dtype=torch.float64)
    shifted=(sampled_q*delta_pi).sum(-1)
    result['checkpoint_probability_shift']=dict(value=float((delta_pi*q.mean(0)).sum()),half_bank_values=[float((delta_pi*x.mean(0)).sum()) for x in [q[:4],q[4:]]],
        bootstrap_95=torch.quantile(shifted,torch.tensor([.025,.975],dtype=torch.float64)).tolist(),
        definition='sum_a (pi1040(a)-pi0(a))*Q_owner(a); identical frozen state, downstream policy fixed to owner checkpoint')
    disconnected=sum(g is None for g in gradients)
    result['disconnected_ST_tapes']=disconnected
    if disconnected:
        result.update(mean_ST_gradient=None,inner_product=None,cosine=None,inner_product_bootstrap_95=None,
            direction='unresolved_disconnected',classification='informative_disconnected_temporal_credit' if result['informative_action_values'] else 'unresolved_action_values')
        return result
    gs=torch.tensor(gradients,dtype=torch.float64)[:,:8]
    mean=gs.mean(0);inner=float(gcf@mean)
    halves=[]
    for qq,gg in [(q[:4],gs[:4]),(q[4:],gs[4:])]:
        _,g=categorical_gradient(logits,qq.mean(0));halves.append(float(g@gg.mean(0)))
    sampled_gcf=pi*(sampled_q-(sampled_q*pi).sum(-1,keepdim=True))
    sampled_st=gs[st_indices].mean(1)
    boot_inner=(sampled_gcf*sampled_st).sum(-1)
    interval=torch.quantile(boot_inner,torch.tensor([.025,.975],dtype=torch.float64)).tolist()
    meaningful=float(gcf.norm())>NORM_EPS and float(mean.norm())>NORM_EPS
    direction='unresolved'
    if meaningful and min(halves)>0 and interval[0]>0:direction='aligned'
    if meaningful and max(halves)<0 and interval[1]<0:direction='opposed'
    result.update(mean_ST_gradient=mean.tolist(),ST_norm=float(mean.norm()),ST_tape_norms=gs.norm(dim=1).tolist(),
        ST_component_tape_sd=gs.std(0).tolist(),exact_zero_ST_tapes=int(gs.eq(0).all(1).sum()),small_connected_ST_tapes=int((gs.norm(dim=1)<=NORM_EPS).sum()),
        inner_product=inner,half_bank_inner_products=halves,inner_product_bootstrap_95=interval,direction=direction,
        cosine=float(inner/(gcf.norm()*mean.norm())) if meaningful else None)
    if not result['informative_action_values']:
        classification='flat_action_values' if result['flat_action_values'] else 'unresolved_action_values'
    elif direction=='opposed':classification='informative_opposed_ST'
    elif direction=='aligned':classification='informative_aligned_ST'
    elif gs.eq(0).all():classification='informative_connected_zero_credit'
    else:classification='informative_unresolved_direction'
    result['classification']=classification
    return result

class CaptureInitialization(Exception):pass

def force_advance_candidate(result,action):
    assert 0<=action<8
    result=dict(result)
    result['selected_action']=torch.tensor([action],device=result['action'].device)
    result['action']=F.one_hot(result['selected_action'],10).to(result['action'])
    result['h_next']=result['candidates'][:,action]
    result['next_pointer']=result['selected_action'].clone()
    return result

def initialized_features(model,state,x_t):
    """Normal macro initialization; intercept before execution, not after commit."""
    original=model.executor.execute_differentiable
    captured={}
    before=gpu_snapshot(state)
    def capture(**kwargs):
        captured.update(kwargs)
        raise CaptureInitialization()
    model.executor.execute_differentiable=capture
    try:
        with torch.no_grad():
            try:model.adaptive_differentiable_step(x_t,state,x_t.new_zeros(1,64),budget=x_t.new_zeros(1,2),tau=2.,training=True)
            except CaptureInitialization:pass
    finally:model.executor.execute_differentiable=original
    assert captured and state.tokens_committed>=0
    assert_same(before,state)
    onehot=x_t.new_zeros(1,9);onehot[:,0]=1
    return dict(h=captured['h0'].clone(),memory_read=captured['memory'],memory_mean=captured['memory_mean'],
                pointer_one_hot=onehot,budget=captured['budget']),captured

class ContinuationPath(StochasticPath):
    """Original macro steps from a cloned fixed token-boundary state."""
    def __init__(self,model,noise,sid,start,forced=None,gradient=False,expected=None):
        super().__init__(model,noise,sid,gradient)
        self.start,self.forced,self.expected=start,forced,expected

    @contextmanager
    def observers(self):
        ex=self.model.executor
        original=ex.differentiable_route_step
        first=True
        def intervene(*args,**kwargs):
            nonlocal first
            result=original(*args,**kwargs)
            if first and self.forced is not None:
                assert kwargs['execution_count']==0 and kwargs['pointer']==0 and not kwargs['just_restarted']
                result=force_advance_candidate(result,self.forced)
            first=False
            return result
        ex.differentiable_route_step=intervene
        try:
            with super().observers():
                observed=ex.differentiable_route_step
                def reconstruct(*args,**kwargs):
                    result=observed(*args,**kwargs);row=self.current[-1]
                    initial=self.token==self.start and row['decision']==0
                    with torch.no_grad():
                        raw=row['raw_controller_logits'][:,:10].to(kwargs['h'])
                        mask=row['legal_successor_mask'].to(raw.device).unsqueeze(0)
                        u=self.tape.get(self.sequence,self.token,row['decision']).to(raw)
                        eps=torch.finfo(raw.dtype).eps;u=u.clamp(eps,1-eps)
                        z=((F.log_softmax(raw.masked_fill(~mask,-torch.inf),-1)-torch.log(-torch.log(u)))/2).softmax(-1)
                        sampled=int(z.argmax(-1))
                        if not(initial and self.forced is not None):assert sampled==row['selected_action']
                        row.update(forced_probe_action=initial and self.forced is not None,
                            unintervened_sampled_action=sampled,noise_reconstruction_verified=True)
                        if initial:
                            assert kwargs['execution_count']==0 and kwargs['pointer']==0 and not kwargs['just_restarted']
                            assert mask[0,:8].all() and not mask[0,8:].any()
                            if self.expected:
                                for key,value in self.expected.items():torch.testing.assert_close(row[{'h':'h_before'}.get(key,key)],copy(value),atol=0,rtol=0)
                    return result
                ex.differentiable_route_step=reconstruct
                try:yield
                finally:ex.differentiable_route_step=observed
        finally:ex.differentiable_route_step=original

    def run_span(self,x,targets,initial,start,stop):
        state=initial.clone();initial_snapshot=gpu_snapshot(initial)
        with anchor_guard(self.model),self.observers():
            for self.token in range(start,stop):
                self.current=[];self.commit_count=self.read_count=0
                self.input_state=state;self.input_snapshot=gpu_snapshot(state);self.input_versions=versions(state)
                incoming=state_record(state)
                out=self.model.adaptive_differentiable_step(x[:,self.token],state,x.new_zeros(1,64),budget=x.new_zeros(1,2),tau=2.,training=True)
                assert torch.isfinite(out['logits']).all()
                P=sum(row['selected_action']!=9 for row in self.current);R=sum(row['restart'] for row in self.current)
                assert out['execution_count']==P==8 and len(out['action_history'])==len(self.current)==8+R
                assert 8<=len(self.current)<=15 and not out['halted'] and not out['safety_cap_hit']
                assert out['action_history']==[('ADVANCE',row['selected_action']) if row['selected_action']<8 else ('REPEAT',row['pointer_before']) if row['selected_action']==8 else ('RESTART',None) for row in self.current]
                assert self.commit_count==self.read_count==1
                assert out['state'].tokens_committed==self.token+1
                assert_same(self.input_snapshot,state)
                self.rows.extend(self.current)
                self.tokens.append(dict(token=self.token,P=P,D=len(self.current),history=out['action_history'],
                    restart_count=R,repeat_count=sum(row['repeat'] for row in self.current),
                    executed_blocks=[row['actual_executed_block'] for row in self.current if row['actual_executed_block'] is not None],
                    termination_reason='execution_budget_reached',commit_count=1,commit_entries=7,
                    incoming_persistent=incoming,outgoing_persistent=state_record(out['state']),graph_state=graph_state(out['state']),
                    output_logits=copy(out['logits']),prediction=int(out['logits'].argmax(-1)),target=int(targets[:,self.token])))
                state=out['state'] # No continuation detach or truncation.
        assert_same(initial_snapshot,initial)
        self.final_state,self.final_logits=state,out['logits'] if stop>start else None
        return self

class CanonicalPrefix(ObservedPath):
    def run_prefix(self,x,targets,stop):
        state=self.model.initial_persistent_state(1,x.device,x.dtype)
        with self.observers():
            for self.token in range(stop):
                self.current=[];self.commit_count=self.read_count=0
                self.input_state,self.incoming_snapshot=state,snapshot(state)
                out=self.model.step(x[:,self.token],state,x.new_zeros(1,64))
                assert len(self.current)==8 and self.commit_count==self.read_count==1
                assert unchanged(self.incoming_snapshot,snapshot(state))
                state=out['state']
                self.rows.extend(self.current)
                self.tokens.append(dict(token=self.token,P=8,D=8,history=[('ADVANCE',i) for i in range(8)],
                    executed_blocks=list(range(8)),restart_count=0,repeat_count=0,commit_count=1,
                    incoming_persistent=state_record(self.input_state),outgoing_persistent=state_record(state)))
        self.final_state=state
        return self

def main():
    assert sys.dont_write_bytecode
    if OUT.exists():raise RuntimeError('Refusing to repeat or overwrite diagnostic artifacts')
    parent=json.loads((PARENT/'manifest.json').read_text());prior=json.loads((PARENT/'results.json').read_text())
    pre=json.loads(PRECHECK.read_text());resolved=json.loads(RESOLUTION.read_text())
    historical_index=json.loads((PARENT/'artifact_index.json').read_text())
    for path,info in historical_index['files'].items():assert sha(ROOT/path)==info['sha256']
    for path,digest in parent['source_sha256'].items():assert sha(ROOT/path)==digest
    att=json.loads((PARENT/'encoding_attestation.json').read_text())
    for path,digest in att['helper_source_sha256'].items():assert sha(ROOT/path)==digest
    assert sha(ROOT/'data/task_a.py')==att['dataset_source_sha256']
    assert git_identity()['tracked_diff_sha256']==pre['git']['tracked_diff_sha256']
    rows=pre['six_sequences'];training=json.loads((PARENT/'training_sequences.json').read_text())['rows']
    panel=json.loads((PARENT/'final_panel.json').read_text())['rows']
    training_extension=[s for s in training if s['sequence_id'] in {r['sequence_id'] for r in rows if r['source']=='training'}]
    assert len(rows)==6 and len(training_extension)==2
    noise_meta=parent['randomness']['tapes']['final_tape_1']
    historical=torch.load(PARENT/noise_meta['file'],map_location='cpu',weights_only=True)
    assert historical['seed']==924088217
    extended=extend_prefix_noise(panel,training_extension,historical['uniforms'],924088217)
    cf_bank,st_bank=banks(rows,4800),banks(rows,4801)
    probe_specs=[dict(probe_id=f'update{update:04d}/{s["sequence_id"]}/{condition}',checkpoint_update=update,
                     sequence_id=s['sequence_id'],source=s['source'],condition=condition,parity=s['label'],
                     token_index=0 if condition=='A' else s['length']-1,human_token_number=1 if condition=='A' else s['length'])
                 for update in [0,1040] for s in rows for condition in ['A','B','C']]
    gen=torch.Generator().manual_seed(4802)
    bootstrap=dict(CF=torch.randint(0,8,(36,2000,8),generator=gen),ST=torch.randint(0,8,(36,2000,8),generator=gen))
    OUT.mkdir(exist_ok=False)
    for filename,value in [('prefix_noise_extension.pt',dict(seed=924088217,uniforms=extended)),('continuation_noise.pt',dict(seed=4800,tapes=cf_bank)),('ST_probe_noise.pt',dict(seed=4801,tapes=st_bank)),('bootstrap_indices.pt',dict(seed=4802,**bootstrap))]:
        with (OUT/filename).open('xb') as handle:torch.save(value,handle)
    protected={str(path):sha(path) for path in ROOT.glob('src/**/*') if path.is_file()}
    protected.update({str(ROOT/path):meta['sha256'] for path,meta in historical_index['files'].items()})
    protected.update({str(ROOT/path):digest for path,digest in att['helper_source_sha256'].items()})
    selected_ck={s['update']:s for s in prior['snapshots'] if s['update'] in [0,1040]}
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol='locked-ST-versus-full-hard-action-values-4700-v1',
        script_sha256=sha(__file__),parent_manifest_sha256=sha(PARENT/'manifest.json'),precheck_sha256=sha(PRECHECK),resolution_sha256=sha(RESOLUTION),
        git=git_identity(),source_sha256=parent['source_sha256'],encoding_attestation_sha256=sha(PARENT/'encoding_attestation.json'),
        checkpoints=selected_ck,backbone=parent['backbone'],six_sequences=rows,probe_order=probe_specs,
        approved_clarifications=dict(A='human ordinal1/internal index0; normal initialization, no previous tokens',
            B='internal L-1 after independent checkpoint-owned sampled prefix 0..L-2; original tape0 for evaluation IDs, appended extension for training IDs',
            C='internal L-1 after fixed canonical prefix; checkpoint continuation begins at this boundary',
            state='normal token initialization captured before first action; D=P=0, eight ADVANCE successors legal'),
        prefix_noise=dict(historical_file=str((PARENT/noise_meta['file']).relative_to(ROOT)),historical_sha256=noise_meta['sha256'],
            seed=924088217,original_sequence_count=96,bitwise_reconstruction_verified=True,same_ID_coordinate_association_verified=True,
            extension_IDs=list(extended),extension_order='original training-manifest order after all 96 original final-panel rows',
            generator='one private CPU torch.Generator(924088217), original ordered FP32 torch.rand[L,15,10] calls, then append missing training rows',
            extension_only_for_prefixes=True,new_training_prefix_provenance=True),
        randomness=dict(CF_seed=4800,ST_seed=4801,bootstrap_seed=4802,tapes_per_bank=8,
            generation_order='tape0..7, then six locked sequence IDs in recorded order; separate private CPU generator per bank',
            key=['bank','tape','sequence','zero_based_token','within_token_decision','successor_action'],
            omitted_keys=['checkpoint','counterfactual_branch','prefix_condition','outcomes'],historical_draws_unchanged=True,
            action_slots=10,fifteen_decisions='legal bound coverage only; no decision cap',
            noise_files={name:sha(OUT/name) for name in ['prefix_noise_extension.pt','continuation_noise.pt','ST_probe_noise.pt','bootstrap_indices.pt']},
            bootstrap='2000 replicates per state; independently resample the eight CF paired-action rows and eight ST vectors; first generate all CF indices then all ST indices'),
        computation=dict(batch=1,dtype='FP32',tau=2.,budget=[0.,0.],presence=0.,P=8,HALT=False,convergence=False,
            CF='force only first ADVANCE successor after constructing real candidates; all later actions original hard stochastic closed loop',
            ST='unforced original hard-forward exhaustive-candidate path and relaxed backward; autograd.grad final CE with respect to original first routing logits',
            loss='original final-sequence parity CE, batch1 mean reduction',gradient_policy=parent['gradient_policy'],
            fixed_state_conditioning='prefix reconstructed without grad; independent full-state clone per continuation; incoming state treated fixed as requested; no suffix-state/cache detach',
            optimization='no optimizer created; no parameter or buffer updates'),
        comparator=dict(pi='softmax(raw legal logits), not softmax(logits/tau); float64 analytic calculation and recorded original FP32 probabilities',
            g_CF='pi*(mean_Q-dot(pi,mean_Q))',validation='differentiate explicit sum softmax(local_logits)*Q with Q held constant',
            direction='positive dot product favors lower expected loss under descent; negative dot opposes it',
            resolution='both half-bank dot products same sign AND 95% full-bank bootstrap interval excludes zero; norms must exceed1e-8 for directional interpretation',
            action_value_resolution='at least one paired action-loss difference has same-sign half-bank means and unadjusted95% interval excluding zero',
            disconnected='record None distinctly; do not fabricate zero gradients or compute full-bank direction with missing gradients',
            zero_vs_small='exact connected zeros recorded separately; small-connected threshold L2<=1e-8',
            saturation='max pi>=0.99 descriptive flag',uncertainty='conditional Monte Carlo, not training-seed/population uncertainty; multiple unadjusted pair intervals'),
        numerical=parent['numerical'],exact_budgets=dict(states=36,counterfactual_continuations=2304,ST_gradient_probes=288),
        historical_exception='preserved bytecode-only identity failure; post-exception recorded diff/cache identities match; all experiment-relevant identities verified',
        limitations=['six preselected sequences and two fixed checkpoints; evaluation panel previously exposed',
            'eight independent tapes per bank; conditional MC uncertainty and unadjusted multiple paired differences',
            'new appended training-prefix randomness; evaluation-prefix historical tensors unchanged',
            'canonical prefix does not imply canonical continuation; continuation uses checkpoint policy',
            'incoming state fixed; this is a local independent-logit benchmark, not complete historical optimizer effect'])
    write_json(OUT/'manifest.json',manifest)
    print(f'MANIFEST SEALED {sha(OUT/"manifest.json")} 96 historical tables bitwise verified; A index0',flush=True)
    torch.set_num_threads(4)
    flags=parent['numerical'];torch.backends.cuda.matmul.allow_tf32=flags['matmul_allow_tf32'];torch.backends.cudnn.allow_tf32=flags['cudnn_allow_tf32']
    torch.use_deterministic_algorithms(flags['deterministic_algorithms'])
    assert str(torch.__version__)==flags['torch_version'] and torch.cuda.get_device_name()==flags['device_name']
    device=torch.device('cuda')
    backbone=torch.load(ROOT/parent['backbone']['path'],map_location='cpu',weights_only=True)
    weights={k:v for k,v in backbone['state_dict'].items() if not k.startswith('controller.')}
    models={}
    for update in [0,1040]:
        model,_=make_model(weights,device,4700,True)
        model.controller.load_state_dict(torch.load(ROOT/selected_ck[update]['path'],map_location='cpu',weights_only=True)['controller_state_dict'],strict=True)
        model.eval();models[update]=model
    before={u:snapshot(model.state_dict()) for u,model in models.items()}
    task=TaskAParity(64,device)
    prefix_tape=tape(dict(historical['uniforms'],**extended),924088217)
    canonical_cache={};policy_cache={};results=[];cf_count=st_count=0
    (OUT/'states').mkdir()
    # Validate historical B incoming states on the four evaluation sequences.
    historical_B={}
    with gzip.open(PARENT/'tokens.jsonl.gz','rt') as handle:
        wanted={s['sequence_id']:s['length']-1 for s in rows if s['source']!='training'}
        for line in handle:
            item=json.loads(line)
            if item['phase']=='final_final_tape_1' and item['sequence_id'] in wanted and item['token']==wanted[item['sequence_id']]:
                historical_B[(0 if item['arm']=='U' else 1040,item['sequence_id'])]=item['incoming_persistent']['sha256']
                if len(historical_B)==8:break
    assert len(historical_B)==8
    try:
        with ExitStack() as stack:
            handles={name:stack.enter_context(gzip.open(OUT/(name+'.jsonl.gz'),'xt')) for name in ['decisions','tokens','runs','prefixes']}
            for index,spec in enumerate(probe_specs):
                s=next(s for s in rows if s['sequence_id']==spec['sequence_id']);update=spec['checkpoint_update'];model=models[update]
                bits=torch.tensor([s['bits']],device=device);x=task.token_features[bits];targets=bits.cumsum(1).remainder(2)
                start=spec['token_index'];initial=model.initial_persistent_state(1,device,torch.float32)
                if spec['condition']=='A':incoming=initial
                elif spec['condition']=='B':
                    key=(update,s['sequence_id'])
                    if key not in policy_cache:
                        with torch.no_grad():path=ContinuationPath(model,prefix_tape,s['sequence_id'],0).run_span(x,targets,initial,0,start)
                        policy_cache[key]=path.final_state.clone()
                        trace_path(handles,'prefix',str(update),dict(s,sequence_id=spec['probe_id']),path)
                        record(handles['prefixes'],dict(probe_id=spec['probe_id'],kind='sampled checkpoint-owned',source='training extension' if s['source']=='training' else 'historical evaluation tape0',routes=[t['history'] for t in path.tokens],commits=len(path.tokens)))
                    incoming=policy_cache[key].clone()
                    if s['source']!='training':assert state_record(incoming)['sha256']==historical_B[key],'Historical B prefix identity mismatch'
                else:
                    if s['sequence_id'] not in canonical_cache:
                        with torch.no_grad():path=CanonicalPrefix(model,model.controller,False).run_prefix(x,targets,start)
                        canonical_cache[s['sequence_id']]=path.final_state.clone()
                        trace_path(handles,'prefix',str(update),dict(s,sequence_id=spec['probe_id']),path)
                        record(handles['prefixes'],dict(probe_id=spec['probe_id'],kind='fixed canonical',routes=[t['history'] for t in path.tokens],commits=len(path.tokens)))
                    incoming=canonical_cache[s['sequence_id']].clone()
                assert incoming.tokens_committed==start and not any(v.requires_grad for v in refs(incoming).values() if torch.is_tensor(v))
                features,captured=initialized_features(model,incoming,x[:,start])
                mask=build_execution_mask(0,0,False,8);assert mask[:8].all() and not mask[8:10].any()
                with torch.no_grad():
                    logits={u:copy(mm.controller(**features)['logits'])[0] for u,mm in models.items()}
                pis={u:logits[u][:8].double().softmax(-1).tolist() for u in models}
                statefile=OUT/'states'/f'probe{index:02d}.pt'
                with statefile.open('xb') as handle:torch.save(dict(incoming_persistent=snapshot(incoming),features={k:copy(v) for k,v in features.items()},h0=copy(captured['h0']),C_prev=copy(captured['C_prev']),context=snapshot(captured['context']),probe=spec),handle)
                values=[[None]*8 for _ in range(8)];correct=[[None]*8 for _ in range(8)];gradients=[];st_runs=[]
                for bank in ['CF','ST']:
                    for k in range(8):
                        for action in range(8) if bank=='CF' else [None]:
                            noise=tape((cf_bank if bank=='CF' else st_bank)[k],4800 if bank=='CF' else 4801)
                            identifier=f'{spec["probe_id"]}/{bank}/tape{k}/action{action}'
                            with torch.set_grad_enabled(bank=='ST'):
                                path=ContinuationPath(model,noise,s['sequence_id'],start,action,bank=='ST',features).run_span(x,targets,incoming,start,s['length'])
                                loss=F.cross_entropy(path.final_logits,torch.tensor([s['label']],device=device))
                                assert torch.isfinite(loss)
                                grad=None
                                if bank=='ST':
                                    assert loss.requires_grad
                                    grad=torch.autograd.grad(loss,path.rows[0]['_logits'],allow_unused=True)[0]
                                    if grad is not None:
                                        assert torch.isfinite(grad).all() and grad[0,8:].eq(0).all()
                                        grad=copy(grad)[0].tolist()
                            value=float(loss.detach());prediction=int(path.final_logits.detach().argmax(-1))
                            if bank=='CF':values[k][action]=value;correct[k][action]=prediction==s['label'];cf_count+=1
                            else:gradients.append(grad);st_count+=1
                            row=dict(run_id=identifier,probe=spec,bank=bank,tape=k,forced_action=action,final_ce=value,prediction=prediction,correct=prediction==s['label'],
                                output_logits=copy(path.final_logits),probe_logit_gradient=grad,gradient_connected=grad is not None if bank=='ST' else None,
                                routes=[t['history'] for t in path.tokens],P=[t['P'] for t in path.tokens],D=[t['D'] for t in path.tokens],
                                restarts=sum(t['restart_count'] for t in path.tokens),repeats=sum(t['repeat_count'] for t in path.tokens),
                                commits=len(path.tokens),bank_purity_checks=path.block_purity_checks,gdn_purity_checks=dict(path.gdn_purity_checks),
                                candidate_constructions=path.candidate_constructions,candidate_block_calls=9*len(path.rows))
                            record(handles['runs'],row)
                            trace_path(handles,bank,str(update),dict(s,sequence_id=identifier),path)
                            if bank=='ST':st_runs.append(encode(row))
                            assert all(p.grad is None for p in model.parameters())
                            del path,loss
                        print(f'BANK progress probe={index+1}/36 {bank} tape={k+1}/8 CF/ST={cf_count}/{st_count}',flush=True)
                measured=compare_banks(logits[update][:8].tolist(),values,gradients,correct,bootstrap['CF'][index],bootstrap['ST'][index],[pis[0],pis[1040]])
                measured.update(probe=spec,state_file=str(statefile.relative_to(ROOT)),state_sha256=sha(statefile),
                    incoming_state=state_record(incoming),raw_logits_both_checkpoints={str(u):v.tolist() for u,v in logits.items()},pi_both_checkpoints=pis,
                    original_FP32_pi=logits[update][:8].softmax(-1).tolist(),CF_final_ce=values,CF_correct=correct,ST_runs=st_runs,
                    legal_successor_actions=list(range(8)),continuation_policy_checkpoint=update)
                results.append(measured);write_json(OUT/f'probe{index:02d}.json',measured)
                assert all(unchanged(before[u],snapshot(mm.state_dict())) for u,mm in models.items())
                print(f'PROBE {index+1}/36 COMPLETE update={update} {s["sequence_id"]} {spec["condition"]} {measured["classification"]} counts CF/ST={cf_count}/{st_count}',flush=True)
        assert cf_count==2304 and st_count==288 and len(results)==36
        assert all(sha(path)==digest for path,digest in protected.items())
        assert sha(__file__)==manifest['script_sha256']
        assert all(sha(OUT/name)==digest for name,digest in manifest['randomness']['noise_files'].items())
        assert git_identity()['tracked_diff_sha256']==manifest['git']['tracked_diff_sha256']
        assert subprocess.run(['git','diff','--check'],capture_output=True).returncode==0
    except Exception as error:
        write_json(OUT/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),completed_states=len(results),counterfactual_continuations=cf_count,ST_gradient_probes=st_count,stop='No retry or added samples'))
        raise
    strata={}
    for item in results:
        s=item['probe'];key=f'{s["checkpoint_update"]}/{s["source"]}/{s["condition"]}'
        strata.setdefault(key,Counter())[item['classification']]+=1
    aggregate=dict(manifest_sha256=sha(OUT/'manifest.json'),counts=dict(states=36,CF=cf_count,ST=st_count),
        classifications=dict(Counter(r['classification'] for r in results)),strata={k:dict(v) for k,v in strata.items()},
        mechanically_verified=True,parameters_buffers_and_historical_artifacts_unchanged=True,git_diff_check_clean=True,
        comparator_autograd_verified_all_states=True,probe_A_internal_index=0,optimizer_created=False,training_updates=0,
        probes=results,limitations=manifest['limitations'])
    write_json(OUT/'results.json',aggregate)
    print('DIAGNOSTIC COMPLETE '+json.dumps(aggregate['classifications']),flush=True)

if __name__=='__main__':
    existed=OUT.exists()
    try:main()
    except Exception as error:
        if not existed:
            OUT.mkdir(exist_ok=True)
            if not (OUT/'failure.json').exists():
                write_json(OUT/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),phase='preparation before any diagnostic continuations',counterfactual_continuations=0,ST_gradient_probes=0,stop='No retry'))
        raise
