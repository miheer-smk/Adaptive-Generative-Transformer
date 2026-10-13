"""One locked diagnostic of existing controller-output derivative paths.

Forward execution is unchanged. Tensor hooks suppress designated controller
output VJPs only during diagnostic autograd.grad evaluations on a retained
graph. These hooks are never used for training.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import gzip, json, math, subprocess, traceback
from collections import Counter
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from data.task_a import TaskAParity
from experiments.diagnose_gatec_st_action_values_4700 import ContinuationPath,tape,compare_banks,NORM_EPS
from experiments.replicate_gatec_locked_imitation import make_model
from experiments.replay_gatec_learned_policy import sha,snapshot,unchanged,state_record,git_identity,copy,fingerprint,encode
from experiments.run_gatec_paired_task_maintenance import refs,trace_path
from experiments.evaluate_gatec_locked_policy import record,write_json

PARENT=ROOT/'experiments/results/gatec_st_action_value_4700_v1'
OUT=ROOT/'experiments/results/gatec_temporal_vjp_4700_update1040_v1'
ATOL,RTOL=1e-6,1e-5
MODES=['exec','token','next','full']

def restore_state(template,flat,device):
    """Restore all serialized fields into the existing eager cache topology."""
    assert set(refs(template))==set(flat),'Saved persistent topology mismatch'
    def visit(value,path=''):
        if torch.is_tensor(value) or value is None or not isinstance(value,(dict,list,tuple)) and not hasattr(value,'__dict__'):
            saved=flat[path]
            return saved.to(device).clone() if torch.is_tensor(saved) else saved
        if isinstance(value,dict):
            for k,v in list(value.items()):value[k]=visit(v,f'{path}/{k}')
        elif isinstance(value,(tuple,list)):
            raise AssertionError('Unexpected persistent list/tuple topology')
        else:
            for k,v in list(value.__dict__.items()):setattr(value,k,visit(v,f'{path}/{k}'))
        return value
    restored=visit(template)
    assert state_record(restored)['sha256']==fingerprint(flat)
    assert not any(v.requires_grad for v in refs(restored).values() if torch.is_tensor(v))
    return restored

class ControllerOutputVJPs:
    """Same graph and forward values; only an explicit output VJP is masked."""
    def __init__(self,logits,tokens,start):
        self.logits,self.tokens,self.start=logits,tokens,start
        self.mode='full';self.handles=[];self.blocked=[]
        assert tokens[0]==start and all(x.requires_grad for x in logits)
    def allowed(self,index):
        if index==0:return True # Probe output remains differentiable in every mode.
        token=self.tokens[index]
        return self.mode=='full' or self.mode=='token' and token==self.start or self.mode=='next' and token<=self.start+1
    def __enter__(self):
        for i,logit in enumerate(self.logits):
            def gate(gradient,index=i):
                if self.allowed(index):return gradient
                self.blocked.append(index)
                return torch.zeros_like(gradient)
            self.handles.append(logit.register_hook(gate))
        return self
    def set_mode(self,mode):
        assert mode in MODES;self.mode=mode;self.blocked=[]
        for logit in self.logits:logit.grad=None # Intermediate diagnostic retention only.
    def __exit__(self,*args):
        for handle in self.handles:handle.remove()

class TemporalPath(ContinuationPath):
    @contextmanager
    def observers(self):
        with super().observers():
            ex=self.model.executor;controller,step=ex.controller,ex.differentiable_route_step
            def capture_controller(**features):
                output=controller(**features)
                self.current[-1]['_adjoint_tensors']={k:features[k] for k in ['h','memory_read','memory_mean']}
                return output
            def capture_step(*args,**kwargs):
                output=step(*args,**kwargs)
                self.current[-1]['_adjoint_tensors'].update(h_next=output['h_next'],h0=kwargs['h0'])
                return output
            ex.controller,ex.differentiable_route_step=capture_controller,capture_step
            try:yield
            finally:ex.controller,ex.differentiable_route_step=controller,step

def status(gradient,requires_grad=True):
    if not requires_grad:return 'fixed_tensor_no_autograd_input'
    if gradient is None:return 'disconnected'
    if gradient.eq(0).all():return 'connected_exact_zero'
    return 'connected_small' if float(gradient.norm())<=NORM_EPS else 'connected_nonzero'

def original_records():
    runs={}
    with gzip.open(PARENT/'runs.jsonl.gz','rt') as h:
        for line in h:
            row=json.loads(line)
            if row['bank']=='ST' and row['probe']['checkpoint_update']==1040:runs[row['run_id']]=row
    assert len(runs)==144
    decisions={k:[] for k in runs};tokens={k:[] for k in runs}
    for name,destination in [('decisions',decisions),('tokens',tokens)]:
        with gzip.open(PARENT/(name+'.jsonl.gz'),'rt') as h:
            for line in h:
                row=json.loads(line)
                if row['sequence_id'] in destination:destination[row['sequence_id']].append(row)
    return runs,decisions,tokens

def verify_forward(path,run,old_decisions,old_tokens,loss):
    assert len(path.rows)==len(old_decisions) and len(path.tokens)==len(old_tokens)
    exact=['token','decision','execution_before','execution_after','pointer_before','pointer_after',
           'just_restarted_before','just_restarted_after','selected_action','decoded_block_target',
           'actual_executed_block','executed_blocks','candidate_block_calls','noise_coordinate']
    tensors=['legal_action_mask','legal_successor_mask','budget','raw_controller_logits','probabilities']
    for new,old in zip(path.rows,old_decisions):
        for k in exact:assert new[k]==old[k],f'Forward metadata changed: {k}'
        for k in tensors:
            actual=copy(new[k]);expected=torch.tensor(old[k],dtype=actual.dtype)
            torch.testing.assert_close(actual,expected,atol=0 if actual.dtype==torch.bool else ATOL,rtol=0 if actual.dtype==torch.bool else RTOL)
    for new,old in zip(path.tokens,old_tokens):
        for k in ['token','P','D','history','restart_count','repeat_count','executed_blocks','commit_count','commit_entries','termination_reason','prediction','target']:
            assert encode(new[k])==old[k],f'Token record changed: {k}'
        for k in ['incoming_persistent','outgoing_persistent']:
            assert new[k]['sha256']==old[k]['sha256'],'Full commit-state identity changed'
        torch.testing.assert_close(copy(new['output_logits']),torch.tensor(old['output_logits']),atol=ATOL,rtol=RTOL)
    torch.testing.assert_close(loss.detach().cpu(),torch.tensor(run['final_ce']),atol=ATOL,rtol=RTOL)
    assert [t['history'] for t in path.tokens]==[[(a,b) for a,b in route] for route in run['routes']]

def measurements(loss,path,start,on_vjp=None):
    logits=[r['_logits'] for r in path.rows];tokens=[r['token'] for r in path.rows]
    modes=['exec','full'] if all(t==start for t in tokens) else MODES
    cumulatives={};boundaries={};full_adjoints=None
    with ControllerOutputVJPs(logits,tokens,start) as isolation:
        for mode in modes:
            isolation.set_mode(mode)
            targets=[logits[0]];lookup={id(logits[0]):0}
            if mode=='full':
                for row in path.rows:
                    for tensor in [row['_logits'],*row['_adjoint_tensors'].values()]:
                        if tensor.requires_grad and id(tensor) not in lookup:
                            lookup[id(tensor)]=len(targets);targets.append(tensor)
            if on_vjp is not None:on_vjp()
            returned=torch.autograd.grad(loss,targets,retain_graph=mode!=modes[-1],allow_unused=True)
            assert all(torch.isfinite(g).all() for g in returned if g is not None)
            cumulatives[mode]=returned[0]
            boundaries[mode]=dict(blocked_controller_outputs=isolation.blocked[:],
                allowed_controller_outputs=[i for i in range(len(logits)) if isolation.allowed(i)],
                first_probe_always_allowed=isolation.allowed(0))
            if mode=='full':
                full_adjoints=[]
                for row in path.rows:
                    def value(tensor):return returned[lookup[id(tensor)]] if tensor.requires_grad else None
                    g=value(row['_logits'])
                    if g is not None:
                        illegal=~row['legal_successor_mask'].to(g.device)
                        assert g[0,:10][illegal].eq(0).all() and g[0,10]==0
                    states={}
                    for key,tensor in row['_adjoint_tensors'].items():
                        sg=value(tensor)
                        states[key]=dict(shape=list(tensor.shape),requires_grad=tensor.requires_grad,has_grad_fn=tensor.grad_fn is not None,
                            status=status(sg,tensor.requires_grad),gradient=copy(sg).tolist() if sg is not None else None,
                            L2=float(sg.norm()) if sg is not None else None)
                    full_adjoints.append(dict(token=row['token'],human_token=row['token']+1,decision=row['decision'],
                        decision_count_before=row['decision'],decision_count_after=row['decision']+1,
                        execution_before=row['execution_before'],execution_after=row['execution_after'],
                        remaining_tokens=path.tokens[-1]['token']-row['token'],remaining_selected_executions_in_token=8-row['execution_after'],
                        pointer=row['pointer_before'],selected_action=row['selected_action'],
                        original_logit_gradient=copy(g)[0].tolist() if g is not None else None,gradient_status=status(g),
                        execution_state_adjoints=states))
    if len(modes)==2:
        assert all(t==start for t in tokens),'B/C equality requires absence of later-token calls'
        cumulatives['token']=cumulatives['next']=cumulatives['full']
        boundaries['token']=boundaries['next']=dict(alias_of='full',reason='suffix contains only probe token; no redundant VJP')
    if any(g is None for g in cumulatives.values()):
        return cumulatives,None,boundaries,full_adjoints,len(modes)
    components=dict(exec=cumulatives['exec'],within=cumulatives['token']-cumulatives['exec'],
        next=cumulatives['next']-cumulatives['token'],distant=cumulatives['full']-cumulatives['next'])
    torch.testing.assert_close(sum(components.values()),cumulatives['full'],atol=ATOL,rtol=RTOL)
    return cumulatives,components,boundaries,full_adjoints,len(modes)

def vector_variance(x):
    centered=x-x.mean(-2,keepdim=True)
    return centered.square().sum(-1).sum(-1)/(x.shape[-2]-1)

def summarize_components(previous,replays,cf_indices,st_indices):
    q=torch.tensor(previous['CF_final_ce'],dtype=torch.float64)
    pi=torch.tensor(previous['pi'],dtype=torch.float64)
    q_boot=q[cf_indices].mean(1);gcf_boot=pi*(q_boot-(q_boot*pi).sum(-1,keepdim=True))
    gcf=torch.tensor(previous['g_CF'],dtype=torch.float64)
    cumulatives={};components={};component_projections={}
    for mode in MODES:
        grads=[r['cumulative_gradients'][mode] for r in replays]
        cumulatives[mode]=compare_banks(previous['raw_logits_both_checkpoints']['1040'][:8],previous['CF_final_ce'],grads,previous['CF_correct'],
            cf_indices,st_indices,[previous['pi_both_checkpoints']['0'],previous['pi_both_checkpoints']['1040']])
    if any(r['components'] is None for r in replays):
        return dict(cumulative_alignment=cumulatives,components=None,classification='unresolved_disconnected_component')
    for key in ['exec','within','next','distant']:
        values=torch.tensor([r['components'][key][:8] for r in replays],dtype=torch.float64)
        mean=values.mean(0);projections=values@gcf
        boot=(values[st_indices].mean(1)*gcf_boot).sum(-1)
        halves=[]
        for sl in [slice(0,4),slice(4,8)]:
            mean_q=q[sl].mean(0);half_cf=pi*(mean_q-(mean_q*pi).sum())
            halves.append(float(half_cf@values[sl].mean(0)))
        interval=torch.quantile(boot,torch.tensor([.025,.975],dtype=torch.float64)).tolist()
        meaningful=float(mean.norm())>NORM_EPS and float(gcf.norm())>NORM_EPS
        direction='positive' if meaningful and min(halves)>0 and interval[0]>0 else 'negative' if meaningful and max(halves)<0 and interval[1]<0 else 'unresolved'
        components[key]=dict(mean=mean.tolist(),mean_L2=float(mean.norm()),tape_L2=values.norm(dim=1).tolist(),
            component_tape_sd=values.std(0).tolist(),projection=float(mean@gcf),projection_by_tape=projections.tolist(),
            projection_halves=halves,projection_bootstrap_95=interval,resolved_projection_sign=direction)
        component_projections[key]=projections
    exe=torch.tensor([x['components']['exec'][:8] for x in replays],dtype=torch.float64)
    full=torch.tensor([x['cumulative_gradients']['full'][:8] for x in replays],dtype=torch.float64)
    rest=full-exe
    cov=((exe-exe.mean(0))*(rest-rest.mean(0))).sum()/7
    ve,vr,vf=vector_variance(exe),vector_variance(rest),vector_variance(full)
    torch.testing.assert_close(vf,ve+vr+2*cov,atol=1e-12,rtol=1e-10)
    paired_diff=vector_variance(full[st_indices])-vector_variance(exe[st_indices])
    interval=torch.quantile(paired_diff,torch.tensor([.025,.975],dtype=torch.float64)).tolist()
    halves=[float(vector_variance(full[sl])-vector_variance(exe[sl])) for sl in [slice(0,4),slice(4,8)]]
    added=min(halves)>0 and interval[0]>0
    # Projection variance also carries uncertainty from the independent CF bank.
    pexe=exe@gcf;pfull=full@gcf;prest=rest@gcf
    projection_cov=((pexe-pexe.mean())*(prest-prest.mean())).sum()/7
    boot_pe=(exe[st_indices]*gcf_boot.unsqueeze(1)).sum(-1)
    boot_pf=(full[st_indices]*gcf_boot.unsqueeze(1)).sum(-1)
    projection_delta=boot_pf.var(1)-boot_pe.var(1)
    projection_ci=torch.quantile(projection_delta,torch.tensor([.025,.975],dtype=torch.float64)).tolist()
    projection_halves=[]
    for sl in [slice(0,4),slice(4,8)]:
        qq=q[sl].mean(0);gg=pi*(qq-(qq*pi).sum())
        projection_halves.append(float((full[sl]@gg).var()-(exe[sl]@gg).var()))
    # Rest cancellation is assessed directly with independently resampled banks.
    rest_boot=(rest[st_indices].mean(1)*gcf_boot).sum(-1)
    rest_ci=torch.quantile(rest_boot,torch.tensor([.025,.975],dtype=torch.float64)).tolist()
    rest_halves=[]
    for sl in [slice(0,4),slice(4,8)]:
        qq=q[sl].mean(0);gg=pi*(qq-(qq*pi).sum())
        rest_halves.append(float(gg@rest[sl].mean(0)))
    cancellation=float(rest.mean(0).norm())>NORM_EPS and float(gcf.norm())>NORM_EPS and max(rest_halves)<0 and rest_ci[1]<0
    exec_useful=previous['informative_action_values'] and cumulatives['exec']['direction']=='aligned'
    full_resolved=cumulatives['full']['direction']=='aligned'
    if not exec_useful:classification='unresolved_execution_only_credit'
    elif (cancellation or added) and not full_resolved:classification='supports_temporal_interference_CE_at_state'
    elif cancellation or added:classification='component_effect_with_resolved_full_alignment'
    elif full_resolved:classification='no_resolved_adverse_temporal_effect'
    else:classification='unresolved_temporal_effect'
    return dict(cumulative_alignment=cumulatives,components=components,classification=classification,
        variance=dict(definition='sample trace covariance; eight paired ST vectors, ddof1',execution=float(ve),remaining=float(vr),full=float(vf),
            execution_remaining_covariance=float(cov),identity_residual=float(vf-ve-vr-2*cov),
            paired_full_minus_execution=float(vf-ve),paired_difference_halves=halves,paired_difference_bootstrap_95=interval,
            resolved_added_vector_variance=added,
            projection_execution=float(pexe.var()),projection_remaining=float(prest.var()),projection_full=float(pfull.var()),
            projection_covariance=float(projection_cov),projection_identity_residual=float(pfull.var()-pexe.var()-prest.var()-2*projection_cov),
            projection_difference_bootstrap_95=projection_ci,projection_difference_halves=projection_halves,
            resolved_added_projection_variance=min(projection_halves)>0 and projection_ci[0]>0),
        cancellation=dict(rest_projection=float(rest.mean(0)@gcf),rest_projection_halves=rest_halves,rest_projection_bootstrap_95=rest_ci,
            resolved_opposing_remaining_projection=cancellation,relative_full_to_execution_mean_L2=float(full.mean(0).norm()/exe.mean(0).norm()) if exe.mean(0).norm()>NORM_EPS else None,
            mean_component_norm_sum=float(sum(torch.tensor(v['mean']).norm() for v in components.values())),
            full_mean_L2=float(full.mean(0).norm()),
            explanation='Norm shrinkage alone is descriptive, not directional cancellation evidence.'),
        eligibility=dict(informative_action_values=previous['informative_action_values'],resolved_aligned_execution_CE_credit=exec_useful,
            resolved_aligned_full_credit=full_resolved,parity_correctness_varies=any(len(set(row))>1 for row in previous['CF_correct']),
            CE_alignment_is_not_parity_learning=True))

def main():
    if OUT.exists():raise RuntimeError('Refusing to repeat or overwrite temporal diagnostic')
    pm=json.loads((PARENT/'manifest.json').read_text());pr=json.loads((PARENT/'results.json').read_text())
    prior_index=json.loads((PARENT/'artifact_index.json').read_text())
    for name,item in prior_index['files'].items():assert sha(PARENT/name)==item['sha256'],name
    assert sha(PARENT/'manifest.json')==pr['manifest_sha256']
    fresh=ROOT/'experiments/results/gatec_fresh_task_discovery_4700_v1'
    hi=json.loads((fresh/'artifact_index.json').read_text())
    for name,entry in hi['files'].items():assert sha(ROOT/name)==entry['sha256'],name
    for path,digest in pm['source_sha256'].items():assert sha(ROOT/path)==digest,path
    for ck in [pm['checkpoints']['1040'],pm['backbone']]:assert sha(ROOT/ck['path'])==ck['sha256']
    assert git_identity()['tracked_diff_sha256']==pm['git']['tracked_diff_sha256']
    selected=[(i,p) for i,p in enumerate(pr['probes']) if p['probe']['checkpoint_update']==1040]
    assert len(selected)==18 and Counter(p['probe']['condition'] for _,p in selected)=={'A':6,'B':6,'C':6}
    noise=torch.load(PARENT/'ST_probe_noise.pt',map_location='cpu',weights_only=True);assert noise['seed']==4801 and len(noise['tapes'])==8
    bootstrap=torch.load(PARENT/'bootstrap_indices.pt',map_location='cpu',weights_only=True);assert bootstrap['seed']==4802
    old_runs,old_decisions,old_tokens=original_records()
    protected={str(path):sha(path) for path in ROOT.glob('src/**/*') if path.is_file()}
    protected.update({str(PARENT/name):item['sha256'] for name,item in prior_index['files'].items()})
    protected.update({str(fresh/'artifact_index.json'):sha(fresh/'artifact_index.json'),str(PARENT/'artifact_index.json'):sha(PARENT/'artifact_index.json')})
    protected.update({str(ROOT/name):entry['sha256'] for name,entry in hi['files'].items()})
    OUT.mkdir(exist_ok=False)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),git=git_identity(),script_sha256=sha(__file__),
        previous_manifest_sha256=sha(PARENT/'manifest.json'),previous_index_sha256=sha(PARENT/'artifact_index.json'),
        controller=pm['checkpoints']['1040'],backbone=pm['backbone'],six_sequences=pm['six_sequences'],
        states=[dict(original_index=i,probe=p['probe'],path=p['state_file'],sha256=p['state_sha256']) for i,p in selected],
        previous_classifications=pr['strata'],previous_all_probes=[dict(probe=p['probe'],classification=p['classification'],direction=p['direction']) for p in pr['probes']],
        prefix_noise=pm['prefix_noise'],randomness=dict(no_new_noise=True,ST_bank_seed=4801,CF_values_from_seed4800=True,
            source_noise_hashes=pm['randomness']['noise_files'],bootstrap_seed=4802,replicates=2000,
            original_36_state_indices_used=True,CF_pairing_preserved=True,ST_components_resampled_jointly=True,banks_resampled_independently=True),
        probe_definitions=pm['approved_clarifications'],computation=pm['computation'],numerical=pm['numerical'],
        graph_boundaries=dict(output='original Controller.forward logits [1,11], before executor successor slicing/masking/ST',
            exec='only probe logits VJP allowed; every later Controller output VJP zeroed by tensor hook',
            token='allow outputs of all controller calls in probe token, exclude controller outputs in later tokens',
            next='allow probe token and immediately following token outputs, exclude outputs beyond that token',
            full='no hook suppression; original ST derivative',
            preserved='all direct execution/state/commit derivatives and existing hard metadata/relaxed-stop-gradient boundaries; no edge reconnected',
            B_C='suffix contains only final probe token; token/next alias full after graph-boundary verification, no redundant VJP',
            adjoints='one full VJP also requests original logits at all calls and available h/h_next/h0/memory-read/memory-mean adjoints; no new action values at those calls'),
        comparison=dict(atol=ATOL,rtol=RTOL,incoming_and_commit_states='exact tensor-derived hashes',
            metadata='exact equality',logits_CE_gradients='original atol1e-6/rtol1e-5',
            components='exec; token-exec; next-token; full-next; sum checked against full'),
        uncertainty=dict(rule='same-sign four-tape halves plus 95% 2000-bootstrap interval excluding zero; meaningful alignment norms >1e-8',
            variance='sample trace covariance ddof1; paired full-minus-exec variance resampled jointly; require both half differences positive and95% lower bound>0 for resolved added variance',
            projections='CF gradient recomputed only from retained values during independent CF resampling; paired ST component vectors share indices',
            cancellation='remaining contribution projection negative in both halves and95% upper bound<0',
            support='informative action values plus resolved aligned execution-only CE credit plus resolved cancellation/added vector variance; strong unreliability support additionally requires full credit not resolved aligned',
            parity='report conditional correctness; CE alignment alone never labeled parity learning',
            intervals='conditional Monte Carlo only, unadjusted across components/states; eight tapes per bank'),
        budgets=dict(forward_replays=144,maximum_VJP_evaluations=384,new_CF_continuations=0,optimizer_steps=0),
        focused_validation=dict(final_passed=32,failed=0,bytecode_writes='disabled; explicit compilation targets /tmp',
            preserved_pre_execution_test_failure=dict(passed=31,failed=1,
                test='test_hooks_preserve_forward_and_keep_probe_differentiable_in_all_modes',
                cause='synthetic fixture dot([.2,-.1],[1,2]) was zero, so its squared loss and every gradient were zero',
                correction='synthetic fixture only: second coordinate .1; no AGT forward or diagnostic measurement had run')),
        historical_bytecode_exception=pm['historical_exception'],protected_files_sha256=protected)
    write_json(OUT/'manifest.json',manifest);print('MANIFEST SEALED '+sha(OUT/'manifest.json'),flush=True)
    flags=pm['numerical'];torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=flags['matmul_allow_tf32'];torch.backends.cudnn.allow_tf32=flags['cudnn_allow_tf32']
    torch.use_deterministic_algorithms(flags['deterministic_algorithms'])
    assert str(torch.__version__)==flags['torch_version'] and torch.cuda.get_device_name()==flags['device_name']
    device=torch.device('cuda');backbone=torch.load(ROOT/pm['backbone']['path'],map_location='cpu',weights_only=True)
    weights={k:v for k,v in backbone['state_dict'].items() if not k.startswith('controller.')}
    model,_=make_model(weights,device,4700,True)
    model.controller.load_state_dict(torch.load(ROOT/manifest['controller']['path'],map_location='cpu',weights_only=True)['controller_state_dict'],strict=True)
    model.eval();before=snapshot(model.state_dict());task=TaskAParity(64,device)
    results=[];forward_count=vjp_count=0
    def counted_vjp():
        nonlocal vjp_count
        vjp_count+=1
        assert vjp_count<=384
    try:
        with ExitStack() as stack:
            handles={name:stack.enter_context(gzip.open(OUT/(name+'.jsonl.gz'),'xt')) for name in ['decisions','tokens','runs','adjoints']}
            for old_index,previous in selected:
                spec=previous['probe'];s=next(s for s in pm['six_sequences'] if s['sequence_id']==spec['sequence_id'])
                saved=torch.load(ROOT/previous['state_file'],map_location='cpu',weights_only=True)
                assert sha(ROOT/previous['state_file'])==previous['state_sha256']
                incoming=restore_state(model.initial_persistent_state(1,device,torch.float32),saved['incoming_persistent'],device)
                assert state_record(incoming)['sha256']==previous['incoming_state']['sha256']
                start=spec['token_index'];assert start==(0 if spec['condition']=='A' else s['length']-1)
                assert incoming.tokens_committed==start
                bits=torch.tensor([s['bits']],device=device);x=task.token_features[bits];targets=bits.cumsum(1).remainder(2)
                replay_results=[]
                for k in range(8):
                    run_id=f'{spec["probe_id"]}/ST/tape{k}/actionNone';old=old_runs[run_id]
                    forward_count+=1
                    assert forward_count<=144
                    path=TemporalPath(model,tape(noise['tapes'][k],4801),s['sequence_id'],start,gradient=True,
                        expected={key:v.to(device) for key,v in saved['features'].items()}).run_span(x,targets,incoming,start,s['length'])
                    loss=F.cross_entropy(path.final_logits,torch.tensor([s['label']],device=device))
                    assert torch.isfinite(loss)
                    verify_forward(path,old,old_decisions[run_id],old_tokens[run_id],loss)
                    cumulative,components,boundaries,adjoints,evaluations=measurements(loss,path,start,counted_vjp)
                    original=old['probe_logit_gradient'];actual=cumulative['full']
                    if original is None:assert actual is None
                    else:torch.testing.assert_close(copy(actual)[0],torch.tensor(original),atol=ATOL,rtol=RTOL)
                    if spec['condition']!='A':
                        assert cumulative['token'] is cumulative['next'] is cumulative['full']
                        assert components is None or components['next'].eq(0).all() and components['distant'].eq(0).all()
                    for g in cumulative.values():
                        if g is not None:assert torch.isfinite(g).all() and g[0,8:].eq(0).all()
                    row=dict(run_id=run_id,probe=spec,tape=k,final_ce=float(loss.detach()),prediction=int(path.final_logits.argmax(-1)),
                        correct=int(path.final_logits.argmax(-1))==s['label'],forward_matches_previous=True,
                        cumulative_gradients={key:copy(g)[0].tolist() if g is not None else None for key,g in cumulative.items()},
                        cumulative_status={key:status(g) for key,g in cumulative.items()},
                        components={key:copy(g)[0].tolist() for key,g in components.items()} if components is not None else None,
                        component_status={key:status(g) for key,g in components.items()} if components is not None else None,
                        boundaries=boundaries,VJP_evaluations=evaluations,controller_calls=len(path.rows),
                        candidate_constructions=path.candidate_constructions,bank_purity_checks=path.block_purity_checks,
                        gdn_purity_checks=dict(path.gdn_purity_checks),commits=len(path.tokens),
                        D=[t['D'] for t in path.tokens],P=[t['P'] for t in path.tokens])
                    replay_results.append(row);record(handles['runs'],row)
                    trace_path(handles,'ST1040_temporal','1040',dict(s,sequence_id=run_id),path)
                    for adj in adjoints:record(handles['adjoints'],dict(run_id=run_id,probe=spec,tape=k,**adj))
                    assert all(p.grad is None for p in model.parameters()) and unchanged(before,snapshot(model.state_dict()))
                    print(f'REPLAY {forward_count}/144 VJP={vjp_count}/384 {spec["probe_id"]} tape{k} VERIFIED',flush=True)
                    del path,loss,cumulative,components,actual,adjoints
                statistics=summarize_components(previous,replay_results,bootstrap['CF'][old_index],bootstrap['ST'][old_index])
                item=dict(probe=spec,original_index=old_index,previous_classification=previous['classification'],
                    incoming_state_sha256=previous['incoming_state']['sha256'],CF_final_ce=previous['CF_final_ce'],CF_correct=previous['CF_correct'],
                    g_CF=previous['g_CF'],pi=previous['pi'],conditional_correctness=previous['conditional_correctness'],replays=replay_results,**statistics)
                results.append(item);write_json(OUT/f'probe{old_index:02d}.json',item)
                print(f'PROBE {len(results)}/18 COMPLETE '+statistics['classification'],flush=True)
        assert forward_count==144 and vjp_count==384 and len(results)==18
        for path,digest in protected.items():assert sha(path)==digest,path
        assert sha(__file__)==manifest['script_sha256']
        assert git_identity()['tracked_diff_sha256']==manifest['git']['tracked_diff_sha256']
        assert subprocess.run(['git','diff','--check'],capture_output=True).returncode==0
    except Exception as error:
        write_json(OUT/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),completed_states=len(results),
            forward_replays=forward_count,VJP_evaluations=vjp_count,stop='No retries or added samples'))
        raise
    aggregate=dict(manifest_sha256=sha(OUT/'manifest.json'),counts=dict(states=18,forward_replays=forward_count,VJP_evaluations=vjp_count,new_CF_continuations=0),
        classifications=dict(Counter(p['classification'] for p in results)),mechanically_verified=True,
        original_forward_and_full_gradients_reproduced=True,parameters_buffers_production_and_historical_artifacts_unchanged=True,
        git_diff_check_clean=True,optimizer_created=False,training_updates=0,probes=results)
    write_json(OUT/'results.json',aggregate);print('DIAGNOSTIC COMPLETE '+json.dumps(aggregate['classifications']),flush=True)

if __name__=='__main__':
    existed=OUT.exists()
    try:main()
    except Exception as error:
        if not existed:
            OUT.mkdir(exist_ok=True)
            if not (OUT/'failure.json').exists():write_json(OUT/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),phase='preparation',forward_replays=0,VJP_evaluations=0))
        raise
