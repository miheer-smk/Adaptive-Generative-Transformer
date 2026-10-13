"""Post-run artifact verification; no model forward or optimizer operation."""
from pathlib import Path
from collections import Counter, defaultdict
import gzip, hashlib, json, math, subprocess, sys
ROOT=Path('/home/iiitn/Desktop/Miheer Projects/AGT/AGT_v3_POC')
sys.path.insert(0,str(ROOT))
import torch
from src.masks import build_execution_mask
torch.set_num_threads(4)
OUT=ROOT/'experiments/results/gatec_fresh_task_discovery_4700_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def finite(value):
    if isinstance(value,float):assert math.isfinite(value)
    elif isinstance(value,dict):
        for item in value.values():finite(item)
    elif isinstance(value,list):
        for item in value:finite(item)
def read(path):
    value=json.loads(path.read_text());finite(value);return value
def stream(name):
    with gzip.open(OUT/(name+'.jsonl.gz'),'rt') as handle:
        for line in handle:
            value=json.loads(line);finite(value);yield value
m=read(OUT/'manifest.json');r=read(OUT/'results.json')
assert sha(OUT/'manifest.json')==r['manifest_sha256']
failure=read(OUT/'failure.json')
assert failure['completed_updates']==1040 and failure['type']=='AssertionError'
assert 'assert all(sha(path)==digest for path,digest in protected.items())' in failure['traceback']
assert r['artifact_integrity_exception']['affected_tracked_path']=='src/__pycache__/executor.cpython-312.pyc'
assert sha(ROOT/m['backbone']['path'])==m['backbone']['sha256']
for path,digest in m['source_sha256'].items():assert sha(ROOT/path)==digest
assert sha(ROOT/'experiments/run_gatec_fresh_task_discovery.py')==m['script_sha256']
assert sha(ROOT/'experiments/run_gatec_paired_task_maintenance.py')==m['observer_sha256']
a=read(OUT/'encoding_attestation.json')
for path,digest in a['helper_source_sha256'].items():assert sha(ROOT/path)==digest
assert sha(ROOT/'data/task_a.py')==a['dataset_source_sha256']
current_diff_sha=hashlib.sha256(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT)).hexdigest()
assert current_diff_sha!=m['git']['tracked_diff_sha256']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==m['git']['head']
assert subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True).returncode==0
inputs=m['inputs']
for path,key in [('training_sequences.json','training_materialized_sha256'),('training_schedule.json','schedule_sha256'),('final_panel.json','final_panel_sha256'),('prior_input_inventory.json','inventory_sha256')]:assert sha(OUT/path)==inputs[key]
training=read(OUT/'training_sequences.json')['rows'];panel=read(OUT/'final_panel.json')['rows']
inventory=read(OUT/'prior_input_inventory.json')
excluded={tuple(b) for rec in inventory['records'] for b in rec['extracted_sequences']}
excluded.update(tuple(s['bits']) for s in training)
assert len(panel)==len({tuple(s['bits']) for s in panel})==96
assert not ({tuple(s['bits']) for s in panel}&excluded)
for rec in inventory['records']:assert sha(ROOT/rec['path'])==rec['sha256']
visits=read(OUT/'training_schedule.json')['visits']
for epoch in range(1,11):
    assert sorted(v['source_index'] for v in visits if v['epoch']==epoch)==list(range(104))
assert [v['update'] for v in visits]==list(range(1,1041))
updates=list(stream('updates'));assert [u['update'] for u in updates]==list(range(1,1041))
assert all(all(g['present'] and g['finite'] for g in u['gradients'].values()) for u in updates)
snapshots=list(stream('snapshots'));assert [s['update'] for s in snapshots]==[0,104,520,1040]
snapweights={}
for s in snapshots:
    assert sha(ROOT/s['path'])==s['sha256']
    ck=torch.load(ROOT/s['path'],map_location='cpu',weights_only=True)
    assert ck['updates']==s['update'] and ck['manifest_sha256']==r['manifest_sha256']
    assert ck['backbone_sha256']==m['backbone']['sha256'] and 'state_dict' not in ck
    assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in ck['controller_state_dict'].values())
    snapweights[s['update']]=ck['controller_state_dict']
initial,final=snapweights[0],snapweights[1040]
assert initial['norm_h.weight'].eq(1).all() and initial['norm_memory_read.weight'].eq(1).all()
assert set(initial)==set(final)
assert torch.equal(initial['mlp.2.weight'][10],final['mlp.2.weight'][10])
assert torch.equal(initial['mlp.2.bias'][10],final['mlp.2.bias'][10])
delta=torch.cat([(final[k]-initial[k]).flatten() for k in initial])
assert float(delta.norm())==r['controller_change']['l2']
tapes={}
for name,info in m['randomness']['tapes'].items():
    assert sha(OUT/info['file'])==info['sha256']
    tape=torch.load(OUT/info['file'],map_location='cpu',weights_only=True)
    assert tape['seed']==info['seed'];tapes[info['seed']]=tape['uniforms']
histories=defaultdict(list);blocks=defaultdict(list);previous={};phase_counts=defaultdict(Counter)
logit_cells=defaultdict(list);st_count=halt_count=0;batch=[];reconstructed_count=0
def reconstruct():
    global reconstructed_count
    if not batch:return
    logits=torch.tensor([x[0] for x in batch],device='cuda',dtype=torch.float32)
    masks=torch.tensor([x[1] for x in batch],device='cuda',dtype=torch.bool)
    uniform=torch.stack([x[2] for x in batch]).to('cuda');eps=torch.finfo(torch.float32).eps
    uniform=uniform.clamp(eps,1-eps);g=-torch.log(-torch.log(uniform))
    z=torch.softmax((torch.log_softmax(logits.masked_fill(~masks,-torch.inf),-1)+g)/2,-1)
    actions=torch.tensor([x[3] for x in batch],device='cuda')
    bad=(z.argmax(-1)!=actions).nonzero().flatten().tolist()
    # Resolve only kernel-shape rounding discrepancies using the exact [1,10]
    # primitive dimensions. This never changes a recorded or executed action.
    for i in bad:
        uu=batch[i][2].to('cuda').unsqueeze(0).clamp(eps,1-eps)
        gg=-torch.log(-torch.log(uu))
        zz=torch.softmax((torch.log_softmax(logits[i:i+1].masked_fill(~masks[i:i+1],-torch.inf),-1)+gg)/2,-1)
        assert int(zz.argmax(-1))==int(actions[i]),'Recorded-noise reconstruction mismatch'
    reconstructed_count+=len(batch);batch.clear()
for d in stream('decisions'):
    key=d['phase'],d['arm'],d['sequence_id'],d['token'];phase=key[:2]
    p=d['pointer_before'];e=d['execution_before'];restart=d.get('just_restarted_before',False)
    prev=previous.get(key,(0,0,False))
    assert (e,p,restart)==prev and d['decision']==len(histories[key])
    mask=build_execution_mask(p,e,restart,8).tolist();action=d['selected_action']
    assert mask==d['legal_action_mask'] and action<10 and mask[action]
    block=action if action<8 else p if action==8 else None
    assert block==d['actual_executed_block']
    assert d['execution_after']==e+int(block is not None)
    assert d['pointer_after']==(action if action<8 else p if action==8 else 0)
    assert d['budget']==[[0.,0.]]
    if 'probabilities' in d:
        probabilities=torch.tensor(d['probabilities']).reshape(-1).tolist()
        assert abs(sum(probabilities)-1)<1e-6
        assert all(prob==0 for prob,legal in zip(probabilities,mask[:10]) if not legal)
    else:assert d['arm']=='F'  # Canonical reference has no autonomous policy.
    if 'noise_coordinate' in d:
        seed,sid,t,j=d['noise_coordinate'];assert (sid,t,j)==(d['sequence_id'],d['token'],d['decision'])
        assert d['candidate_constructions']==1 and d['candidate_block_calls']==list(range(8))+[p]
        batch.append((d['raw_controller_logits'][0][:10],mask[:10],tapes[seed][sid][t,j],action))
        if len(batch)==8192:reconstruct()
        st_count+=1;phase_counts[phase]['candidate_constructions']+=1;phase_counts[phase]['candidate_block_calls']+=9
    else:assert not d.get('candidate_constructions',0) and not d.get('candidate_block_calls',[])
    if d['phase']=='training' and d['arm']=='T':
        grad=torch.tensor(d['retained_logit_gradient'][0]);assert grad.shape==(11,) and grad[10]==0
        legal=torch.tensor(mask[:10]);assert grad[:10][~legal].eq(0).all()
        logit_cells[(d['token'],d['decision'])].append(float(grad.norm()));halt_count+=1
    histories[key].append(action)
    if block is not None:blocks[key].append(block)
    previous[key]=(d['execution_after'],d['pointer_after'],action==9)
    phase_counts[phase]['decisions']+=1;phase_counts[phase]['executions']+=int(block is not None)
reconstruct();assert st_count==reconstructed_count
prev_hash={};token_counts=Counter();version_checks=0
for t in stream('tokens'):
    seqkey=t['phase'],t['arm'],t['sequence_id'];key=(*seqkey,t['token']);actions=histories[key]
    assert t['P']==len(blocks[key])==8 and t['D']==len(actions)==8+actions.count(9)
    assert 8<=t['D']<=15 and t['executed_blocks']==blocks[key]
    assert t['restart_count']==actions.count(9) and t['repeat_count']==actions.count(8)
    expected_reason='canonical_sweep_completed' if t['arm']=='F' else 'execution_budget_reached'
    assert t['commit_count']==1 and t['commit_entries']==7 and t['termination_reason']==expected_reason
    if seqkey in prev_hash:assert prev_hash[seqkey]==t['incoming_persistent']['sha256']
    else:assert t['token']==0
    prev_hash[seqkey]=t['outgoing_persistent']['sha256'];token_counts[seqkey]+=1
    for scope,version in [('incoming_persistent',t['token']),('outgoing_persistent',t['token']+1)]:
        state=t[scope]['summary'];versions=[v for k,v in state.items() if k.endswith('/version')]
        assert len(versions)==7 and all(v==version for v in versions) and state['/tokens_committed']==version
        version_checks+=len(versions)
    if t['phase']=='training' and t['arm']=='T':assert t['graph_state']['/s']['requires_grad'] and t['graph_state']['/memory']['requires_grad']
sequences=list(stream('sequences'))
for s in sequences:
    key=s['phase'],s['arm'],s['sequence_id'];assert token_counts[key]==s['length']
    if s['phase']=='training' or s['phase'].startswith('final_final_tape_'):assert s['noise_reconstructions']==sum(s['D'])
for mode,arms in r['final'].items():
    for arm,splits in arms.items():
        counts=phase_counts[('final_'+mode,arm)]
        for field in ['decisions','executions','candidate_constructions','candidate_block_calls']:
            assert counts[field]==sum(metrics[field] for metrics in splits.values())
for arm in ['U','T']:
    counts=phase_counts[('training',arm)]
    for field in ['decisions','executions','candidate_constructions','candidate_block_calls']:
        assert counts[field]==sum(metrics[arm][field] for metrics in r['training_by_epoch'].values())
for row in updates:
    assert row['logit_gradient_coverage']['decisions']==sum(row['T']['D'])
assert halt_count==r['gradient_statistics']['logit_decisions']
verification=dict(measurements_verified=True,full_file_identity_verified=False,new_model_runs=0,new_optimizer_updates=0,exact_training_updates=1040,unchanged_U_updates=0,
    manifest_sha256=r['manifest_sha256'],source_and_input_hashes_unchanged=True,tracked_diff_unchanged=False,
    artifact_integrity_exception=r['artifact_integrity_exception'],current_tracked_diff_sha256=current_diff_sha,
    final_panel_distinct=96,known_exact_training_prior_overlap=0,finite_json_and_checkpoint_values=True,
    finite_controller_gradients=True,all_four_snapshots_fixed=True,HALT_output_row_unchanged=True,
    sampled_action_reconstructions=st_count,selected_block_executions=sum(v['executions'] for v in phase_counts.values()),
    observed_tokens=sum(token_counts.values()),observed_decisions=sum(v['decisions'] for v in phase_counts.values()),
    one_commit_per_token_verified=True,cache_version_checks=version_checks,
    token_state_continuity_verified=True,execution_legality_accounting_verified=True,
    training_HALT_exact_zero_gradient_decisions=halt_count,illegal_logit_gradients_zero=True,
    parameter_displacement_L2=float(delta.norm()),git_diff_check_clean=True,
    per_phase_counts={f'{phase}/{arm}':dict(counts) for (phase,arm),counts in phase_counts.items()},
    retained_logit_gradient_by_token_decision={f'{t}/{d}':dict(n=len(values),mean=sum(values)/len(values),minimum=min(values),maximum=max(values),nonzero=sum(v>0 for v in values)) for (t,d),values in logit_cells.items()})
with (OUT/'verification.json').open('x') as handle:json.dump(verification,handle,indent=2,allow_nan=False);handle.write('\n')
print(json.dumps({k:v for k,v in verification.items() if k not in ['per_phase_counts','retained_logit_gradient_by_token_decision']},indent=2))
