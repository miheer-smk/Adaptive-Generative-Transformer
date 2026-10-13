"""Offline validation of the locked experiment; no model forward or optimizer step."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import gzip,json,hashlib,math,subprocess
from collections import Counter,defaultdict
import torch
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.masks import build_execution_mask
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(x):
    if isinstance(x,float):assert math.isfinite(x)
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,list):
        for v in x:finite(v)
def read(p):
    x=json.loads(Path(p).read_text());finite(x);return x
def rows(name):
    with gzip.open(OUT/f'{name}.jsonl.gz','rt') as h:
        for line in h:
            x=json.loads(line);finite(x);yield x
def write(p,x):
    with Path(p).open('x') as h:json.dump(x,h,indent=2,allow_nan=False);h.write('\n')
torch.set_num_threads(4)
m=read(OUT/'manifest.json');r=read(OUT/'results.json')
assert not (OUT/'failure.json').exists(),'Preserved failure: do not certify a failed experiment'
assert sha(OUT/'manifest.json')==r['manifest_sha256']
assert sha(ROOT/'experiments/run_gatec_paired_gradient_average_4900.py')==m['script_sha256']
for path,digest in m['protected_file_sha256'].items():assert sha(path)==digest,path
encoding=read(ROOT/'experiments/results/gatec_fresh_task_discovery_4700_v1/encoding_attestation.json')
assert sha(ROOT/'data/task_a.py')==encoding['dataset_source_sha256']
for path,digest in encoding['helper_source_sha256'].items():assert sha(ROOT/path)==digest,path
for name,digest in m['identities'].items():assert sha(OUT/name)==digest
for name,record in m['randomness']['files'].items():assert sha(OUT/record['path'])==record['sha256']
for path,record in read(OUT/'artifact_index.json')['files'].items():assert sha(OUT/path)==record['sha256']
assert subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True).returncode==0
assert hashlib.sha256(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT)).hexdigest()==m['git']['tracked_diff_sha256']
assert r['exact_optimizer_steps']==dict(C=520,T=520) and r['exact_training_rollouts']==dict(C=1040,T=1040)
visits=read(OUT/'training_schedule.json')['rows'];assert [v['update'] for v in visits]==list(range(1,521))
for epoch in range(1,6):assert sorted(v['source_index'] for v in visits if v['epoch']==epoch)==list(range(104))
updates=list(rows('updates'));assert [u['update'] for u in updates]==list(range(1,521))
for u in updates:
    for arm in ['C','T']:
        a=u['arms'][arm]
        assert a['optimizer_steps']==u['update'] and a['gradient_pair']['finite']
        assert len(a['rollouts'])==len(a['losses'])==2
        assert all(a['rollouts'][i]['CE']==a['losses'][i] for i in range(2))
        if arm=='C':
            assert a['gradient_pair']['max_absolute_difference']<=1e-6
            assert a['rollouts'][0]['routes']==a['rollouts'][1]['routes']
            assert a['losses'][0]==a['losses'][1]
panel=read(OUT/'confirmation_panel.json')['rows'];assert len(panel)==len({tuple(s['bits']) for s in panel})==96
inventory=read(OUT/'prior_input_inventory.json')
excluded={tuple(bits) for rec in inventory['records'] for bits in rec['extracted_sequences']}
assert not ({tuple(s['bits']) for s in panel}&excluded)
for rec in inventory['records']:assert sha(ROOT/rec['path'])==rec['sha256']
initial=torch.load(ROOT/m['initial']['path'],map_location='cpu',weights_only=True)['controller_state_dict']
for snap in r['snapshots']:
    assert sha(ROOT/snap['path'])==snap['sha256']
    ck=torch.load(ROOT/snap['path'],map_location='cpu',weights_only=True)
    assert ck['updates']==snap['update'] and ck['arm']==snap['arm'] and ck['manifest_sha256']==r['manifest_sha256']
    state=ck['controller_state_dict'];assert state.keys()==initial.keys()
    assert all(torch.isfinite(v).all() and v.dtype==torch.float32 for v in state.values())
    if snap['update']==0:assert all(torch.equal(v,initial[k]) for k,v in state.items())
    assert torch.equal(state['mlp.2.weight'][10],initial['mlp.2.weight'][10])
    assert torch.equal(state['mlp.2.bias'][10],initial['mlp.2.bias'][10])
    delta=torch.cat([(state[k]-initial[k]).flatten() for k in initial])
    assert float(delta.norm())==snap['controller_change']['l2']
history=defaultdict(list);blocks=defaultdict(list);metadata={};counts=defaultdict(Counter);gradient_calls=0
for d in rows('decisions'):
    key=d['phase'],d['arm'],d['sequence_id'],d['token'];phase=key[:2]
    p,e,j=d['pointer_before'],d['execution_before'],d['just_restarted_before']
    assert (p,e,j)==metadata.get(key,(0,0,False)) and d['decision']==len(history[key])
    mask=build_execution_mask(p,e,j,8).tolist();action=d['selected_action']
    assert mask==d['legal_action_mask'] and action<10 and mask[action]
    actual=action if action<8 else p if action==8 else None
    assert actual==d['actual_executed_block'] and d['execution_after']==e+int(actual is not None)
    assert d['pointer_after']==(action if action<8 else p if action==8 else 0)
    assert d['budget']==[[0.,0.]]
    if d['phase'].startswith('training_rollout') or d['phase'].startswith('confirmation_tape'):
        assert d['candidate_constructions']==1 and d['candidate_block_calls']==list(range(8))+[p]
        counts[phase]['candidate_constructions']+=1;counts[phase]['candidate_block_calls']+=9
    else:assert not d.get('candidate_constructions',0) and not d.get('candidate_block_calls',[])
    if d['phase'].startswith('training_rollout'):
        coord=d['noise_coordinate'];assert coord['token']==d['token'] and coord['decision']==d['decision']
        assert coord['lane']==(0 if d['arm']=='C' or d['phase']=='training_rollout0' else 1)
        grad=torch.tensor(d['retained_logit_gradient'][0]);assert grad.shape==(11,) and torch.isfinite(grad).all() and grad[10]==0
        assert grad[:10][~torch.tensor(mask[:10])].eq(0).all();gradient_calls+=1
    history[key].append(action)
    if actual is not None:blocks[key].append(actual)
    metadata[key]=(d['pointer_after'],d['execution_after'],action==9)
    counts[phase]['decisions']+=1;counts[phase]['executions']+=int(actual is not None)
token_counts=Counter();state_hash={};versions=0
for t in rows('tokens'):
    seqkey=t['phase'],t['arm'],t['sequence_id'];key=(*seqkey,t['token']);actions=history[key]
    assert t['P']==len(blocks[key])==8 and t['D']==len(actions)==8+actions.count(9)
    assert t['executed_blocks']==blocks[key] and t['restart_count']==actions.count(9) and t['repeat_count']==actions.count(8)
    assert t['commit_count']==1 and t['commit_entries']==7
    assert t['termination_reason']==('canonical_sweep_completed' if t['arm']=='F' else 'execution_budget_reached')
    assert t['token']==token_counts[seqkey]
    if t['token']:assert state_hash[seqkey]==t['incoming_persistent']['sha256']
    state_hash[seqkey]=t['outgoing_persistent']['sha256'];token_counts[seqkey]+=1
    for scope,expected in [('incoming_persistent',t['token']),('outgoing_persistent',t['token']+1)]:
        state=t[scope]['summary'];v=[x for k,x in state.items() if k.endswith('/version')]
        assert len(v)==7 and all(x==expected for x in v) and state['/tokens_committed']==expected;versions+=7
    if t['phase'].startswith('training_rollout'):assert t['graph_state']['/s']['requires_grad'] and t['graph_state']['/memory']['requires_grad']
sequence_counts=Counter()
for s in rows('sequences'):
    phase=(s['phase'] if s['phase']!='training' else f"training_rollout{s['rollout']}")
    key=phase,s['arm'],s['sequence_id'];assert token_counts[key]==s['length']
    sequence_counts[phase,s['arm']]+=1
    if s['phase']=='training':assert s['gradient_coverage']['HALT_exact_zero'] and s['gradient_coverage']['illegal_exact_zero']
assert all(sequence_counts[f'training_rollout{j}',a]==520 for j in [0,1] for a in ['C','T'])
assert sequence_counts['confirmation_reference','F']==96
assert all(sequence_counts['confirmation_'+mode,a]==96 for mode in ['deterministic','tape0','tape1','tape2','tape3'] for a in ['U','C','T'])
bs=torch.load(OUT/'bootstrap_indices.pt',map_location='cpu',weights_only=True)['indices']
evals={mode:read(OUT/f'evaluation_{mode}.json') for mode in ['deterministic','tape0','tape1','tape2','tape3']}
for split in ['val','ood_test']:
    ids=[i for i,s in enumerate(panel) if s['split']==split]
    acc=[];ce=[]
    for mode in ['tape0','tape1','tape2','tape3']:
        acc.append([int(evals[mode]['T'][i]['correct'])-int(evals[mode]['C'][i]['correct']) for i in ids])
        ce.append([evals[mode]['T'][i]['CE']-evals[mode]['C'][i]['CE'] for i in ids])
    for values,name in [(acc,'stochastic_accuracy'),(ce,'stochastic_CE')]:
        paired=torch.tensor(values,dtype=torch.float64).mean(0)
        interval=torch.quantile(paired[bs[split]].mean(1),torch.tensor([.025,.975],dtype=torch.float64)).tolist()
        assert interval==r['paired'][split][name]['bootstrap_95'] and float(paired.mean())==r['paired'][split][name]['mean']
verification=dict(exact_training_rollouts=dict(C=1040,T=1040),exact_updates=dict(C=520,T=520),new_model_calls=0,new_optimizer_steps=0,
    duplicated_control_gradients_verified=True,maximum_C_gradient_difference=r['gradient_statistics']['C']['maximum_pair_difference'],
    averaged_gradient_arithmetic_verified_in_scheduled_updates=True,finite_controller_gradients=True,retained_training_logit_gradients=gradient_calls,
    legality_accounting_verified=True,one_commit_per_completed_token=True,cache_version_checks=versions,persistent_state_continuity_verified=True,
    observed_tokens=sum(token_counts.values()),observed_decisions=sum(c['decisions'] for c in counts.values()),
    selected_block_executions=sum(c['executions'] for c in counts.values()),candidate_block_calls=sum(c['candidate_block_calls'] for c in counts.values()),
    historical_checkpoint_and_production_identity_verified=True,frozen_noncontroller_verified_by_runtime=True,HALT_output_parameters_unchanged=True,
    snapshots=[0,104,260,520],bootstrap_intervals_reproduced=True,git_diff_check='clean',historical_exception=m['historical_integrity_exception'],
    task_generator_and_historical_helper_identities_verified=True,dataset_source_sha256=encoding['dataset_source_sha256'],
    counts={f'{phase}/{arm}':dict(c) for (phase,arm),c in counts.items()},sequence_counts={f'{phase}/{arm}':n for (phase,arm),n in sequence_counts.items()})
write(OUT/'verification.json',verification)
write(OUT/'verified_artifact_index.json',dict(files={str(p.relative_to(OUT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file()}))
print(json.dumps({k:v for k,v in verification.items() if k not in ['counts','sequence_counts']},indent=2))
