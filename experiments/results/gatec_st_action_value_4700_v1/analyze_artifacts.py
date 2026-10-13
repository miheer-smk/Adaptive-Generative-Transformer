"""Offline artifact checks and reporting; never constructs or executes a model."""
import sys
sys.dont_write_bytecode = True
import gzip, hashlib, json, math, subprocess
from collections import Counter, defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.diagnose_gatec_st_action_values_4700 import compare_banks
from experiments.replay_gatec_learned_policy import fingerprint
from src.masks import build_execution_mask

OUT = Path(__file__).resolve().parent
def load(name): return json.loads((OUT/name).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name, value):
    with (OUT/name).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
def lines(name):
    with gzip.open(OUT/(name+'.jsonl.gz'), 'rt') as handle:
        for line in handle: yield json.loads(line)
def finite(value):
    if torch.is_tensor(value): assert torch.isfinite(value).all()
    elif isinstance(value, float): assert math.isfinite(value)
    elif isinstance(value, dict):
        for v in value.values(): finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value: finite(v)
def same(a,b):
    if isinstance(a, dict):
        assert set(a)==set(b)
        for k in a: same(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b): same(x,y)
    elif isinstance(a,float):
        assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-14), (a,b)
    else: assert a==b, (a,b)

def main():
    assert not (OUT/'failure.json').exists(), 'Preserved failure; do not analyze as completed'
    m,r=load('manifest.json'),load('results.json')
    assert sha(OUT/'manifest.json')==r['manifest_sha256']
    assert r['counts']==dict(states=36,CF=2304,ST=288)
    finite(r)
    assert sha(ROOT/'experiments/diagnose_gatec_st_action_values_4700.py')==m['script_sha256']
    for path,digest in m['source_sha256'].items(): assert sha(ROOT/path)==digest
    for ck in m['checkpoints'].values(): assert sha(ROOT/ck['path'])==ck['sha256']
    assert sha(ROOT/m['backbone']['path'])==m['backbone']['sha256']
    for name,digest in m['randomness']['noise_files'].items(): assert sha(OUT/name)==digest
    banks={bank:torch.load(OUT/name,map_location='cpu',weights_only=True)['tapes']
           for bank,name in [('CF','continuation_noise.pt'),('ST','ST_probe_noise.pt')]}
    finite(banks)
    bootstrap=torch.load(OUT/'bootstrap_indices.pt',map_location='cpu',weights_only=True)
    probes={p['probe']['probe_id']:p for p in r['probes']}
    runs={x['run_id']:x for x in lines('runs')}
    assert len(runs)==2592
    by_probe=defaultdict(list)
    for item in runs.values():
        finite(item); by_probe[item['probe']['probe_id']].append(item)
    gradients=[]
    for index,p in enumerate(r['probes']):
        spec=p['probe']; pid=spec['probe_id']; rs=by_probe[pid]
        assert len(rs)==72 and Counter(x['bank'] for x in rs)=={'CF':64,'ST':8}
        assert {(x['tape'],x['forced_action']) for x in rs if x['bank']=='CF'}=={(k,a) for k in range(8) for a in range(8)}
        sr=sorted((x for x in rs if x['bank']=='ST'),key=lambda x:x['tape'])
        assert [x['tape'] for x in sr]==list(range(8))
        gs=[x['probe_logit_gradient'] for x in sr]; gradients.extend(gs)
        for g in gs:
            if g is not None: assert len(g)==11 and g[8:]==[0,0,0]
        fresh=compare_banks(p['raw_logits_both_checkpoints'][str(spec['checkpoint_update'])][:8],p['CF_final_ce'],gs,p['CF_correct'],
                            bootstrap['CF'][index],bootstrap['ST'][index],[p['pi_both_checkpoints']['0'],p['pi_both_checkpoints']['1040']])
        for k,v in fresh.items(): same(v,p[k])
        assert spec['token_index']==(0 if spec['condition']=='A' else next(s['length'] for s in m['six_sequences'] if s['sequence_id']==spec['sequence_id'])-1)
        assert sha(ROOT/p['state_file'])==p['state_sha256']
        state=torch.load(ROOT/p['state_file'],map_location='cpu',weights_only=True)
        finite(state)
        assert fingerprint(state['incoming_persistent'])==p['incoming_state']['sha256']
        assert state['incoming_persistent']['/tokens_committed']==spec['token_index']
        assert torch.equal(state['features']['h'],state['h0'])
        assert state['features']['pointer_one_hot'].argmax(-1).item()==0
        assert state['features']['budget'].eq(0).all()
        assert p['legal_successor_actions']==list(range(8))
    # Stream full trace; no model execution. Check every recorded logical noise key.
    token_rows={}; count=Counter(); action_counts=Counter()
    for d in lines('decisions'):
        if d['phase']=='prefix': continue # runtime prefix purity/hash checks recorded separately
        item=runs[d['sequence_id']]; spec=item['probe']; key=(d['sequence_id'],d['token'])
        z=token_rows.setdefault(key,dict(D=0,P=0,R=0,repeat=0,pointer=0,restarted=False,actions=[]))
        assert d['decision']==z['D'] and d['execution_before']==z['P']
        assert d['pointer_before']==z['pointer'] and d['just_restarted_before']==z['restarted']
        mask=build_execution_mask(z['pointer'],z['P'],z['restarted'],8).tolist()
        assert mask==d['legal_action_mask'] and mask[:10]==d['legal_successor_mask']
        a=d['selected_action']; assert a<10 and mask[a]
        assert d['candidate_block_calls']==list(range(8))+[z['pointer']]
        assert d['budget']==[[0.,0.]] and d['candidate_constructions']==1
        seed=4800 if item['bank']=='CF' else 4801
        assert d['noise_coordinate']==[seed,spec['sequence_id'],d['token'],d['decision']]
        u=banks[item['bank']][item['tape']][spec['sequence_id']][d['token'],d['decision']]
        eps=torch.finfo(torch.float32).eps;u=u.clamp(eps,1-eps)
        raw=torch.tensor(d['raw_controller_logits'][0][:10],dtype=torch.float32)
        legal=torch.tensor(mask[:10]);lp=F.log_softmax(raw.masked_fill(~legal,-torch.inf),-1)
        sampled=int(((lp-torch.log(-torch.log(u)))/2).argmax())
        intervention=item['bank']=='CF' and d['token']==spec['token_index'] and d['decision']==0
        if intervention: assert a==item['forced_action'];count['forced_decisions']+=1
        else: assert a==sampled;count['unforced_noise_reconstructions']+=1
        block=a if a<8 else z['pointer'] if a==8 else None
        assert d['actual_executed_block']==d['decoded_block_target']==block
        assert d['executed_blocks']==([] if block is None else [block])
        z['D']+=1;z['P']+=int(a!=9);z['R']+=int(a==9);z['repeat']+=int(a==8)
        z['pointer']=a if a<8 else z['pointer'] if a==8 else 0;z['restarted']=a==9
        z['actions'].append(['ADVANCE',a] if a<8 else ['REPEAT',block] if a==8 else ['RESTART',None])
        assert d['execution_after']==z['P'] and d['pointer_after']==z['pointer'] and d['just_restarted_after']==z['restarted']
        count['continuation_decisions']+=1;action_counts[str(a)]+=1
    history=defaultdict(list); predecessor={}; commits=0
    for t in lines('tokens'):
        if t['phase']=='prefix': continue
        run=runs[t['sequence_id']];key=(t['sequence_id'],t['token']);z=token_rows.pop(key)
        assert t['P']==z['P']==8 and t['D']==z['D']==8+z['R'] and 8<=z['D']<=15
        assert t['restart_count']==z['R'] and t['repeat_count']==z['repeat'] and t['history']==z['actions']
        assert t['commit_count']==1 and t['commit_entries']==7 and t['termination_reason']=='execution_budget_reached'
        incoming=t['incoming_persistent'];outgoing=t['outgoing_persistent']
        assert incoming['summary']['/tokens_committed']==t['token']
        assert outgoing['summary']['/tokens_committed']==t['token']+1
        for k,v in outgoing['summary'].items():
            if k.endswith('/version'): assert v==t['token']+1
        expected=predecessor.get(t['sequence_id'],probes[run['probe']['probe_id']]['incoming_state']['sha256'])
        assert incoming['sha256']==expected
        predecessor[t['sequence_id']]=outgoing['sha256'];history[t['sequence_id']].append(t['history'])
        if run['bank']=='ST':
            assert t['graph_state']['/s']['requires_grad'] and t['graph_state']['/memory']['requires_grad']
        commits+=1
    assert not token_rows
    for run_id,item in runs.items():
        assert history[run_id]==item['routes'] and item['commits']==len(history[run_id])
        decisions=sum(item['D']);assert item['candidate_constructions']==decisions
        assert item['candidate_block_calls']==item['bank_purity_checks']==9*decisions
    assert commits==20160
    raw_diff=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT)
    assert hashlib.sha256(raw_diff).hexdigest()==m['git']['tracked_diff_sha256']
    assert subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True).returncode==0
    verification=dict(offline_only=True,model_calls=0,additional_continuations=0,additional_ST_probes=0,
        manifest_sha256=sha(OUT/'manifest.json'),counts=dict(r['counts'],completed_continuation_tokens=commits,**dict(count)),
        benchmark_bootstrap_recomputed_all_36=True,source_backbone_checkpoint_noise_state_hashes_verified=True,
        original_tracked_diff_unchanged=True,git_diff_check_clean=True,
        connected_gradient_probes=sum(g is not None for g in gradients),disconnected_gradient_probes=sum(g is None for g in gradients),
        exact_zero_gradient_probes=sum(g is not None and all(v==0 for v in g) for g in gradients),
        small_connected_gradient_probes=sum(g is not None and sum(v*v for v in g)**.5<=1e-8 for g in gradients),
        action_counts=dict(action_counts),physical_candidate_bank_calls=sum(x['candidate_block_calls'] for x in runs.values()),
        GDN_purity_checks={k:sum(x['gdn_purity_checks'].get(k,0) for x in runs.values()) for k in ['4','5']},
        interpretation='Continuation selected-block accounting is distinct from nine physical speculative bank calls per routing decision. Purity was asserted at runtime; offline accounting and noise reconstructions are additional checks.')
    write('verification.json',verification)
    print(json.dumps(verification,indent=2))

if __name__=='__main__':main()
