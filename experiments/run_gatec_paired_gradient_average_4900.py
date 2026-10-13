"""Locked two-arm, 520-update full-ST gradient-averaging intervention."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse,gzip,json,hashlib,time,traceback,subprocess
from collections import Counter
from contextlib import ExitStack
from datetime import datetime,timezone
import torch
import torch.nn.functional as F
from data.task_a import TaskAParity
from experiments.run_gatec_paired_task_maintenance import IndexedNoise,parameter_change,trace_path,action_probability_statistics
from experiments.run_gatec_fresh_task_discovery import (
    VerifiedStochasticPath,anchor_guard,gradient_coverage,prior_inputs,recorded_bits)
from experiments.replicate_gatec_locked_imitation import make_model,bits_from_x,panel_identity
from experiments.replay_gatec_learned_policy import (
    ObservedPath,BACKBONE,BACKBONE_SHA,sha,copy,snapshot,unchanged,fingerprint,git_identity,encode)
from experiments.evaluate_gatec_locked_policy import record,write_json

PARENT=ROOT/'experiments/results/gatec_fresh_task_discovery_4700_v1'
OLD_DIAGNOSTIC=ROOT/'experiments/results/gatec_st_action_value_4700_v1'
OUT=ROOT/'experiments/results/gatec_paired_gradient_average_4900_v1'
INITIAL=ROOT/'experiments/runs/stage3_fresh_task_4700_update0000_locked_v1.pt'
INITIAL_SHA='ae8ff9256afd573331719cf15a3f82d21f70181891400640739a0415c82d6279'
SNAPS={arm:{n:ROOT/f'experiments/runs/stage3_gradient_average_4900_{arm}_update{n:04d}_locked_v1.pt'
            for n in [0,104,260,520]} for arm in ['C','T']}

class LaneNoise(IndexedNoise):
    def __init__(self,draws,lane,bank='training',seed=4900):
        self.draws,self.lane,self.bank,self.seed=draws,lane,bank,seed

def confirmation_panel(excluded):
    task=TaskAParity(64,'cpu');rows=[];attempts=[];used=set()
    for split,lengths,quota,seed in [('val',[16],16,4901),('ood_test',range(17,33),2,4902)]:
        task_split='val' if split=='val' else 'test'
        task._eval_generators[task_split].manual_seed(seed)
        for length in lengths:
            balance=Counter()
            while sum(balance.values())<2*quota:
                batch=task.make_batch(1,length,task_split)
                bits=bits_from_x(batch.x,task)[0];label=int(batch.target[0]);key=tuple(bits)
                reason='known_input_overlap' if key in excluded else 'panel_duplicate' if key in used else 'parity_quota_full' if balance[label]>=quota else 'accepted'
                attempts.append(dict(split=split,length=length,bits=bits,label=label,reason=reason))
                if reason=='accepted':
                    rows.append(dict(sequence_id=f'{split}/L{length}/row{sum(balance.values())}',split=split,length=length,bits=bits,label=label))
                    used.add(key);balance[label]+=1
            assert balance==Counter({0:quota,1:quota})
    assert len(rows)==len(used)==96 and not (used&excluded)
    return rows,attempts

def training_lanes(visits):
    generator=torch.Generator().manual_seed(4900);lanes={0:{},1:{}}
    for s in visits:
        for lane in [0,1]:
            lanes[lane][s['sequence_id']]=torch.rand(s['length'],15,10,generator=generator,dtype=torch.float32)
    return lanes

def stratified_indices(rows,generator,replicates=10000):
    groups={}
    for i,s in enumerate(rows):groups.setdefault((s['length'],s['label']),[]).append(i)
    pieces=[]
    for key in sorted(groups):
        ids=torch.tensor(groups[key],dtype=torch.long)
        pieces.append(ids[torch.randint(len(ids),(replicates,len(ids)),generator=generator)])
    return torch.cat(pieces,1)

def average_gradients(first,second):
    assert first.keys()==second.keys()
    return {k:(first[k]+second[k])*.5 for k in first}

def gradient_pair(first,second):
    a=torch.cat([v.flatten() for v in first.values()]);b=torch.cat([v.flatten() for v in second.values()])
    return dict(first_l2=float(a.norm()),second_l2=float(b.norm()),dot=float(a@b),difference_l2=float((a-b).norm()),
                max_absolute_difference=float((a-b).abs().max()),averaged_l2=float(((a+b)*.5).norm()),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))

def path_metrics(spec,path):
    action_probability_statistics(path.rows)
    final=path.tokens[-1];actions=Counter(r['selected_action'] for r in path.rows)
    for t in path.tokens:
        assert t['P']==len(t['executed_blocks'])==8 and t['D']==8+t['restart_count'] and 8<=t['D']<=15
        assert t['commit_count']==1 and t['commit_entries']==7 and t['termination_reason'] in ['execution_budget_reached','canonical_sweep_completed']
    return dict(**spec,prediction=final['prediction'],correct=final['prediction']==spec['label'],CE=final['prefix_ce'],
        logits=encode(final['output_logits']),D=[t['D'] for t in path.tokens],P=[t['P'] for t in path.tokens],
        restart_count=sum(t['restart_count'] for t in path.tokens),repeat_count=sum(t['repeat_count'] for t in path.tokens),
        unique_blocks=[len(set(t['executed_blocks'])) for t in path.tokens],
        routes=[[r['selected_action'] for r in path.rows if r['token']==t['token']] for t in path.tokens],
        executed_blocks=[t['executed_blocks'] for t in path.tokens],
        canonical_action_tokens=sum([r['selected_action'] for r in path.rows if r['token']==t['token']]==list(range(8)) for t in path.tokens),
        canonical_block_tokens=sum(t['executed_blocks']==list(range(8)) for t in path.tokens),
        exact_sequence_canonical=all([r['selected_action'] for r in path.rows if r['token']==t['token']]==list(range(8)) for t in path.tokens),
        action_counts=dict(actions),commits=len(path.tokens),candidate_constructions=getattr(path,'candidate_constructions',0),
        candidate_block_calls=sum(len(r.get('candidate_block_calls',[])) for r in path.rows),
        bank_purity_checks=path.block_purity_checks,gdn_purity_checks=dict(path.gdn_purity_checks),
        action_entropy_sum=sum(r.get('entropy',0) for r in path.rows),
        minimum_selected_margin=min((r.get('selected_raw_logit_margin',float('inf')) for r in path.rows)),
        termination_reasons=[t['termination_reason'] for t in path.tokens])

def summarize(rows,per_length=True):
    n=len(rows);tokens=sum(r['length'] for r in rows);decisions=sum(sum(r['D']) for r in rows)
    result=dict(sequences=n,accuracy=sum(r['correct'] for r in rows)/n,mean_CE=sum(r['CE'] for r in rows)/n,
        tokens=tokens,decisions=decisions,executions=8*tokens,mean_D=decisions/tokens,maximum_D=max(max(r['D']) for r in rows),
        restart_count=sum(r['restart_count'] for r in rows),repeat_count=sum(r['repeat_count'] for r in rows),
        canonical_action_fraction=sum(r['canonical_action_tokens'] for r in rows)/tokens,
        canonical_block_fraction=sum(r['canonical_block_tokens'] for r in rows)/tokens,
        sequence_canonical_fraction=sum(r['exact_sequence_canonical'] for r in rows)/n,
        D_distribution=dict(Counter(d for r in rows for d in r['D'])),P_distribution=dict(Counter(p for r in rows for p in r['P'])),
        unique_block_distribution=dict(Counter(i for r in rows for i in r['unique_blocks'])),
        route_frequencies=dict(Counter(','.join(map(str,route)) for r in rows for route in r['routes'])),
        candidate_constructions=sum(r['candidate_constructions'] for r in rows),candidate_block_calls=sum(r['candidate_block_calls'] for r in rows),
        commits=sum(r['commits'] for r in rows),purity_checks=sum(r['bank_purity_checks'] for r in rows),
        mean_action_entropy=sum(r['action_entropy_sum'] for r in rows)/decisions)
    if per_length:result['per_length']={str(L):summarize([r for r in rows if r['length']==L],False) for L in sorted({r['length'] for r in rows})}
    return result

def paired_interval(values,indices):
    values=torch.tensor(values,dtype=torch.float64)
    return dict(mean=float(values.mean()),bootstrap_95=torch.quantile(values[indices].mean(1),torch.tensor([.025,.975],dtype=torch.float64)).tolist())

def main():
    if OUT.exists() or any(p.exists() for v in SNAPS.values() for p in v.values()):raise RuntimeError('Refusing repetition or overwrite')
    torch.set_num_threads(4)
    parent=json.loads((PARENT/'manifest.json').read_text());oldresults=json.loads((PARENT/'results.json').read_text())
    assert sha(PARENT/'manifest.json')==oldresults['manifest_sha256']
    for p,h in parent['source_sha256'].items():assert sha(ROOT/p)==h,p
    assert sha(INITIAL)==INITIAL_SHA and sha(BACKBONE)==BACKBONE_SHA
    assert sha(PARENT/'training_sequences.json')==parent['inputs']['training_materialized_sha256']
    assert sha(PARENT/'training_schedule.json')==parent['inputs']['schedule_sha256']
    previous=json.loads((OLD_DIAGNOSTIC/'manifest.json').read_text())
    assert sha(ROOT/'experiments/run_gatec_paired_task_maintenance.py')==parent['observer_sha256']
    assert sha(ROOT/'experiments/run_gatec_fresh_task_discovery.py')==parent['script_sha256']
    for path,h in previous['source_sha256'].items():assert sha(ROOT/path)==h
    assert git_identity()['tracked_diff_sha256']==previous['git']['tracked_diff_sha256']
    training=json.loads((PARENT/'training_sequences.json').read_text())['rows']
    schedule=json.loads((PARENT/'training_schedule.json').read_text());visits=schedule['visits'][:520]
    assert len(training)==104 and len(visits)==520
    for s in visits:
        source=training[s['source_index']]
        assert s['bits']==source['bits'] and s['label']==source['label'] and s['source_sequence_id']==source['sequence_id']
    assert [s['update'] for s in visits]==list(range(1,521))
    for epoch in range(1,6):assert sorted(s['source_index'] for s in visits if s['epoch']==epoch)==list(range(104))
    assert parent['training']['clipping'] is None and parent['training']['tau']==2 and parent['training']['budget']==[0.,0.]
    assert parent['gradient_policy']['policy'].startswith('unchanged run_diff_sequence full-sequence BPTT')
    excluded,inventory,unknown=prior_inputs()
    # Existing helper's scan includes all named arrays in bounded JSON/sequence files;
    # additionally retain compressed sequence summaries from prior experiments.
    for path in sorted((ROOT/'experiments/results').rglob('sequences*.jsonl.gz')):
        found=[]
        with gzip.open(path,'rt') as h:
            for line in h:found.extend(recorded_bits(json.loads(line)))
        unique=sorted({bits for _,bits in found});excluded.update(unique)
        inventory.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),distinct_sequences=len(unique),extracted_sequences=[list(b) for b in unique]))
    excluded.update(tuple(s['bits']) for s in training)
    panel,generation=confirmation_panel(excluded);lanes=training_lanes(visits)
    seeds=torch.randint(0,2**31-1,(4,),generator=torch.Generator().manual_seed(4903)).tolist()
    eval_tapes=[IndexedNoise(panel,seed) for seed in seeds]
    boot_generator=torch.Generator().manual_seed(4904)
    bootstrap={split:stratified_indices([s for s in panel if s['split']==split],boot_generator) for split in ['val','ood_test']}
    stage1=torch.load(BACKBONE,map_location='cpu',weights_only=True);saved=torch.load(INITIAL,map_location='cpu',weights_only=True)
    weights={k:v for k,v in stage1['state_dict'].items() if not k.startswith('controller.')};initial=saved['controller_state_dict']
    assert saved['updates']==0 and saved['backbone_sha256']==BACKBONE_SHA
    assert fingerprint(snapshot(initial))==parent['initialization']['state_fingerprint']
    device=torch.device('cuda');models={}
    for arm in ['F','U','C','T']:
        models[arm],_=make_model(weights,device,4700,arm in ['C','T'])
        models[arm].controller.load_state_dict(initial,strict=True)
    assert models['T'].controller.input_dim==395 and models['T'].controller.num_actions==11
    numerical=dict(torch_version=str(torch.__version__),device='cuda',device_name=torch.cuda.get_device_name(),matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    assert numerical==parent['numerical'],(numerical,parent['numerical'])
    optimizers={arm:torch.optim.AdamW(models[arm].controller.parameters(),lr=1e-4,weight_decay=0.,betas=(.9,.999),eps=1e-8) for arm in ['C','T']}
    assert all(not o.state and encode(o.defaults)==parent['training']['optimizer_defaults'] for o in optimizers.values())
    protected={str(p):sha(p) for directory in [ROOT/'src',ROOT/'experiments/runs',PARENT,OLD_DIAGNOSTIC] for p in directory.rglob('*') if p.is_file()}
    identity=git_identity();OUT.mkdir(exist_ok=False)
    write_json(OUT/'training_schedule.json',dict(source_sha256=sha(PARENT/'training_schedule.json'),rows=visits))
    write_json(OUT/'confirmation_panel.json',dict(rows=panel,generation=generation,seeds=[4901,4902],identity=panel_identity(panel)))
    write_json(OUT/'prior_input_inventory.json',dict(records=inventory,unknown=unknown,exclusion_count=len(excluded),Stage1_exposure='unknown'))
    for name,data in [('training_noise',dict(seed=4900,lanes=lanes)),('evaluation_noise',dict(seed=4903,tape_seeds=seeds,uniforms=[t.draws for t in eval_tapes])),('bootstrap_indices',dict(seed=4904,indices=bootstrap))]:
        with (OUT/f'{name}.pt').open('xb') as h:torch.save(data,h)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol='single bounded two-independent-versus-duplicated full-ST gradients experiment4900-v1',git=identity,script_sha256=sha(__file__),
        historical_manifest_sha256=sha(PARENT/'manifest.json'),initial=dict(path=str(INITIAL.relative_to(ROOT)),sha256=INITIAL_SHA,controller_fingerprint=fingerprint(snapshot(initial))),
        backbone=dict(path=str(BACKBONE.relative_to(ROOT)),sha256=BACKBONE_SHA),source_sha256=parent['source_sha256'],protected_file_sha256=protected,
        training=dict(updates_per_arm=520,rollouts_per_update=2,rollouts_per_arm=1040,total_training_rollouts=2080,epochs=5,visit_source=str((PARENT/'training_schedule.json').relative_to(ROOT)),training_source=str((PARENT/'training_sequences.json').relative_to(ROOT)),
            optimizer=encode(optimizers['T'].defaults),loss='mean of two original final-sequence parity CEs via arithmetic gradient mean; each original full gradient measured separately',
            clipping=None,scheduler=None,auxiliary_loss=None,batch=1,tau=2.,budget=[0.,0.],gradient_policy=parent['gradient_policy'],
            resets='independent sequence initialization per rollout; ordinary within-sequence evolution/commits',frozen='every non-controller parameter',C='lane0 twice',T='lane0 then lane1',
            snapshot_updates=[0,104,260,520],outcome_checkpoint=520,selection='fixed final checkpoint; no intermediate outcome evaluation'),
        randomness=dict(training_seed=4900,training_recipe='CPU Generator4900; historical visits in order; lane0 then lane1; FP32 rand[length,15,10]; tables are sealed, not consumed as a route-dependent flat stream',
            training_key='bank/update/source_sequence_ID/lane/token/decision/successor_action_slot; arm excluded',
            evaluation_seed=4903,evaluation_tape_seeds=seeds,evaluation_recipe='four integer seeds from CPU Generator4903; each original IndexedNoise draws panel-order FP32[length,15,10] tensors',
            evaluation_key='bank/tape/sequence/token/decision/action; arm/checkpoint excluded',decision_axis='legal maximum D15 coverage, not a stopping cap',
            files={n:dict(path=f'{n}.pt',sha256=sha(OUT/f'{n}.pt')) for n in ['training_noise','evaluation_noise','bootstrap_indices']}),
        panel=dict(seeds=[4901,4902],identity=panel_identity(panel),known_input_overlap=0,Stage1_exposure='unknown; unretained historical input exposure cannot be excluded'),
        evaluation=dict(paths=['F fixed canonical','U untouched initial','C final duplicated','T final independent'],modes=['deterministic','four separate paired ST tapes'],prediction_ensembling=False),
        uncertainty=dict(replicates=10000,seed=4904,unit='paired sequence; four stochastic outcomes kept together',strata='length and parity (val only length16)',conditional='one initialization and four locked noise tapes'),
        decision_rules=dict(helped='on each panel pooled stochastic T-C>=.10, paired95 interval excludes0, positive on>=3/4 tapes; deterministic T>=C on both panels',
            useful='additionally T>=.90 deterministic and pooled stochastic on each panel, and CE lower than untouched U for both deterministic and pooled stochastic panels',
            chance='valid mechanics and chance-level T means this repair failed within the declared budget'),
        identities={name:sha(OUT/name) for name in ['training_schedule.json','confirmation_panel.json','prior_input_inventory.json']},numerical=numerical,
        historical_integrity_exception=previous['historical_exception'],focused_tests='10 passed: seven new diagnostic-local tests plus three existing paired-noise tests; no Task-A training in tests; production tests unchanged')
    write_json(OUT/'manifest.json',manifest);manifest_sha=sha(OUT/'manifest.json')
    print('MANIFEST SEALED '+manifest_sha,flush=True)
    initial_model=snapshot(models['U'].state_dict());task=TaskAParity(64,device);updates=[];snapshots=[];evaluation={};reference=[];started=time.monotonic();rollouts=Counter()
    def check_frozen():
        assert unchanged(initial_model,snapshot(models['U'].state_dict()))
        for model in models.values():
            assert all(torch.equal(model.state_dict()[k].detach().cpu(),v) for k,v in weights.items())
            assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('controller.'))
    def inputs(s):
        bits=torch.tensor([s['bits']],device=device);return task.token_features[bits],bits.cumsum(1).remainder(2)
    def run(s,arm,tape=None,gradient=False):
        x,targets=inputs(s)
        with anchor_guard(models[arm]):
            return VerifiedStochasticPath(models[arm],tape,s['sequence_id'],gradient).run(x,targets) if tape is not None else ObservedPath(models[arm],models[arm].controller,arm!='F').run(x,targets)
    def save(n):
        check_frozen()
        for arm in ['C','T']:
            path=SNAPS[arm][n];payload=dict(controller_state_dict={k:copy(v) for k,v in models[arm].controller.state_dict().items()},updates=n,arm=arm,
                backbone_sha256=BACKBONE_SHA,initial_sha256=INITIAL_SHA,manifest_sha256=manifest_sha,gradient_policy=parent['gradient_policy'],optimizer_defaults=encode(optimizers[arm].defaults),tau=2.,budget=[0.,0.],selection='predetermined; outcome520only')
            with path.open('xb') as h:torch.save(payload,h)
            snapshots.append(dict(arm=arm,update=n,path=str(path.relative_to(ROOT)),sha256=sha(path),controller_change=parameter_change(initial,models[arm].controller)))
        print('SNAPSHOTS SAVED '+str(n),flush=True)
    with ExitStack() as stack:
        handles={key:stack.enter_context(gzip.open(OUT/f'{key}.jsonl.gz','xt',compresslevel=1)) for key in ['decisions','tokens','sequences','updates']}
        try:
            save(0)
            for spec in visits:
                update=dict(update=spec['update'],epoch=spec['epoch'],sequence_id=spec['sequence_id'],source_sequence_id=spec['source_sequence_id'],arms={})
                for arm in ['C','T']:
                    model=models[arm];model.train();opt=optimizers[arm];before={k:copy(v) for k,v in model.controller.state_dict().items()};gs=[];losses=[];ms=[];paths=[];tick=time.monotonic()
                    for j,lane in enumerate([0,0] if arm=='C' else [0,1]):
                        opt.zero_grad(set_to_none=True)
                        with torch.enable_grad():
                            path=run(spec,arm,LaneNoise(lanes[lane],lane),True);loss=F.cross_entropy(path.final_logits,torch.tensor([spec['label']],device=device))
                            assert torch.isfinite(loss) and loss.requires_grad
                            loss.backward()
                        rollouts[arm]+=1
                        grad={k:p.grad.detach().clone() for k,p in model.controller.named_parameters() if p.grad is not None}
                        assert set(grad)==set(initial) and all(torch.isfinite(g).all() for g in grad.values())
                        assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('controller.'))
                        coverage=gradient_coverage(path);metrics=path_metrics(spec,path)
                        for row in path.rows:row['noise_coordinate']=dict(bank='training',master=4900,update=spec['update'],sequence=spec['source_sequence_id'],lane=lane,token=row['token'],decision=row['decision'],slots=list(range(10)))
                        trace_path(handles,f'training_rollout{j}',arm,spec,path)
                        record(handles['sequences'],dict(phase='training',arm=arm,rollout=j,lane=lane,**metrics,gradient_coverage=coverage))
                        assert unchanged(snapshot(before),snapshot(model.controller.state_dict()))
                        gs.append(grad);losses.append(float(loss.detach()));ms.append(metrics)
                        paths.append(dict(logits=copy(path.final_logits),route=metrics['routes']))
                        del path,loss
                    if arm=='C':
                        assert paths[0]['route']==paths[1]['route']
                        torch.testing.assert_close(paths[0]['logits'],paths[1]['logits'],atol=1e-6,rtol=1e-5)
                        for k in gs[0]:torch.testing.assert_close(gs[0][k],gs[1][k],atol=1e-6,rtol=1e-5)
                    averaged=average_gradients(gs[0],gs[1]);stat=gradient_pair(gs[0],gs[1]);assert stat['finite']
                    for k,p in model.controller.named_parameters():
                        p.grad=averaged[k]
                        torch.testing.assert_close(copy(p.grad),(copy(gs[0][k])+copy(gs[1][k]))*.5,atol=1e-6,rtol=1e-5)
                    opt.step()
                    assert all(torch.isfinite(p).all() for p in model.controller.parameters())
                    assert all(int(s['step'])==spec['update'] for s in opt.state.values())
                    update['arms'][arm]=dict(rollouts=ms,losses=losses,gradient_pair=stat,parameter_displacement=parameter_change(before,model.controller),
                        controller_fingerprint=fingerprint(snapshot(model.controller.state_dict())),elapsed_seconds=time.monotonic()-tick,optimizer_steps=spec['update'])
                    del gs,averaged,paths
                updates.append(update);record(handles['updates'],update)
                if spec['update']==1 or spec['update']%10==0:
                    print(f"UPDATE {spec['update']}/520 C/T mean CE {sum(update['arms']['C']['losses'])/2:.6g}/{sum(update['arms']['T']['losses'])/2:.6g} elapsed={time.monotonic()-started:.1f}s",flush=True)
                if spec['update'] in [104,260,520]:save(spec['update'])
            assert dict(rollouts)==dict(C=1040,T=1040) and len(updates)==520
            for arm in ['C','T']:optimizers[arm].zero_grad(set_to_none=True)
            check_frozen()
            for model in models.values():
                model.eval()
                for p in model.parameters():p.requires_grad_(False)
            for i,s in enumerate(panel):
                with torch.no_grad():path=run(s,'F');metric=path_metrics(s,path)
                reference.append(metric);trace_path(handles,'confirmation_reference','F',s,path);record(handles['sequences'],dict(phase='confirmation_reference',arm='F',**metric));del path
            write_json(OUT/'reference.json',reference)
            for mode,tape in [('deterministic',None)]+[(f'tape{i}',t) for i,t in enumerate(eval_tapes)]:
                evaluation[mode]={arm:[] for arm in ['U','C','T']}
                for i,s in enumerate(panel):
                    for arm in ['U','C','T']:
                        with torch.no_grad():path=run(s,arm,tape);metric=path_metrics(s,path)
                        evaluation[mode][arm].append(metric);trace_path(handles,'confirmation_'+mode,arm,s,path)
                        record(handles['sequences'],dict(phase='confirmation_'+mode,arm=arm,**metric));del path
                    if (i+1)%8==0:print(f'EVALUATION {mode} {i+1}/96 elapsed={time.monotonic()-started:.1f}s',flush=True)
                write_json(OUT/f'evaluation_{mode}.json',evaluation[mode])
            check_frozen()
            for p,h in protected.items():assert sha(p)==h,p
            assert git_identity()['tracked_diff_sha256']==identity['tracked_diff_sha256']
            assert sha(__file__)==manifest['script_sha256']
            for name,row in manifest['randomness']['files'].items():assert sha(OUT/row['path'])==row['sha256']
            for name,h in manifest['identities'].items():assert sha(OUT/name)==h
            assert subprocess.run(['git','diff','--check'],capture_output=True).returncode==0
        except Exception as error:
            write_json(OUT/'failure.json',dict(error=str(error),type=type(error).__name__,traceback=traceback.format_exc(),completed_updates=len(updates),training_rollouts=dict(rollouts),last_sequence=locals().get('spec'),stop='No replacement, tuning, extension or hidden retry'))
            raise
    aggregated={mode:{arm:{split:summarize([r for r in rows if r['split']==split]) for split in ['val','ood_test']} for arm,rows in arms.items()} for mode,arms in evaluation.items()}
    pooled={arm:{split:summarize([r for mode,arms in evaluation.items() if mode!='deterministic' for r in arms[arm] if r['split']==split]) for split in ['val','ood_test']} for arm in ['U','C','T']}
    paired={}
    for split in ['val','ood_test']:
        positions=[i for i,s in enumerate(panel) if s['split']==split]
        acc=[];ces=[];tapes=[]
        for mode in [f'tape{i}' for i in range(4)]:
            a=[int(evaluation[mode]['T'][i]['correct'])-int(evaluation[mode]['C'][i]['correct']) for i in positions]
            q=[evaluation[mode]['T'][i]['CE']-evaluation[mode]['C'][i]['CE'] for i in positions]
            acc.append(a);ces.append(q);tapes.append(dict(tape=mode,accuracy_T_minus_C=sum(a)/len(a),CE_T_minus_C=sum(q)/len(q)))
        paired[split]=dict(stochastic_accuracy=paired_interval(torch.tensor(acc,dtype=torch.float64).mean(0).tolist(),bootstrap[split]),
            stochastic_CE=paired_interval(torch.tensor(ces,dtype=torch.float64).mean(0).tolist(),bootstrap[split]),individual_tapes=tapes,
            deterministic_accuracy_T_minus_C=aggregated['deterministic']['T'][split]['accuracy']-aggregated['deterministic']['C'][split]['accuracy'])
    helped=all(paired[s]['stochastic_accuracy']['mean']>=.10 and paired[s]['stochastic_accuracy']['bootstrap_95'][0]>0 and sum(t['accuracy_T_minus_C']>0 for t in paired[s]['individual_tapes'])>=3 and paired[s]['deterministic_accuracy_T_minus_C']>=0 for s in ['val','ood_test'])
    useful=all(aggregated['deterministic']['T'][s]['accuracy']>=.9 and pooled['T'][s]['accuracy']>=.9 and aggregated['deterministic']['T'][s]['mean_CE']<aggregated['deterministic']['U'][s]['mean_CE'] and pooled['T'][s]['mean_CE']<pooled['U'][s]['mean_CE'] for s in ['val','ood_test'])
    result=dict(manifest_sha256=manifest_sha,snapshots=snapshots,reference={s:summarize([r for r in reference if r['split']==s]) for s in ['val','ood_test']},evaluation=aggregated,pooled_stochastic=pooled,paired=paired,
        intervention_helped=helped,useful_policy_T=useful,useful_policy_repair=helped and useful,
        training_by_pass={str(epoch):dict(updates_per_arm=104,rollouts_per_arm=208,duplicated_control_is_not_independent=True,
            metrics={arm:summarize([r for u in updates if u['epoch']==epoch for r in u['arms'][arm]['rollouts']]) for arm in ['C','T']}) for epoch in range(1,6)},
        exact_optimizer_steps={arm:520 for arm in ['C','T']},exact_training_rollouts=dict(rollouts),elapsed_seconds=time.monotonic()-started,
        controller_change={arm:parameter_change(initial,models[arm].controller) for arm in ['C','T']},
        gradient_statistics={arm:dict(minimum_average_norm=min(u['arms'][arm]['gradient_pair']['averaged_l2'] for u in updates),maximum_average_norm=max(u['arms'][arm]['gradient_pair']['averaged_l2'] for u in updates),
            mean_pair_difference=sum(u['arms'][arm]['gradient_pair']['difference_l2'] for u in updates)/520,maximum_pair_difference=max(u['arms'][arm]['gradient_pair']['difference_l2'] for u in updates),
            nonzero_parameter_updates=sum(u['arms'][arm]['parameter_displacement']['changed_values']>0 for u in updates)) for arm in ['C','T']},
        invariants='all observed legality/accounting/candidate-purity/GDN/cache/version/one-commit/frozen checks verified',historical_exception=previous['historical_exception'],
        limitations=['one initialization; four fixed sampling tapes, not training seed replication','bootstrap conditional on panel/initialization/tapes','unknown Stage-1 exposure','duplicate control rollouts are not independent observations'])
    write_json(OUT/'results.json',result)
    lines=['# Gate-C paired full-ST gradient averaging4900','',f'Manifest: `{manifest_sha}`. Exactly520 updates and1040 training rollouts per arm; final520 only. No tuning, extensions or replacement seeds.','',f'Intervention helped: **{helped}**. Useful treatment policy: **{useful}**. Combined useful-policy repair: **{helped and useful}**.','',
        '| Panel | Path | Deterministic accuracy/CE | Pooled stochastic accuracy/CE |','|---|---|---|---|']
    for split in ['val','ood_test']:
        lines.append(f"| {split} | F | {result['reference'][split]['accuracy']:.6f} / {result['reference'][split]['mean_CE']:.6g} | fixed canonical reference |")
        for arm in ['U','C','T']:
            d=aggregated['deterministic'][arm][split];s=pooled[arm][split]
            lines.append(f"| {split} | {arm} | {d['accuracy']:.6f} / {d['mean_CE']:.6g} | {s['accuracy']:.6f} / {s['mean_CE']:.6g} |")
    lines+=['','## Paired effects','']
    for split,p in paired.items():lines.append(f"{split}: pooled accuracy T-C={p['stochastic_accuracy']}; CE T-C={p['stochastic_CE']}; individual tapes={p['individual_tapes']}; deterministic T-C={p['deterministic_accuracy_T_minus_C']}.")
    lines+=['','All individual tape, length and route results are retained. No prediction ensembling. Paired10,000 sequence bootstrap resamples preserve all four outcomes together, stratified by parity and OOD length. Conditional uncertainty concerns one initialization and four tapes. Duplicate control equality verifies deterministic coupling, not zero sampling variance.','',
        f"Gradient statistics: {result['gradient_statistics']}. Controller displacement: C={result['controller_change']['C']['l2']}, T={result['controller_change']['T']['l2']}.",
        '', 'Both arms use original full-sequence BPTT through frozen computations, no added detach and no clipping. Each rollout uses an independent initial persistent state. Parameters remain fixed across a pair; two original gradients are averaged before one fresh AdamW step. No auxiliary/imitation loss.','',
        'Mechanical invariants and frozen/source/historical identities verified. Historical bytecode exception retained. Results do not establish general seed reproducibility, adaptive depth, useful diversity or superiority over canonical routing. Any chance-level result is a failure of this repair under the declared budget; ordinary poor performance was not used for stopping.','']
    with (OUT/'report.md').open('x') as h:h.write('\n'.join(lines))
    write_json(OUT/'artifact_index.json',dict(files={str(p.relative_to(OUT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file()},snapshots=snapshots))
    print('EXPERIMENT COMPLETE; helped='+str(helped)+' useful_T='+str(useful),flush=True)

if __name__=='__main__':main()
