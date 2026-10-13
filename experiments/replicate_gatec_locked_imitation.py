"""Exactly two fresh imitation seeds, then one locked paired replay panel."""
from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from data.task_a import TaskAParity
from src.agt import AGTv3
from src.masks import apply_action_mask, build_execution_mask
from experiments.run_stage3_route_imitation import set_seed
from experiments.replay_gatec_learned_policy import (
    ObservedPath, POLICY, POLICY_SHA, BACKBONE, BACKBONE_SHA, LABELS,
    sha, snapshot, unchanged, encode, fingerprint, git_identity, copy,
)
from experiments.evaluate_gatec_locked_policy import sequence_metrics, aggregate, json_hash, write_json, record

SEEDS = (4401, 4402)
UPDATES, BATCH_SIZE, PANEL_SEED = 5000, 1024, 4500
OUT = ROOT / 'experiments/results/gatec_imitation_seed_replication_4401_4402_v1'
RECIPE = ROOT / 'experiments/run_stage3_route_imitation.py'
NEW_CHECKPOINTS = {seed: ROOT / f'experiments/runs/stage3_imitation_replication_{seed}_locked_v1.pt' for seed in SEEDS}


def assignment_name(node):
    return node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None


def collection_statements():
    """Execute the fingerprinted original collection code, not a reimplementation."""
    tree = ast.parse(RECIPE.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    start = next(i for i, node in enumerate(main.body) if assignment_name(node) == 'block_inputs')
    end = next(i for i, node in enumerate(main.body) if assignment_name(node) == 'decision_masks')
    statements = main.body[start:end+1]
    assert not any(isinstance(n, ast.Attribute) and n.attr in ['backward', 'zero_grad']
                   for statement in statements for n in ast.walk(statement))
    return statements


def make_model(weights, device, seed, train_controller=False):
    # Match original seed placement and complete AGTv3 constructor consumption.
    set_seed(seed)
    initial_cpu_rng, initial_cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=2)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    assert set(missing) == {k for k in model.state_dict() if k.startswith('controller.')} and not unexpected
    model.to(device=device, dtype=torch.float32).eval()
    model.executor.max_execution_depth = 8
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.controller.parameters():
        parameter.requires_grad_(train_controller)
    initialization = dict(controller_state_dict={k:copy(v) for k,v in model.controller.state_dict().items()},
        seed=seed, cpu_rng_before_constructor=initial_cpu_rng, cuda_rng_before_constructor=initial_cuda_rng,
        cpu_rng_before_training=torch.get_rng_state().clone(), cuda_rng_before_training=torch.cuda.get_rng_state(device).clone())
    return model, initialization


def bits_from_x(x, task):
    matches = (x.unsqueeze(-2) == task.token_features.view(1,1,2,64)).all(-1)
    assert matches.sum(-1).eq(1).all()
    return matches.long().argmax(-1).cpu().tolist()


@torch.no_grad()
def materialize_teacher(model, device):
    task = TaskAParity(64, device)
    data, original = [], task.make_fixed_set
    def capture(*args, **kwargs):
        batch = original(*args, **kwargs)
        assert kwargs['split'] == 'train' and kwargs['size'] == 8
        data.append(dict(length=batch.length, bits=bits_from_x(batch.x, task), targets=batch.target.cpu().tolist()))
        return batch
    task.make_fixed_set = capture
    before_cpu, before_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
    scope = dict(model=model, dim=64, device=device, task=task, torch=torch, build_execution_mask=build_execution_mask)
    module = ast.Module(body=collection_statements(), type_ignores=[])
    exec(compile(module, str(RECIPE), 'exec'), scope)
    assert torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_cuda,torch.cuda.get_rng_state(device))
    assert [d['length'] for d in data] == list(range(4,17))
    tensors = {key:scope[key] for key in ['H','M','MM','P','B','Y','decision_masks']}
    assert tensors['Y'].numel() == 8320 and tensors['B'].eq(0).all()
    assert all(torch.isfinite(value).all() for value in tensors.values())
    indices = []
    for batch in data:
        for token in range(batch['length']):
            for block in range(8):
                for sample in range(8):
                    indices.append(dict(length=batch['length'], token=token, block=block, sample=sample))
    assert len(indices)==8320 and tensors['Y'].cpu().tolist()==[i['block'] for i in indices]
    assert tensors['P'].argmax(-1).cpu().tolist()==[0 if i['block']==0 else i['block']-1 for i in indices]
    return tensors, data, indices


def make_panel(device):
    """Existing generator/encoding, accepting two samples per parity per length."""
    task = TaskAParity(64, device)
    rows, generation = [], []
    for split, lengths, stream_seed in [('val',range(4,17),PANEL_SEED),('ood_test',range(17,33),PANEL_SEED+1)]:
        task_split = 'val' if split=='val' else 'test'
        task._eval_generators[task_split].manual_seed(stream_seed)
        for length in lengths:
            balance = Counter()
            while sum(balance.values()) < 4:
                batch = task.make_batch(1,length,task_split)
                bits, label = bits_from_x(batch.x,task)[0], int(batch.target[0])
                accepted = balance[label] < 2
                generation.append(dict(split=split,length=length,bits=bits,label=label,accepted=accepted))
                if accepted:
                    rows.append(dict(sequence_id=f'{split}/L{length}/row{sum(balance.values())}',
                        split=split,length=length,bits=bits,label=label))
                    balance[label]+=1
            assert balance==Counter({0:2,1:2})
    assert len(rows)==116
    return rows, generation


def prior_evaluation_sets(paths):
    records, unresolved = [], []
    for path in paths:
        if path.stat().st_size>25_000_000:
            continue
        try:
            value=json.loads(path.read_text())
        except (ValueError,UnicodeError):
            continue
        if not isinstance(value,dict): continue
        sets={}
        for key,batches in value.get('evaluation_data',{}).items():
            if isinstance(batches,list):
                sets['evaluation_data.'+key]=[bits for batch in batches for bits in batch['bits']]
        if isinstance(value.get('sequences'),list) and all(isinstance(r,dict) and 'bits' in r for r in value['sequences']):
            sets['sequences']=[r['bits'] for r in value['sequences']]
        if isinstance(value.get('bits'),list): sets['bits']=value['bits']
        if isinstance(value.get('dataset'),dict) and 'bits' in value['dataset']:
            sets['dataset.bits']=value['dataset']['bits']
        if 'dataset_source' in value and 'datasets' in value:
            source=ROOT/value['dataset_source']['path']
            if source.exists():
                source_data=json.loads(source.read_text())
                for key in value['datasets']:
                    batches=source_data.get('evaluation_data',{}).get(key,[])
                    if batches: sets['resolved_manifest.'+key]=[bits for batch in batches for bits in batch['bits']]
        if sets:
            records.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),sets=sets))
        elif 'heldout' in value or ('dataset' in value and 'bits' not in value['dataset']):
            unresolved.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),reason='evaluation bits not serialized'))
    return records, unresolved


def overlap(panel, training, prior):
    training_bits={tuple(bits) for batch in training for bits in batch['bits']}
    return dict(training={split:dict(count=sum(tuple(r['bits']) in training_bits for r in panel if r['split']==split),
        sequence_ids=[r['sequence_id'] for r in panel if r['split']==split and tuple(r['bits']) in training_bits]) for split in ['val','ood_test']},
        prior_evaluations=[dict(path=p['path'],sha256=p['sha256'],sets={key:dict(
            count=sum(tuple(r['bits']) in {tuple(b) for b in bits} for r in panel),
            panel_sequence_ids=[r['sequence_id'] for r in panel if tuple(r['bits']) in {tuple(b) for b in bits}])
            for key,bits in p['sets'].items()}) for p in prior])


def panel_identity(rows):
    return {split:dict(sequences=len(selected),tokens=sum(r['length'] for r in selected),
        bits_sha256=json_hash([r['bits'] for r in selected]),labels_sha256=json_hash([r['label'] for r in selected]),
        ordered_sequence_label_sha256=json_hash(selected), counts_by_length=dict(Counter(r['length'] for r in selected)),
        label_balance_by_length={str(length):dict(Counter(r['label'] for r in selected if r['length']==length)) for length in sorted({r['length'] for r in selected})},
        duplicate_full_sequences=len(selected)-len({tuple(r['bits']) for r in selected}))
        for split in ['val','ood_test'] for selected in [[r for r in rows if r['split']==split]]}


def train_one(model, optimizer, initialization, tensors, seed, manifest_sha):
    torch.set_rng_state(initialization['cpu_rng_before_training'])
    torch.cuda.set_rng_state(initialization['cuda_rng_before_training'],next(model.parameters()).device)
    backbone_before=snapshot({k:v for k,v in model.state_dict().items() if not k.startswith('controller.')})
    initial_fingerprint=fingerprint(snapshot(model.controller.state_dict()))
    assert initial_fingerprint==fingerprint(snapshot(initialization['controller_state_dict']))
    sampled, sampler_hash = [], hashlib.sha256()
    with (OUT/f'training_{seed}.jsonl').open('x') as trace:
        for update in range(1,UPDATES+1):
            idx=torch.randint(0,tensors['Y'].numel(),(BATCH_SIZE,),device=next(model.parameters()).device)
            cpu_idx=idx.cpu(); sampled.append(cpu_idx)
            sampler_hash.update(bytes(cpu_idx.contiguous().view(torch.uint8).flatten().tolist()))
            logits=model.controller(h=tensors['H'][idx],memory_read=tensors['M'][idx],memory_mean=tensors['MM'][idx],
                pointer_one_hot=tensors['P'][idx],budget=tensors['B'][idx])['logits']
            masked=apply_action_mask(logits,tensors['decision_masks'][tensors['Y'][idx]])
            loss=F.cross_entropy(masked,tensors['Y'][idx])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.controller.parameters())
            optimizer.step()
            record(trace,dict(seed=seed,update=update,imitation_ce=loss.item()))
            if update==1 or update%500==0:
                print(f'TRAIN seed={seed} update={update}/{UPDATES} CE={loss.item():.8g}',flush=True)
    assert unchanged(backbone_before,snapshot({k:v for k,v in model.state_dict().items() if not k.startswith('controller.')}))
    indices_file=OUT/f'minibatch_indices_{seed}.pt'
    with indices_file.open('xb') as handle: torch.save(torch.stack(sampled),handle)
    metadata=dict(seed=seed,updates=UPDATES,selection_rule='final predetermined update, no evaluation-based selection',
        initial_controller_fingerprint=initial_fingerprint, final_controller_fingerprint=fingerprint(snapshot(model.controller.state_dict())),
        optimizer='AdamW',optimizer_defaults=optimizer.defaults,batch_size=BATCH_SIZE,
        minibatch_index_stream_sha256=sampler_hash.hexdigest(),minibatch_indices_file=indices_file.name,
        minibatch_indices_file_sha256=sha(indices_file),minibatch_sampling='global CUDA torch.randint, uniform with replacement',
        cpu_rng_final_fingerprint=fingerprint(snapshot(torch.get_rng_state())),
        cuda_rng_final_fingerprint=fingerprint(snapshot(torch.cuda.get_rng_state(next(model.parameters()).device))),
        backbone_unchanged=True,backbone_sha256=BACKBONE_SHA,dim=64,budget=[0.,0.],manifest_sha256=manifest_sha,
        feature_dataset_sha256=sha(OUT/'common_teacher_features.pt'),training_sequences_sha256=sha(OUT/'training_sequences.json'))
    with NEW_CHECKPOINTS[seed].open('xb') as handle:
        torch.save(dict(controller_state_dict={k:copy(v) for k,v in model.controller.state_dict().items()},**metadata),handle)
    metadata['checkpoint_path']=str(NEW_CHECKPOINTS[seed].relative_to(ROOT))
    metadata['checkpoint_sha256']=sha(NEW_CHECKPOINTS[seed])
    write_json(OUT/f'training_{seed}.json',metadata)
    optimizer.zero_grad(set_to_none=True)
    for p in model.parameters():p.requires_grad_(False)
    return metadata


@torch.no_grad()
def offline(model,tensors):
    logits=model.controller(h=tensors['H'],memory_read=tensors['M'],memory_mean=tensors['MM'],
        pointer_one_hot=tensors['P'],budget=tensors['B'])['logits']
    masked=apply_action_mask(logits,tensors['decision_masks'][tensors['Y']])
    predictions=masked.argmax(-1)
    return dict(accuracy=float(predictions.eq(tensors['Y']).float().mean()),
        ce=float(F.cross_entropy(masked,tensors['Y'])),
        errors_by_position=[int((predictions[tensors['Y']==d]!=d).sum()) for d in range(8)])


@torch.no_grad()
def teacher_view(reference,controller,device):
    rows=[]
    for old in reference.rows:
        row=dict(old)
        features={key:row[source].to(device) for key,source in [
            ('h','h_before'),('memory_read','memory_read'),('memory_mean','memory_mean'),
            ('pointer_one_hot','pointer_one_hot'),('budget','budget')]}
        logits=controller(**features)['logits']
        row['raw_controller_logits']=copy(logits)
        row['normalized_features']=copy(controller._prepare_inputs(**features))
        row['teacher_selected_action']=int(logits[:,:10].masked_fill(~row['legal_successor_mask'].to(device),-torch.inf).argmax(-1))
        row['teacher_action_correct']=row['teacher_selected_action']==row['decision']
        rows.append(row)
    return SimpleNamespace(rows=rows,tokens=reference.tokens,block_purity_checks=reference.block_purity_checks,
                           gdn_purity_checks=reference.gdn_purity_checks)


@torch.no_grad()
def replay(reference_model,policies,panel):
    task=TaskAParity(64,'cuda')
    results={name:{split:[] for split in ['val','ood_test']} for name in policies}
    snapshots={name:snapshot(m.state_dict()) for name,m in dict(reference=reference_model,**policies).items()}
    cpu_rng,cuda_rng=torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
    with (OUT/'replay_decisions.jsonl').open('x') as decisions,(OUT/'replay_tokens.jsonl').open('x') as tokens,(OUT/'first_divergence.jsonl').open('x') as first:
        for index,spec in enumerate(panel):
            bits=torch.tensor([spec['bits']],device='cuda',dtype=torch.long)
            x,targets=task.token_features[bits],bits.cumsum(1).remainder(2)
            a=ObservedPath(reference_model,policies['historical_3305'].controller,False).run(x,targets)
            def trace(path_name,path):
                fields=['token','decision','execution_before','execution_after','pointer_before','pointer_after',
                    'just_restarted_before','just_restarted_after','legal_action_mask','legal_successor_mask',
                    'raw_controller_logits','selected_action','selected_action_label','decoded_block_target',
                    'actual_executed_block','executed_blocks','restart','repeat','budget']
                for row in path.rows:
                    record(decisions,dict(sequence_id=spec['sequence_id'],split=spec['split'],path=path_name,**{k:row[k] for k in fields}))
                for token in path.tokens:
                    record(tokens,dict(sequence_id=spec['sequence_id'],split=spec['split'],path=path_name,**encode(token)))
            trace('reference',a)
            for name,model in policies.items():
                view=teacher_view(a,model.controller,'cuda')
                b=ObservedPath(model,model.controller,True).run(x,targets)
                result=sequence_metrics(spec,view,b)
                results[name][spec['split']].append(result)
                trace(name,b)
                record(first,dict(path=name,sequence_id=spec['sequence_id'],split=spec['split'],**result['divergence']))
                teacher_correct=sum(r['teacher_action_correct'] for r in view.rows)
                print(f'REPLAY {index+1}/116 {spec["sequence_id"]} {name}: teacher={teacher_correct}/{len(view.rows)} canonical={result["canonical_action_tokens"]}/{spec["length"]} CE={result["learned_ce"]:.8g}',flush=True)
                del view,b
            del a
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
    assert all(unchanged(snapshots[name],snapshot(model.state_dict())) for name,model in dict(reference=reference_model,**policies).items())
    return results


def main():
    if OUT.exists() or any(p.exists() for p in NEW_CHECKPOINTS.values()):
        raise RuntimeError('Refusing to overwrite an experiment or checkpoint')
    assert sha(POLICY)==POLICY_SHA and sha(BACKBONE)==BACKBONE_SHA
    old_manifest=json.loads((ROOT/'experiments/runs/stage3_validation_manifest_v1.json').read_text())
    for path in ['experiments/run_stage3_route_imitation.py','src/controller.py','src/masks.py','src/agt.py','src/blocks.py']:
        assert sha(ROOT/path)==old_manifest['source_sha256'][path],f'Recipe reproducibility blocker: {path}'
    generator_historical_sha=hashlib.sha256(subprocess.check_output(['git','show','bd87568:data/task_a.py'])).hexdigest()
    assert sha(ROOT/'data/task_a.py')==generator_historical_sha,'Training generator reproducibility blocker'
    assert sha(POLICY.with_suffix('.json'))==old_manifest['artifacts_sha256']['experiments/runs/stage3_imitation_replication_3305_v1.json']
    historical=json.loads(POLICY.with_suffix('.json').read_text())
    assert historical['seed']==3305 and historical['steps']==UPDATES and historical['batch_size']==BATCH_SIZE
    assert historical['lr']==.001 and historical['weight_decay']==0. and historical['budget']=='zeros'
    historical_record=json.loads((ROOT/'experiments/runs/stage3_replication_3305_3307_100_v1.json').read_text())
    assert str(torch.__version__)==historical_record['torch_version'],'Historical optimizer runtime not recoverable in this environment'
    protected={str(p):sha(p) for p in list((ROOT/'src').rglob('*'))+list((ROOT/'experiments/runs').rglob('*')) if p.is_file()}
    prior_paths=sorted(list((ROOT/'experiments/runs').rglob('*.json'))+list((ROOT/'experiments/results').rglob('*.json')))
    priors,unknown_priors=prior_evaluation_sets(prior_paths)
    identity=git_identity();torch.set_num_threads(4);device=torch.device('cuda')
    stage1=torch.load(BACKBONE,map_location='cpu',weights_only=True)
    weights={k:v for k,v in stage1['state_dict'].items() if not k.startswith('controller.')}
    teacher,_=make_model(weights,device,3305)
    tensors,training,row_indices=materialize_teacher(teacher,device)
    panel,generation=make_panel(device)
    models,initializations,optimizers={},{},{}
    for seed in SEEDS:
        models[seed],initializations[seed]=make_model(weights,device,seed,True)
        optimizers[seed]=torch.optim.AdamW(models[seed].controller.parameters(),lr=1e-3,weight_decay=0.)
        assert all(torch.equal(models[seed].state_dict()[k],teacher.state_dict()[k]) for k in weights)
    assert not unchanged(snapshot(initializations[4401]['controller_state_dict']),snapshot(initializations[4402]['controller_state_dict']))
    OUT.mkdir(parents=True,exist_ok=False)
    write_json(OUT/'training_sequences.json',dict(generator_seed=TaskAParity.TASK_SEED,feature_seed=TaskAParity.FEATURE_SEED,batches=training))
    with (OUT/'common_teacher_features.pt').open('xb') as handle:
        torch.save(dict(tensors={k:copy(v) for k,v in tensors.items()},row_indices=row_indices),handle)
    write_json(OUT/'panel.json',dict(generation_seed=PANEL_SEED,derived_split_seeds=dict(val=PANEL_SEED,ood_test=PANEL_SEED+1),rows=panel,generation_attempts=generation))
    for seed in SEEDS:
        with (OUT/f'initialization_{seed}.pt').open('xb') as handle:torch.save(initializations[seed],handle)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),experiment='exactly two fresh canonical-imitation training seeds, then locked no-update panel replay',
        controller_seeds=list(SEEDS),historical_control=dict(path=str(POLICY.relative_to(ROOT)),sha256=POLICY_SHA),
        backbone=dict(path=str(BACKBONE.relative_to(ROOT)),sha256=BACKBONE_SHA,frozen=True,tensor_count=len(weights)),
        git=identity,script_sha256=sha(__file__),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src').glob('*.py')},
        reconstruction=dict(generating_script=str(RECIPE.relative_to(ROOT)),script_sha256=sha(RECIPE),
            matched_historical_script_hash=True,matched_feature_path_controller_mask_block_hashes=True,
            generator_sha256=sha(ROOT/'data/task_a.py'),historical_generator_git_revision='bd87568',
            historical_generator_matches=True,
            training_data_algorithm='fresh TaskAParity TASK_SEED=3001, eight sequences per ascending length 4..16; exact original collection AST executed',
            feature_rows='token-group -> block -> eight batch samples, original H/M/MM/P/B/Y construction',
            raw_features_reused=True,normalization='inside each trainable real Controller, never cached normalized features',
            collection_leaves_global_rng_unchanged=True,torch_version=str(torch.__version__),historical_runtime_version_matches=True,
            adamw_implementation_sha256=hashlib.sha256(inspect.getsource(torch.optim.AdamW).encode()).hexdigest(),
            linear_initialization_source=inspect.getsource(torch.nn.Linear.reset_parameters),
            historical_training_content_hash_missing=True,training_content_reconstructed_from_fingerprinted_script_and_generator=True),
        training=dict(lengths=[4,16],sequences=104,tokens=1040,route_rows=8320,updates=UPDATES,batch_size=BATCH_SIZE,
            optimizer='AdamW',optimizer_defaults=optimizers[4401].defaults,
            optimizer_default_provenance='original constructor inspected; same recorded PyTorch 2.10.0+cu130 implementation, defaults captured from instantiated optimizer',
            objective='unchanged source-masked eleven-action canonical imitation cross-entropy; no HALT targets',
            initialization='original set_seed before full AGTv3 constructor, fresh six Controller tensors; no historical controller load',
            batch_sampling='torch.randint on CUDA global generator, uniform with replacement, full index history saved per seed',
            mode='model.eval as original; six controller parameters trainable, all backbone parameters frozen',
            budget_fraction=0.,budget_presence_flag=0.,selection_rule='final update 5000 only; no panel selection, early stop or extension',
            checkpoint_paths={str(s):str(p.relative_to(ROOT)) for s,p in NEW_CHECKPOINTS.items()}),
        common_training=dict(sequence_file='training_sequences.json',sequence_file_sha256=sha(OUT/'training_sequences.json'),
            feature_file='common_teacher_features.pt',feature_file_sha256=sha(OUT/'common_teacher_features.pt'),
            tensor_fingerprint=fingerprint(snapshot(tensors)),
            initialization_files={str(s):dict(path=f'initialization_{s}.pt',sha256=sha(OUT/f'initialization_{s}.pt'),
                controller_fingerprint=fingerprint(snapshot(initializations[s]['controller_state_dict']))) for s in SEEDS}),
        panel=dict(file='panel.json',sha256=sha(OUT/'panel.json'),generation_seed=PANEL_SEED,
            generation='existing TaskA make_batch with independently seeded val/test streams, accept exactly two labels of each parity per length; no overlap exclusion or deduplication',
            identity=panel_identity(panel),overlap=overlap(panel,training,priors),unresolved_prior_evaluation_inputs=unknown_priors,
            role='evaluation only after both final checkpoints fixed; not used in training, selection, early stopping or tuning'),
        replay=dict(paths=['fixed reference','historical_3305','fresh_4401','fresh_4402'],batch=1,dtype='float32',device=str(device),
            device_name=torch.cuda.get_device_name(device),budget=[0.,0.],actual_execution_budget=8,deterministic_legal_argmax=True,
            halt_excluded=True,convergence_unused=True,teacher_forcing=False,fallback=False,decision_cap=None,
            independent_sequence_resets=True,completed_token_production_evaluator=True,atol=1e-6,rtol=1e-5),
        strict_success='BOTH fresh final-update checkpoints canonical on EVERY evaluated token and equivalent reference computation',
        limitations=['historical training features were not hashed at creation; deterministic reconstruction is source-verified',
            'historical implicit numerical runtime flags and optimizer backend flags were not logged; same framework/device family and explicit current flags recorded',
            'exact Stage-1 training contents and unretained prior probe overlaps unknown',
            'short-input panel overlaps are disclosed; no blanket unseen-sequence claim',
            'this tests imitation replication only, not task-loss routing, route diversity, topology or adaptive depth'],
        numerical_flags=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cudnn_benchmark=torch.backends.cudnn.benchmark,
            float32_matmul_precision=torch.get_float32_matmul_precision()))
    write_json(OUT/'manifest.json',manifest);manifest_sha=sha(OUT/'manifest.json')
    print(f'MANIFEST LOCKED {manifest_sha}',flush=True)
    training_results={}
    try:
        for seed in SEEDS:
            training_results[str(seed)]=train_one(models[seed],optimizers[seed],initializations[seed],tensors,seed,manifest_sha)
        # No held-out evaluation occurs until both predetermined final checkpoints exist.
        positive,_=make_model(weights,device,3305)
        positive_checkpoint=torch.load(POLICY,map_location='cpu',weights_only=True)
        positive.controller.load_state_dict(positive_checkpoint['controller_state_dict'],strict=True)
        policies={'historical_3305':positive}
        for seed in SEEDS:
            saved=torch.load(NEW_CHECKPOINTS[seed],map_location='cpu',weights_only=True)
            assert unchanged(snapshot(saved['controller_state_dict']),snapshot(models[seed].controller.state_dict()))
            policies[f'fresh_{seed}']=models[seed]
        offline_results={name:offline(model,tensors) for name,model in policies.items()}
        with torch.no_grad():results=replay(teacher,policies,panel)
    except Exception as error:
        write_json(OUT/'failure.json',dict(error=str(error),type=type(error).__name__,traceback=traceback.format_exc(),
            completed_training=training_results,continuation='STOP; no replacement seed, extension, tuning or repair'))
        raise
    summaries={}
    for name,splits in results.items():
        summaries[name]={}
        for split,records in splits.items():
            summary=aggregate(records)
            summary['per_length']={str(length):aggregate([r for r in records if r['length']==length]) for length in sorted({r['length'] for r in records})}
            summaries[name][split]=summary
            write_json(OUT/f'{name}_{split}.json',dict(summary=summary,sequences=records))
    checks=dict(production_and_preexisting_artifacts_unchanged=all(sha(p)==digest for p,digest in protected.items()),
        historical_controller_unchanged=sha(POLICY)==POLICY_SHA,backbone_file_unchanged=sha(BACKBONE)==BACKBONE_SHA,
        tracked_diff_unchanged=git_identity()['tracked_diff_sha256']==identity['tracked_diff_sha256'],
        new_checkpoints_unchanged_during_replay=all(sha(NEW_CHECKPOINTS[int(seed)])==r['checkpoint_sha256'] for seed,r in training_results.items()),
        all_backbones_identical_and_unchanged=all(torch.equal(model.state_dict()[k],weights[k].to(device)) for model in [teacher,*policies.values()] for k in weights),
        parameter_gradients_absent=all(p.grad is None and not p.requires_grad for model in [teacher,*policies.values()] for p in model.parameters()),
        diff_check_clean=__import__('subprocess').run(['git','diff','--check'],capture_output=True).returncode==0,
        candidate_constructions=0,terminal_commit_and_purity='checked by actual production replay observers')
    assert all(v for k,v in checks.items() if k not in ['candidate_constructions','terminal_commit_and_purity'])
    success=all(summaries[f'fresh_{s}'][split]['exact_sequence_canonical_fraction']==1 and
        summaries[f'fresh_{s}'][split]['paired_max_abs_ce_delta']==0 and
        all(v is None for v in summaries[f'fresh_{s}'][split]['first_divergence'].values())
        for s in SEEDS for split in ['val','ood_test'])
    write_json(OUT/'results.json',dict(manifest_sha256=manifest_sha,strict_replication_success=success,
        training=training_results,offline_training=offline_results,evaluation=summaries,invariants=checks,
        interpretation='Canonical imitation replication only; absolute parity metrics are separate from reference equivalence.',
        artifact_hashes={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
    lines=['# Gate-C bounded imitation seed replication','',f'Strict replication success: **{success}**.',
        f'Manifest SHA-256: `{manifest_sha}`. Fresh seeds: 4401, 4402; 5,000 updates each; final checkpoints only.',
        '', '| Policy | Split | Teacher accuracy | Canonical token / sequence recovery | Reference / learned accuracy | Reference / learned CE | D mean/max | RESTART / REPEAT |',
        '|---|---|---|---|---|---|---|---|']
    for name,splits in summaries.items():
        for split,m in splits.items():
            lines.append(f'| {name} | {split} | {m["teacher_accuracy"]} | {m["canonical_action_route_fraction"]} / {m["exact_sequence_canonical_fraction"]} | {m["reference_accuracy"]} / {m["learned_accuracy"]} | {m["reference_mean_ce"]:.12g} / {m["learned_mean_ce"]:.12g} | {m["mean_D"]} / {m["maximum_D"]} | {m["restart_count"]} / {m["repeat_count"]} |')
    lines.extend(['','Balanced panel: four samples per length, two per parity, at every length 4–32. Panel fixed before training; no evaluation-based checkpoint selection.',
        '', 'Training and prior-evaluation overlaps are in the manifest with exact panel IDs. Overlapping short inputs are not unseen-sequence evaluation. Stage-1 exposure and unretained historical probe membership remain unknown.',
        '', 'Both fresh controllers were independently initialized by the original full-model initialization procedure. Raw teacher features were reused from the fingerprinted original collection code. Full minibatch indices, RNG states, initialization weights, training losses, final weights and hashes are retained.',
        '', 'Replay uses the unchanged completed-token production executor. Selected actions, actual bank calls, masks, P/D accounting, in-flight committed-state/GDN purity, seven-entry version increments and exactly one terminal commit per completed token were checked. No candidate construction during deterministic inference.',
        '', 'While complete action prefixes agree, the established state/feature comparison standard and exact persistent/output checks apply. Post-route-divergence states are not compared as equivalent decision-index computations.',
        '', 'Absolute parity and reference equivalence are distinct. Reproducing a reference error would preserve imitation equivalence but limit a perfect-parity claim.',
        '', 'This bounded result does not establish task-loss discovery, useful diversity, state-conditioned topology, adaptive depth, learned HALT or superiority. No production change, backbone update, checkpoint overwrite, additional seed, tuning, or commit.', ''])
    (OUT/'report.md').write_text('\n'.join(lines))
    print(f'EXPERIMENT COMPLETE strict_replication_success={success} {OUT}',flush=True)


if __name__=='__main__':main()
