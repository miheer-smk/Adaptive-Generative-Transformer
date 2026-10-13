"""One preregistered fresh-controller task-loss experiment; no core changes."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import gzip
import inspect
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from data.task_a import TaskAParity
from experiments.run_gatec_paired_task_maintenance import (
    IndexedNoise, StochasticPath, add_measurements, summary, trace_path,
    parameter_change, paired_divergence, versions, gpu_snapshot, assert_same,
)
from experiments.replicate_gatec_locked_imitation import make_model, panel_identity, bits_from_x
from experiments.replay_gatec_learned_policy import (
    ObservedPath, BACKBONE, BACKBONE_SHA, sha, copy, snapshot, unchanged,
    fingerprint, git_identity, encode,
)
from experiments.evaluate_gatec_locked_policy import write_json, record
from experiments.stage3_measurements import parameter_gradients

PARENT = ROOT / 'experiments/results/gatec_imitation_seed_replication_4401_4402_v1'
MAINTENANCE = ROOT / 'experiments/results/gatec_paired_task_maintenance_4401_v1'
OUT = ROOT / 'experiments/results/gatec_fresh_task_discovery_4700_v1'
SNAPSHOTS = {i: ROOT / f'experiments/runs/stage3_fresh_task_4700_update{i:04d}_locked_v1.pt'
             for i in (0, 104, 520, 1040)}
TAU = 2.0


def recorded_bits(value, location='$'):
    """Read named binary-sequence arrays, including nested manifest records."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('bits', 'sequence_bits', 'input_bits') and isinstance(item, list):
                def collect(array, path):
                    if isinstance(array, list) and 4 <= len(array) <= 32 and all(type(b) is int and b in (0, 1) for b in array):
                        found.append((path, tuple(array)))
                    elif isinstance(array, list):
                        for i, sub in enumerate(array):
                            collect(sub, f'{path}/{i}')
                collect(item, f'{location}/{key}')
            found.extend(recorded_bits(item, f'{location}/{key}'))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            if isinstance(item, (dict, list)):
                found.extend(recorded_bits(item, f'{location}/{i}'))
    return found


def prior_inputs():
    paths = sorted(set((ROOT/'experiments/results').rglob('*.json')) |
                   set((ROOT/'experiments/runs').glob('*.json')))
    # Sequence summaries retain input identities; decision/tensor traces do not.
    paths += sorted(p for p in (ROOT/'experiments/results').rglob('*.jsonl')
                    if p.name.startswith('sequences'))
    excluded, records, unresolved = set(), [], []
    for path in paths:
        if OUT in path.parents:
            continue
        found = []
        try:
            if path.suffix == '.jsonl':
                with path.open() as handle:
                    for i, line in enumerate(handle):
                        if line.strip():
                            found.extend(recorded_bits(json.loads(line), f'line/{i+1}'))
            else:
                value = json.loads(path.read_text())
                found = recorded_bits(value)
                if isinstance(value, dict) and not found and any(k in value for k in ('heldout', 'evaluation', 'datasets', 'dataset')):
                    unresolved.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                                           reason='No recoverable named raw binary sequences in this record; any referenced available project records are scanned separately.'))
        except (ValueError, UnicodeError) as error:
            unresolved.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), reason=str(error)))
            continue
        unique = sorted(set(bits for _, bits in found))
        excluded.update(unique)
        records.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                            recovered_occurrences=len(found), distinct_sequences=len(unique),
                            extracted_sequences=[list(bits) for bits in unique],
                            named_array_locations=sorted(set(loc for loc, _ in found))))
    return excluded, records, unresolved


def fresh_panel(excluded, seed=4703):
    task = TaskAParity(64, 'cpu')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    task._eval_generators['val'] = generator
    task._eval_generators['test'] = generator
    rows, attempts, used = [], [], set()
    for split, lengths, quota in [('val', [16], 16), ('ood_test', range(17, 33), 2)]:
        for length in lengths:
            balance = Counter()
            while sum(balance.values()) < 2*quota:
                batch = task.make_batch(1, length, 'val' if split == 'val' else 'test')
                bits, label = bits_from_x(batch.x, task)[0], int(batch.target[0])
                key = tuple(bits)
                reason = ('prior_or_training_exact_match' if key in excluded else
                          'duplicate_in_final_panel' if key in used else
                          'parity_quota_full' if balance[label] >= quota else 'accepted')
                attempts.append(dict(split=split, length=length, bits=bits, label=label, reason=reason))
                if reason == 'accepted':
                    rows.append(dict(sequence_id=f'{split}/L{length}/row{sum(balance.values())}',
                                     split=split, length=length, bits=bits, label=label))
                    used.add(key)
                    balance[label] += 1
            assert balance == Counter({0: quota, 1: quota})
    assert len(rows) == len(used) == 96 and not (used & excluded)
    return rows, attempts


def training_visits(rows, seed=4701):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    visits, permutations = [], []
    for epoch in range(1, 11):
        permutation = torch.randperm(len(rows), generator=generator).tolist()
        assert sorted(permutation) == list(range(104))
        permutations.append(permutation)
        for position, index in enumerate(permutation):
            row = rows[index]
            visits.append(dict(row, sequence_id=f'epoch{epoch:02d}/{row["sequence_id"]}',
                               source_sequence_id=row['sequence_id'], source_index=index,
                               epoch=epoch, epoch_position=position, update=len(visits)+1))
    assert len(visits) == 1040
    return visits, permutations


@contextmanager
def anchor_guard(model):
    original = model.executor.execute_differentiable
    signature = inspect.signature(original)
    def guarded(*args, **kwargs):
        anchor = signature.bind(*args, **kwargs).arguments['h0']
        before = gpu_snapshot(anchor)
        version = versions(anchor)
        result = original(*args, **kwargs)
        assert versions(anchor) == version
        assert_same(before, anchor)
        return result
    model.executor.execute_differentiable = guarded
    try:
        yield
    finally:
        model.executor.execute_differentiable = original


class VerifiedStochasticPath(StochasticPath):
    @contextmanager
    def observers(self):
        with super().observers():
            original = self.model.executor.differentiable_route_step
            def reconstructed(*args, **kwargs):
                result = original(*args, **kwargs)
                row = self.current[-1]
                # Diagnostic reconstruction only; neither returned tensors nor
                # the optimization graph are replaced or detached here.
                with torch.no_grad():
                    logits = kwargs['action_logits'][:, :10]
                    mask = row['legal_successor_mask'].to(logits.device).unsqueeze(0)
                    uniform = self.tape.get(self.sequence, self.token, row['decision']).to(logits)
                    eps = torch.finfo(logits.dtype).eps
                    uniform = uniform.clamp(eps, 1-eps)
                    gumbel = -torch.log(-torch.log(uniform))
                    relaxed = F.softmax((F.log_softmax(logits.masked_fill(~mask, -torch.inf), -1)+gumbel)/TAU, -1)
                    assert torch.equal(relaxed.argmax(-1), result['selected_action']), 'Sealed-noise action reconstruction failed'
                row['noise_reconstruction_verified'] = True
                return result
            self.model.executor.differentiable_route_step = reconstructed
            try:
                yield
            finally:
                self.model.executor.differentiable_route_step = original


def gradient_coverage(path):
    cells = defaultdict(list)
    for row in path.rows:
        gradient = row.pop('_logits').grad
        assert gradient is not None and torch.isfinite(gradient).all()
        gradient = copy(gradient)[0]
        assert gradient[10] == 0
        legal = row['legal_successor_mask']
        assert gradient[:10][~legal].eq(0).all()
        row['retained_logit_gradient'] = gradient.unsqueeze(0)
        cells[(row['token'], row['decision'])].append(float(gradient.norm()))
    return dict(decisions=len(path.rows), nonzero_decisions=sum(float(row['retained_logit_gradient'].norm()) > 0 for row in path.rows),
                legal_nonzero_decisions=sum(float(row['retained_logit_gradient'][0,:10][row['legal_successor_mask']].norm()) > 0 for row in path.rows),
                HALT_exact_zero=True, illegal_exact_zero=True,
                mean_logit_gradient_l2=sum(float(row['retained_logit_gradient'].norm()) for row in path.rows)/len(path.rows),
                token_decision_l2={f'{t}/{d}':values[0] for (t,d),values in cells.items()})


def expanded_summary(rows):
    result = summary(rows)
    decisions = sum(sum(row['D']) for row in rows)
    for field in ['action_entropy', 'selected_logit_margin']:
        result['mean_'+field] = sum(row[field+'_sum'] for row in rows)/decisions
    return result


def measurements(spec, reference, path, controller):
    result = add_measurements(spec, reference, path, controller)
    result['action_entropy_sum'] = sum(row['entropy'] for row in path.rows)
    result['selected_logit_margin_sum'] = sum(row['selected_raw_logit_margin'] for row in path.rows)
    result['noise_reconstructions'] = sum(row.get('noise_reconstruction_verified', False) for row in path.rows)
    return result


def main():
    if OUT.exists() or any(p.exists() for p in SNAPSHOTS.values()):
        raise RuntimeError('Refusing to repeat the experiment or overwrite any artifact')
    torch.set_num_threads(4)
    assert sha(BACKBONE) == BACKBONE_SHA
    maintenance_manifest = MAINTENANCE/'manifest_reporting_correction.json'
    old = json.loads(maintenance_manifest.read_text())
    assert all(sha(ROOT/p) == digest for p,digest in old['source_sha256'].items())
    assert old['training']['gradient_policy'].startswith('unchanged run_diff_sequence full-sequence BPTT')
    assert sha(PARENT/'training_sequences.json') == old['inputs']['training_sha256']
    data = json.loads((PARENT/'training_sequences.json').read_text())
    training = []
    for batch in data['batches']:
        for i,(bits,label) in enumerate(zip(batch['bits'],batch['targets'])):
            assert len(bits) == batch['length'] and sum(bits)%2 == label
            training.append(dict(sequence_id=f'train/L{len(bits)}/row{i}',split='train',length=len(bits),bits=bits,label=label))
    assert len(training) == 104 and Counter(row['length'] for row in training) == Counter({i:8 for i in range(4,17)})
    excluded, prior_records, unresolved = prior_inputs()
    excluded.update(tuple(row['bits']) for row in training)
    final_panel, generation = fresh_panel(excluded)
    development = json.loads((PARENT/'panel.json').read_text())['rows']
    visits, permutations = training_visits(training)
    tape_seeds = torch.randint(0, 2**31-1, (4,), generator=torch.Generator().manual_seed(4704)).tolist()
    tapes = {'training': IndexedNoise(visits,4702),
             **{f'final_tape_{i+1}':IndexedNoise(final_panel,seed) for i,seed in enumerate(tape_seeds)}}
    device = torch.device('cuda')
    stage1 = torch.load(BACKBONE,map_location='cpu',weights_only=True)
    weights = {k:v for k,v in stage1['state_dict'].items() if not k.startswith('controller.')}
    u, initialization = make_model(weights,device,4700,False)
    initial_controller = {k:copy(v) for k,v in u.controller.state_dict().items()}
    t, _ = make_model(weights,device,4700,True)
    t.controller.load_state_dict(initial_controller,strict=True)
    reference, _ = make_model(weights,device,4700,False)
    models = {'U':u,'T':t}
    assert u.controller.input_dim == 395 and u.controller.num_actions == 11
    assert torch.equal(initial_controller['norm_h.weight'],torch.ones(64))
    assert torch.equal(initial_controller['norm_memory_read.weight'],torch.ones(64))
    assert unchanged(snapshot(u.state_dict()),snapshot(t.state_dict()))
    initial_u = snapshot(u.state_dict())
    identity = git_identity()
    protected = {str(p):sha(p) for directory in [ROOT/'src',ROOT/'experiments/runs',PARENT,MAINTENANCE]
                 for p in directory.rglob('*') if p.is_file()}
    optimizer = torch.optim.AdamW(t.controller.parameters(),lr=.0001,weight_decay=0.,betas=(.9,.999),eps=1e-8)
    assert not optimizer.state
    OUT.mkdir(parents=True,exist_ok=False)
    write_json(OUT/'training_sequences.json',dict(rows=training))
    write_json(OUT/'training_schedule.json',dict(seed=4701,epochs=10,permutations=permutations,visits=visits))
    write_json(OUT/'final_panel.json',dict(seed=4703,rows=final_panel,generation=generation))
    write_json(OUT/'prior_input_inventory.json',dict(records=prior_records,unresolved=unresolved,
               conservative_exclusion='all recoverable named binary inputs, including historical training arrays, from bounded project records; no final-panel outcomes consulted'))
    with (OUT/'initialization.pt').open('xb') as handle:
        torch.save(initialization,handle)
    for key,tape in tapes.items():
        with (OUT/f'noise_{key}.pt').open('xb') as handle:
            torch.save(dict(seed=tape.seed,uniforms=tape.draws),handle)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol='single-fresh-controller-task-discovery-4700-v1',
        git=identity,script_sha256=sha(__file__),source_sha256=old['source_sha256'],
        observer_sha256=sha(ROOT/'experiments/run_gatec_paired_task_maintenance.py'),
        backbone=dict(path=str(BACKBONE.relative_to(ROOT)),sha256=BACKBONE_SHA,noncontroller_tensors=len(weights)),
        initialization=dict(seed=4700,procedure='original set_seed immediately before full AGTv3 construction, then load only Stage-1 non-controller weights; clone fresh controller to U/T',
            controller='395->256->11',normalization='fresh unit RMSNorm gains',no_checkpoint_controller_loaded=True,no_bias_override=True,
            state_fingerprint=fingerprint(snapshot(initial_controller)),file_sha256=sha(OUT/'initialization.pt')),
        training=dict(epochs=10,updates=1040,sequences_per_epoch=104,lengths=[4,16],batch=1,dtype='float32',
            loss='final-sequence parity CE only',optimizer='fresh AdamW',optimizer_defaults=optimizer.defaults,
            tau=2.,budget=[0.,0.],budget_fraction=0.,budget_presence_flag=0.,
            trainable_parameters=[name for name,p in t.named_parameters() if p.requires_grad],U_updates=0,
            scheduler=None,clipping=None,auxiliary_loss=None,teacher_features_used_for_training=False,
            recomputed_trajectories=True,independent_sequence_reset=True),
        gradient_policy=dict(maintenance_manifest_sha256=sha(maintenance_manifest),policy=old['training']['gradient_policy'],
            horizon='entire current sequence: 4..16 tokens in training; no inter-sequence graph',
            production_detach_locations=['src/routing.py:137 hard - relaxed.detach() + relaxed; ST relaxed stop-gradient',
                                        'src/executor.py:323 action.detach().argmax; hard pointer/just_restarted/P/mask metadata'],
            unused_detach='src/convergence.py detach helper is outside execute_differentiable',
            clone_semantics='state.clone, cache.clone and memory write clones preserve autograd',
            attention='GQA retains complete sequence; SWA window 256 retains 255 committed rows; no within-sequence eviction at these lengths',
            convolution='GDN retains conv_kernel-1=3 inputs; existing finite convolution receptive field, no added gradient detach',
            frozen_computations='T frozen backbone executes with autograd enabled',
            diagnostic_detaches='copy/snapshot/probability/report tensors only; returned execution/final CE graph is not replaced',
            anchor='original Prelude h_t,0 is passed unchanged to all successor blocks; observed for in-flight mutation'),
        execution=dict(P=8,D='8+RESTART in [8,15]',halting_disabled=True,convergence_disabled=True,decision_cap=None,
            stochastic='unchanged hard-forward ST-Gumbel with all ten candidates, nine speculative bank calls per decision',
            deterministic='selected-only legal argmax',fallback=None,route_supervision=None),
        inputs=dict(training_source=str((PARENT/'training_sequences.json').relative_to(ROOT)),training_source_sha256=sha(PARENT/'training_sequences.json'),
            training_materialized_sha256=sha(OUT/'training_sequences.json'),schedule_sha256=sha(OUT/'training_schedule.json'),
            development_source=str((PARENT/'panel.json').relative_to(ROOT)),development_sha256=sha(PARENT/'panel.json'),
            final_panel_sha256=sha(OUT/'final_panel.json'),final_panel_identity=panel_identity(final_panel),
            inventory_sha256=sha(OUT/'prior_input_inventory.json'),training_overlap=0,available_prior_overlap=0,
            distinct_final_sequences=96,final_generation_seed=4703,rejection_counts=dict(Counter(r['reason'] for r in generation)),
            rejection_rule='reject exact training/prior recorded input match, panel duplicate, or filled parity quota; accept in generator order',
            final_panel_usage='sealed before outcomes; only evaluated after update1040; no checkpoint selection'),
        randomness=dict(order_seed=4701,training_seed=4702,final_panel_seed=4703,evaluation_root_seed=4704,
            evaluation_tape_seeds=tape_seeds,derivation='four integer seeds from CPU torch.Generator(4704), in recorded order',
            training_coordinate='epoch/sequence ID/token/decision/ten successor action slots; new sealed uniforms on every visit',
            pairing='same visit/tape coordinates in U and T regardless of decision counts',
            fifteen_positions='legal P=8 maximum D coverage; not a stopping cap',
            injection='reuse diagnostic-only IndexedNoise hook around original torch.rand_like; original ST math unchanged',
            reconstruction='original masked log-softmax + clamped uniforms + Gumbel/tau softmax reconstructs every sampled hard action',
            tapes={key:dict(seed=tape.seed,file=f'noise_{key}.pt',sha256=sha(OUT/f'noise_{key}.pt')) for key,tape in tapes.items()}),
        snapshots=dict(updates=[0,104,520,1040],paths={str(i):str(p.relative_to(ROOT)) for i,p in SNAPSHOTS.items()},
            development='deterministic U/T diagnostics on existing exposed validation/OOD panel at each snapshot; no updates/selection',
            selection='update1040 irrespective of all metrics'),
        final_evaluation=dict(paths=['F canonical reference','U unchanged fresh policy','T final policy'],
            modes=['deterministic','four paired fixed ST-Gumbel tapes for U/T'],batch=1,dtype='float32',
            aggregate='equal-weight pooling over all four tapes, with each tape also reported'),
        strong_criterion='valid execution/finite optimization; T deterministic accuracy=1 on both fresh splits, higher accuracy and lower CE than U in both deterministic splits and both aggregate sampled splits',
        stopping='mechanical violation or nonfinite only; ordinary performance failure continues through exactly1040 updates',
        limitations=['single controller-training initialization; four tapes replicate sampling, not controller seeds',
            'fresh final inputs exclude exact available recorded training/prior input matches; unknown Stage-1 training inputs and unretained historical probes cannot be excluded',
            'prior-input exclusion is conservative and also excludes recoverable historical training arrays',
            'development panel was previously exposed and is used only at fixed snapshots; final panel is not used for selection',
            'fixed frozen backbone specialized for canonical routing; no architecture or backbone adaptation'],
        numerical=dict(torch_version=str(torch.__version__),device=str(device),device_name=torch.cuda.get_device_name(),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    write_json(OUT/'manifest.json',manifest)
    manifest_sha = sha(OUT/'manifest.json')
    print(f'MANIFEST LOCKED {manifest_sha}',flush=True)
    canonical = {};task = TaskAParity(64,device);updates = [];snapshot_records = [];development_results = {}
    def inputs(spec):
        bits = torch.tensor([spec['bits']],device=device,dtype=torch.long)
        return task.token_features[bits],bits.cumsum(1).remainder(2)
    def ref_for(spec):
        key = tuple(spec['bits'])
        if key not in canonical:
            x,targets = inputs(spec)
            with torch.no_grad():
                canonical[key] = ObservedPath(reference,u.controller,False).run(x,targets)
        return canonical[key]
    def run(spec,arm,tape=None,gradient=False):
        x,targets = inputs(spec);model = models[arm]
        with anchor_guard(model):
            return VerifiedStochasticPath(model,tape,spec['sequence_id'],gradient).run(x,targets) if tape is not None else ObservedPath(model,model.controller,True).run(x,targets)
    def check_frozen():
        assert unchanged(initial_u,snapshot(u.state_dict()))
        assert all(torch.equal(model.state_dict()[key].detach().cpu(),weights[key]) for model in [reference,u,t] for key in weights)
        assert all(p.grad is None for name,p in t.named_parameters() if not name.startswith('controller.'))
    def save_snapshot(update,handle):
        check_frozen()
        path = SNAPSHOTS[update]
        payload = dict(controller_state_dict={k:copy(v) for k,v in t.controller.state_dict().items()},
            controller_seed=4700,updates=update,backbone_sha256=BACKBONE_SHA,manifest_sha256=manifest_sha,
            selection='predetermined snapshot; final selection is1040',optimizer_defaults=optimizer.defaults,
            tau=2.,budget=[0.,0.],gradient_policy=manifest['gradient_policy'])
        with path.open('xb') as checkpoint_handle:torch.save(payload,checkpoint_handle)
        info = dict(update=update,path=str(path.relative_to(ROOT)),sha256=sha(path),change=parameter_change(initial_controller,t.controller))
        snapshot_records.append(info);record(handle,info)
        print(f'SNAPSHOT FIXED update={update} SHA={info["sha256"]}',flush=True)
    def measure(spec,arm,phase,tape=None):
        with torch.no_grad():
            path = run(spec,arm,tape)
            measured = measurements(spec,ref_for(spec),path,models[arm].controller)
        trace_path(handles,phase,arm,spec,path)
        record(handles['sequences'],dict(phase=phase,arm=arm,**measured))
        return measured
    def dev(update):
        phase=f'development_update{update}'
        development_results[str(update)] = {}
        for arm in ['U','T']:
            development_results[str(update)][arm] = [measure(s,arm,phase) for s in development]
        print(f'DEVELOPMENT COMPLETE update={update}',flush=True)
    final_results = {};reference_results = []
    with ExitStack() as stack:
        handles = {key:stack.enter_context(gzip.open(OUT/f'{key}.jsonl.gz','xt',compresslevel=1))
                   for key in ['decisions','tokens','sequences','updates','snapshots','paired_divergence']}
        try:
            save_snapshot(0,handles['snapshots']);dev(0)
            for spec in visits:
                u.train();t.train()
                ref = ref_for(spec)
                with torch.no_grad():
                    upath = run(spec,'U',tapes['training'])
                    um = measurements(spec,ref,upath,u.controller)
                optimizer.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    tpath = run(spec,'T',tapes['training'],True)
                    loss = F.cross_entropy(tpath.final_logits,torch.tensor([spec['label']],device=device))
                    assert torch.isfinite(loss) and loss.requires_grad
                    loss.backward()
                gradients = parameter_gradients(t)
                assert all(g['present'] and g['finite'] for g in gradients.values())
                assert all(p.grad is None for name,p in t.named_parameters() if not name.startswith('controller.'))
                coverage = gradient_coverage(tpath)
                with torch.no_grad():tm = measurements(spec,ref,tpath,t.controller)
                before = {k:copy(v) for k,v in t.controller.state_dict().items()}
                optimizer.step()
                assert all(torch.isfinite(p).all() for p in t.controller.parameters())
                row = dict(update=spec['update'],epoch=spec['epoch'],sequence_id=spec['sequence_id'],
                    source_sequence_id=spec['source_sequence_id'],U=um,T=tm,gradients=gradients,logit_gradient_coverage=coverage,
                    gradient_l2=sum(g['l2']**2 for g in gradients.values())**.5,
                    controller_update=parameter_change(before,t.controller),paired_divergence=paired_divergence(upath,tpath))
                updates.append(row);record(handles['updates'],row)
                for arm,path,metrics in [('U',upath,um),('T',tpath,tm)]:
                    trace_path(handles,'training',arm,spec,path)
                    record(handles['sequences'],dict(phase='training',arm=arm,**metrics))
                record(handles['paired_divergence'],dict(phase='training',update=spec['update'],**row['paired_divergence']))
                if spec['update']%10 == 0 or spec['update']==1:
                    print(f'UPDATE {spec["update"]}/1040 epoch={spec["epoch"]} CE U/T={um["learned_ce"]:.8g}/{tm["learned_ce"]:.8g} grad={row["gradient_l2"]:.7g}',flush=True)
                del upath,tpath,loss,ref
                if spec['update'] in SNAPSHOTS:
                    save_snapshot(spec['update'],handles['snapshots']);dev(spec['update'])
            assert len(updates)==1040
            optimizer.zero_grad(set_to_none=True)
            check_frozen()
            for model in models.values():
                model.eval()
                for parameter in model.parameters():parameter.requires_grad_(False)
            final_hash = sha(SNAPSHOTS[1040])
            for spec in final_panel:
                ref = ref_for(spec)
                target = spec['label'];tok = ref.tokens[-1]
                reference_results.append(dict(**spec,prediction=tok['prediction'],ce=tok['prefix_ce'],correct=tok['prediction']==target,
                    output_logits=copy(tok['output_logits']),commits=len(ref.tokens),P=[x['P'] for x in ref.tokens],D=[x['D'] for x in ref.tokens]))
                trace_path(handles,'final_reference','F',spec,ref)
            write_json(OUT/'final_reference.json',dict(sequences=reference_results))
            modes = [('deterministic',None)]+[(key,tape) for key,tape in tapes.items() if key!='training']
            for mode,tape in modes:
                final_results[mode] = {arm:[] for arm in ['U','T']}
                for i,spec in enumerate(final_panel):
                    for arm in ['U','T']:
                        final_results[mode][arm].append(measure(spec,arm,'final_'+mode,tape))
                    if (i+1)%8==0:
                        print(f'FINAL {mode} {i+1}/96 COMPLETE',flush=True)
                write_json(OUT/f'final_{mode}.json',final_results[mode])
            check_frozen()
            assert sha(SNAPSHOTS[1040])==final_hash
            assert all(sha(path)==digest for path,digest in protected.items())
            assert all(sha(OUT/v['file'])==v['sha256'] for v in manifest['randomness']['tapes'].values())
            assert git_identity()['tracked_diff_sha256']==identity['tracked_diff_sha256']
            assert subprocess.run(['git','diff','--check'],capture_output=True).returncode==0
        except Exception as error:
            write_json(OUT/'failure.json',dict(error=str(error),type=type(error).__name__,traceback=traceback.format_exc(),
                completed_updates=len(updates),last_sequence=locals().get('spec'),stop='No retry, tuning, replacement or extension'))
            raise
    split_summaries = {mode:{arm:{split:expanded_summary([s for s in rows if s['split']==split])
                       for split in ['val','ood_test']} for arm,rows in arms.items()} for mode,arms in final_results.items()}
    sampled_aggregate = {arm:{split:expanded_summary([s for mode,arms in final_results.items() if mode!='deterministic'
                        for s in arms[arm] if s['split']==split]) for split in ['val','ood_test']} for arm in ['U','T']}
    epoch_summaries = {str(epoch):{arm:expanded_summary([row[arm] for row in updates if row['epoch']==epoch])
                       for arm in ['U','T']} for epoch in range(1,11)}
    development_summaries = {update:{arm:{split:expanded_summary([s for s in rows if s['split']==split])
                           for split in ['val','ood_test']} for arm,rows in arms.items()} for update,arms in development_results.items()}
    strong = all(split_summaries['deterministic']['T'][split]['learned_accuracy']==1. and
                 split_summaries['deterministic']['T'][split]['learned_accuracy']>split_summaries['deterministic']['U'][split]['learned_accuracy'] and
                 split_summaries['deterministic']['T'][split]['learned_mean_ce']<split_summaries['deterministic']['U'][split]['learned_mean_ce'] and
                 sampled_aggregate['T'][split]['learned_accuracy']>sampled_aggregate['U'][split]['learned_accuracy'] and
                 sampled_aggregate['T'][split]['learned_mean_ce']<sampled_aggregate['U'][split]['learned_mean_ce'] for split in ['val','ood_test'])
    report = dict(manifest_sha256=manifest_sha,snapshots=snapshot_records,final=split_summaries,
        sampled_aggregate=sampled_aggregate,training_by_epoch=epoch_summaries,development_by_snapshot=development_summaries,
        controller_change=parameter_change(initial_controller,t.controller),
        gradient_statistics=dict(minimum=min(row['gradient_l2'] for row in updates),maximum=max(row['gradient_l2'] for row in updates),
            mean=sum(row['gradient_l2'] for row in updates)/1040,finite_updates=1040,
            nonzero_parameter_updates=sum(row['controller_update']['changed_values']>0 for row in updates),
            logit_decisions=sum(row['logit_gradient_coverage']['decisions'] for row in updates),
            logit_nonzero_decisions=sum(row['logit_gradient_coverage']['nonzero_decisions'] for row in updates)),
        strong_reference_level_discovery=strong,
        invariants=dict(mechanical_validity=True,optimization_path_validity=True,noise_reconstructed_all_actions=True,
            U_unchanged=True,backbone_and_production_historical_files_unchanged=True,exact_updates=1040,
            no_extra_detach=True,anchor_preserved=True,purity_and_one_commit_verified=True,git_diff_check_clean=True),
        limitations=manifest['limitations'])
    write_json(OUT/'results.json',report)
    lines=['# Fresh-controller task-loss routing: locked seed4700 experiment','',
           f'Strong reference-level discovery criterion: **{strong}**. Exactly1040 controller-only CE updates; U unchanged.',
           f'Manifest SHA-256 `{manifest_sha}`. Final checkpoint is update1040 irrespective of outcomes.','',
           '| Evaluation | Split | U accuracy / CE | T accuracy / CE | Canonical tokens U / T |','|---|---|---|---|---|']
    for mode,arms in split_summaries.items():
        for split in ['val','ood_test']:
            a,b=arms['U'][split],arms['T'][split]
            lines.append(f'| {mode} | {split} | {a["learned_accuracy"]:.6g} / {a["learned_mean_ce"]:.12g} | {b["learned_accuracy"]:.6g} / {b["learned_mean_ce"]:.12g} | {a["canonical_action_route_fraction"]:.6g} / {b["canonical_action_route_fraction"]:.6g} |')
    for split in ['val','ood_test']:
        a,b=sampled_aggregate['U'][split],sampled_aggregate['T'][split]
        lines.append(f'| all four tapes, equal weight | {split} | {a["learned_accuracy"]:.6g} / {a["learned_mean_ce"]:.12g} | {b["learned_accuracy"]:.6g} / {b["learned_mean_ce"]:.12g} | {a["canonical_action_route_fraction"]:.6g} / {b["canonical_action_route_fraction"]:.6g} |')
    lines.extend(['',f'Gradient and parameter measurements: {report["gradient_statistics"]}; total controller L2 displacement {report["controller_change"]["l2"]}.',
        '', 'Training-set adaptation is reported epoch by epoch against paired U noise. Fresh length16 generalization and longer-length transfer are separate results. The four noise tapes replicate stochastic sampling only. Ordinary performance failures did not alter the update count or final checkpoint.',
        '', 'Full sequence BPTT, zero budget/presence, eight actual executions, hard metadata transitions and the original ST estimator were preserved. All non-controller parameters remained frozen; no gradients were detached from the optimization path. Prelude anchors, selected successors, masks, counter semantics, cache purity/versions and one commit per token were checked.',
        '', 'The final panel excluded exact matches to reconstructed training and recoverable prior records before outcomes. Unknown Stage1 training membership and unretained historical probe membership remain limitations. Canonical trajectories were diagnostics only, never task-training targets, repaired actions or substituted states.',
        '', 'A noncanonical route is useful only to the extent measured task performance supports it. Tiny CE changes alone do not establish useful discovery. No population or multi-controller-seed claim, adaptive depth, HALT, architectural change or superiority claim follows from this experiment.',
        '', 'All raw per-sequence training losses, actual variable-length routes, gradients and parameter changes are in compressed JSONL. Per-tape and aggregate results include all outcomes, first divergences and candidate counts. Production source, historical checkpoints and backbone weights were unchanged. No commit or subsequent experiment.', ''])
    with (OUT/'report.md').open('x') as handle:handle.write('\n'.join(lines))
    print(f'EXPERIMENT COMPLETE strong_discovery={strong} {OUT}',flush=True)


if __name__=='__main__':
    main()
