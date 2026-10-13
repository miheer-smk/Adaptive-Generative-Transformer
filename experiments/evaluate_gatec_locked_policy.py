"""One locked evaluation of persisted validation/OOD bits; no optimization."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import random
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from data.task_a import TaskAParity
from src.agt import AGTv3
from experiments.replay_gatec_learned_policy import (
    ObservedPath, POLICY, POLICY_SHA, BACKBONE, BACKBONE_SHA, LABELS,
    sha, snapshot, unchanged, first_difference, encode, git_identity, fingerprint,
)

DATA = ROOT / 'experiments/runs/stage3_replication_3305_3307_100_v1.json'
DATA_SHA = 'd32e988558f7cfa3bc8e0a34e11c29e8acfb6aea25dc5f4839839719dcac3beb'
CANONICAL = [('ADVANCE', i) for i in range(8)]
CONTROL = [0, 1, 1, 0, 1, 1, 0, 0]


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    with path.open('x') as handle:
        json.dump(encode(value), handle, indent=2, allow_nan=False)
        handle.write('\n')


def record(handle, value):
    handle.write(json.dumps(encode(value), allow_nan=False, separators=(',', ':')) + '\n')
    handle.flush()


def declared_data(report, key):
    rows = []
    for batch_index, batch in enumerate(report['evaluation_data'][key]):
        assert len(batch['bits']) == len(batch['targets']) == 4
        for row_index, (bits, label) in enumerate(zip(batch['bits'], batch['targets'])):
            assert all(b in (0, 1) for b in bits) and sum(bits) % 2 == label
            rows.append(dict(sequence_id=f'{key}/batch{batch_index}/row{row_index}',
                             length=len(bits), bits=bits, label=label))
    expected = range(4, 17) if key == 'val' else range(17, 33)
    assert Counter(r['length'] for r in rows) == Counter({n: 4 for n in expected})
    return rows


def dataset_identity(rows, report):
    sequences = {tuple(r['bits']) for r in rows}
    overlap = {seed: sum(tuple(r['bits']) in {tuple(bits) for batch in batches for bits in batch['bits']}
                         for r in rows) for seed, batches in report['training_data'].items()}
    return dict(sequences=len(rows), tokens=sum(r['length'] for r in rows),
                ordered_sequence_label_sha256=json_hash(rows),
                bits_sha256=json_hash([r['bits'] for r in rows]),
                labels_sha256=json_hash([r['label'] for r in rows]),
                counts_by_length=dict(Counter(r['length'] for r in rows)),
                label_balance=dict(Counter(r['label'] for r in rows)),
                duplicate_sequences=len(rows)-len(sequences),
                original_control_sequence_ids=[r['sequence_id'] for r in rows if r['bits'] == CONTROL],
                historical_task_loss_training_overlap_counts=overlap,
                selected_imitation_training_overlap='UNKNOWN: exact training bits not serialized',
                stage1_training_overlap='UNKNOWN: exact Stage-1 training bits not serialized')


def prefix_comparison(reference, learned):
    """Align states only up to the first autonomous action divergence."""
    result = {key: None for key in ['first_action_divergence', 'first_execution_divergence',
        'first_state_feature_divergence', 'first_output_logit_divergence', 'first_task_disagreement',
        'first_selection_execution_mismatch']}
    identical_prefix = True
    ar = {t: [r for r in reference.rows if r['token'] == t] for t in range(len(reference.tokens))}
    br = {t: [r for r in learned.rows if r['token'] == t] for t in range(len(learned.tokens))}
    for a, b in zip(reference.tokens, learned.tokens):
        t = a['token']
        if identical_prefix:
            assert unchanged(a['_incoming'], b['_incoming']), 'Persistent difference before route divergence'
            for d, row in enumerate(br[t]):
                teacher = ar[t][d]
                keys = ['h_before', 'memory_read', 'memory_mean', 'pointer_one_hot', 'budget',
                        'normalized_features', 'raw_controller_logits', 'legal_action_mask']
                diff = first_difference(snapshot({k: teacher[k] for k in keys}), snapshot({k: row[k] for k in keys}))
                assert diff is None, f'State/feature/logit mismatch before action divergence: {diff}'
                if row['selected_action'] != teacher['selected_action']:
                    result['first_action_divergence'] = dict(token=t, decision=d,
                        execution_before=row['execution_before'], canonical_action=teacher['selected_action'],
                        selected_action=row['selected_action'], canonical_features={k: teacher[k] for k in keys},
                        learned_features={k: row[k] for k in keys}, equal_preaction_features=True)
                    postdiff = first_difference(snapshot(teacher['h_after']), snapshot(row['h_after']))
                    if postdiff:
                        result['first_state_feature_divergence'] = dict(token=t, decision=d,
                            scope='successors of first different action; counterfactual states, not equivalent computation', **postdiff)
                    identical_prefix = False
                    break
                assert first_difference(snapshot(teacher['h_after']), snapshot(row['h_after'])) is None
            if identical_prefix:
                assert len(ar[t]) == len(br[t]) == 8
                assert unchanged(a['_outgoing'], b['_outgoing']), 'Commit mismatch with identical route prefix'
                assert torch.equal(a['output_logits'], b['output_logits']), 'Output mismatch with identical route prefix'
        if result['first_execution_divergence'] is None:
            for p, (expected, actual) in enumerate(zip(a['executed_blocks'], b['executed_blocks'])):
                if expected != actual:
                    result['first_execution_divergence'] = dict(token=t, execution_position=p,
                        canonical_block=expected, actual_block=actual,
                        scope='ordered executed-block signatures, not aligned hidden states after divergence')
                    break
        if result['first_state_feature_divergence'] is None and not identical_prefix:
            diff = first_difference(a['_outgoing'], b['_outgoing'])
            if diff:
                result['first_state_feature_divergence'] = dict(token=t, scope='token boundary after legitimate action divergence', **diff)
        if result['first_output_logit_divergence'] is None:
            diff = first_difference(snapshot(a['output_logits']), snapshot(b['output_logits']))
            if diff:
                result['first_output_logit_divergence'] = dict(token=t, **diff)
        if result['first_task_disagreement'] is None and t == len(reference.tokens)-1 and a['prediction'] != b['prediction']:
            result['first_task_disagreement'] = dict(token=t, reference=a['prediction'], learned=b['prediction'], target=b['target'])
    return result


def sequence_metrics(spec, reference, learned):
    a, b = reference.tokens[-1], learned.tokens[-1]
    for path in [reference, learned]:
        for token in path.tokens:
            assert token['P'] == len(token['executed_blocks']) == 8
            assert token['D'] == 8 + token['restart_count'] and 8 <= token['D'] <= 15
            assert token['commit_count'] == 1 and token['commit_entries'] == 7
        for row in path.rows:
            assert row['budget'].tolist() == [[0., 0.]]
            assert torch.isfinite(row['raw_controller_logits']).all()
            assert row['legal_action_mask'][row['selected_action']]
    return dict(**spec, reference_prediction=a['prediction'], learned_prediction=b['prediction'],
        reference_ce=a['prefix_ce'], learned_ce=b['prefix_ce'], ce_delta=b['prefix_ce']-a['prefix_ce'],
        prediction_disagreement=a['prediction'] != b['prediction'],
        canonical_action_tokens=sum(t['history'] == CANONICAL for t in learned.tokens),
        canonical_block_tokens=sum(t['executed_blocks'] == list(range(8)) for t in learned.tokens),
        exact_sequence_canonical_recovery=all(t['history'] == CANONICAL for t in learned.tokens),
        D=[t['D'] for t in learned.tokens], P=[t['P'] for t in learned.tokens],
        restart_count=sum(t['restart_count'] for t in learned.tokens),
        repeat_count=sum(t['repeat_count'] for t in learned.tokens),
        unique_blocks_per_token=[len(set(t['executed_blocks'])) for t in learned.tokens],
        routes=[','.join(str(r['selected_action']) for r in learned.rows if r['token'] == t['token']) for t in learned.tokens],
        block_routes=[','.join(map(str, t['executed_blocks'])) for t in learned.tokens],
        action_counts=dict(Counter(r['selected_action'] for r in learned.rows)),
        block_counts=dict(Counter(i for t in learned.tokens for i in t['executed_blocks'])),
        teacher_correct_by_position=[sum(r['teacher_action_correct'] for r in reference.rows if r['decision'] == d) for d in range(8)],
        teacher_decisions=len(reference.rows), commits_reference=len(reference.tokens), commits_learned=len(learned.tokens),
        bank_purity_checks=dict(reference=reference.block_purity_checks, learned=learned.block_purity_checks),
        gdn_purity_checks=dict(reference=dict(reference.gdn_purity_checks), learned=dict(learned.gdn_purity_checks)),
        divergence=prefix_comparison(reference, learned))


def aggregate(records):
    if not records:
        return dict(sequences=0)
    n, tokens = len(records), sum(r['length'] for r in records)
    decisions = sum(sum(r['D']) for r in records)
    actions, blocks = Counter(), Counter()
    for r in records:
        actions.update(r['action_counts']); blocks.update(r['block_counts'])
    first = {}
    for kind in records[0]['divergence']:
        first[kind] = next((dict(sequence_id=r['sequence_id'], **r['divergence'][kind]) for r in records if r['divergence'][kind] is not None), None)
    teacher_correct = [sum(r['teacher_correct_by_position'][d] for r in records) for d in range(8)]
    return dict(sequences=n, tokens=tokens, decisions=decisions, executions=8*tokens,
        reference_accuracy=sum(r['reference_prediction'] == r['label'] for r in records)/n,
        learned_accuracy=sum(r['learned_prediction'] == r['label'] for r in records)/n,
        reference_mean_ce=sum(r['reference_ce'] for r in records)/n,
        learned_mean_ce=sum(r['learned_ce'] for r in records)/n,
        paired_prediction_disagreements=sum(r['prediction_disagreement'] for r in records),
        paired_mean_ce_delta=sum(r['ce_delta'] for r in records)/n,
        paired_max_abs_ce_delta=max(abs(r['ce_delta']) for r in records),
        canonical_action_route_fraction=sum(r['canonical_action_tokens'] for r in records)/tokens,
        canonical_executed_block_fraction=sum(r['canonical_block_tokens'] for r in records)/tokens,
        exact_sequence_canonical_fraction=sum(r['exact_sequence_canonical_recovery'] for r in records)/n,
        teacher_accuracy=sum(teacher_correct)/(8*tokens), teacher_correct_by_position=teacher_correct,
        teacher_errors_by_position=[tokens-v for v in teacher_correct], teacher_decisions=8*tokens,
        restart_count=sum(r['restart_count'] for r in records), repeat_count=sum(r['repeat_count'] for r in records),
        mean_D=decisions/tokens, maximum_D=max(max(r['D']) for r in records),
        D_distribution=dict(Counter(d for r in records for d in r['D'])),
        P_distribution=dict(Counter(p for r in records for p in r['P'])),
        unique_blocks_per_token_distribution=dict(Counter(v for r in records for v in r['unique_blocks_per_token'])),
        action_counts=dict(actions), action_frequencies={k: v/decisions for k, v in actions.items()},
        block_execution_counts=dict(blocks), block_utilization={k: v/(8*tokens) for k, v in blocks.items()},
        route_frequencies=dict(Counter(route for r in records for route in r['routes'])),
        executed_block_route_frequencies=dict(Counter(route for r in records for route in r['block_routes'])),
        first_divergence=first,
        commit_counts=dict(reference=tokens, learned=tokens),
        bank_purity_checks={path: sum(r['bank_purity_checks'][path] for r in records) for path in ['reference','learned']},
        gdn_purity_checks={path: {str(i): sum(r['gdn_purity_checks'][path].get(i, 0) for r in records) for i in [4,5]} for path in ['reference','learned']})


def split_report(output, key, results, mechanical_failure=None):
    metrics = aggregate(results)
    metrics['per_length'] = {str(length): aggregate([r for r in results if r['length'] == length]) for length in sorted({r['length'] for r in results})}
    report = dict(split=key, completed=mechanical_failure is None, mechanical_failure=mechanical_failure,
        metrics=metrics, sequences=results,
        verdicts=dict(mechanical='VERIFIED' if mechanical_failure is None else 'FAILED',
            canonical_imitation='exact recovery on all evaluated sequences' if metrics.get('exact_sequence_canonical_fraction') == 1 else 'incomplete recovery',
            generalization='behavior on historically reused declared evaluation sequences; qualified by manifest provenance',
            parity='reference and learned accuracy/CE reported separately; no trainability claim'))
    write_json(output / f'{key}.json', report)
    lines = [f'# Locked Gate-C {key} evaluation', '', f'Mechanical execution: {report["verdicts"]["mechanical"]}.', '',
        '| Metric | Result |', '|---|---|']
    for field in ['sequences','tokens','reference_accuracy','learned_accuracy','reference_mean_ce','learned_mean_ce',
                  'paired_prediction_disagreements','paired_mean_ce_delta','canonical_action_route_fraction',
                  'canonical_executed_block_fraction','exact_sequence_canonical_fraction','teacher_accuracy',
                  'mean_D','maximum_D','restart_count','repeat_count','P_distribution']:
        lines.append(f'| {field} | {metrics.get(field)} |')
    lines.extend(['', '## Results by length', '', '| Length | Sequences | Reference accuracy / CE | Learned accuracy / CE | Canonical tokens | Teacher accuracy |', '|---|---|---|---|---|---|'])
    for length, m in metrics['per_length'].items():
        lines.append(f'| {length} | {m["sequences"]} | {m["reference_accuracy"]} / {m["reference_mean_ce"]:.12g} | {m["learned_accuracy"]} / {m["learned_mean_ce"]:.12g} | {m["canonical_action_route_fraction"]} | {m["teacher_accuracy"]} |')
    lines.extend(['', 'First divergences are recorded in the JSON and first-divergence JSONL. State comparisons assert equality only while full autonomous action prefixes agree.',
        '', 'Teacher measurements evaluate the policy on reference states without affecting reference or learned execution. Primary CE/accuracy use only the final token of each sequence.',
        '', 'These sequences were used in prior diagnostics. Exact overlap with imitation and Stage-1 training data is unknown. OOD lengths exceed declared training lengths. This frozen supervised checkpoint result does not establish task-loss route discovery, adaptive depth, useful diversity, HALT benefit, or superiority.', ''])
    (output / f'{key}.md').write_text('\n'.join(lines))
    return report


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'experiments/results/gatec_locked_policy_validation_ood_seed3305_v1')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Refusing to overwrite evaluation artifacts')
    assert sha(POLICY) == POLICY_SHA and sha(BACKBONE) == BACKBONE_SHA and sha(DATA) == DATA_SHA
    protected = {str(p): sha(p) for p in list((ROOT/'src').rglob('*')) + list((ROOT/'experiments/runs').glob('*')) if p.is_file()}
    identity = git_identity()
    report = json.loads(DATA.read_text())
    datasets = {key: declared_data(report, key) for key in ['val', 'ood_test']}
    policy = torch.load(POLICY, map_location='cpu', weights_only=True)
    assert 'state_dict' not in policy and policy['backbone_sha256'] == BACKBONE_SHA
    stage1 = torch.load(BACKBONE, map_location='cpu', weights_only=True)
    weights = {k: v for k, v in stage1['state_dict'].items() if not k.startswith('controller.')}
    random.seed(3305); torch.manual_seed(3305); torch.cuda.manual_seed_all(3305); torch.set_num_threads(4)
    device = torch.device('cuda')
    models = [AGTv3(dim=64, num_memory_slots=16, vocab_size=2) for _ in range(2)]
    expected = {k: list(v.shape) for k, v in models[1].controller.state_dict().items()}
    assert expected == {k: list(v.shape) for k, v in policy['controller_state_dict'].items()}
    assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in policy['controller_state_dict'].values())
    for model in models:
        missing, unexpected = model.load_state_dict(weights, strict=False)
        assert set(missing) == {k for k in model.state_dict() if k.startswith('controller.')} and not unexpected
        model.executor.max_execution_depth = 8
    models[1].controller.load_state_dict(policy['controller_state_dict'], strict=True)
    for model in models:
        model.to(device=device, dtype=torch.float32).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    assert all(torch.equal(models[0].state_dict()[k], models[1].state_dict()[k]) for k in weights)
    before = [snapshot(model.state_dict()) for model in models]
    task = TaskAParity(64, device)
    companion = POLICY.with_suffix('.json')
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), protocol='locked-persisted-validation-then-ood-no-update-v1',
        checkpoint=dict(path=str(POLICY.relative_to(ROOT)), sha256=POLICY_SHA, metadata={k:v for k,v in policy.items() if k != 'controller_state_dict'},
                        parameter_shapes=expected, loaded_only='controller_state_dict', companion_sha256=sha(companion)),
        backbone=dict(path=str(BACKBONE.relative_to(ROOT)), sha256=BACKBONE_SHA, noncontroller_tensor_count=len(weights),
                      identical_loaded_backbones=True, loaded_tensor_fingerprint=fingerprint(snapshot(weights))),
        git=identity, script_sha256=sha(__file__), observer_sha256=sha(ROOT/'experiments/replay_gatec_learned_policy.py'),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src').glob('*.py')},
        configuration=dict(device=str(device), device_name=torch.cuda.get_device_name(device), torch_version=str(torch.__version__),
            dim=64, memory_slots=16, controller='395->256->11', action_slots=LABELS, batch=1, dtype='float32',
            actual_budget_fraction=0., actual_budget_presence_flag=0., budget_feature=[0.,0.], execution_budget=8,
            routing='deterministic legal argmax; production selected-only execution', training=False,
            halt_successor_excluded=True, convergence_not_used_by_fixed_depth_path=True, adaptive_depth=False,
            optimizer_exists=False, optimizer_steps=0, decision_cap=None, teacher_forcing=False, fallback=False,
            sequence_reset='fresh identical initial_persistent_state; each path retains its own state within sequence',
            task_metrics='final-token CE and final-token parity only', atol=1e-6, rtol=1e-5,
            token_feature_seed=TaskAParity.FEATURE_SEED, token_feature_sha256=fingerprint(snapshot(task.token_features))),
        dataset_source=dict(path=str(DATA.relative_to(ROOT)), sha256=DATA_SHA, historical_git_head=report['git_head'],
            arrays=['evaluation_data.val','evaluation_data.ood_test'], order='stored batch then stored row; unchanged',
            data_generation=False, filtering=False, resampling=False),
        datasets={key:dataset_identity(rows, report) for key,rows in datasets.items()},
        val_ood_full_sequence_overlap=len({tuple(r['bits']) for r in datasets['val']} & {tuple(r['bits']) for r in datasets['ood_test']}),
        provenance=dict(objective='controller-only supervised canonical route imitation', seed=policy['seed'], steps=policy['steps'],
            selection='final requested 5000-update checkpoint; not selected by this evaluation',
            preparation_backbone_frozen='recorded by preparation script; checkpoint itself contains no alternate backbone',
            preparation_training_lengths=[4,16], preparation_sequences=104, preparation_route_rows=8320,
            before_execution_accounting_correction=True,
            correction_caveat='canonical teacher trajectories have D=P and zero budget; no restart budget discrepancy in those teacher inputs',
            earlier_heldout_probes='13 validation and 16 OOD sequences, one per length; only counts stored, not recoverable bits; distinct from present persisted 52/64 sets',
            historical_task_loss_overlap_caveat='overlaps refer to later task-loss diagnostics, not weight updates of the selected pure-imitation checkpoint',
            validation_blind=False, evaluation_sets_previously_used=True,
            ood_imitation_full_sequence_training_overlap='excluded by declared disjoint length ranges; exact training content not retained',
            unknowns=['exact imitation and Stage-1 training sequences/overlaps', 'independent blind checkpoint-selection protocol',
                      'exact membership of earlier one-per-length probes', 'post-preparation backbone tensor hash in original run']),
        preregistered_order=['val','ood_test'], stop_rule='mechanical violation stops before OOD; scientific discrepancies do not change protocol')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir/'manifest.json', manifest)
    manifest_sha = sha(args.output_dir/'manifest.json')
    print(f'MANIFEST LOCKED {manifest_sha}', flush=True)
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
    reports, mechanical_failure = {}, None
    with (args.output_dir/'decisions.jsonl').open('x') as decisions, (args.output_dir/'tokens.jsonl').open('x') as tokens, (args.output_dir/'first_divergence.jsonl').open('x') as divergence:
        for key in ['val','ood_test']:
            if mechanical_failure:
                reports[key] = dict(completed=False, reason='NOT RUN: mechanical failure in validation')
                write_json(args.output_dir/f'{key}.json', reports[key])
                (args.output_dir/f'{key}.md').write_text('# OOD evaluation\n\nNOT RUN: validation mechanical failure.\n')
                break
            results = []
            for index, spec in enumerate(datasets[key]):
                a = b = None
                try:
                    bits = torch.tensor([spec['bits']], device=device, dtype=torch.long)
                    x, targets = task.token_features[bits], bits.cumsum(1).remainder(2)
                    assert int(targets[0,-1]) == spec['label']
                    a = ObservedPath(models[0], models[1].controller, False)
                    b = ObservedPath(models[1], models[1].controller, True)
                    a.run(x, targets); b.run(x, targets)
                    result = sequence_metrics(spec, a, b)
                    results.append(result)
                    for path_name, path in [('reference',a),('learned',b)]:
                        for row in path.rows:
                            fields = ['token','decision','execution_before','execution_after','pointer_before','pointer_after',
                                'just_restarted_before','just_restarted_after','legal_action_mask','legal_successor_mask',
                                'raw_controller_logits','selected_action','selected_action_label','decoded_block_target',
                                'actual_executed_block','executed_blocks','restart','repeat','budget']
                            record(decisions, dict(split=key, sequence_id=spec['sequence_id'], path=path_name,
                                **{k:row[k] for k in fields},
                                **({k:row[k] for k in ['teacher_selected_action','teacher_action_correct']} if path_name == 'reference' else {})))
                        for token in path.tokens:
                            record(tokens, dict(split=key, sequence_id=spec['sequence_id'], path=path_name, **encode(token)))
                    record(divergence, dict(split=key, sequence_id=spec['sequence_id'], **result['divergence']))
                    print(f'{key} {index+1}/{len(datasets[key])} L={spec["length"]}: P=8, Dmax={max(result["D"])}, CE ref/policy={result["reference_ce"]:.8g}/{result["learned_ce"]:.8g}', flush=True)
                except Exception as error:
                    mechanical_failure = dict(split=key, sequence_id=spec['sequence_id'], type=type(error).__name__,
                        message=str(error), traceback=traceback.format_exc(),
                        partial_reference=[] if a is None else a.rows, partial_learned=[] if b is None else b.rows)
                    write_json(args.output_dir/'mechanical_failure.json', mechanical_failure)
                    print(f'MECHANICAL FAILURE: {key} {spec["sequence_id"]} {error}', flush=True)
                    break
            reports[key] = split_report(args.output_dir, key, results, mechanical_failure)
            print(f'{key} COMPLETE: {len(results)} sequences', flush=True)
    invariant = dict(weights_unchanged=all(unchanged(s, snapshot(m.state_dict())) for s,m in zip(before,models)),
        parameters_frozen_gradients_absent=all(not p.requires_grad and p.grad is None for m in models for p in m.parameters()),
        protected_files_unchanged=all(sha(p)==h for p,h in protected.items()),
        tracked_diff_unchanged=git_identity()['tracked_diff_sha256']==identity['tracked_diff_sha256'],
        rng_states_unchanged=torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state(device)),
        diff_check_clean=subprocess.run(['git','diff','--check'],capture_output=True).returncode==0,
        optimizer_steps=0, candidate_constructions=0, checkpoint_unchanged=sha(POLICY)==POLICY_SHA,
        dataset_unchanged=sha(DATA)==DATA_SHA, backbone_file_unchanged=sha(BACKBONE)==BACKBONE_SHA)
    assert all(v for k,v in invariant.items() if k not in ['optimizer_steps','candidate_constructions'])
    aggregate_report = dict(manifest_path='manifest.json', manifest_sha256=manifest_sha,
        checkpoint_sha256=POLICY_SHA, backbone_sha256=BACKBONE_SHA,
        mechanical_validity='VERIFIED' if mechanical_failure is None else 'FAILED',
        evaluations={k:{field:v for field,v in r.items() if field!='sequences'} for k,r in reports.items()},
        invariants=invariant, original_control='NOT INCLUDED in either declared split; not re-evaluated',
        limitations=['one historical supervised checkpoint, historically reused evaluation sets',
            'exact imitation/Stage-1 training overlap unknown', 'no task-loss routing, adaptive depth, HALT or superiority claim',
            'candidate construction not exercised; selected-block GDN/cache purity checked'],
        artifacts={p.name:sha(p) for p in args.output_dir.iterdir() if p.is_file()})
    write_json(args.output_dir/'aggregate.json', aggregate_report)
    print(json.dumps(encode({k:r.get('metrics',r) for k,r in reports.items()}), indent=2, allow_nan=False))
    print(f'ARTIFACTS {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
