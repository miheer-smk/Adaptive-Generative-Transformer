"""One frozen, deterministic paired replay; observers never select/repair actions."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.task_a import TaskAParity
from src.agt import AGTv3
import src.agt as agt_module
import src.executor as executor_module
from src.masks import build_execution_mask
from src.terminal_commit import build_commit_plan
from experiments.audit_gatec_canonical_replay import snapshot, unchanged, first_difference, concise

POLICY = ROOT / 'experiments/runs/stage3_imitation_replication_3305_v1.pt'
POLICY_SHA = '259e2b86786c6cd037e3ba0db71d91fd05cdb76753bdb462be414bd1b9400df3'
BACKBONE = ROOT / 'experiments/runs/stage1_parity_seed3305_best.pt'
BACKBONE_SHA = '53b56e7e90f7eb7a6570107b6dfe3a297caac48d637fe27e5f00fbe94c6f09c3'
CONTROL = ROOT / 'experiments/results/gatec_execution_accounting_v1/gatec_canonical_replay.json'
LABELS = [f'ADVANCE({i})' for i in range(8)] + ['REPEAT', 'RESTART', 'HALT']
ATOL, RTOL = 1e-6, 1e-5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy(tensor):
    return tensor.detach().cpu().clone()


def fingerprint(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        digest.update(key.encode())
        if torch.is_tensor(value):
            digest.update(str((str(value.dtype), list(value.shape))).encode())
            digest.update(bytes(value.contiguous().view(torch.uint8).flatten().tolist()))
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def state_record(state):
    values = snapshot(state)
    return dict(sha256=fingerprint(values), summary=concise(values))


def git_identity():
    return dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
                status=subprocess.check_output(['git', 'status', '--short'], text=True),
                tracked_diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'])).hexdigest())


def encode(value):
    if torch.is_tensor(value):
        return value.tolist()
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()
                if not (isinstance(k, str) and k.startswith('_'))}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


class ObservedPath:
    def __init__(self, model, teacher_controller, learned):
        self.model, self.teacher_controller, self.learned = model, teacher_controller, learned
        self.tokens, self.rows = [], []
        self.block_purity_checks, self.gdn_purity_checks = 0, Counter()

    def finish_row(self, h_after):
        row = self.current[-1]
        expected = [] if row['selected_action'] == 9 else [row['decoded_block_target']]
        assert row['executed_blocks'] == expected, 'Selected action / actual bank execution mismatch'
        if row['selected_action'] == 9:
            torch.testing.assert_close(copy(h_after), row['h_before'], atol=0, rtol=0)
        elif 'h_after' in row:
            torch.testing.assert_close(copy(h_after), row['h_after'], atol=0, rtol=0)
        row['h_after'] = copy(h_after)
        row['actual_executed_block'] = expected[0] if expected else None

    @contextmanager
    def observers(self):
        model = self.model
        original_controller = model.executor.controller
        original_block = model.executor._run_block
        original_mask = executor_module.build_execution_mask
        original_commit = agt_module.terminal_commit
        original_read = model.memory.read
        original_candidates = model.executor.candidate_successors
        handles = []

        def read(*args, **kwargs):
            result = original_read(*args, **kwargs)
            self.read_count += 1
            return result

        def commit(work, **kwargs):
            before = snapshot(work)
            assert unchanged(before, self.incoming_snapshot), 'In-flight persistent mutation'
            plan = build_commit_plan(work, **kwargs)
            assert unchanged(before, snapshot(work)), 'Commit-plan mutation'
            result = original_commit(work, **kwargs)
            assert len(plan.entries) == 7
            torch.testing.assert_close(result[0].s, plan.s, atol=0, rtol=0)
            torch.testing.assert_close(result[0].memory, plan.memory, atol=0, rtol=0)
            for cache, version, payload in plan.entries:
                assert cache.version == version + 1
                for actual, expected in zip(cache.get_committed_context(), payload):
                    if actual is not None:
                        if actual.shape != expected.shape:
                            actual = actual[:, :, -1:, :]
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.commit_count += 1
            self.commit_entries = len(plan.entries)
            return result

        def controller(**features):
            if self.current:
                self.finish_row(features['h'])
            result = original_controller(**features)  # Returned unchanged.
            self.current.append(dict(token=self.token, decision=len(self.current),
                                     h_before=copy(features['h']),
                                     memory_read=copy(features['memory_read']),
                                     memory_mean=copy(features['memory_mean']),
                                     pointer_one_hot=copy(features['pointer_one_hot']),
                                     budget=copy(features['budget']), raw_controller_logits=copy(result['logits']),
                                     normalized_features=copy(model.controller._prepare_inputs(**features)),
                                     executed_blocks=[]))
            return result

        def mask(*args, **kwargs):
            result = original_mask(*args, **kwargs)  # Real source mask, unchanged.
            row = self.current[-1]
            pointer, executions, restarted = kwargs['pointer'], kwargs['execution_count'], kwargs['just_restarted']
            assert executions == sum(len(r['executed_blocks']) for r in self.current[:-1])
            assert int(row['pointer_one_hot'].argmax(-1)) == pointer
            if len(self.current) > 1:
                previous = self.current[-2]
                assert (pointer, executions, restarted) == (
                    previous['pointer_after'], previous['execution_after'], previous['just_restarted_after'])
            else:
                assert (pointer, executions, restarted) == (0, 0, False)
            successor_mask = result[:10].cpu()
            logits = row['raw_controller_logits'][:, :10]
            # Reconstruct selection for observation only; never feed it to execution.
            selected = int(logits.masked_fill(~successor_mask, -torch.inf).argmax(-1))
            assert result[selected] and selected != 10
            probabilities = logits.masked_fill(~successor_mask, -torch.inf).softmax(-1)
            row.update(pointer_before=pointer, execution_before=executions,
                       just_restarted_before=restarted, legal_action_mask=copy(result),
                       legal_successor_mask=successor_mask, selected_action=selected,
                       selected_action_label=LABELS[selected],
                       decoded_block_target=selected if selected < 8 else pointer if selected == 8 else None,
                       pointer_after=selected if selected < 8 else pointer if selected == 8 else 0,
                       execution_after=executions + int(selected != 9),
                       just_restarted_after=selected == 9, repeat=selected == 8, restart=selected == 9,
                       probabilities=probabilities)
            return result

        def block(block_index, h, h0, memory, context, **kwargs):
            before = snapshot(dict(h=h, h0=h0, R=memory, C=context))
            result = original_block(block_index, h, h0, memory, context, **kwargs)
            assert unchanged(before, snapshot(dict(h=h, h0=h0, R=memory, C=context)))
            assert unchanged(self.incoming_snapshot, snapshot(self.input_state))
            row = self.current[-1]
            row['executed_blocks'].append(block_index)
            row['h_after'], row['h0'] = copy(result), copy(h0)
            self.block_purity_checks += 1
            if model.blocks.blocks[block_index].species == 'GDN':
                self.gdn_purity_checks[block_index] += 1
            return result

        def no_candidates(*args, **kwargs):
            raise AssertionError('Deterministic inference constructed unselected candidates')

        def reference_pre(index):
            def hook(module, args, kwargs):
                h, h0, memory = args[0].squeeze(1), kwargs['h0'].squeeze(1), kwargs['R'].squeeze(1)
                d = len(self.current)
                assert index == d
                pointer = 0 if d == 0 else d - 1
                onehot = h.new_zeros(1, 9)
                onehot[:, pointer] = 1
                features = dict(h=h, memory_read=memory, memory_mean=self.input_state.memory.mean(1),
                                pointer_one_hot=onehot, budget=h.new_zeros(1, 2))
                logits = self.teacher_controller(**features)['logits']
                legal = build_execution_mask(pointer, d, False, 8).to(h.device)
                prediction = int(logits[:, :10].masked_fill(~legal[:10], -torch.inf).argmax(-1))
                self.current.append(dict(token=self.token, decision=d, execution_before=d, execution_after=d+1,
                    pointer_before=pointer, pointer_after=index, just_restarted_before=False, just_restarted_after=False,
                    selected_action=index, selected_action_label=LABELS[index], decoded_block_target=index,
                    actual_executed_block=index, executed_blocks=[index], repeat=False, restart=False,
                    legal_action_mask=copy(legal), legal_successor_mask=copy(legal[:10]),
                    raw_controller_logits=copy(logits), teacher_selected_action=prediction,
                    teacher_action_correct=prediction == index,
                    normalized_features=copy(self.teacher_controller._prepare_inputs(**features)),
                    h_before=copy(h), h0=copy(h0), memory_read=copy(memory),
                    memory_mean=copy(features['memory_mean']), pointer_one_hot=copy(onehot), budget=copy(features['budget']),
                    _inputs=snapshot(dict(h=args[0], h0=kwargs['h0'], R=kwargs['R'], C=kwargs.get('C_prev')))))
            return hook

        def reference_post(index):
            def hook(module, args, kwargs, output):
                row = self.current[-1]
                assert unchanged(row['_inputs'], snapshot(dict(h=args[0], h0=kwargs['h0'], R=kwargs['R'], C=kwargs.get('C_prev'))))
                assert unchanged(self.incoming_snapshot, snapshot(self.input_state))
                row['h_after'] = copy((output[0] if isinstance(output, tuple) else output).squeeze(1))
                self.block_purity_checks += 1
                if module.species == 'GDN':
                    self.gdn_purity_checks[index] += 1
            return hook

        try:
            model.memory.read, agt_module.terminal_commit = read, commit
            if self.learned:
                model.executor.controller, model.executor._run_block = controller, block
                model.executor.candidate_successors = no_candidates
                executor_module.build_execution_mask = mask
            else:
                for i, bank_block in enumerate(model.blocks.blocks):
                    handles.append(bank_block.register_forward_pre_hook(reference_pre(i), with_kwargs=True))
                    handles.append(bank_block.register_forward_hook(reference_post(i), with_kwargs=True))
            yield
        finally:
            for handle in handles:
                handle.remove()
            model.executor.controller, model.executor._run_block = original_controller, original_block
            model.executor.candidate_successors = original_candidates
            executor_module.build_execution_mask, agt_module.terminal_commit = original_mask, original_commit
            model.memory.read = original_read

    def run(self, x, targets):
        state = self.model.initial_persistent_state(1, x.device, x.dtype)
        compatibility = x.new_zeros(1, 64)
        with self.observers():
            for self.token in range(x.shape[1]):
                self.current, self.commit_count, self.read_count = [], 0, 0
                self.input_state, self.incoming_snapshot = state, snapshot(state)
                incoming = state_record(state)
                if self.learned:
                    out = self.model.adaptive_differentiable_step(x[:, self.token], state, compatibility,
                        budget=x.new_zeros(1, 2), training=False, tau=1.)
                    self.finish_row(out['h_terminal'])
                    executions = out['execution_count']
                    history = out['action_history']
                    assert not out['halted'] and not out['safety_cap_hit']
                else:
                    out = self.model.step(x[:, self.token], state, compatibility)
                    executions = len(self.current)
                    history = [('ADVANCE', i) for i in range(8)]
                assert executions == sum(len(r['executed_blocks']) for r in self.current) == 8
                assert len(history) == len(self.current)
                assert history == [('ADVANCE', r['selected_action']) if r['selected_action'] < 8 else
                                   ('REPEAT', r['pointer_before']) if r['selected_action'] == 8 else
                                   ('RESTART', None) for r in self.current]
                assert self.commit_count == 1 and self.read_count == 1
                assert unchanged(self.incoming_snapshot, snapshot(state))
                assert out['state'].tokens_committed == state.tokens_committed + 1
                self.rows.extend(self.current)
                self.tokens.append(dict(token=self.token, P=executions, D=len(history), history=history,
                    termination_reason='execution_budget_reached' if self.learned else 'canonical_sweep_completed',
                    restart_count=sum(r['restart'] for r in self.current), repeat_count=sum(r['repeat'] for r in self.current),
                    executed_blocks=[r['actual_executed_block'] for r in self.current if r['actual_executed_block'] is not None],
                    commit_count=self.commit_count, commit_entries=self.commit_entries,
                    incoming_persistent=incoming, outgoing_persistent=state_record(out['state']),
                    _incoming=self.incoming_snapshot, _outgoing=snapshot(out['state']),
                    output_logits=copy(out['logits']), prediction=int(out['logits'].argmax(-1)), target=int(targets[:, self.token]),
                    prefix_ce=F.cross_entropy(out['logits'], targets[:, self.token]).item()))
                state = out['state']
        return self


def comparisons(reference, learned):
    result = dict(first_action_divergence=None, first_execution_divergence=None,
                  first_state_divergence=None, first_execution_state_divergence=None,
                  first_persistent_divergence=None, first_output_divergence=None)
    for a, b in zip(reference.tokens, learned.tokens):
        t = a['token']
        ar = [r for r in reference.rows if r['token'] == t]
        br = [r for r in learned.rows if r['token'] == t]
        for d, row in enumerate(br):
            teacher = ar[d] if d < len(ar) else None
            if result['first_action_divergence'] is None and (teacher is None or row['selected_action'] != teacher['selected_action']):
                result['first_action_divergence'] = dict(token=t, decision=d, execution_before=row['execution_before'],
                    selected=row['selected_action_label'], canonical=teacher['selected_action_label'] if teacher else None,
                    raw_controller_logits=row['raw_controller_logits'], legal_action_mask=row['legal_action_mask'],
                    input_difference=first_difference(snapshot({k: row[k] for k in ['h_before', 'memory_read', 'memory_mean', 'pointer_one_hot', 'budget']}),
                        snapshot({k: teacher[k] for k in ['h_before', 'memory_read', 'memory_mean', 'pointer_one_hot', 'budget']})) if teacher else 'No canonical decision at this index',
                    normalized_feature_difference=first_difference(snapshot(row['normalized_features']), snapshot(teacher['normalized_features'])) if teacher else None)
            if teacher is not None and result['first_state_divergence'] is None:
                mismatch = first_difference(snapshot({k: row[k] for k in ['h_before', 'h_after', 'memory_read', 'memory_mean']}),
                                            snapshot({k: teacher[k] for k in ['h_before', 'h_after', 'memory_read', 'memory_mean']}))
                if mismatch:
                    result['first_state_divergence'] = dict(token=t, decision=d, alignment='decision index', **mismatch)
        be = [r for r in br if r['actual_executed_block'] is not None]
        for p, (teacher, row) in enumerate(zip(ar, be)):
            if result['first_execution_divergence'] is None and row['actual_executed_block'] != teacher['actual_executed_block']:
                result['first_execution_divergence'] = dict(token=t, execution_index=p, decision=row['decision'],
                    canonical_block=teacher['actual_executed_block'], actual_block=row['actual_executed_block'])
            if result['first_execution_state_divergence'] is None:
                mismatch = first_difference(snapshot(teacher['h_after']), snapshot(row['h_after']))
                if mismatch:
                    result['first_execution_state_divergence'] = dict(token=t, execution_index=p, decision=row['decision'], **mismatch)
        for boundary in ['incoming', 'outgoing']:
            if result['first_persistent_divergence'] is None:
                mismatch = first_difference(a['_' + boundary], b['_' + boundary])
                if mismatch:
                    result['first_persistent_divergence'] = dict(token=t, boundary=boundary, **mismatch)
        if result['first_output_divergence'] is None:
            mismatch = first_difference(snapshot(a['output_logits']), snapshot(b['output_logits']))
            if mismatch or a['prediction'] != b['prediction']:
                result['first_output_divergence'] = dict(token=t, **(mismatch or dict(field='prediction')))
    return result


def metrics(path):
    tokens, rows = path.tokens, path.rows
    canonical = sum(t['history'] == [('ADVANCE', i) for i in range(8)] for t in tokens)
    blocks = [block for t in tokens for block in t['executed_blocks']]
    return dict(completed_tokens=len(tokens), decisions=len(rows), executions=len(blocks),
        closed_loop_canonical_route_fraction=canonical / len(tokens),
        sequence_exact_canonical_route_recovery=canonical == len(tokens),
        restart_count=sum(t['restart_count'] for t in tokens), repeat_count=sum(t['repeat_count'] for t in tokens),
        action_counts=dict(Counter(r['selected_action_label'] for r in rows)),
        action_frequencies={a: n / len(rows) for a, n in Counter(r['selected_action_label'] for r in rows).items()},
        block_execution_counts=dict(Counter(blocks)),
        block_execution_frequencies={str(b): n / len(blocks) for b, n in Counter(blocks).items()},
        unique_blocks=sorted(set(blocks)), unique_blocks_per_token=[len(set(t['executed_blocks'])) for t in tokens],
        mean_D=sum(t['D'] for t in tokens) / len(tokens), maximum_D=max(t['D'] for t in tokens),
        P_distribution=dict(Counter(t['P'] for t in tokens)),
        sequence_parity_accuracy=float(tokens[-1]['prediction'] == tokens[-1]['target']),
        final_token_mean_ce=tokens[-1]['prefix_ce'],
        prefix_parity_accuracy=sum(t['prediction'] == t['target'] for t in tokens) / len(tokens),
        prefix_ce_mean=sum(t['prefix_ce'] for t in tokens) / len(tokens),
        per_length={'8': dict(sequences=1, final_token_accuracy=float(tokens[-1]['prediction'] == tokens[-1]['target']),
                             final_token_ce=tokens[-1]['prefix_ce'])})


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'experiments/results/gatec_learned_policy_replay_seed3305_v1')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory exists; never overwrite a replay')
    assert sha(POLICY) == POLICY_SHA and sha(BACKBONE) == BACKBONE_SHA
    protected = {str(p): sha(p) for p in list((ROOT / 'src').rglob('*')) + list((ROOT / 'experiments/runs').glob('*.pt')) if p.is_file()}
    identity = git_identity()
    torch.set_num_threads(4)
    torch.manual_seed(3305)
    device = torch.device(args.device)
    stage1 = torch.load(BACKBONE, map_location='cpu', weights_only=True)
    checkpoint = torch.load(POLICY, map_location='cpu', weights_only=True)
    assert 'state_dict' not in checkpoint  # No alternate backbone is accepted.
    weights = {k: v for k, v in stage1['state_dict'].items() if not k.startswith('controller.')}
    assert checkpoint['backbone_sha256'] == BACKBONE_SHA
    assert set(checkpoint['controller_state_dict']) == {
        'norm_h.weight', 'norm_memory_read.weight', 'mlp.0.weight', 'mlp.0.bias', 'mlp.2.weight', 'mlp.2.bias'}
    reference_model, learned_model = [AGTv3(dim=64, num_memory_slots=16, vocab_size=2) for _ in range(2)]
    expected = learned_model.controller.state_dict()
    assert {k: tuple(v.shape) for k, v in expected.items()} == {k: tuple(v.shape) for k, v in checkpoint['controller_state_dict'].items()}
    assert all(torch.isfinite(v).all() and v.dtype == torch.float32 for v in checkpoint['controller_state_dict'].values())
    for model in [reference_model, learned_model]:
        missing, unexpected = model.load_state_dict(weights, strict=False)
        assert set(missing) == {k for k in model.state_dict() if k.startswith('controller.')} and not unexpected
        model.executor.max_execution_depth = 8
    learned_model.controller.load_state_dict(checkpoint['controller_state_dict'], strict=True)
    for model in [reference_model, learned_model]:
        model.to(device=device, dtype=torch.float32).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    assert all(torch.equal(reference_model.state_dict()[k], learned_model.state_dict()[k]) for k in weights)
    model_snapshots = [snapshot(model.state_dict()) for model in [reference_model, learned_model]]
    task = TaskAParity(64, device)
    batch = task.make_fixed_set('train', 8, 1)
    bits = torch.cdist(batch.x.reshape(-1, 64), task.token_features).argmin(-1).reshape(1, 8)
    targets = bits.cumsum(1).remainder(2)
    control = json.loads(CONTROL.read_text())
    assert bits.tolist() == control['bits'] and int(batch.target) == int(targets[:, -1])
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
    a = ObservedPath(reference_model, learned_model.controller, False).run(batch.x, targets)
    print('Reference completed: eight canonical executions/token.', flush=True)
    b = ObservedPath(learned_model, learned_model.controller, True).run(batch.x, targets)
    print('Learned production rollout completed; no forced or fallback actions.', flush=True)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert cuda_rng is None or torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    pair = comparisons(a, b)
    # Match the stored canonical scientific control exactly, not only by CE.
    prior = control['paths']['A']
    assert all(torch.equal(r['h_before'], torch.tensor(old['h_before'])) and
               torch.equal(r['h_after'], torch.tensor(old['h']))
               for r, old in zip(a.rows, prior['steps']))
    assert all(torch.equal(t['output_logits'], torch.tensor(old['logits'])) and
               t['prefix_ce'] == old['ce'] for t, old in zip(a.tokens, prior['tokens']))
    assert all(t['outgoing_persistent']['summary'] == old['commit'] and
               t['incoming_persistent']['summary'] == old['commit_pre']
               for t, old in zip(a.tokens, prior['tokens']))
    assert a.tokens[-1]['prefix_ce'] == 0.0014546061865985394
    assert all(unchanged(before, snapshot(model.state_dict())) for model, before in zip([reference_model, learned_model], model_snapshots))
    assert all(p.grad is None and not p.requires_grad for model in [reference_model, learned_model] for p in model.parameters())
    assert all(sha(p) == digest for p, digest in protected.items())
    assert git_identity()['tracked_diff_sha256'] == identity['tracked_diff_sha256']
    assert subprocess.run(['git', 'diff', '--check'], capture_output=True).returncode == 0
    teacher_correct = sum(r['teacher_action_correct'] for r in a.rows)
    report = dict(protocol='one-sequence-no-update-production-closed-loop-v1', git=identity,
        execution_note='Initial attempt completed both paths and invariant checks but failed report encoding on integer metric keys. Only the encoder was corrected; the identical deterministic protocol was repeated to write these artifacts.',
        script_sha256=sha(__file__), checkpoint=dict(path=str(POLICY), sha256=POLICY_SHA, seed=checkpoint['seed'],
            supervised_steps=checkpoint['steps'], loaded_only='controller_state_dict',
            parameter_shapes={k: list(v.shape) for k, v in checkpoint['controller_state_dict'].items()}),
        backbone=dict(path=str(BACKBONE), sha256=BACKBONE_SHA, identical_noncontroller_tensors=True, tensor_count=len(weights)),
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'src').glob('*.py')},
        configuration=dict(device=str(device), torch_version=str(torch.__version__), batch_size=1, dtype='float32',
            execution_budget=8, budget=[0., 0.], training=False, routing='legal argmax over ten non-HALT successors',
            action_slots=LABELS, optimizer_created=False, optimizer_steps=0, gradients_enabled=False,
            parameters_frozen=True, teacher_forcing=False, fallback=False, decision_cap=None, atol=ATOL, rtol=RTOL),
        dataset=dict(task='Task-A parity', split='train control; not a held-out generalization test', length=8, sequences=1,
            task_seed=TaskAParity.TASK_SEED, feature_seed=TaskAParity.FEATURE_SEED, bits=bits.tolist(),
            prefix_targets=targets.tolist(), final_target=batch.target.tolist(),
            input_sha256=fingerprint(snapshot(batch.x)), control_path=str(CONTROL), control_sha256=sha(CONTROL)),
        teacher_trajectory=dict(scope='64 canonical decisions on this control only; not the full imitation dataset',
            correct=teacher_correct, decisions=len(a.rows), accuracy=teacher_correct / len(a.rows)),
        fixed_metrics=metrics(a), learned_metrics=metrics(b), first_divergence=pair,
        invariants=dict(selected_actions_legal=True, selection_matches_execution=True, execution_accounting=True,
            restart_zero_compute=True, one_terminal_commit_per_token=True, all_context_entries_committed=True,
            commit_payloads_recomputed_from_precommit_state=True, input_persistent_state_unchanged=True,
            actual_candidate_constructions=0, unselected_candidate_execution=False,
            selected_block_purity_checks=dict(reference=a.block_purity_checks, learned=b.block_purity_checks),
            gdn_committed_state_buffer_purity_checks=dict(reference=dict(a.gdn_purity_checks), learned=dict(b.gdn_purity_checks)),
            reference_matches_stored_control_exactly=True, weights_unchanged=True, protected_files_unchanged=True,
            tracked_diff_unchanged=True, diff_check_clean=True, rng_states_unchanged=True, parameter_gradients_absent=True),
        fixed_trace=dict(decisions=a.rows, tokens=a.tokens), learned_trace=dict(decisions=b.rows, tokens=b.tokens),
        limitations=['One historical supervised checkpoint, one train control, one length.',
            'No task-loss trainability, generalization, efficiency, route diversity, adaptive-depth, or HALT claim.',
            'Exhaustive candidate construction is not exercised during deterministic inference.'])
    report = encode(report)
    lm = report['learned_metrics']
    report['outcome'] = 'A: canonical route recovered and parity preserved' if lm['sequence_exact_canonical_route_recovery'] and lm['sequence_parity_accuracy'] else \
        'B: noncanonical legal route and parity preserved on this control' if lm['sequence_parity_accuracy'] else \
        'C: valid mechanical replay, task preservation failed on this control'
    lines = ['# Gate-C first no-update learned-policy replay', '', report['outcome'], '',
        f"Controller `{POLICY.name}` SHA-256 `{POLICY_SHA}`; frozen Stage-1 backbone SHA-256 `{BACKBONE_SHA}`.",
        f"HEAD `{identity['head']}`; branch `{identity['branch']}`. Full worktree identity and source hashes are in JSON.",
        '', 'One train control of length eight; FP32, batch one, zero budget, deterministic production inference, P=8. No forced actions, fallback, optimizer, or decision cap.',
        '', f"Bits: `{bits.tolist()}`; final target `{int(batch.target)}`.",
        f"Teacher-trajectory decisions: {teacher_correct}/{len(a.rows)} correct. This is this control's teacher-trajectory accuracy, not a new offline-dataset measurement.",
        '', '| Path | Final prediction | Final-token CE | Sequence parity accuracy | Canonical tokens | Decisions | Executions | RESTART | REPEAT |',
        '|---|---|---|---|---|---|---|---|---|']
    for label, path, m in [('Fixed', a, report['fixed_metrics']), ('Learned', b, lm)]:
        lines.append(f"| {label} | {path.tokens[-1]['prediction']} | {m['final_token_mean_ce']:.17g} | {m['sequence_parity_accuracy']} | {m['closed_loop_canonical_route_fraction']} | {m['decisions']} | {m['executions']} | {m['restart_count']} | {m['repeat_count']} |")
    lines.extend(['', '| Token | D | P | Commits | Route | Prediction | Prefix CE |', '|---|---|---|---|---|---|---|'])
    for t in b.tokens:
        lines.append(f"| {t['token']} | {t['D']} | {t['P']} | {t['commit_count']} | {t['history']} | {t['prediction']} | {t['prefix_ce']:.9g} |")
    lines.extend(['', f"Learned D mean/max: {lm['mean_D']}/{lm['maximum_D']}; P distribution: {lm['P_distribution']}; unique blocks: {lm['unique_blocks']}.",
        f"Prefix readout accuracy/mean CE (secondary): {lm['prefix_parity_accuracy']}/{lm['prefix_ce_mean']:.9g}. Primary task CE is final-token only.",
        '', 'First divergences: `' + json.dumps(report['first_divergence']) + '`.',
        '', 'VERIFIED: legal selected actions match actual bank execution, P=8 accounting, zero-compute RESTART semantics, frozen in-flight committed state, seven terminal-derived context entries, and one commit per completed token.',
        f"Selected-block purity checks: fixed {a.block_purity_checks}, learned {b.block_purity_checks}; GDN checks: {dict(b.gdn_purity_checks)}. Candidate construction was not invoked; no new exhaustive-candidate purity result is claimed.",
        'VERIFIED: reference live states/logits/CE exactly match the stored canonical control; weights and protected files unchanged; git diff --check clean.',
        '', 'This establishes behavior of this supervised controller on one control only. It does not establish task-loss route discovery, general parity competence, generalization, useful diversity, adaptive depth, learned HALT, or superiority.', ''])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'replay.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    with (args.output_dir / 'replay.md').open('x') as handle:
        handle.write('\n'.join(lines))
    print(json.dumps(dict(outcome=report['outcome'], teacher=report['teacher_trajectory'],
                         fixed=report['fixed_metrics'], learned=lm, first_divergence=report['first_divergence']), indent=2))
    print(args.output_dir / 'replay.json', args.output_dir / 'replay.md')


if __name__ == '__main__':
    main()
