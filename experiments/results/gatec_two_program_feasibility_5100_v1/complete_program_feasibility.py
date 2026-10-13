"""Research-only, frozen two-program feasibility study. No training or selector.

All execution, masking, cloning and commits are calls to existing primitives.
Observers are sequential and restored in finally blocks; they never repair routes.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from collections import Counter
from contextlib import ExitStack
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import shutil
import subprocess
import time
import traceback
from unittest.mock import patch

import torch
import torch.nn.functional as F

from data.task_a import TaskAParity
from src.agt import AGTv3
import src.agt as agt_module
import src.executor as executor_module
import src.terminal_commit as commit_module
from src.masks import build_execution_mask
from experiments.audit_gatec_canonical_replay import first_difference
from experiments.run_gatec_paired_task_maintenance import refs, versions, gpu_snapshot, assert_same
from experiments.run_gatec_fresh_task_discovery import recorded_bits
from experiments.replay_gatec_learned_policy import BACKBONE, BACKBONE_SHA, git_identity

OUT = ROOT / 'experiments/results/gatec_two_program_feasibility_5100_v1'
ACQUISITION_SUFFIXES = ('00000000', '10000000', '01010101', '11010101')
CONFIRMATION_SUFFIXES = ('00110011', '10110011', '01101001', '11101001')
SCHEDULES = ('P0', 'P1', 'alternating_P0_first')
HORIZONS = (1, 4, 8)
ATOL, RTOL = 1e-6, 1e-5


class MechanicalViolation(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise MechanicalViolation(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def tensor_bytes(value):
    """Exact bytes of an owned contiguous CPU tensor, without NumPy."""
    value = value.detach().cpu().contiguous()
    return ctypes.string_at(value.data_ptr(), value.numel() * value.element_size())


def tensor_hash(value):
    return hashlib.sha256(str((str(value.dtype), list(value.shape))).encode() + tensor_bytes(value)).hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(refs(state).items()):
        digest.update(key.encode())
        if torch.is_tensor(value):
            digest.update(str((str(value.dtype), list(value.shape))).encode())
            digest.update(tensor_bytes(value))
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def state_metadata(state):
    return {k: dict(shape=list(v.shape), dtype=str(v.dtype)) if torch.is_tensor(v) else v
            for k, v in refs(state).items()}


def exact_state(snapshot, state):
    try:
        assert_same(snapshot, state)
    except AssertionError as error:
        raise MechanicalViolation(str(error)) from error


def close_difference(left, right):
    # The archived comparator handles exact metadata and established FP tolerances.
    a = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in refs(left).items()}
    b = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in refs(right).items()}
    return first_difference(a, b)


def assert_disjoint(left, right):
    a, b = refs(left), refs(right)
    require(a.keys() == b.keys(), 'Clone topology differs')
    for key, value in a.items():
        other = b[key]
        if torch.is_tensor(value) and value.numel():
            require(value.untyped_storage().data_ptr() != other.untyped_storage().data_ptr(),
                    f'Parent/sibling tensor storage alias: {key}')


@dataclass(frozen=True)
class Program:
    name: str
    actions: tuple[int, ...]

    def __post_init__(self):
        if type(self.actions) is not tuple:
            raise ValueError('Program actions must be an immutable tuple')
        self.validate()

    def validate(self):
        pointer, count, restarted = 0, 0, False
        rows = []
        for decision, action in enumerate(self.actions):
            if type(action) is not int or action not in range(10):
                raise ValueError('Only non-HALT successor actions are allowed')
            if count >= 8:
                raise ValueError('Surplus program actions after P=8')
            legal = build_execution_mask(pointer, count, restarted, 8)
            if not bool(legal[action]):
                raise ValueError(f'Illegal action {action} at decision {decision}')
            rows.append(dict(decision=decision, pointer_before=pointer,
                             execution_before=count, action=action, mask=legal.tolist()))
            if action < 8:
                pointer, restarted = action, False
                count += 1
            elif action == 8:
                restarted = False
                count += 1
            else:
                pointer, restarted = 0, True
            rows[-1].update(pointer_after=pointer, execution_after=count,
                            just_restarted_after=restarted)
        if count != 8:
            raise ValueError(f'Incomplete program: P={count}, expected 8')
        return tuple(rows)


P0 = Program('P0', tuple(range(8)))
P1 = Program('P1', (0, 1, 2, 3, 4, 5, 8, 7))
PROGRAMS = (P0, P1)


class ScriptedProvider:
    def __init__(self, program):
        self.program, self.index = program, 0

    def __call__(self, **features):
        if self.index >= len(self.program.actions):
            raise MechanicalViolation('Exhausted script: no fallback is allowed')
        action = self.program.actions[self.index]
        self.index += 1
        logits = features['h'].new_full((1, 11), -100.)
        logits[:, action] = 100.
        return {'logits': logits}


class CompleteToken:
    """Sequential observer around real complete-token calls, including reference.

    No routing, operator or commit implementation is replaced. Metadata below is
    observation/validation, not a second execution implementation.
    """
    def __init__(self, model, writer=None):
        self.model, self.writer = model, writer
        self.completed, self.block_calls, self.commit_calls = 0, 0, 0
        self.gdn_checks, self.plan_purity_checks = Counter(), 0
        self.first_failure = None

    @torch.no_grad()
    def run(self, x, parent, program=P0, *, reference=False, label=0, case=None):
        require(x.shape == (1, 64) and x.dtype == torch.float32, 'Wrong token shape/dtype')
        require(not parent._consumed, 'Cannot continue a consumed state')
        program.validate()
        model, rows, handles = self.model, [], []
        provider = ScriptedProvider(program)
        parent_values, parent_versions = gpu_snapshot(parent), versions(parent)
        commits, reads, plans = [], [], []
        original_block = model.executor._run_block
        original_mask = executor_module.build_execution_mask
        original_commit = agt_module.terminal_commit
        original_plan = commit_module.build_commit_plan
        original_read = model.memory.read

        def controller(**features):
            if rows:
                rows[-1]['h_after'] = features['h'].detach().clone()
            result = provider(**features)
            rows.append(dict(decision=len(rows), h_before=features['h'].detach().clone(),
                memory_read=features['memory_read'].detach().clone(),
                memory_mean=features['memory_mean'].detach().clone(),
                pointer_one_hot=features['pointer_one_hot'].detach().clone(),
                budget=features['budget'].detach().clone(), logits=result['logits'].detach().clone(),
                actual_blocks=[]))
            require(bool(features['budget'].eq(0).all()), 'Budget/presence changed')
            return result

        def mask(*args, **kwargs):
            result = original_mask(*args, **kwargs)
            row = rows[-1]
            requested = program.actions[row['decision']]
            require(bool(result[requested]), 'Requested program action is illegal')
            selected = int(row['logits'][:, :10].masked_fill(~result[:10], -torch.inf).argmax(-1))
            require(selected == requested, 'Mask caused scripted action fallback')
            p, pointer = kwargs['execution_count'], kwargs['pointer']
            require(p == sum(len(r['actual_blocks']) for r in rows[:-1]), 'Counter/call mismatch')
            expected = program.validate()[row['decision']]
            require(pointer == expected['pointer_before'] and p == expected['execution_before'],
                    'Production pointer/count differs from validated descriptor')
            row.update(action=requested, pointer_before=pointer,
                       pointer_after=expected['pointer_after'], execution_before=p,
                       execution_after=expected['execution_after'],
                       just_restarted_before=kwargs['just_restarted'],
                       just_restarted_after=expected['just_restarted_after'], legal_mask=result.tolist())
            return result

        def check_bank(index, h, h0, memory, context, invoke):
            row = rows[-1]
            target = row['action'] if row['action'] < 8 else row['pointer_before']
            require(row['action'] != 9 and index == target, 'Selection/execution mismatch')
            inputs = gpu_snapshot(dict(h=h, h0=h0, memory=memory))
            context_versions = versions(context)
            result = invoke()
            exact_state(inputs, dict(h=h, h0=h0, memory=memory))
            require(context_versions == versions(context), 'In-flight context mutation')
            require(parent_versions == versions(parent), 'Parent mutation during bank execution')
            row['actual_blocks'].append(index)
            row['h_after'], row['h0'] = result.detach().clone(), h0.detach().clone()
            self.block_calls += 1
            if model.blocks.blocks[index].species == 'GDN':
                self.gdn_checks[index] += 1
            return result

        def block(block_index, h, h0, memory, context, **kwargs):
            return check_bank(block_index, h, h0, memory, context,
                              lambda: original_block(block_index, h, h0, memory, context, **kwargs))

        def reference_pre(index):
            def hook(module, args, kwargs):
                d = len(rows)
                require(index == d and index == program.actions[d], 'Reference is not canonical')
                h, h0, memory = args[0].squeeze(1), kwargs['h0'].squeeze(1), kwargs['R'].squeeze(1)
                expected = program.validate()[d]
                rows.append(dict(**expected, h_before=h.detach().clone(), h0=h0.detach().clone(),
                    memory_read=memory.detach().clone(), memory_mean=parent.memory.mean(1).detach().clone(),
                    actual_blocks=[], just_restarted_before=False, legal_mask=expected['mask'],
                    _input_values=gpu_snapshot(dict(args=args, kwargs=kwargs)),
                    _context_versions=versions(kwargs.get('C_prev'))))
            return hook

        def reference_post(index):
            def hook(module, args, kwargs, output):
                row = rows[-1]
                exact_state(row.pop('_input_values'), dict(args=args, kwargs=kwargs))
                require(row.pop('_context_versions') == versions(kwargs.get('C_prev')), 'Reference context mutation')
                require(parent_versions == versions(parent), 'Reference parent mutation')
                row['actual_blocks'] = [index]
                row['h_after'] = (output[0] if isinstance(output, tuple) else output).squeeze(1).detach().clone()
                self.block_calls += 1
                if module.species == 'GDN':
                    self.gdn_checks[index] += 1
            return hook

        def read(*args, **kwargs):
            reads.append(1)
            return original_read(*args, **kwargs)

        def plan(work, **kwargs):
            exact_state(parent_values, work)
            before = gpu_snapshot(work)
            result = original_plan(work, **kwargs)
            exact_state(before, work)
            plans.append(result)
            self.plan_purity_checks += 1
            return result

        def commit(work, **kwargs):
            require(work is not parent, 'Commit attempted on parent')
            exact_state(parent_values, work)
            assert_disjoint(parent, work)
            result = original_commit(work, **kwargs)
            new_state, built = result
            require(built.state_ref is work and built.token_index == parent.tokens_committed, 'Commit plan ownership/index')
            require(len(built.entries) == 7, 'Missing terminal-derived context entry')
            require(new_state.tokens_committed == parent.tokens_committed + 1 and not new_state._consumed,
                    'Returned state count/consumption')
            require(torch.equal(new_state.s, built.s) and torch.equal(new_state.memory, built.memory), 'Commit state/payload mismatch')
            for cache, version, payload in built.entries:
                require(cache.version == version + 1, 'Cache version did not advance once')
                for actual, expected in zip(cache.get_committed_context(), payload):
                    if actual is not None:
                        if actual.shape != expected.shape:
                            actual = actual[:, :, -1:, :]
                        require(torch.equal(actual, expected), 'Terminal cache payload mismatch')
            commits.append(1)
            self.commit_calls += 1
            return result

        def no_candidates(*args, **kwargs):
            raise MechanicalViolation('Unselected candidates executed in deterministic program replay')

        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(model.memory, 'read', read))
                stack.enter_context(patch.object(agt_module, 'terminal_commit', commit))
                stack.enter_context(patch.object(commit_module, 'build_commit_plan', plan))
                stack.enter_context(patch.object(model.executor, 'candidate_successors', no_candidates))
                if reference:
                    require(program == P0, 'Only canonical program has fixed-reference mode')
                    for i, module in enumerate(model.blocks.blocks):
                        handles.extend([module.register_forward_pre_hook(reference_pre(i), with_kwargs=True),
                                        module.register_forward_hook(reference_post(i), with_kwargs=True)])
                else:
                    stack.enter_context(patch.object(model.executor, 'controller', controller))
                    stack.enter_context(patch.object(model.executor, '_run_block', block))
                    stack.enter_context(patch.object(executor_module, 'build_execution_mask', mask))
                compatibility = x.new_zeros(1, 64)
                if reference:
                    out = model.step(x, parent, compatibility)
                    history = [('ADVANCE', i) for i in range(8)]
                else:
                    out = model.adaptive_differentiable_step(x, parent, compatibility,
                        budget=x.new_zeros(1, 2), training=False, tau=1.)
                    rows[-1]['h_after'] = out['h_terminal'].detach().clone()
                    history = out['action_history']
                    require(out['execution_count'] == 8 and not out['halted'] and not out['safety_cap_hit'],
                            'Incorrect completion')
                    require(provider.index == len(program.actions), 'Script not exactly exhausted')
            require(len(commits) == len(plans) == len(reads) == 1, 'Token requires one read, plan and commit')
            require(len(rows) == len(program.actions) == len(history), 'Decision-history mismatch')
            expected_history = [('ADVANCE', r['action']) if r['action'] < 8 else
                                ('REPEAT', r['pointer_before']) if r['action'] == 8 else ('RESTART', None)
                                for r in rows]
            require(history == expected_history, 'History differs from requested descriptor')
            for row in rows:
                expected = [] if row['action'] == 9 else [row['action'] if row['action'] < 8 else row['pointer_before']]
                require(row['actual_blocks'] == expected, 'Actual calls differ from requested action')
                if row['action'] == 9:
                    require(torch.equal(row['h_before'], row['h_after']), 'RESTART transformed live h')
            require(sum(len(r['actual_blocks']) for r in rows) == 8, 'Actual bank executions are not eight')
            exact_state(parent_values, parent)
            require(parent_versions == versions(parent), 'Parent storage/version identity changed')
            assert_disjoint(parent, out['state'])
            require(all(torch.isfinite(t).all() for t in [out['logits'], out['state'].s, out['state'].memory]), 'Nonfinite output')
            self.completed += 1
            logits = out['logits'].detach().cpu()
            summary = dict(case=case, token_index=parent.tokens_committed, program=program.name,
                reference=reference, P=8, D=len(rows), restart_count=program.actions.count(9),
                repeat_count=program.actions.count(8), executed_blocks=[b for r in rows for b in r['actual_blocks']],
                termination_reason='canonical_sweep_completed' if reference else 'execution_budget_reached',
                commit_count=1, commit_entries=7, memory_reads=1, candidate_constructions=0,
                label=int(label), prediction=int(logits.argmax(-1)), correct=int(logits.argmax(-1)) == int(label),
                CE=float(F.cross_entropy(logits, torch.tensor([label]))), logits=logits.tolist(),
                probabilities=logits.softmax(-1).tolist(), incoming_state_sha256=state_digest(parent),
                outgoing_state_sha256=state_digest(out['state']), state_metadata=state_metadata(out['state']))
            if self.writer is not None:
                serial_rows = [{k: v.detach().cpu().tolist() if torch.is_tensor(v) else v for k, v in row.items()}
                               for row in rows]
                self.writer(dict(**summary, decisions=serial_rows))
            return out, summary, rows
        except Exception as error:
            self.first_failure = dict(case=case, token=parent.tokens_committed, error=repr(error),
                                      observed_actions=[r.get('action') for r in rows],
                                      observed_block_calls=[r.get('actual_blocks') for r in rows])
            raise
        finally:
            for handle in handles:
                handle.remove()


def input_inventory(directory):
    """Named raw sequence records, with bounded project-only provenance."""
    excluded, records, unknown = set(), [], []
    paths = sorted(set((ROOT / 'experiments/results').rglob('*.json')) |
                   set((ROOT / 'experiments/runs').glob('*.json')) |
                   set((ROOT / 'experiments/results').rglob('*.jsonl')) |
                   set((ROOT / 'experiments/results').rglob('*.jsonl.gz')))
    for path in paths:
        if directory in path.parents:
            continue
        found = []
        try:
            if path.name.endswith('.jsonl.gz') or path.suffix == '.jsonl':
                opener = gzip.open if path.name.endswith('.gz') else open
                with opener(path, 'rt') as handle:
                    for n, line in enumerate(handle):
                        if line.strip():
                            found.extend(recorded_bits(json.loads(line), f'line/{n+1}'))
            else:
                found = recorded_bits(json.loads(path.read_text()))
        except (ValueError, UnicodeError) as error:
            unknown.append(dict(path=str(path.relative_to(ROOT)), reason=str(error)))
        unique = sorted({b for _, b in found})
        excluded.update(unique)
        records.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                            recovered_occurrences=len(found), sequences=[list(b) for b in unique]))
    return excluded, records, unknown


def materialize(directory):
    require(not (directory / 'input_manifest.json').exists(), 'Inputs already sealed; do not overwrite')
    excluded, inventory, unknown = input_inventory(directory)
    task, rows, attempts, used = TaskAParity(64, 'cpu'), [], [], set()
    for partition, length, seed in [('acquisition', 16, 5100), ('confirmation', 16, 5101),
                                    ('longer_confirmation', 24, 5102)]:
        split = 'val' if length == 16 else 'test'
        task._eval_generators[split].manual_seed(seed)
        if sum(1 for b in excluded | used if len(b) == length and sum(b) % 2 == 0) > 2 ** (length - 1) - 16 or \
           sum(1 for b in excluded | used if len(b) == length and sum(b) % 2 == 1) > 2 ** (length - 1) - 16:
            raise MechanicalViolation(f'Input exclusion blocker at length {length}')
        balance = Counter()
        while sum(balance.values()) < 32:
            batch = task.make_batch(1, length, split)
            matches = (batch.x.unsqueeze(-2) == task.token_features.view(1, 1, 2, 64)).all(-1)
            require(bool(matches.sum(-1).eq(1).all()), 'Encoding ambiguous')
            bits = matches.long().argmax(-1).flatten().tolist()
            label, key = int(batch.target), tuple(bits)
            reason = ('known_exact_input_overlap' if key in excluded else 'partition_duplicate' if key in used else
                      'parity_quota_full' if balance[label] >= 16 else 'accepted')
            attempts.append(dict(partition=partition, bits=bits, label=label, reason=reason))
            if reason != 'accepted':
                continue
            suffixes = ACQUISITION_SUFFIXES if partition == 'acquisition' else CONFIRMATION_SUFFIXES
            seq_id = f'{partition}/L{length}/row{sum(balance.values()):02d}'
            full = [tuple(bits + [int(b) for b in suffix]) for suffix in suffixes]
            overlaps = [dict(suffix=suffix, exact_complete_overlap=sequence in excluded,
                             shared_prefix_occurrences=sum(old[:length] == key for old in excluded if len(old) >= length))
                        for suffix, sequence in zip(suffixes, full)]
            rows.append(dict(sequence_id=seq_id, partition=partition, length=length, seed=seed,
                bits=bits, label=label, bits_sha256=hashlib.sha256(bytes(bits)).hexdigest(),
                encoded_sha256=tensor_hash(batch.x), suffixes=list(suffixes), overlaps=overlaps))
            used.add(key)
            balance[label] += 1
    require(len(rows) == len(used) == 96 and not (used & excluded), 'Input panel exclusion/size failure')
    tensors = dict(token_features=task.token_features, prefixes={r['sequence_id']:task.token_features[torch.tensor(r['bits'])].unsqueeze(0) for r in rows},
                   suffixes={s:task.token_features[torch.tensor([int(b) for b in s])].unsqueeze(0)
                             for s in ACQUISITION_SUFFIXES + CONFIRMATION_SUFFIXES})
    with (directory / 'encoded_inputs.pt').open('xb') as handle:
        torch.save(tensors, handle)
    write_json(directory / 'prior_input_inventory.json', dict(records=inventory, unknown=unknown,
        excluded_contents=len(excluded), limitations=['Unknown/unretained Stage-1 exposure',
            'Named raw binary arrays extracted; unrecorded or unreconstructable generator exposure remains unknown',
            'Shared suffix structure is deliberate; shared-prefix and complete overlaps are reported separately']))
    write_json(directory / 'generation_attempts.json', attempts)
    write_json(directory / 'input_manifest.json', dict(rows=rows, token_feature_seed=3002,
        generator='TaskAParity.make_batch; original CPU torch.randint sampler; val/test generator explicitly seeded',
        sequence_seeds=[5100,5101,5102], balanced_parity_each_partition={0:16,1:16},
        original_control=dict(bits=[0,1,1,0,1,1,0,0], expected_CE=0.0014546061865985394),
        encoded_inputs_sha256=sha(directory / 'encoded_inputs.pt'),
        inventory_sha256=sha(directory / 'prior_input_inventory.json'),
        attempts_sha256=sha(directory / 'generation_attempts.json'),
        acquisition_suffixes=ACQUISITION_SUFFIXES, confirmation_suffixes=CONFIRMATION_SUFFIXES,
        horizons=HORIZONS, future_observations='all indices1..8; primary1,4,8', schedules=SCHEDULES,
        original_control_in_new_prefix_panel=False, overlap_rule='Reject exact retained input matches and panel duplicates; report other overlaps'))
    write_json(directory / 'protocol_manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        protocol='Latest finalized conversation specification; frozen two-program feasibility only',
        git=git_identity(), backbone=dict(path=str(BACKBONE.relative_to(ROOT)),sha256=BACKBONE_SHA),
        programs={p.name:dict(actions=p.actions, validation=p.validate()) for p in PROGRAMS},
        numerical=dict(batch=1,dtype='float32',device='cuda',budget=[0.,0.],presence=0,HALT=False,convergence=False,
                       ST_Gumbel=False,optimizer=None,selector=None,trainable_parameters=0),
        thresholds=dict(atol=ATOL,rtol=RTOL,no_additional_parity_errors=True,
            single_positive_CE_max=.05,mean_positive_CE_upper_bound=.01,total_variation_max=.01,
            selector_authorization='all declared current/future fixture answers correct, both programs safe and informative'),
        uncertainty=dict(replicates=10000,seed=5103,resampling_unit='prefix ID within parity; all conditions/suffixes together',
            acceptance='mean per-prefix worst positive mean-suffix CE deterioration; one-sided97.5% bound for each confirmation panel',
            per_prefix_statistic='maximum of positive current CE difference and positive mean-suffix future CE differences across history, schedule, and primary horizon',
            limitations='conditional on frozen model, fixed structured suffix bank and sampled prefix population; zero errors do not establish zero risk'),
        gate_order=['P0 archival equivalence','mechanical/isolation','current correctness','future/switch contract','alternative usefulness'],
        history_conditions=[0,1],boundary_programs=[0,1],max_completed_tokens=40848,
        maximum_future_continuations=4608,complete_future_tokens=36864,
        input_manifest_sha256=sha(directory / 'input_manifest.json'),
        research_source_sha256=sha(__file__),focused_tests_sha256=sha(ROOT/'tests/test_complete_program_feasibility.py'),
        no_training=True, historical_integrity_exception=json.loads((directory/'integrity_before.json').read_text())['historical_exception']))
    print('SEALED INPUTS: 96 distinct prefixes; 16/16 parity each partition.', flush=True)


def load_frozen(device):
    require(sha(BACKBONE) == BACKBONE_SHA, 'Backbone file hash mismatch')
    checkpoint = torch.load(BACKBONE, map_location='cpu', weights_only=True)
    weights = {k:v for k,v in checkpoint['state_dict'].items() if not k.startswith('controller.')}
    torch.manual_seed(3305)
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=2)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    require(set(missing) == {k for k in model.state_dict() if k.startswith('controller.')} and not unexpected,
            'Backbone keys mismatch; no silent fills')
    model.to(device=device, dtype=torch.float32).eval()
    model.executor.max_execution_depth = 8
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def compare_outputs(left, right, left_rows, right_rows):
    result = dict(first_state_divergence=close_difference(left['state'], right['state']),
                  first_logit_divergence=close_difference(left['logits'], right['logits']))
    for index, (a,b) in enumerate(zip(left_rows,right_rows)):
        difference = close_difference({k:a[k] for k in ['h_before','h_after','h0','memory_read','memory_mean']},
                                      {k:b[k] for k in ['h_before','h_after','h0','memory_read','memory_mean']})
        if difference:
            result['first_transition_divergence'] = dict(decision=index, **difference)
            break
    else:
        result['first_transition_divergence'] = None
    result['metadata_equal'] = all({k:a[k] for k in ['action','pointer_before','pointer_after','execution_before','execution_after','actual_blocks']} ==
                                  {k:b[k] for k in ['action','pointer_before','pointer_after','execution_before','execution_after','actual_blocks']}
                                  for a,b in zip(left_rows,right_rows)) and len(left_rows)==len(right_rows)
    return result


def paired_case(left, right, case, horizon):
    lp,rp = torch.tensor(left['probabilities'],dtype=torch.float64),torch.tensor(right['probabilities'],dtype=torch.float64)
    return dict(case=case,horizon=horizon,P0=left,P1=right,
        CE_delta=right['CE']-left['CE'],total_variation=float((lp-rp).abs().sum()*.5),
        additional_error=left['correct'] and not right['correct'],
        corrected_reference_error=not left['correct'] and right['correct'],
        shared_error=not left['correct'] and not right['correct'],
        prediction_disagreement=left['prediction'] != right['prediction'])


def bootstrap_prefix_worst(rows, prefixes):
    by_id={p['sequence_id']:[] for p in prefixes}
    for row in rows:
        by_id[row['sequence_id']].append(row)
    values=[]
    for prefix in prefixes:
        cells=by_id[prefix['sequence_id']]
        current=[r['CE_delta'] for r in cells if r['horizon']==0]
        grouped={}
        for r in cells:
            if r['horizon'] in HORIZONS:
                key=(r['history_program'],r['schedule'],r['horizon'])
                grouped.setdefault(key,[]).append(r['CE_delta'])
        require(len(current)==2 and len(grouped)==18 and all(len(v)==4 for v in grouped.values()),
                'Incomplete paired contract cells')
        values.append(max([0.]+current+[sum(v)/len(v) for v in grouped.values()]))
    return values


def uncertainty_table(comparisons, prefixes):
    generator=torch.Generator(device='cpu').manual_seed(5103)
    results={}
    for partition in ['acquisition','confirmation','longer_confirmation']:
        panel=[p for p in prefixes if p['partition']==partition]
        values=torch.tensor(bootstrap_prefix_worst(comparisons,panel),dtype=torch.float64)
        groups=[[i for i,p in enumerate(panel) if p['label']==label] for label in [0,1]]
        ids=torch.cat([torch.tensor(g)[torch.randint(len(g),(10000,len(g)),generator=generator)] for g in groups],1)
        samples=values[ids].mean(1)
        results[partition]=dict(prefix_count=len(panel),per_prefix_worst=values.tolist(),
            mean=float(values.mean()),bootstrap_95=torch.quantile(samples,torch.tensor([.025,.975],dtype=torch.float64)).tolist(),
            one_sided_97_5_upper=float(torch.quantile(samples,.975)),
            accepted_upper_bound=bool(torch.quantile(samples,.975)<=.01),
            constant_statistic=bool(values.eq(values[0]).all()),
            caveat='Approximate conditional prefix bootstrap; constant outcomes do not prove population equivalence')
    return results


def check_integrity(directory):
    baseline=json.loads((directory/'integrity_before.json').read_text())
    changed=[p for p,h in baseline['protected_files'].items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    result=dict(protected_count=len(baseline['protected_files']),changed=changed,
        tracked_diff_unchanged=git_identity()['tracked_diff_sha256']==baseline['tracked_diff_sha256'],
        historical_exception=baseline['historical_exception'])
    require(not changed and result['tracked_diff_unchanged'],'Protected artifacts/source changed')
    return result


@torch.no_grad()
def execute(directory, device='cuda'):
    manifest=json.loads((directory/'protocol_manifest.json').read_text())
    inputs=json.loads((directory/'input_manifest.json').read_text())
    require(sha(directory/'input_manifest.json')==manifest['input_manifest_sha256'],'Input manifest changed')
    require(sha(directory/'encoded_inputs.pt')==inputs['encoded_inputs_sha256'],'Encoded inputs changed')
    require(sha(__file__)==manifest['research_source_sha256'],'Research implementation changed after sealing')
    require(not (directory/'token_traces.jsonl.gz').exists(),'Refusing repeat/overwrite')
    check_integrity(directory)
    tensors=torch.load(directory/'encoded_inputs.pt',map_location='cpu',weights_only=True)
    torch.set_num_threads(4)
    model=load_frozen(torch.device(device))
    before=gpu_snapshot(model.state_dict())
    torch_cpu_rng=torch.get_rng_state().clone()
    torch_cuda_rng=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    started=time.monotonic()
    comparisons,current,roots,canonical_checks=[],[],[],[]
    failure=None;archival=None;runner=None
    with gzip.open(directory/'token_traces.jsonl.gz','xt') as trace:
        def record(value):
            trace.write(json.dumps(value,allow_nan=False)+'\n')
        runner=CompleteToken(model,record)
        try:
            # Gate 1 runs before any P1 forward, using the original audited input.
            bits=inputs['original_control']['bits'];state_ref=model.initial_persistent_state(1,device,torch.float32)
            state_adapter=state_ref.clone();control=[]
            prior=json.loads((ROOT/'experiments/results/gatec_execution_accounting_v1/gatec_canonical_replay.json').read_text())['paths']['A']
            for token,bit in enumerate(bits):
                x=tensors['token_features'][bit].to(device).unsqueeze(0);label=sum(bits[:token+1])%2
                a,sa,ar=runner.run(x,state_ref,P0,reference=True,label=label,case=dict(gate='archival',arm='reference',token=token))
                b,sb,br=runner.run(x,state_adapter,P0,label=label,case=dict(gate='archival',arm='adapter',token=token))
                comparison=compare_outputs(a,b,ar,br)
                require(comparison['metadata_equal'] and all(comparison[k] is None for k in ['first_state_divergence','first_logit_divergence','first_transition_divergence']),
                        f'P0 adapter equivalence failed at token {token}: {comparison}')
                torch.testing.assert_close(a['logits'].cpu(),torch.tensor(prior['tokens'][token]['logits']),atol=ATOL,rtol=RTOL)
                for d,row in enumerate(ar):
                    old=prior['steps'][token*8+d]
                    torch.testing.assert_close(row['h_before'].cpu(),torch.tensor(old['h_before']),atol=ATOL,rtol=RTOL)
                    torch.testing.assert_close(row['h_after'].cpu(),torch.tensor(old['h']),atol=ATOL,rtol=RTOL)
                control.append(dict(token=token,comparison=comparison,reference=sa,adapter=sb))
                state_ref,state_adapter=a['state'],b['state']
            require(math.isclose(control[-1]['reference']['CE'],inputs['original_control']['expected_CE'],abs_tol=ATOL,rel_tol=RTOL),
                    'Archived control CE mismatch')
            archival=dict(status='VERIFIED',transitions=64,final_CE=control[-1]['reference']['CE'],control=control)
            write_json(directory/'archival_gate.json',archival)
            print('GATE 1 VERIFIED: P0 archival equivalence; CE',archival['final_CE'],flush=True)

            for index,prefix in enumerate(inputs['rows']):
                sequence_id=prefix['sequence_id'];raw=prefix['bits'];x=tensors['prefixes'][sequence_id].to(device)
                incoming=[]
                for hist in [0,1]:
                    state=model.initial_persistent_state(1,device,torch.float32)
                    for token in range(prefix['length']-1):
                        out,_,_=runner.run(x[:,token],state,PROGRAMS[hist],reference=hist==0,
                            label=sum(raw[:token+1])%2,case=dict(sequence_id=sequence_id,phase='prefix',history_program=hist,token=token))
                        state=out['state']
                    incoming.append(state)
                assert_disjoint(incoming[0],incoming[1])
                for hist,parent in enumerate(incoming):
                    parent_copy=gpu_snapshot(parent)
                    sibling_copy=gpu_snapshot(incoming[1-hist])
                    parent_hash=state_digest(parent)
                    states,summaries,rowsets=[],[],[]
                    for program in PROGRAMS:
                        out,summary,rows=runner.run(x[:,-1],parent,program,label=prefix['label'],
                            case=dict(sequence_id=sequence_id,phase='boundary',history_program=hist,program=program.name))
                        exact_state(parent_copy,parent);exact_state(sibling_copy,incoming[1-hist])
                        if states:
                            assert_disjoint(states[0],out['state'])
                        states.append(out['state']);summaries.append(summary);rowsets.append(rows)
                        roots.append(dict(sequence_id=sequence_id,history_program=hist,program=program.name,
                                          incoming_sha256=parent_hash,outgoing_sha256=state_digest(out['state'])))
                    # Supplied-state P0 must also agree with existing reference from P1 histories.
                    ref,refsummary,refrows=runner.run(x[:,-1],parent,P0,reference=True,label=prefix['label'],
                        case=dict(sequence_id=sequence_id,phase='P0_supplied_state_check',history_program=hist))
                    equivalent=compare_outputs(dict(state=states[0],logits=torch.tensor(summaries[0]['logits'])),ref,rowsets[0],refrows)
                    require(equivalent['metadata_equal'] and all(equivalent[k] is None for k in ['first_state_divergence','first_logit_divergence','first_transition_divergence']),
                            'P0 supplied-state reference mismatch')
                    canonical_checks.append(dict(sequence_id=sequence_id,history_program=hist,comparison=equivalent))
                    paired=paired_case(summaries[0],summaries[1],dict(sequence_id=sequence_id,history_program=hist),0)
                    paired.update(sequence_id=sequence_id,partition=prefix['partition'],history_program=hist,schedule=None,suffix=None)
                    comparisons.append(paired);current.append(paired)
                    # Store the actual complete incoming and committed root states, not detached reconstruction substitutes.
                    with (directory/f'states_{index:02d}_history{hist}.pt').open('xb') as handle:
                        torch.save(dict(incoming=parent.clone(),P0=states[0].clone(),P1=states[1].clone()),handle)
                    root_snapshots=[gpu_snapshot(s) for s in states]
                    for suffix in prefix['suffixes']:
                        sx=tensors['suffixes'][suffix].to(device);suffix_bits=[int(b) for b in suffix]
                        for schedule in SCHEDULES:
                            branch=[s.clone() for s in states]
                            assert_disjoint(branch[0],branch[1])
                            for h in range(1,9):
                                program_id=0 if schedule=='P0' else 1 if schedule=='P1' else (h-1)%2
                                summaries_future=[]
                                for root_program in [0,1]:
                                    other_values=gpu_snapshot(branch[1-root_program])
                                    label=(sum(raw)+sum(suffix_bits[:h]))%2
                                    out,sm,_=runner.run(sx[:,h-1],branch[root_program],PROGRAMS[program_id],reference=program_id==0,
                                        label=label,case=dict(sequence_id=sequence_id,phase='future',history_program=hist,
                                            root_program=root_program,suffix=suffix,schedule=schedule,horizon=h))
                                    exact_state(other_values,branch[1-root_program])
                                    branch[root_program]=out['state'];summaries_future.append(sm)
                                comparison=paired_case(*summaries_future,dict(sequence_id=sequence_id,history_program=hist,
                                    suffix=suffix,schedule=schedule),h)
                                comparison.update(sequence_id=sequence_id,partition=prefix['partition'],history_program=hist,
                                                  suffix=suffix,schedule=schedule)
                                comparisons.append(comparison)
                            for s,values in zip(states,root_snapshots):
                                exact_state(values,s)
                    exact_state(parent_copy,parent);exact_state(sibling_copy,incoming[1-hist])
                if (index+1)%4==0:
                    trace.flush()
                    print(f'PREFIX {index+1}/96 COMPLETE: {runner.completed} token transitions; elapsed {time.monotonic()-started:.1f}s',flush=True)
            require(runner.completed==40848,'Completed transition budget differs from specification')
            require(runner.block_calls==8*runner.completed and runner.commit_calls==runner.completed,'Global bank/commit accounting')
            exact_state(before,model.state_dict())
            require(all(p.grad is None and not p.requires_grad for p in model.parameters()),'Unexpected parameter gradient/update')
            require(torch.equal(torch_cpu_rng,torch.get_rng_state()),'Evaluation consumed unsealed RNG')
            require(torch_cuda_rng is None or torch.equal(torch_cuda_rng,torch.cuda.get_rng_state()),'Evaluation consumed CUDA RNG')
        except Exception as error:
            failure=dict(error=repr(error),traceback=traceback.format_exc(),first_mechanical=runner.first_failure,
                         completed_tokens=runner.completed,elapsed_seconds=time.monotonic()-started)
            write_json(directory/'failure.json',failure)
        finally:
            integrity=check_integrity(directory)
            exact_state(before,model.state_dict())
            write_json(directory/'integrity_after.json',dict(**integrity,parameters_unchanged=True,
                no_optimizer=True,no_training=True,no_gradients=all(p.grad is None for p in model.parameters())))
    if failure:
        write_json(directory/'results.json',dict(status='STOPPED',failure=failure,archival_gate=archival,
            completed_tokens=runner.completed,unexecuted_conditions='all conditions after first failure',integrity=integrity))
        (directory/'report.md').write_text('# Two-program feasibility: STOPPED\n\n'+json.dumps(failure,indent=2)+'\n\nNo training, selector or follow-up acquisition was run.\n')
        return
    bootstrap=uncertainty_table(comparisons,inputs['rows'])
    failures=[dict(sequence_id=r['sequence_id'],history_program=r['history_program'],horizon=r['horizon'],
        suffix=r['suffix'],schedule=r['schedule'],CE_delta=r['CE_delta'],total_variation=r['total_variation'],
        additional_error=r['additional_error']) for r in comparisons if r['additional_error'] or r['CE_delta']>.05 or r['total_variation']>.01]
    absolute_errors=[r for r in comparisons if not r['P0']['correct'] or not r['P1']['correct']]
    result=dict(status='REJECTED' if failures or absolute_errors or not all(bootstrap[k]['accepted_upper_bound'] for k in ['confirmation','longer_confirmation']) else 'BOUNDED_FEASIBILITY',
        archival_gate=dict(status=archival['status'],transitions=64,final_CE=archival['final_CE']),
        mechanical=dict(status='VERIFIED',completed_tokens=runner.completed,actual_bank_calls=runner.block_calls,
            commits=runner.commit_calls,plan_purity_checks=runner.plan_purity_checks,GDN_checks=dict(runner.gdn_checks),
            candidate_constructions=0,parent_and_sibling_isolation=True),
        current_task=dict(cases=len(current),P0_accuracy=sum(r['P0']['correct'] for r in current)/len(current),
            P1_accuracy=sum(r['P1']['correct'] for r in current)/len(current),
            P0_mean_CE=sum(r['P0']['CE'] for r in current)/len(current),P1_mean_CE=sum(r['P1']['CE'] for r in current)/len(current),
            additional_errors=sum(r['additional_error'] for r in current)),
        future=dict(paired_cases=len(comparisons)-len(current),additional_errors=sum(r['additional_error'] for r in comparisons if r['horizon']>0),
                    shared_errors=sum(r['shared_error'] for r in comparisons if r['horizon']>0)),
        violations=dict(cases=len(failures),absolute_error_cases=len(absolute_errors),first=failures[0] if failures else None,
            maximum_positive_CE_delta=max(0.,max(r['CE_delta'] for r in comparisons)),maximum_TV=max(r['total_variation'] for r in comparisons)),
        bootstrap=bootstrap,elapsed_seconds=time.monotonic()-started,integrity=integrity,
        interpretation='Fixed P1 rejected if any declared criterion fails; no substitution, extra samples or selector training.',
        limitations=['Finite suffixes and schedules do not establish universal equivalence',
            'Unknown archival Stage-1 input exposure remains', 'Full-sequence P1 prefixes and root replacements are distinct interventions',
            'Equal execution counts do not imply equal FLOPs or latency'],no_training=True,no_selector=True,no_commit=True)
    write_json(directory/'results.json',result)
    write_json(directory/'comparisons.json',comparisons)
    write_json(directory/'violations.json',failures)
    write_json(directory/'root_states.json',roots)
    write_json(directory/'P0_supplied_state_checks.json',canonical_checks)
    with (directory/'comparison_table.csv').open('x') as handle:
        handle.write('sequence_id,history,schedule,suffix,horizon,P0_correct,P1_correct,P0_CE,P1_CE,CE_delta,TV\n')
        for r in comparisons:
            handle.write(','.join(map(str,[r['sequence_id'],r['history_program'],r['schedule'],r['suffix'],r['horizon'],
                r['P0']['correct'],r['P1']['correct'],r['P0']['CE'],r['P1']['CE'],r['CE_delta'],r['total_variation']]))+'\n')
    report=['# Frozen two-program feasibility study','',f"Verdict: **{result['status']}**.",'',
        'P0: A0,A1,A2,A3,A4,A5,A6,A7. P1: A0,A1,A2,A3,A4,A5,REPEAT,A7.',
        'No optimizer, training, selector, ST sampling, parameter updates, production edits or commit.', '',
        f"Archival P0 equivalence VERIFIED; 64 transitions; original-control CE {archival['final_CE']:.17g}.",
        f"Mechanics VERIFIED: {runner.completed} completed tokens, {runner.block_calls} selected bank calls, {runner.commit_calls} commits.",
        'Each branch has P8/D8; P1 repeats GDN5 and omits FFN6. Parent and siblings remain unchanged. All seven commit entries and GDN buffers are checked.', '',
        '## Current-task results','',json.dumps(result['current_task'],indent=2),'',
        '## Future and switching results','',json.dumps(result['future'],indent=2),'',json.dumps(result['violations'],indent=2),'',
        '## Conditional uncertainty','',json.dumps(bootstrap,indent=2),'',
        'Bootstrap unit: prefix within parity. All branches, suffixes, horizons and schedules retained together. 10,000 resamples, seed5103.',
        'A constant binary outcome/degenerate interval does not prove zero population risk.', '',
        '## Interpretation','',
        'Current correctness, committed-state compatibility and selector usefulness are separate claims. Any failed threshold rejects this fixed candidate.',
        'Reference errors are retained and prevent the absolute task-correctness gate; they are not automatically P1-induced failures.',
        'P1 numerical/state differences alone do not show useful diversity. No replacement program or acquisition is authorized.', '',
        '## Provenance and limitations','',
        'Input contents and encodings were sealed before any program evaluation. Exact retained input matches and duplicates were rejected; other overlaps were recorded.',
        'Unknown Stage-1 exposure and the archived bytecode integrity exception remain disclosed and unchanged.',
        'Finite tests do not prove universal state equivalence, novelty, selector benefit or task-loss discovery.', '',
        'Protected-file and full-parameter verification: VERIFIED. See integrity_after.json.',
        f'Elapsed evaluation time: {result["elapsed_seconds"]:.1f}s.']
    with (directory/'report.md').open('x') as handle:
        handle.write('\n'.join(report)+'\n')
    print('FINAL VERDICT:',result['status'],'; current P1 accuracy',result['current_task']['P1_accuracy'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['seal','run'])
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    args=parser.parse_args()
    if args.phase=='seal':
        materialize(args.output_dir)
        for source in [Path(__file__),ROOT/'tests/test_complete_program_feasibility.py']:
            target=args.output_dir/source.name
            with target.open('xb') as out,source.open('rb') as original:
                shutil.copyfileobj(original,out)
    else:
        execute(args.output_dir,args.device)


if __name__=='__main__':
    main()
