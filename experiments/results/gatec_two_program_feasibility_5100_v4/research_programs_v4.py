"""Research-only, frozen two-program feasibility study. No training or selector.

All execution, masking, cloning and commits are calls to existing primitives.
Observers are sequential and restored in finally blocks; they never repair routes.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
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
from experiments.audit_gatec_canonical_replay import first_difference, snapshot, concise
from experiments.run_gatec_paired_task_maintenance import refs, versions, gpu_snapshot, assert_same
from experiments.run_gatec_fresh_task_discovery import recorded_bits
from experiments.replay_gatec_learned_policy import BACKBONE, BACKBONE_SHA, git_identity

OUT = ROOT / 'experiments/results/gatec_two_program_feasibility_5100_v4'
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
        archival_commit = {}
        persistent_summary = concise(snapshot(parent))
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
            selected = int(row['logits'][:, :10].masked_fill(~result[:10].to(device=row['logits'].device), -torch.inf).argmax(-1))
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
            row['persistent'] = persistent_summary
            row['context'] = concise(snapshot(context))
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
                    _context_versions=versions(kwargs.get('C_prev')),
                    persistent=persistent_summary, context=concise(snapshot(parent.ctx))))
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
            archival_commit['commit_pre'] = concise(snapshot(work))
            archival_commit['terminal'] = kwargs['h_terminal'].squeeze(1).detach().cpu().clone()
            result = original_commit(work, **kwargs)
            archival_commit['commit'] = concise(snapshot(result[0]))
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
            summary['archival_commit'] = {k:v.tolist() if torch.is_tensor(v) else v for k,v in archival_commit.items()}
            return out, summary, rows
        except Exception as error:
            self.first_failure = dict(case=case, token=parent.tokens_committed, error=repr(error),
                                      observed_actions=[r.get('action') for r in rows],
                                      observed_block_calls=[r.get('actual_blocks') for r in rows])
            raise
        finally:
            for handle in handles:
                handle.remove()


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
