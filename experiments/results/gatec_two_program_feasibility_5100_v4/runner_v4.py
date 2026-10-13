"""V4 locked, no-update study. One invocation, explicit sequential barriers."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DIR))
import gzip
import json
import math
import time
import traceback
from collections import Counter
import torch
from research_programs_v4 import (
    CompleteToken, P0, P1, PROGRAMS, require, sha, write_json, load_frozen,
    compare_outputs, paired_case, uncertainty_table, gpu_snapshot, exact_state,
    state_digest, assert_disjoint, check_integrity, tensor_hash, ATOL, RTOL,
    SCHEDULES, HORIZONS, BACKBONE, MechanicalViolation,
)
from archival_monitor_v4 import reconstruct, observe

V1 = ROOT / 'experiments/results/gatec_two_program_feasibility_5100_v1'
ARCHIVE = ROOT / 'experiments/results/gatec_execution_accounting_v1/gatec_canonical_replay.json'


def strict_compare(expected, actual, path='root'):
    """Complete schema, exact discrete/null topology, locked floating tolerance."""
    if expected is None:
        require(actual is None, f'{path}: null mismatch')
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and expected.keys() == actual.keys(), f'{path}: missing/unmatched keys')
        for key in expected:
            strict_compare(expected[key], actual[key], f'{path}/{key}')
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(expected) == len(actual), f'{path}: shape/record length')
        for index, (a, b) in enumerate(zip(expected, actual)):
            strict_compare(a, b, f'{path}/{index}')
    elif isinstance(expected, float):
        require(isinstance(actual, (float, int)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isclose(expected, actual, abs_tol=ATOL, rel_tol=RTOL),
                f'{path}: numeric mismatch expected={expected!r} actual={actual!r}')
    else:
        require(type(expected) is type(actual) and expected == actual,
                f'{path}: discrete mismatch expected={expected!r} actual={actual!r}')


class Barriers:
    def __init__(self):
        self.archival = None
        self.p0_current = None
        self.all_current = None
        self.future_calls = 0

    def require_p1_history(self):
        require(self.archival is True and self.p0_current is True,
                'P1 history requires archival and all P0→P0 current acceptance')

    def future(self, callback):
        require(self.archival is True and self.p0_current is True and self.all_current is True,
                'Future work blocked by failed, missing or unresolved current prerequisite')
        self.future_calls += 1
        return callback()


class CurrentRejection(RuntimeError):
    pass


def screen(summary, *, paired=None):
    checks = dict(current_correct=bool(summary['correct']),
        executions=summary['P'] == 8, commits=summary['commit_count'] == 1)
    if paired is not None:
        checks.update(no_additional_error=not paired['additional_error'],
            positive_CE=paired['CE_delta'] <= .05,
            total_variation=paired['total_variation'] <= .01)
    return dict(checks=checks, accepted=all(checks.values()))


def as_list(tensor):
    return tensor.detach().cpu().tolist()


def archival_records(model, summary, rows):
    """Exact original schema; monitor output supplied separately, never script logits."""
    steps = []
    for row in rows:
        onehot = row['h_before'].new_zeros(1, 9)
        onehot[:, row['pointer_before']] = 1
        prepared = model.controller._prepare_inputs(h=row['h_before'],
            memory_read=row['memory_read'], memory_mean=row['memory_mean'],
            pointer_one_hot=onehot, budget=row['h_before'].new_zeros(1, 2))
        steps.append(dict(token=summary['token_index'], decision_count=row['decision'],
            execution_count=row['execution_before'], pointer_before=row['pointer_before'],
            selected_action=row['action'], pointer_after=row['pointer_after'],
            just_restarted=row['just_restarted_before'],
            legal_actions=[i for i, legal in enumerate(row['legal_mask']) if legal],
            h_before=as_list(row['h_before']), h=as_list(row['h_after']),
            h0=as_list(row['h0']), memory_read=as_list(row['memory_read']),
            memory_mean=as_list(row['memory_mean']), features=as_list(prepared),
            persistent=row['persistent'], context=row['context']))
    commit = summary['archival_commit']
    token = dict(token=summary['token_index'], terminal=commit['terminal'],
        commit=commit['commit'], commit_pre=commit['commit_pre'], logits=summary['logits'],
        prediction=summary['prediction'], ce=summary['CE'], commit_count=summary['commit_count'],
        commit_entries=summary['commit_entries'], read_calls=summary['memory_reads'],
        decision_count=summary['D'], execution_count=summary['P'],
        candidate_block_calls=len(summary['executed_blocks']))
    return steps, token


def equivalent(a, b, ar, br):
    require(len(ar) == len(br) == 8, 'Within-v4 truncated transition record')
    comparison = compare_outputs(a, b, ar, br)
    require(comparison['metadata_equal'] and all(comparison[k] is None for k in
        ['first_state_divergence', 'first_logit_divergence', 'first_transition_divergence']),
        f'Within-v4 full reference/adapter mismatch: {comparison}')
    # Masks, flags and original feature operands are also same-function targets.
    for left, right in zip(ar, br):
        strict_compare(left['legal_mask'], right['legal_mask'], 'within_v4/mask')
        strict_compare(left['just_restarted_before'], right['just_restarted_before'], 'within_v4/restart')
    return comparison


def verify_retained_inputs(inputs, tensors):
    require(len(inputs['rows']) == 96 and inputs['horizons'] == list(HORIZONS), 'Locked input scope')
    require(inputs['schedules'] == list(SCHEDULES), 'Locked schedules differ')
    for row in inputs['rows']:
        require(tensor_hash(tensors['prefixes'][row['sequence_id']]) == row['encoded_sha256'],
                f"Retained encoded input mismatch {row['sequence_id']}")
    counts = Counter(row['partition'] for row in inputs['rows'])
    require(counts == dict(acquisition=32, confirmation=32, longer_confirmation=32), 'Panel counts')


@torch.no_grad()
def execute():
    require(not (DIR/'study_started.json').exists(), 'Attempt already started; no resume')
    seal = json.loads((DIR/'implementation_seal.json').read_text())
    for path, digest in seal['files'].items():
        require(sha(DIR/path) == digest, f'Sealed v4 implementation changed: {path}')
    check_integrity(DIR)
    require(torch.cuda.is_available(), 'Required CUDA environment unavailable')
    test = json.loads((DIR/'test_results.json').read_text())
    require(test['focused_exit_code'] == 0 and test['substrate_exit_code'] == 0
            and test['cuda_executed'] is True, 'Required test gate not passed')
    original = json.loads((V1/'protocol_manifest.json').read_text())
    inputs = json.loads((V1/'input_manifest.json').read_text())
    require(sha(V1/'input_manifest.json') == original['input_manifest_sha256'], 'Input manifest identity')
    require(sha(V1/'encoded_inputs.pt') == inputs['encoded_inputs_sha256'], 'Encoded input identity')
    tensors = torch.load(V1/'encoded_inputs.pt', map_location='cpu', weights_only=True)
    verify_retained_inputs(inputs, tensors)
    torch.set_num_threads(4)
    model = load_frozen(torch.device('cuda'))
    weights = gpu_snapshot(model.state_dict())
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = [torch.cuda.get_rng_state(i).clone() for i in range(torch.cuda.device_count())]
    start = time.monotonic()
    write_json(DIR/'study_started.json', dict(device='cuda', torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        backbone_sha256=sha(BACKBONE), input_manifest_sha256=sha(V1/'input_manifest.json'),
        encoded_inputs_sha256=sha(V1/'encoded_inputs.pt'), protocol=original,
        amendments_sha256=sha(DIR/'protocol_amendment.md'), batch=1, dtype='float32',
        budget=[0,0], trainable_parameters=0, no_optimizer=True))
    gate = Barriers()
    current, comparisons, control, ledgers = [], [], [], []
    prefix_states, root_states, root_summaries = {}, {}, {}
    boundary = 'P0 archival equivalence'
    failure = None
    monitor_result = None
    with gzip.open(DIR/'token_traces.jsonl.gz', 'xt') as trace:
        def record(value):
            trace.write(json.dumps(value, allow_nan=False)+'\n')
        runner = CompleteToken(model, record)

        def run(x, parent, program, label, case, reference=False):
            require(runner.completed < original['max_completed_tokens'], 'Original study compute cap')
            row = dict(case=case, program=program.name, reference=reference,
                incoming_token=parent.tokens_committed, status='STARTED')
            ledgers.append(row)
            calls, commits = runner.block_calls, runner.commit_calls
            try:
                result = runner.run(x, parent, program, reference=reference, label=label, case=case)
                row['status'] = 'COMPLETED'
                row['D'], row['P'] = result[1]['D'], result[1]['P']
                return result
            except Exception as error:
                row['status'], row['failure'] = 'FAILED', repr(error)
                raise
            finally:
                row['actual_bank_calls'] = runner.block_calls - calls
                row['actual_commits'] = runner.commit_calls - commits

        def accepted(summary, case, paired=None, canonical=None):
            result = screen(summary, paired=paired)
            current.append(dict(case=case, summary=summary, acceptance=result,
                canonical_same_input=canonical, paired_same_history=paired))
            if not result['accepted']:
                raise CurrentRejection(f'Mandatory current screen failed: {case}: {result}')

        try:
            archive = json.loads(ARCHIVE.read_text())
            # Monitor reproduction uses retained control feature records, not new rollouts.
            monitor_result = dict(status='NOT_REPLAYABLE', role='non-gating monitor diagnostic')
            try:
                monitor = reconstruct('cuda')
                first_mismatch, count = None, 0
                for name in ['A', 'C']:
                    require(len(archive['paths'][name]['steps']) == 64, 'Monitor archive truncated')
                    for index, old in enumerate(archive['paths'][name]['steps']):
                        observed = observe(monitor, old, 'cuda')
                        count += 1
                        try:
                            strict_compare(old['controller_logits'], observed, f'monitor/{name}/{index}')
                        except MechanicalViolation as error:
                            if first_mismatch is None:
                                first_mismatch = dict(path=name, step=index, error=str(error),
                                    expected=old['controller_logits'], actual=observed)
                monitor_result.update(status='REPLAY_MISMATCH' if first_mismatch else 'REPRODUCED',
                    evaluated_records=count, first_mismatch=first_mismatch,
                    parameter_sha256=state_digest(monitor.state_dict()),
                    provenance='original stage3_measurements.load_model(3401, cuda); full constructor RNG ordering; isolated RNG; retained features')
                del monitor
            except Exception as error:
                monitor_result.update(reason=repr(error))
            write_json(DIR/'monitor_diagnostic.json', monitor_result)

            state_a = model.initial_persistent_state(1, 'cuda', torch.float32)
            state_c = state_a.clone()
            bits = inputs['original_control']['bits']
            require(len(bits) == 8, 'Original control length')
            for name in ['A','C']:
                require(len(archive['paths'][name]['steps']) == 64
                    and len(archive['paths'][name]['tokens']) == 8, 'Truncated archived records')
            for token, bit in enumerate(bits):
                x = tensors['token_features'][bit].cuda().unsqueeze(0)
                label = sum(bits[:token+1]) % 2
                a, sa, ar = run(x, state_a, P0, label, dict(phase='archival', arm='A', token=token), True)
                c, sc, cr = run(x, state_c, P0, label, dict(phase='archival', arm='C', token=token))
                full = equivalent(a, c, ar, cr)
                for name, summary, rows in [('A',sa,ar), ('C',sc,cr)]:
                    observed_steps, observed_token = archival_records(model, summary, rows)
                    for decision, observed in enumerate(observed_steps):
                        old = archive['paths'][name]['steps'][token*8+decision]
                        require(set(old) == set(observed) | {'controller_logits'}, 'Required archived step schema changed')
                        strict_compare({k:v for k,v in old.items() if k != 'controller_logits'}, observed,
                                       f'archive/{name}/steps/{token*8+decision}')
                    strict_compare(archive['paths'][name]['tokens'][token], observed_token,
                                   f'archive/{name}/tokens/{token}')
                    with (DIR/'archival_comparisons.jsonl').open('a') as f:
                        f.write(json.dumps(dict(path=name, token=token, historical_full_tensors='AGREED',
                            historical_summaries='AGREED', full_within_v4=full, observed_steps=observed_steps,
                            observed_token=observed_token), allow_nan=False)+'\n')
                control.append(dict(token=token, A=sa, C=sc, full_within_v4=full))
                state_a, state_c = a['state'], c['state']
            require(math.isclose(control[-1]['A']['CE'], inputs['original_control']['expected_CE'],
                                abs_tol=ATOL, rel_tol=RTOL), 'Final archival control CE')
            gate.archival = True
            write_json(DIR/'archival_gate.json', dict(status='ACCEPTED', transitions_per_path=64,
                tokens_per_path=8, final_CE=control[-1]['A']['CE'], full_record_coverage=True,
                historical_full_tensor_agreement=True, historical_summary_agreement=True,
                full_within_v4_reference_adapter_agreement=True, monitor=monitor_result))
            print('P0 ARCHIVAL ACCEPTED:', control[-1]['A']['CE'], flush=True)

            boundary = 'P0→P0 current screen across all panels'
            for index, prefix in enumerate(inputs['rows']):
                sid = prefix['sequence_id']
                x = tensors['prefixes'][sid].cuda()
                state = model.initial_persistent_state(1, 'cuda', torch.float32)
                for t in range(prefix['length']-1):
                    out, _, _ = run(x[:,t], state, P0, sum(prefix['bits'][:t+1])%2,
                        dict(phase='prefix', sequence_id=sid, history_program=0, token=t), True)
                    state = out['state']
                prefix_states[(sid,0)] = state
                case = dict(phase='current', sequence_id=sid, history_program=0, program=0)
                a, sa, ar = run(x[:,-1], state, P0, prefix['label'], case)
                ref, _, rr = run(x[:,-1], state, P0, prefix['label'],
                    dict(phase='supplied_reference', sequence_id=sid, history_program=0), True)
                equivalent(a, ref, ar, rr)
                accepted(sa, case, canonical=sa)
                root_states[(sid,0,0)], root_summaries[(sid,0,0)] = a['state'], sa
                if (index+1)%8 == 0:
                    print(f'P0 CURRENT {index+1}/96: {runner.completed} tokens', flush=True)
            gate.p0_current = True
            write_json(DIR/'p0_current_gate.json', dict(status='ACCEPTED', cases=96, conjunction=True))

            boundary = 'remaining current screen; P1 histories and boundary combinations'
            for index, prefix in enumerate(inputs['rows']):
                gate.require_p1_history()
                sid = prefix['sequence_id']; x = tensors['prefixes'][sid].cuda()
                state = model.initial_persistent_state(1, 'cuda', torch.float32)
                for t in range(prefix['length']-1):
                    out, _, _ = run(x[:,t], state, P1, sum(prefix['bits'][:t+1])%2,
                        dict(phase='prefix', sequence_id=sid, history_program=1, token=t))
                    state = out['state']
                prefix_states[(sid,1)] = state
                assert_disjoint(state, prefix_states[(sid,0)])
                # Same-history numerical contrast; fully canonical target remains separately recorded.
                canonical = root_summaries[(sid,0,0)]
                for hist in [0,1]:
                    parent = prefix_states[(sid,hist)]
                    sibling = prefix_states[(sid,1-hist)]
                    sibling_saved = gpu_snapshot(sibling)
                    if hist == 1:
                        case = dict(phase='current', sequence_id=sid, history_program=hist, program=0)
                        a, sa, ar = run(x[:,-1], parent, P0, prefix['label'], case)
                        ref, _, rr = run(x[:,-1], parent, P0, prefix['label'],
                            dict(phase='supplied_reference', sequence_id=sid, history_program=hist), True)
                        equivalent(a, ref, ar, rr)
                        accepted(sa, case, canonical=canonical)
                        root_states[(sid,hist,0)], root_summaries[(sid,hist,0)] = a['state'], sa
                    case = dict(phase='current', sequence_id=sid, history_program=hist, program=1)
                    b, sb, _ = run(x[:,-1], parent, P1, prefix['label'], case)
                    sa = root_summaries[(sid,hist,0)]
                    paired = paired_case(sa, sb, dict(sequence_id=sid, history_program=hist), 0)
                    paired.update(sequence_id=sid, partition=prefix['partition'], history_program=hist,
                                  schedule=None, suffix=None)
                    comparisons.append(paired)
                    assert_disjoint(root_states[(sid,hist,0)], b['state'])
                    exact_state(sibling_saved, sibling)
                    accepted(sb, case, paired, canonical)
                    root_states[(sid,hist,1)], root_summaries[(sid,hist,1)] = b['state'], sb
                with (DIR/f'committed_states_{index:02d}.pt').open('xb') as f:
                    torch.save(dict(incoming={hist:prefix_states[(sid,hist)].clone() for hist in [0,1]},
                        roots={(hist,prog):root_states[(sid,hist,prog)].clone() for hist in [0,1] for prog in [0,1]}), f)
                if (index+1)%8 == 0:
                    print(f'ALL CURRENT {index+1}/96: {runner.completed} tokens', flush=True)
            require(len(current) == 384 and all(c['acceptance']['accepted'] for c in current), 'Current conjunction incomplete')
            gate.all_current = True
            write_json(DIR/'all_current_gate.json', dict(status='ACCEPTED', cases=384, conjunction=True,
                no_current_bootstrap=True))

            boundary = 'future compatibility'
            for prefix in inputs['rows']:
                sid = prefix['sequence_id']
                for hist in [0,1]:
                    roots = [root_states[(sid,hist,prog)] for prog in [0,1]]
                    root_values = [gpu_snapshot(s) for s in roots]
                    for suffix in prefix['suffixes']:
                        sx = tensors['suffixes'][suffix].cuda()
                        for schedule in SCHEDULES:
                            branch = [s.clone() for s in roots]
                            assert_disjoint(*branch)
                            for h in range(1,9):
                                prog = 0 if schedule == 'P0' else 1 if schedule == 'P1' else (h-1)%2
                                future = []
                                for root in [0,1]:
                                    other = gpu_snapshot(branch[1-root])
                                    label = (sum(prefix['bits'])+sum(int(b) for b in suffix[:h]))%2
                                    result = gate.future(lambda: run(sx[:,h-1], branch[root], PROGRAMS[prog], label,
                                        dict(phase='future', sequence_id=sid, history_program=hist,
                                            root_program=root, suffix=suffix, schedule=schedule, horizon=h), prog==0))
                                    branch[root] = result[0]['state']; future.append(result[1])
                                    exact_state(other, branch[1-root])
                                pair = paired_case(*future, dict(sequence_id=sid, history_program=hist,
                                    suffix=suffix, schedule=schedule), h)
                                pair.update(sequence_id=sid, partition=prefix['partition'], history_program=hist,
                                            suffix=suffix, schedule=schedule)
                                comparisons.append(pair)
                                require(all(sm['correct'] for sm in future) and not pair['additional_error']
                                    and pair['CE_delta'] <= .05 and pair['total_variation'] <= .01,
                                    f'Mandatory future case failed {pair["case"]} horizon{h}')
                            for old, root in zip(root_values, roots):
                                exact_state(old, root)
            require(runner.completed == original['max_completed_tokens'], 'Full study token accounting')
            boundary = 'original final joint acceptance'
            uncertainty = uncertainty_table(comparisons, inputs['rows'])
            write_json(DIR/'joint_uncertainty.json', uncertainty)
            require(all(uncertainty[p]['accepted_upper_bound'] for p in ['confirmation','longer_confirmation']),
                    'Original final joint confidence acceptance failed')
            boundary = 'completed original locked study'
        except Exception as error:
            failure = dict(error=repr(error), traceback=traceback.format_exc(), boundary=boundary,
                first_mechanical=runner.first_failure, completed_tokens=runner.completed,
                future_calls=gate.future_calls, elapsed_seconds=time.monotonic()-start,
                classification='CURRENT_CANDIDATE_REJECTED' if isinstance(error, CurrentRejection) else 'STOPPED')
            write_json(DIR/'failure.json', failure)
        finally:
            trace.flush()
            exact_state(weights, model.state_dict())
            require(all(p.grad is None and not p.requires_grad for p in model.parameters()), 'Parameter/gradient invariant')
            require(torch.equal(cpu_rng, torch.get_rng_state()), 'Unsealed CPU RNG consumption')
            require(all(torch.equal(state, torch.cuda.get_rng_state(i)) for i,state in enumerate(cuda_rng)),
                    'Unsealed CUDA RNG consumption')
            integrity = check_integrity(DIR)
            write_json(DIR/'integrity_after.json', dict(**integrity, parameters_unchanged=True,
                no_gradients=True, no_training=True, no_optimizer=True))
    write_json(DIR/'execution_ledger.json', dict(calls=ledgers, completed_tokens=runner.completed,
        bank_calls=runner.block_calls, commit_calls=runner.commit_calls,
        phases=dict(Counter(row['case']['phase'] for row in ledgers if row['status']=='COMPLETED')),
        future_calls=gate.future_calls, gdn_checks=dict(runner.gdn_checks), plan_purity_checks=runner.plan_purity_checks,
        historical_v1=dict(completed_reference_tokens=1, bank_calls=8, commits=1),
        historical_v2=0, historical_v3=0, tests='separate test_results.json',
        elapsed_seconds=time.monotonic()-start))
    write_json(DIR/'current_cases.json', current)
    write_json(DIR/'paired_comparisons.json', comparisons)
    results = dict(status='STOPPED' if failure else 'COMPLETED', boundary=boundary, failure=failure,
        P0_archival=gate.archival, P0_current=gate.p0_current, entire_current_screen=gate.all_current,
        monitor=monitor_result, future_calls=gate.future_calls, completed_tokens=runner.completed,
        bank_calls=runner.block_calls, commits=runner.commit_calls, integrity=integrity)
    write_json(DIR/'results.json', results)
    report = '# V4 two-program feasibility\n\n'+json.dumps(results, indent=2)+'\n\n'
    report += ('Engineering checks passed on completed calls; failed calls remain separately accounted. '
        'Historical tensor comparisons and historical summary comparisons are separate from full within-v4 reference/adapter evidence. '
        'Monitor status is non-gating and does not authorize a claim of complete diagnostic reproduction. '
        'No future or final-joint verdict is made for unexecuted stages. '
        'No gradients, training, optimizer, selector, production changes, checkpoint updates or commits occurred. '
        'V1 remains STOPPED BEFORE P1; v2 is inspection-only; v3 remains specification-barrier stop.\n')
    if failure and failure['classification']=='CURRENT_CANDIDATE_REJECTED':
        report += '\nP1 candidate rejected at mandatory current screening; future compatibility and final joint acceptance remain UNVERIFIED. This is bounded evidence, not a general program-portfolio rejection.\n'
    with (DIR/'report.md').open('x') as f:
        f.write(report)
    print(json.dumps({k:v for k,v in results.items() if k not in ['integrity','failure']},indent=2), flush=True)


if __name__ == '__main__':
    execute()
