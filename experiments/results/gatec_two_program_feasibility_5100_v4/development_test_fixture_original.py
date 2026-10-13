"""Synthetic unit fixtures: no new study inputs or scientific outcomes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import copy
import pytest
import torch
import research_programs_v4 as programs
from runner_v4 import Barriers, strict_compare, screen, archival_records
from src.agt import AGTv3


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_mask_device_observation_and_complete_P0(device, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable: study must not execute')
    torch.set_num_threads(4)
    torch.manual_seed(99)
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=2).to(device).eval()
    model.executor.max_execution_depth = 8
    for p in model.parameters():
        p.requires_grad_(False)
    mask_calls, observation_calls = [], []
    original_mask = programs.executor_module.build_execution_mask
    original_fill = torch.Tensor.masked_fill
    def capture_mask(*args, **kwargs):
        result = original_mask(*args, **kwargs)
        assert result.device.type == 'cpu' and result.dtype == torch.bool and result.shape == (11,)
        mask_calls.append((result, result.clone()))
        return result
    def capture_fill(logits, mask, value):
        if logits.shape == (1,10):
            saved = logits.clone()
            assert mask.device == logits.device and mask.dtype == torch.bool and mask.shape == (10,)
            assert torch.equal(mask.cpu(), ~mask_calls[-1][1][:10])
            result = original_fill(logits, mask, value)
            assert torch.equal(logits, saved)
            observation_calls.append(int(result.argmax(-1)))
            return result
        return original_fill(logits, mask, value)
    monkeypatch.setattr(programs.executor_module, 'build_execution_mask', capture_mask)
    monkeypatch.setattr(torch.Tensor, 'masked_fill', capture_fill)
    parent = model.initial_persistent_state(1, device, torch.float32)
    saved = programs.gpu_snapshot(parent)
    runner = programs.CompleteToken(model)
    out, summary, rows = runner.run(torch.zeros(1,64,device=device), parent, programs.P0)
    assert len(mask_calls) == 8
    for result, before in mask_calls:
        assert torch.equal(result, before) and result.device.type == 'cpu'
        assert torch.equal(torch.where(result)[0], torch.where(before)[0])
    assert all(action in range(8) for action in observation_calls)
    # Both observer and production selection are observed; no fallback dispatch.
    assert observation_calls == [action for action in range(8) for _ in range(2)]
    assert [row['action'] for row in rows] == list(range(8))
    assert [row['actual_blocks'] for row in rows] == [[i] for i in range(8)]
    assert summary['P'] == summary['D'] == runner.block_calls == 8
    assert summary['commit_count'] == runner.commit_calls == 1
    assert out['state'].tokens_committed == 1
    programs.exact_state(saved, parent)
    assert model.executor.controller is model.controller


@pytest.mark.parametrize('field', ['archival','p0_current','all_current'])
@pytest.mark.parametrize('value', [None, False])
def test_missing_failed_or_unresolved_gate_executes_zero_future_calls(field, value):
    gate = Barriers()
    gate.archival = gate.p0_current = gate.all_current = True
    setattr(gate, field, value)
    calls = []
    with pytest.raises(programs.MechanicalViolation, match='blocked'):
        gate.future(lambda: calls.append('forbidden'))
    assert calls == [] and gate.future_calls == 0


def test_P1_history_permission_is_distinct_from_P1_acceptance():
    gate = Barriers()
    with pytest.raises(programs.MechanicalViolation):
        gate.require_p1_history()
    gate.archival = gate.p0_current = True
    gate.require_p1_history()
    with pytest.raises(programs.MechanicalViolation):
        gate.future(lambda: None)
    assert gate.all_current is None


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'null', 'version', 'window', 'commit', 'shape', 'summary'])
def test_strict_archival_comparator_detects_all_required_classes(mutation):
    expected = {'token':0, 'commit_count':1, 'commit':{'/ctx/kv/2/version':1,
        '/ctx/swa/0/num_committed_tokens':1, '/ctx/gdn/4/committed_buffer':None,
        '/s':{'shape':[1,64], 'l2':2., 'max_abs':.2}}}
    actual = copy.deepcopy(expected)
    if mutation == 'missing': del actual['commit']['/ctx/kv/2/version']
    if mutation == 'extra': actual['unknown'] = 1
    if mutation == 'null': actual['commit']['/ctx/gdn/4/committed_buffer'] = []
    if mutation == 'version': actual['commit']['/ctx/kv/2/version'] += 1
    if mutation == 'window': actual['commit']['/ctx/swa/0/num_committed_tokens'] += 1
    if mutation == 'commit': actual['commit_count'] = 2
    if mutation == 'shape': actual['commit']['/s']['shape'] = [64]
    if mutation == 'summary': actual['commit']['/s']['l2'] = 3.
    with pytest.raises(programs.MechanicalViolation):
        strict_compare(expected, actual)
    strict_compare(expected, copy.deepcopy(expected))


def test_current_screen_has_original_thresholds_without_new_confidence_gate():
    summary = dict(P=8, commit_count=1, correct=True)
    pair = dict(CE_delta=.05, total_variation=.01, additional_error=False)
    assert screen(summary, paired=pair)['accepted']
    for key in ['CE_delta','total_variation']:
        bad = dict(pair); bad[key] += 1e-8
        assert not screen(summary, paired=bad)['accepted']
    bad = dict(summary, correct=False)
    assert not screen(bad)['accepted']


def test_monitor_current_and_script_targets_separate_and_summary_evidence_labeled():
    source = Path(__file__).with_name('runner_v4.py').read_text()
    assert "set(observed) | {'controller_logits'}" in source
    assert "first_mismatch" in source and "'REPLAY_MISMATCH'" in source
    assert 'historical_summaries' in source and 'full_within_v4' in source
    assert 'old[\'controller_logits\'], observed' in source
    matrix = __import__('json').loads(Path(__file__).with_name('requirements_matrix_annotated.json').read_text())
    assert len(matrix['requirements']) == 620
    for row in matrix['requirements']:
        binding = row['v4_binding']
        assert binding['producer'] and binding['target'] and binding['gate_role']
        if row['requirement'] == 'archived_step_controller_logits':
            assert binding['evidence_type'] == 'monitor diagnostic'
            assert 'NON-GATING' in binding['gate_role']
        if row.get('field_schema',{}).get('top_level_field') in ['commit','commit_pre','context','persistent']:
            assert binding['evidence_type'] == 'summary'
