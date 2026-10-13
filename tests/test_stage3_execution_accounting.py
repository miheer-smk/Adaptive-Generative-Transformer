"""Realized Stage-3 depth counts block executions, including REPEAT, excluding RESTART."""
import pytest
import torch

import src.executor as executor_module
from src.executor import AdaptiveExecutor
from experiments.audit_gatec_canonical_replay import snapshot, unchanged


class ScriptedController:
    def __init__(self, actions):
        self.actions = actions
        self.inputs = []

    def __call__(self, **features):
        action = self.actions[len(self.inputs)]  # Extra decisions fail immediately.
        self.inputs.append(features['h'].detach().clone())
        logits = features['h'].new_full((1, 11), -100.)
        logits[:, action] = 100.
        return {'logits': logits.requires_grad_()}


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('actions', [list(range(8)), [0, 9] * 7 + [0], [4] + [8] * 7])
def test_exact_execution_budget_and_restart_prefix_counts(monkeypatch, training, actions):
    class Block:
        def __call__(self, x, **kwargs):
            return x + 1

    class Bank:
        blocks = [Block() for _ in range(8)]
        def __call__(self, index, x, **kwargs):
            return self.blocks[index](x, **kwargs)

    controller = ScriptedController(actions)
    executor = AdaptiveExecutor(Bank(), controller, max_execution_depth=8)
    masks = []
    original_mask = executor_module.build_execution_mask

    def observe_mask(*args, **kwargs):
        mask = original_mask(*args, **kwargs)
        masks.append((kwargs['execution_count'], kwargs['just_restarted'], mask.clone()))
        return mask

    monkeypatch.setattr(executor_module, 'build_execution_mask', observe_mask)
    h = torch.zeros(1, 64)
    result = executor.execute_differentiable(h, h, training=training)
    assert result.execution_count == 8
    assert len(result.action_history) == len(actions) == len(controller.inputs)
    assert sum(action != 'RESTART' for action, _ in result.action_history) == 8
    assert torch.equal(result.h_terminal, torch.full_like(h, 8.))
    assert not result.halted and not result.safety_cap_hit
    # Training constructs the same mask twice: executor and route primitive.
    if training:
        assert masks[::2] and len(masks) == 2 * len(actions)
        masks = masks[::2]
    assert len(masks) == len(actions)
    executed = 0
    for decision, (action, (count, restarted, mask)) in enumerate(zip(actions, masks)):
        assert count == executed
        assert restarted == (decision > 0 and actions[decision - 1] == 9)
        assert mask[action] and mask[10]  # Source HALT legality is preserved.
        assert torch.equal(controller.inputs[decision], torch.full_like(h, float(executed)))
        if action != 9:
            executed += 1
    assert executed == 8
    if 9 in actions:
        assert len(actions) == 15
        assert sum(action != 9 for action in actions[:8]) == 4
    if training:
        result.h_terminal.sum().backward()


@pytest.mark.parametrize('training', [False, True])
def test_restart_repeat_budget_preserves_real_state_and_single_commit(monkeypatch, training):
    import src.agt as agt_module
    from src.agt import AGTv3
    from src.terminal_commit import build_commit_plan

    torch.manual_seed(3401)
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=2)
    model.executor.max_execution_depth = 8
    # Establish nonempty committed GDN/convolution/attention context first.
    with torch.no_grad():
        state = model.step(torch.randn(1, 64), model.initial_persistent_state(1, 'cpu'),
                           torch.zeros(1, 64))['state']
    before = snapshot(state)
    weights = snapshot(model.state_dict())
    actions = [4, 8, 9, 4, 8, 9, 4, 8, 9, 4, 8]
    controller = ScriptedController(actions)
    model.executor.controller = controller
    original_block = model.executor._run_block
    original_commit = agt_module.terminal_commit
    block_calls = []
    commit_calls = []

    def checked_block(*args, **kwargs):
        inputs = snapshot((args, kwargs))
        result = original_block(*args, **kwargs)
        assert unchanged(inputs, snapshot((args, kwargs)))
        assert unchanged(before, snapshot(state))
        block_calls.append(1)
        return result

    def checked_commit(work, **kwargs):
        assert unchanged(before, snapshot(work))
        plan = build_commit_plan(work, **kwargs)
        assert unchanged(before, snapshot(work))
        assert len(plan.entries) == 7
        result = original_commit(work, **kwargs)
        torch.testing.assert_close(result[0].s, plan.s, atol=0, rtol=0)
        torch.testing.assert_close(result[0].memory, plan.memory, atol=0, rtol=0)
        for cache, version, payload in plan.entries:
            assert cache.version == version + 1
            for actual, expected in zip(cache.get_committed_context(), payload):
                if actual is not None:
                    if actual.shape != expected.shape:
                        actual = actual[:, :, -1:, :]
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        commit_calls.append(1)
        return result

    monkeypatch.setattr(model.executor, '_run_block', checked_block)
    monkeypatch.setattr(agt_module, 'terminal_commit', checked_commit)
    with torch.no_grad():
        result = model.adaptive_differentiable_step(torch.randn(1, 64), state,
                                                    torch.zeros(1, 64), training=training)
    assert result['execution_count'] == 8
    assert len(result['action_history']) == 11
    assert sum(k == 'RESTART' for k, _ in result['action_history']) == 3
    assert sum(k == 'REPEAT' for k, _ in result['action_history']) == 4
    assert len(block_calls) == (99 if training else 8)
    assert len(commit_calls) == 1
    assert result['state'].tokens_committed == state.tokens_committed + 1
    assert unchanged(before, snapshot(state))
    assert unchanged(weights, snapshot(model.state_dict()))
