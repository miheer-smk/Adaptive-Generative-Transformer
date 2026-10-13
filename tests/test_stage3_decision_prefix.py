"""Decision-prefix controls stop at the observation boundary without commit."""
import pytest
import torch

from experiments.audit_gatec_canonical_replay import snapshot, unchanged
from experiments.stage3_decision_prefix import observe_decision_prefix, observe_token_prefix
from src.agt import AGTv3
from src.executor import AdaptiveExecutor


class Script:
    def __init__(self, actions):
        self.actions, self.calls = actions, 0

    def __call__(self, **features):
        action = self.actions[self.calls]  # Any unsupplied continuation fails.
        self.calls += 1
        logits = features['h'].new_full((1, 11), -torch.inf)
        logits[:, action] = 0.
        return {'logits': logits.requires_grad_()}


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('actions', [list(range(8)), [0, 9, 0, 1, 2, 3, 4, 5]])
def test_prefix_preserves_real_routing_and_graph_without_commit(monkeypatch, training, actions):
    import src.agt as agt_module
    torch.manual_seed(3401)
    model = AGTv3(dim=64, num_memory_slots=16, vocab_size=2)
    model.executor.max_execution_depth = 8
    compatibility = torch.zeros(1, 64)
    with torch.no_grad():
        state = model.step(torch.randn(1, 64), model.initial_persistent_state(1, 'cpu'), compatibility)['state']
    before, weights = snapshot(state), snapshot(model.state_dict())
    x = torch.randn(1, 64, requires_grad=True)
    memory = model.memory.read(state.s, state.memory)[0]
    h0 = model.prelude(x, state.s, memory, compatibility)
    # Canonical production completes at D=P=8; compare exact live state/noise.
    if actions == list(range(8)):
        model.executor.controller = Script(actions)
        torch.manual_seed(45)
        expected = model.executor.execute_differentiable(h0, memory, context=state.ctx,
                                                        memory_mean=state.memory.mean(1), training=training)
    model.executor.controller = Script(actions)
    monkeypatch.setattr(agt_module, 'terminal_commit', lambda *a, **k: pytest.fail('Prefix committed'))
    monkeypatch.setattr(model.executor, 'execute_differentiable',
                        lambda *a, **k: pytest.fail('Prefix requested production completion'))
    block_calls = []
    original_block = model.executor._run_block
    def checked_block(*args, **kwargs):
        inputs = snapshot((args, kwargs))
        value = original_block(*args, **kwargs)
        assert unchanged(inputs, snapshot((args, kwargs)))
        assert unchanged(before, snapshot(state))
        block_calls.append(args[0] if args else kwargs['block_index'])
        return value
    monkeypatch.setattr(model.executor, '_run_block', checked_block)
    torch.manual_seed(45)
    prefix = observe_token_prefix(model, x, state, compatibility,
                                  decision_horizon=len(actions), training=training, token=12)
    assert model.executor.controller.calls == prefix.decision_count == len(actions)
    assert prefix.execution_count == sum(action != 9 for action in actions)
    assert not prefix.committed and prefix.termination_reason == 'observation_horizon'
    assert not hasattr(prefix, 'state') and not hasattr(prefix, 'h_terminal')
    assert [r['selected_action'] for r in prefix.rows] == actions
    assert [(r['token'], r['decision']) for r in prefix.rows] == [(12, d) for d in range(len(actions))]
    if actions == list(range(8)):
        assert prefix.action_history == expected.action_history
        torch.testing.assert_close(prefix.h_observed, expected.h_terminal, atol=0, rtol=0)
    if not training:
        assert block_calls == [action for action in actions if action != 9]
    for row in prefix.rows:
        assert row['selected_action'] in row['legal_actions']
        assert row['execution_after'] - row['execution_before'] == (row['selected_action'] != 9)
    assert prefix.h_observed.grad_fn is not None
    prefix.h_observed.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.norm() > 0
    assert unchanged(before, snapshot(state))
    assert unchanged(weights, snapshot(model.state_dict()))


@pytest.mark.parametrize('actions', [[8], [9], [0, 9, 9], [0, 9, 8], [2, 1], [10]])
def test_supplied_illegal_or_halt_action_cannot_fall_back(actions):
    class Block:
        def __call__(self, x, **kwargs):
            return x + 1
    class Bank:
        blocks = [Block() for _ in range(8)]
        def __call__(self, index, x, **kwargs):
            return self.blocks[index](x, **kwargs)
    executor = AdaptiveExecutor(Bank(), Script(actions))
    h = torch.zeros(1, 64)
    with pytest.raises(ValueError, match='no finite legal successor'):
        observe_decision_prefix(executor, h, h, decision_horizon=len(actions), training=False)


def test_scripted_policy_rejects_unsupplied_continuation():
    from experiments.diagnose_stage3_taskA_gradient_flow import ScriptedPolicy
    class Block:
        def __call__(self, x, **kwargs):
            return x
    class Bank:
        blocks = [Block() for _ in range(8)]
        def __call__(self, index, x, **kwargs):
            return self.blocks[index](x, **kwargs)
    executor = AdaptiveExecutor(Bank(), ScriptedPolicy(9))
    h = torch.zeros(1, 64)
    prefix = observe_decision_prefix(executor, h, h, decision_horizon=8, training=False, token=7)
    assert prefix.execution_count == 7 and executor.controller.calls == 8
    with pytest.raises(RuntimeError, match='no continuation supplied'):
        executor.controller(h=h, pointer_one_hot=torch.nn.functional.one_hot(torch.tensor([7]), 9))
