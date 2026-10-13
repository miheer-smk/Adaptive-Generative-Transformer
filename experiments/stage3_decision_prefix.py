"""Diagnostic observation horizons, never completed-token execution or commit."""
from dataclasses import dataclass

import torch

from src.masks import apply_action_mask, build_execution_mask


@dataclass
class DecisionPrefix:
    h_observed: torch.Tensor
    decision_count: int
    execution_count: int
    action_history: list
    rows: list
    token: int
    observation_horizon: int
    termination_reason: str = "observation_horizon"
    committed: bool = False


def observe_decision_prefix(executor, h0, memory, *, decision_horizon,
                            context=None, memory_mean=None, budget=None,
                            tau=1.0, training=True, token=0):
    """Observe exactly D decisions using existing Stage-3 routing primitives.

    Training calls the real ST successor step without detaching live state.
    Inference masks legal successors and executes only the selected block.
    This helper neither completes a token nor returns/commits persistent state,
    even when the observed execution count happens to equal the token budget.
    """
    if not isinstance(decision_horizon, int) or decision_horizon < 0:
        raise ValueError("decision_horizon must be a nonnegative integer")
    if h0.ndim != 2 or h0.shape[0] != 1 or memory.shape != h0.shape:
        raise ValueError("Prefix pilot requires matching [1, D] h0 and memory")
    memory_mean = h0.new_zeros(1, 256) if memory_mean is None else memory_mean
    budget = h0.new_zeros(1, 2) if budget is None else budget
    if memory_mean.shape != (1, 256) or budget.shape != (1, 2):
        raise ValueError("Expected [1, 256] memory_mean and [1, 2] budget")
    num_blocks = len(executor.block_bank.blocks)
    h, pointer, restarted, executions = h0.clone(), 0, False, 0
    history, rows = [], []
    for decision in range(decision_horizon):
        onehot = h.new_zeros(1, num_blocks + 1)
        onehot[:, pointer] = 1
        logits = executor.controller(h=h, memory_read=memory,
                                     memory_mean=memory_mean,
                                     pointer_one_hot=onehot, budget=budget)["logits"]
        mask = build_execution_mask(pointer, executions, restarted, num_blocks).to(logits.device)
        if training:
            result = executor.differentiable_route_step(
                h=h, h0=h0, memory=memory, context=context, pointer=pointer,
                action_logits=logits, tau=tau, training=True,
                execution_count=executions, just_restarted=restarted)
            selected = int(result["selected_action"][0].item())
            h_next = result["h_next"]
        else:
            masked = apply_action_mask(logits[:, :num_blocks + 2],
                                       mask[:num_blocks + 2].unsqueeze(0))
            selected = int(masked.argmax(-1)[0].item())
            if not torch.isfinite(masked[0, selected]):
                raise ValueError("Prefix policy has no finite legal successor")
            h_next = h
            if selected <= num_blocks:
                block = selected if selected < num_blocks else pointer
                h_next = executor._run_block(block, h, h0, memory, context)
        if selected >= num_blocks + 2 or not mask[selected]:
            raise RuntimeError("Prefix selected an illegal/HALT successor")
        row = dict(token=token, decision=decision, execution_before=executions,
                   pointer_before=pointer, just_restarted_before=restarted,
                   selected_action=selected,
                   legal_actions=[i for i in range(num_blocks + 2) if mask[i]])
        h = h_next
        if selected < num_blocks:
            pointer, restarted = selected, False
            executions += 1
            history.append(("ADVANCE", pointer))
        elif selected == num_blocks:
            restarted = False
            executions += 1
            history.append(("REPEAT", pointer))
        else:
            pointer, restarted = 0, True
            history.append(("RESTART", None))
        row.update(execution_after=executions, pointer_after=pointer,
                   just_restarted_after=restarted)
        rows.append(row)
    return DecisionPrefix(h, len(history), executions, history, rows, token, decision_horizon)


def observe_token_prefix(model, x_t, state, C_prev, *, decision_horizon, **kwargs):
    """The real memory read/Prelude, followed by an uncommitted observation."""
    work = state.clone()
    memory = model.memory.read(work.s, work.memory)[0]
    h0 = model.prelude(x_t=x_t, H_prev=work.s, memory_read=memory, C_prev=C_prev)
    return observe_decision_prefix(model.executor, h0, memory, context=work.ctx,
                                   memory_mean=work.memory.mean(1),
                                   decision_horizon=decision_horizon, **kwargs)
