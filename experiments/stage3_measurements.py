"""Read-only measurements for the locked fixed-depth D64 routing pilot."""
from __future__ import annotations

import hashlib
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch

from experiments.run_stage3_taskA_controller import (
    AGTv3, CHECKPOINT, set_seed,
)
from src.masks import build_execution_mask


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(seed, device, controller_checkpoint=None):
    set_seed(seed)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    model = AGTv3(dim=int(checkpoint["dim"]), num_memory_slots=16, vocab_size=2)
    weights = {k: v for k, v in checkpoint["state_dict"].items()
               if not k.startswith("controller.")}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if set(missing) != {k for k in model.state_dict() if k.startswith("controller.")} or unexpected:
        raise RuntimeError("Backbone transfer differs from the controller-only contract")
    if controller_checkpoint is not None:
        controller = torch.load(controller_checkpoint, map_location="cpu")
        if controller["dim"] != model.dim or controller.get("routing_decisions") != 8:
            raise ValueError("Controller checkpoint lacks compatible depth/dimension metadata")
        if controller.get("budget") != "zeros":
            raise ValueError("Controller checkpoint budget differs from the experiment")
        model.controller.load_state_dict(controller["controller_state_dict"], strict=True)
    model.to(device).eval()
    model.executor.max_execution_depth = 8
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.controller.parameters():
        parameter.requires_grad_(True)
    return model


def tensor_stats(tensor):
    if tensor is None:
        return {"present": False}
    detached = tensor.detach().float()
    return dict(present=True, finite=bool(torch.isfinite(detached).all()),
                l2=detached.norm().item(), rms=detached.square().mean().sqrt().item(),
                max_abs=detached.abs().max().item(),
                nonzero=int(torch.count_nonzero(detached)), numel=detached.numel())


def parameter_gradients(model):
    return {name: dict(**tensor_stats(parameter.grad),
                       parameter_l2=parameter.detach().float().norm().item())
            for name, parameter in model.controller.named_parameters()}


@contextmanager
def count_block_calls(model):
    """Count executor block invocations; terminal-commit projections are separate."""
    original = model.executor._run_block
    counter = {"block_calls": 0}

    def measured(*args, **kwargs):
        counter["block_calls"] += 1
        return original(*args, **kwargs)

    model.executor._run_block = measured
    try:
        yield counter
    finally:
        model.executor._run_block = original


def route_metrics(histories, num_blocks=8):
    """Analyze supplied action histories, including decision prefixes.

    Each history retains the boundary supplied by the caller. Legacy `tokens`
    and `*_per_token` fields count/average those histories, not proven completed
    tokens. `exact_tokens` counts histories equal to the full canonical sweep;
    it does not infer completion of a caller's configured execution budget.
    `incorrect_decisions` compares observed positions with that sweep; positions
    beyond the sweep are extra decisions, not fabricated ADVANCE destinations.

    Action rates use observed decisions. Block-use frequencies use observed
    ADVANCE/REPEAT executions. No history length or completion budget is imposed.
    """
    actions = Counter()
    by_decision = []
    routes = Counter()
    unique_blocks = []
    first_errors = Counter()
    errors = restart_errors = illegal = 0
    confusion = []
    block_executions = [0] * num_blocks
    exact = 0
    for history in histories:
        pointer, restarted, executions = 0, False, 0
        visited = set()
        route = []
        first = None
        for decision, (kind, index) in enumerate(history):
            action = index if kind == "ADVANCE" else {
                "REPEAT": num_blocks, "RESTART": num_blocks + 1,
                "HALT": num_blocks + 2,
            }[kind]
            if decision == len(by_decision):
                by_decision.append(Counter())
                confusion.append([0] * (num_blocks + 3))
            mask = build_execution_mask(pointer, executions, restarted, num_blocks)
            illegal += int(not mask[action] or kind == "HALT")
            confusion[decision][action] += 1
            key = str((kind, index))
            actions[key] += 1
            by_decision[decision][key] += 1
            route.append(key)
            if decision >= num_blocks or (kind, index) != ("ADVANCE", decision):
                errors += 1
                restart_errors += int(kind == "RESTART")
                if first is None:
                    first = decision
            if kind == "ADVANCE":
                pointer, restarted = index, False
                visited.add(index)
                block_executions[index] += 1
                executions += 1
            elif kind == "REPEAT":
                visited.add(pointer)
                block_executions[pointer] += 1
                executions += 1
                restarted = False
            elif kind == "RESTART":
                pointer, restarted = 0, True
        exact += int(len(history) == num_blocks and first is None)
        if first is not None:
            first_errors[str(first)] += 1
        unique_blocks.append(len(visited))
        routes[" | ".join(route)] += 1
    tokens = len(histories)
    total = sum(actions.values())
    restart = sum(n for action, n in actions.items() if "RESTART" in action)
    repeat = sum(n for action, n in actions.items() if "REPEAT" in action)
    executed = sum(block_executions)
    return dict(tokens=tokens, decisions=total, executions=executed,
                restarts=restart, repeats=repeat, exact_tokens=exact,
                exact_route_fraction=exact/tokens if tokens else None,
                actions=dict(actions), per_decision_actions=[dict(c) for c in by_decision],
                per_decision_observations=[sum(c.values()) for c in by_decision],
                confusion=confusion, incorrect_decisions=errors, illegal_decisions=illegal,
                restart_fraction_of_errors=restart_errors/errors if errors else 0.0,
                restart_rate=restart/total if total else 0.0,
                repeat_rate=repeat/total if total else 0.0,
                selected_block_calls=executed,
                selected_block_calls_per_token=executed/tokens if tokens else None,
                block_execution_counts=block_executions,
                block_execution_frequencies=[n/executed if executed else 0.0
                                             for n in block_executions],
                unique_blocks_per_token=sum(unique_blocks)/tokens if tokens else None,
                unique_blocks_histogram=dict(Counter(map(str, unique_blocks))),
                first_error_decision=dict(first_errors), route_histogram=dict(routes))
