"""Observation alignment, autograd, and fixed-backbone sensitivity diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.task_a import TaskAParity
from experiments.run_stage3_taskA_controller import run_diff_sequence, run_fixed_sequence, set_seed
from experiments.stage3_measurements import (
    CHECKPOINT, file_sha256, load_model, parameter_gradients, route_metrics, tensor_stats,
)
from src.masks import build_execution_mask


@torch.no_grad()
def check_observation_alignment(model, x):
    """Compare teacher block inputs and actual controller observations token by token."""
    expected = []
    state = model.initial_persistent_state(1, x.device, x.dtype)
    context = torch.zeros(1, model.dim, device=x.device, dtype=x.dtype)
    handles = []
    token_inputs = {}

    def teacher_hook(index):
        def hook(module, args):
            pointer = torch.zeros(1, 9, device=x.device, dtype=x.dtype)
            pointer[:, max(index - 1, 0)] = 1
            expected.append(dict(h=args[0][:, 0].clone(),
                                 memory_read=token_inputs["memory_read"].clone(),
                                 memory_mean=token_inputs["memory_mean"].clone(),
                                 pointer_one_hot=pointer,
                                 budget=torch.zeros(1, 2, device=x.device, dtype=x.dtype)))
        return hook

    for i, block in enumerate(model.blocks.blocks):
        handles.append(block.register_forward_pre_hook(teacher_hook(i)))
    try:
        for t in range(x.shape[1]):
            token_inputs["memory_read"] = model.memory.read(state.s, state.memory)[0]
            token_inputs["memory_mean"] = state.memory.mean(1)
            state = model.step(x[:, t], state, context)["state"]
    finally:
        for handle in handles:
            handle.remove()

    observed = []
    probabilities = []

    def controller_hook(module, args, kwargs):
        observed.append({key: kwargs[key].detach().clone() for key in expected[0]})

    handle = model.controller.register_forward_pre_hook(controller_hook, with_kwargs=True)
    try:
        logits, histories = run_diff_sequence(model, x, 2.0, False)
    finally:
        handle.remove()
    assert len(expected) == len(observed) == x.shape[1] * 8
    differences = {key: max((a[key]-b[key]).abs().max().item()
                            for a, b in zip(expected, observed)) for key in expected[0]}
    for row, features in enumerate(expected):
        decision = row % 8
        logits = model.controller(**features)["logits"][:, :10]
        mask = build_execution_mask(max(decision-1, 0), decision, False, 8)[:10].to(x.device)
        probs = torch.softmax(logits.masked_fill(~mask, -torch.inf), -1)
        probabilities.append(probs[0, decision].item())
    return dict(rows=len(expected), max_absolute_difference=differences,
                exact_alignment=all(value == 0 for value in differences.values()),
                minimum_teacher_action_probability=min(probabilities),
                routes=route_metrics(histories))


def gradient_trial(model, x, target, noise_seed, backbone_probe=False):
    original = model.executor.differentiable_route_step
    retained = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(backbone_probe or name.startswith("controller."))
    model.zero_grad(set_to_none=True)
    token_input = x.detach().clone().requires_grad_(True)

    def traced_step(**kwargs):
        result = original(**kwargs)
        for tensor in (kwargs["action_logits"], result["candidates"], result["h_next"],
                       kwargs["h"], kwargs["h0"]):
            if tensor.requires_grad:
                tensor.retain_grad()
        mask = build_execution_mask(kwargs["pointer"], kwargs["execution_count"],
                                    kwargs["just_restarted"], 8)[:10].to(x.device)
        index = int(result["selected_action"].item())
        retained.append(dict(logits=kwargs["action_logits"], candidates=result["candidates"],
                             h_next=result["h_next"], h=kwargs["h"], h0=kwargs["h0"],
                             selected=index, mask=mask,
                             hard_forward_exact=torch.equal(result["h_next"].detach(),
                                                            result["candidates"][:, index].detach()),
                             illegal_forward_mass=result["probs"][:, ~mask].abs().sum().item(),
                             successor_count=result["candidates"].shape[1]))
        return result

    model.executor.differentiable_route_step = traced_step
    set_seed(noise_seed)
    try:
        logits, histories = run_diff_sequence(model, token_input, 2.0, True)
        loss = F.cross_entropy(logits, target)
        loss.backward()
    finally:
        model.executor.differentiable_route_step = original
    steps = []
    for i, row in enumerate(retained):
        grad = row["logits"].grad
        candidate_grad = row["candidates"].grad
        selected = row["selected"]
        other = torch.arange(10, device=x.device) != selected
        steps.append(dict(token=i//8, decision=i%8, selected=selected,
                          logits_gradient=tensor_stats(grad),
                          selected_successor_gradient=tensor_stats(candidate_grad[:, selected]),
                          unselected_successor_gradient=tensor_stats(candidate_grad[:, other]),
                          live_gradient=tensor_stats(row["h"].grad),
                          anchor_gradient=tensor_stats(row["h0"].grad),
                          illegal_logit_gradient_max=grad[:, :10][:, ~row["mask"]].abs().max().item()
                          if (~row["mask"]).any() else 0.0,
                          halt_logit_gradient_max=grad[:, 10].abs().max().item(),
                          **{key: row[key] for key in ("hard_forward_exact", "illegal_forward_mass", "successor_count")}))
    gradients = parameter_gradients(model)
    vector = torch.cat([p.grad.detach().reshape(-1) for p in model.controller.parameters()])
    backbone = {}
    if backbone_probe:
        for name, parameter in model.named_parameters():
            if not name.startswith("controller."):
                group = ".".join(name.split(".")[:3]) if name.startswith("blocks.blocks.") else name.split(".")[0]
                entry = backbone.setdefault(group, dict(tensors=0, with_gradient=0, nonzero=0, finite=True, squared_l2=0.0))
                entry["tensors"] += 1
                if parameter.grad is not None:
                    stats = tensor_stats(parameter.grad)
                    entry["with_gradient"] += 1
                    entry["nonzero"] += int(stats["nonzero"] > 0)
                    entry["finite"] &= stats["finite"]
                    entry["squared_l2"] += stats["l2"] ** 2
    return dict(noise_seed=noise_seed, ce=loss.item(), controller_gradients=gradients,
                token_input_gradient=tensor_stats(token_input.grad), steps=steps,
                backbone_probe=backbone, routes=route_metrics(histories)), vector.cpu()


class ScriptedSweep:
    """Diagnostic controller only; no parameters or optimizer changes."""
    def __init__(self, actions):
        self.actions = actions
        self.calls = 0

    def __call__(self, **kwargs):
        index = self.actions[self.calls % 8]
        self.calls += 1
        logits = kwargs["h"].new_full((1, 11), -100.0)
        logits[:, index] = 100.0
        return {"logits": logits}


def compact_report(report):
    """Retain gradient magnitudes and invariants without every retained tensor row."""
    def compact_trial(trial):
        result = {key: value for key,value in trial.items() if key != "steps"}
        steps = trial["steps"]
        result["invariants"] = dict(
            steps=len(steps), hard_forward_exact=all(s["hard_forward_exact"] for s in steps),
            finite_logit_gradients=all(s["logits_gradient"]["finite"] for s in steps),
            nonzero_logit_steps=sum(s["logits_gradient"]["l2"] > 0 for s in steps),
            illegal_forward_mass_max=max(s["illegal_forward_mass"] for s in steps),
            illegal_logit_gradient_max=max(s["illegal_logit_gradient_max"] for s in steps),
            halt_logit_gradient_max=max(s["halt_logit_gradient_max"] for s in steps),
            unselected_successor_gradient_max=max(s["unselected_successor_gradient"]["l2"] for s in steps),
        )
        result["gradient_l2_by_decision"] = []
        for decision in range(8):
            rows = [s for s in steps if s["decision"] == decision]
            metrics = {}
            for name in ("logits_gradient", "selected_successor_gradient", "live_gradient", "anchor_gradient"):
                values = [s[name]["l2"] for s in rows]
                metrics[name] = dict(mean=sum(values)/len(values), minimum=min(values), maximum=max(values))
            result["gradient_l2_by_decision"].append(metrics)
        return result

    result = {key: value for key,value in report.items() if key != "arms"}
    result["arms"] = {}
    for arm, values in report["arms"].items():
        result["arms"][arm] = {key: value for key,value in values.items()
                               if key not in ("trials", "backbone_structural_probe")}
        result["arms"][arm]["trials"] = [compact_trial(trial) for trial in values["trials"]]
        result["arms"][arm]["backbone_structural_probe"] = compact_trial(values["backbone_structural_probe"])
    return result


@torch.no_grad()
def route_sensitivity(model, cases):
    original = model.executor.controller
    policies = dict(canonical=list(range(8)), skip_first_swa=[1,2,3,4,5,6,7,8],
                    skip_first_gdn=[0,1,2,3,5,6,7,8], repeat_first_gdn=[0,1,2,3,4,8,5,6],
                    restart_middle=[0,1,2,3,9,0,1,2])
    result = {}
    try:
        for name, actions in policies.items():
            correct = total = 0
            ce = []
            histories = []
            for batch in cases:
                for i in range(batch.target.numel()):
                    model.executor.controller = ScriptedSweep(actions)
                    logits, route = run_diff_sequence(model, batch.x[i:i+1], 2.0, False)
                    correct += int((logits.argmax(-1) == batch.target[i:i+1]).sum())
                    ce.append(F.cross_entropy(logits, batch.target[i:i+1]).item())
                    histories.extend(route)
                    total += 1
            result[name] = dict(accuracy=correct/total, ce=sum(ce)/total, examples=total,
                                routes=route_metrics(histories))
    finally:
        model.executor.controller = original
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.exists() or summary_path.exists():
        parser.error("Output exists; choose a new run path")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = TaskAParity(64, device)
    batch = task.make_fixed_set("val", 8, 1)
    report = dict(hypothesis="route expressivity versus task-loss credit assignment",
                  backbone=str(CHECKPOINT), backbone_sha256=file_sha256(CHECKPOINT),
                  warmstart=str(args.warmstart), warmstart_sha256=file_sha256(args.warmstart),
                  dim=64, decisions=8, tau=2.0, budget="zeros", trials=args.trials, arms={})
    for arm in ("random", "imitation"):
        model = load_model(3305, device, args.warmstart if arm == "imitation" else None)
        alignment = check_observation_alignment(model, batch.x)
        trials, vectors = [], []
        for i in range(args.trials):
            trial, vector = gradient_trial(model, batch.x, batch.target, 4305+i)
            trials.append(trial)
            vectors.append(vector)
        vectors = torch.stack(vectors)
        norms = vectors.norm(dim=1)
        normalized = F.normalize(vectors, dim=1)
        cosines = normalized @ normalized.T
        off_diagonal = ~torch.eye(args.trials, dtype=torch.bool)
        probe, _ = gradient_trial(model, batch.x, batch.target, 4305, backbone_probe=True)
        report["arms"][arm] = dict(observation_alignment=alignment, trials=trials,
                                   controller_gradient_l2_mean=norms.mean().item(),
                                   controller_gradient_l2_std=norms.std(unbiased=False).item(),
                                   pairwise_gradient_cosine_mean=cosines[off_diagonal].mean().item(),
                                   mean_gradient_l2=vectors.mean(0).norm().item(),
                                   gradient_dispersion_l2=(vectors-vectors.mean(0)).square().mean(0).sum().sqrt().item(),
                                   backbone_structural_probe=probe)
        print(arm, report["arms"][arm]["controller_gradient_l2_mean"], flush=True)
    # Independent route interventions on a balanced fixed-size sample.
    task = TaskAParity(64, device)
    cases = [task.make_fixed_set("val", length, 8) for length in (4,8,12,16)]
    model = load_model(3305, device)
    report["route_sensitivity"] = route_sensitivity(model, cases)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = compact_report(report)
    summary["raw_trace_sha256"] = file_sha256(args.output)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
