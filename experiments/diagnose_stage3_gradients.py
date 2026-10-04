"""Observation alignment, autograd, and fixed-backbone sensitivity diagnostics."""
from __future__ import annotations

import argparse
import json
import math
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
from src.executor import AdaptiveExecutor


def metadata_credit_probe():
    """Toy counterexample isolating hard metadata credit; not an AGTv3 benchmark.

    REPEAT at an identity block and RESTART have identical live successors,
    yet select different future legal destinations. The ST hidden-state
    mixture has no derivative through that discrete pointer/mask effect.
    """
    class AddBlock:
        def __init__(self, amount):
            self.amount = amount

        def __call__(self, x, **kwargs):
            return x + self.amount

    class Bank:
        blocks = [AddBlock(value) for value in (100.,0.,0.,0.,0.,0.,0.,1.)]

        def __call__(self, index, x, **kwargs):
            return self.blocks[index](x, **kwargs)

    class BranchController:
        def __init__(self, values):
            self.branch = torch.tensor(values, requires_grad=True)
            self.calls = 0

        def __call__(self, **kwargs):
            logits = kwargs["h"].new_full((1,11), -torch.inf)
            if self.calls == 1:
                # These two legal successor states are both the live h.
                logits[0,8:10] = self.branch
            elif self.calls == 2:
                # ADVANCE(0) is legal only after the preceding RESTART.
                logits[0,0] = 100.
                logits[0,7] = 0.
            else:
                action = {0:6, 3:9, 4:6, 5:8, 6:9, 7:6}[self.calls]
                logits[0,action] = 100.
            self.calls += 1
            return {"logits": logits}

    # Fix equal Gumbel draws to isolate the forward branch comparison;
    # never change noise sampling in the actual Task-A experiments.
    original_rand_like = torch.rand_like
    results = []
    try:
        torch.rand_like = lambda tensor: torch.full_like(tensor, 0.5)
        for values in ([1.,0.], [0.,1.]):
            controller = BranchController(values)
            executor = AdaptiveExecutor(Bank(),controller,max_execution_depth=8)
            h0 = torch.zeros(1,2,requires_grad=True)
            result = executor.execute_differentiable(h0,torch.zeros_like(h0),training=True)
            loss = result.h_terminal.square().mean()
            loss.backward()
            probabilities = controller.branch.detach().double().softmax(-1)
            conditional_losses = torch.tensor([1.,10000.], dtype=torch.float64)
            expected_loss = (probabilities * conditional_losses).sum()
            expected_gradient = probabilities * (conditional_losses - expected_loss)
            results.append(dict(branch_logits=values, terminal=result.h_terminal.detach().tolist(),
                                loss=loss.item(), branch_gradient=controller.branch.grad.tolist(),
                                branch_probabilities=probabilities.tolist(),
                                expected_loss_resampling_branch_only=expected_loss.item(),
                                expected_categorical_gradient_resampling_branch_only=expected_gradient.tolist(),
                                history=result.action_history))
    finally:
        torch.rand_like = original_rand_like
    return dict(scope="eight-block toy identity/additive bank, eight fixed decisions; no model redesign",
                expected_gradient_scope="ideal categorical draw at the branch only; all other noise draws held fixed",
                results=results,
                conclusion="hidden-state ST mixing cannot credit a pure pointer/mask effect when candidate h states coincide")


@torch.no_grad()
def metadata_dependency_profile(model, x):
    """Real-bank contrasts on teacher states; does not alter a routing policy.

    Holding h fixed while changing pointer/mask isolates dependence of the
    next controller choice on metadata. This is not a causal Task-A loss
    intervention. First-layer projections measure input effects, not rewards.
    """
    state = model.initial_persistent_state(1,x.device,x.dtype)
    compatibility = torch.zeros(1,model.dim,device=x.device,dtype=x.dtype)
    rows = []
    context_unchanged = True
    for token in range(x.shape[1]):
        work = state.clone()
        versions = {(kind,key):cache.version for kind,mapping in
                    (("kv",work.ctx.kv),("swa",work.ctx.swa),("gdn",work.ctx.gdn)) for key,cache in mapping.items()}
        memory = model.memory.read(work.s,work.memory)[0]
        mean = work.memory.mean(1)
        h0 = model.prelude(x[:,token],work.s,memory,compatibility)
        h = h0
        for index in range(8):
            h = model.executor._run_block(index,h,h0,memory,work.ctx)
            # After a further routing decision, at least one decision must
            # remain for its pointer/mask effect to influence this token.
            if index >= 6:
                continue
            repeated = model.executor._run_block(index,h,h0,memory,work.ctx)
            p_repeat = h.new_zeros(1,9)
            p_repeat[:,index] = 1
            p_restart = h.new_zeros(1,9)
            p_restart[:,0] = 1
            mask_repeat = build_execution_mask(index,index+2,False,8)[:10].to(x.device)
            mask_restart = build_execution_mask(0,index+2,True,8)[:10].to(x.device)

            def selection(hidden,pointer,mask):
                logits = model.controller(h=hidden,memory_read=memory,memory_mean=mean,
                                          pointer_one_hot=pointer,budget=h.new_zeros(1,2))["logits"][:, :10]
                choice = int(logits.masked_fill(~mask,-torch.inf).argmax(-1))
                assert mask[choice]
                return choice

            held_repeat = selection(h,p_repeat,mask_repeat)
            restart = selection(h,p_restart,mask_restart)
            actual_repeat = selection(repeated,p_repeat,mask_repeat)
            weights = model.controller.mlp[0].weight
            pointer_start = 2*model.dim + model.controller.memory_dim
            live_effect = F.linear(model.controller.norm_h(repeated)-model.controller.norm_h(h),
                                   weights[:, :model.dim]).norm().item()
            pointer_effect = F.linear(p_repeat-p_restart,weights[:,pointer_start:pointer_start+9]).norm().item()
            rows.append(dict(token=token,block=index,species=model.blocks.blocks[index].species,
                             future_decisions=6-index,
                             repeat_relative_live_change=(repeated-h).norm().item()/max(h.norm().item(),1e-12),
                             normalized_live_first_layer_change_l2=live_effect,
                             pointer_first_layer_change_l2=pointer_effect,
                             mask_symmetric_difference=int((mask_repeat ^ mask_restart).sum()),
                             held_h_repeat_choice=held_repeat,restart_choice=restart,actual_repeat_choice=actual_repeat,
                             metadata_only_changes_choice=held_repeat != restart))
        context_unchanged &= all(cache.version == versions[(kind,key)] for kind,mapping in
                                 (("kv",work.ctx.kv),("swa",work.ctx.swa),("gdn",work.ctx.gdn)) for key,cache in mapping.items())
        state = model.step(x[:,token],state,compatibility)["state"]
    if not context_unchanged:
        raise RuntimeError("In-flight profile changed context versions")
    return dict(scope="real-bank teacher-state metadata dependence; no optimizer or causal task-loss claim",
                context_versions_unchanged=context_unchanged,rows=rows,
                metadata_choice_changes=sum(row["metadata_only_changes_choice"] for row in rows),
                observations=len(rows))


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
    log_probabilities = []

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
        log_probabilities.append(F.log_softmax(logits.masked_fill(~mask, -torch.inf), -1)[0, decision].item())
    return dict(rows=len(expected), max_absolute_difference=differences,
                exact_alignment=all(value == 0 for value in differences.values()),
                minimum_teacher_action_probability=min(probabilities),
                log_nominal_teacher_trajectory_probability=sum(log_probabilities),
                nominal_teacher_trajectory_probability=math.exp(sum(log_probabilities)),
                nominal_canonical_token_probabilities=[math.exp(sum(log_probabilities[i:i+8]))
                                                       for i in range(0, len(log_probabilities), 8)],
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
    # Independent route interventions with equal sample counts per length.
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
