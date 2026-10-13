"""Retained-logit measurements only: locked Task-A routing, no optimization."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.task_a import TaskAParity
from experiments.run_stage3_taskA_controller import run_diff_sequence, run_fixed_sequence, set_seed
from experiments.stage3_measurements import CHECKPOINT, file_sha256, load_model, parameter_gradients
from src.masks import build_execution_mask
from experiments.stage3_decision_prefix import observe_token_prefix

SEED, DEPTH, TAU = 3401, 8, 2.0
LABELS = [f"ADVANCE({i})" for i in range(8)] + ["REPEAT", "RESTART", "HALT"]


class RetainedController:
    def __init__(self, controller):
        self.controller, self.rows, self.tokens = controller, [], []
        self.token, self.decision = -1, 0

    def __call__(self, **kwargs):
        output = self.controller(**kwargs)
        logits = output["logits"]
        logits.retain_grad()
        if self.token < 0:
            raise RuntimeError("Controller capture requires an observed executor token boundary")
        self.rows.append(dict(token=self.token, decision=self.decision,
                              logits=logits, pointer=int(kwargs["pointer_one_hot"].argmax(-1).item()),
                              pointer_one_hot_requires_grad=kwargs["pointer_one_hot"].requires_grad))
        self.decision += 1
        return output


@contextmanager
def capture(model):
    original_controller = model.executor.controller
    original_step = model.executor.differentiable_route_step
    original_execute = model.executor.execute_differentiable
    proxy = RetainedController(original_controller)

    def observed_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        row = proxy.rows[-1]
        mask = build_execution_mask(kwargs["pointer"], kwargs["execution_count"],
                                    kwargs["just_restarted"], 8).to(kwargs["h"].device)
        selected = int(result["selected_action"].item())
        row.update(mask=mask, just_restarted=kwargs["just_restarted"],
                   execution_before=kwargs["execution_count"],
                   execution_after=kwargs["execution_count"] + int(selected < 9),
                   terminal_decision=False,
                   probabilities=result["probs"].detach(), selected=int(result["selected_action"].item()),
                   next_pointer=int(result["next_pointer"].item()),
                   selected_action_requires_grad=result["selected_action"].requires_grad,
                   next_pointer_requires_grad=result["next_pointer"].requires_grad,
                   next_pointer_grad_fn=result["next_pointer"].grad_fn is not None,
                   mask_requires_grad=mask.requires_grad)
        return result

    def observed_execute(*args, **kwargs):
        # AGTv3 invokes the executor exactly once for each actual input token.
        proxy.token += 1
        proxy.decision = 0
        start = len(proxy.rows)
        result = original_execute(*args, **kwargs)
        rows = proxy.rows[start:]
        if len(rows) != len(result.action_history):
            raise RuntimeError("Captured controller decisions differ from token history")
        if rows:
            rows[-1]["terminal_decision"] = True
        proxy.tokens.append(dict(token=proxy.token, history=list(result.action_history),
                                 decision_count=len(rows), execution_count=result.execution_count))
        return result

    model.executor.execute_differentiable = observed_execute
    model.executor.controller = proxy
    model.executor.differentiable_route_step = observed_step
    try:
        yield proxy
    finally:
        model.executor.controller = original_controller
        model.executor.differentiable_route_step = original_step
        model.executor.execute_differentiable = original_execute


def measured_example(model, x, target, example):
    model.zero_grad(set_to_none=True)
    with capture(model) as proxy:
        prediction, histories = run_diff_sequence(model, x, TAU, True)
        loss = F.cross_entropy(prediction, target)
        loss.backward()
    rows = []
    for observed in proxy.rows:
        logits = observed["logits"]
        if logits.grad is None:
            raise RuntimeError("A retained logit tensor has no loss gradient")
        gradient = logits.grad.detach()[0]
        probabilities = observed["probabilities"][0]
        mask = observed["mask"]
        positive = probabilities > 0
        entropy = -(probabilities[positive] * probabilities[positive].log()).sum().item()
        full_probs = logits.detach()[0].masked_fill(~mask, -torch.inf).softmax(-1)
        selected = observed["selected"]
        if not mask[selected] or selected == 10:
            raise RuntimeError("Illegal/HALT forward action")
        kind, block = histories[observed["token"]][observed["decision"]]
        if selected != (block if kind == "ADVANCE" else {"REPEAT": 8, "RESTART": 9}[kind]):
            raise RuntimeError("Captured selection differs from real executor history")
        rows.append(dict(example=example, token=observed["token"], decision=observed["decision"],
                         execution_before=observed["execution_before"],
                         execution_after=observed["execution_after"],
                         terminal_decision=observed["terminal_decision"],
                         pointer_before=observed["pointer"], just_restarted=observed["just_restarted"],
                         legal_successor_actions=[i for i in range(10) if mask[i]],
                         logits=logits.detach()[0].tolist(), gradient=gradient.tolist(),
                         gradient_l2=gradient.norm().item(), max_abs_gradient=gradient.abs().max().item(),
                         legal_gradient_l2=gradient[:10][mask[:10]].norm().item(),
                         illegal_gradient_max=gradient[:10][~mask[:10]].abs().max().item()
                         if (~mask[:10]).any() else 0.0,
                         halt_gradient=gradient[10].item(), probabilities=probabilities.tolist(),
                         entropy=entropy, max_legal_probability=probabilities.max().item(),
                         pre_exclusion_halt_probability=full_probs[10].item(), selected_action=selected,
                         next_pointer=observed["next_pointer"],
                         metadata={key:observed[key] for key in
                                   ["pointer_one_hot_requires_grad", "selected_action_requires_grad",
                                    "next_pointer_requires_grad", "next_pointer_grad_fn", "mask_requires_grad"]}))
    return dict(example=example, ce=loss.item(), target=target.item(), prediction=int(prediction.argmax(-1)),
                histories=histories, controller_gradients=parameter_gradients(model), decisions=rows)


class ScriptedPolicy:
    """One replacement in an explicit token's eight-decision prefix; no fallback."""
    def __init__(self, intervention=None, token=7):
        self.intervention, self.calls, self.pointer, self.restarted = intervention, 0, 0, False
        self.token, self.executions = token, 0

    def __call__(self, **kwargs):
        decision = self.calls
        if decision >= DEPTH:
            raise RuntimeError("Scripted decision prefix exhausted; no continuation supplied")
        action = decision
        if self.intervention is not None and (self.token, decision) == (7, 4):
            action = self.intervention
        self.pointer = int(kwargs['pointer_one_hot'].argmax(-1).item())
        mask = build_execution_mask(self.pointer, self.executions, self.restarted, 8)
        if not mask[action] or action == 10:
            raise RuntimeError("Script requested an illegal action")
        logits = kwargs["h"].new_full((1, 11), -100.0)
        logits[:, action] = 100.0
        if action < 8:
            self.pointer, self.restarted = action, False
        elif action == 8:
            self.restarted = False
        else:
            self.pointer, self.restarted = 0, True
        self.executions += action != 9
        self.calls += 1
        return {"logits": logits}


@torch.no_grad()
def route_sensitivity(model, x, target):
    """One prescribed substitution in token7's uncommitted D=8 prefix.

    The first seven canonical tokens provide identical committed input state.
    CE uses only the existing final norm/head on the observed h: a prefix
    readout, not a completed-token result. No terminal commit is performed on
    any observed prefix, including the canonical P=8 control.
    """
    if x.shape != (1, 8, model.dim):
        raise ValueError("Sensitivity control requires one eight-token sequence")
    state = model.initial_persistent_state(1, x.device, x.dtype)
    compatibility = x.new_zeros(1, model.dim)
    for token in range(7):
        state = model.step(x[:, token], state, compatibility)['state']
    original = model.executor.controller
    results = []
    try:
        for intervention in [None, 8, 9]:
            model.executor.controller = ScriptedPolicy(intervention)
            prefix = observe_token_prefix(model, x[:, 7], state, compatibility,
                                           decision_horizon=8, training=False, token=7)
            histories = [[('ADVANCE', d) for d in range(8)] for _ in range(7)] + [prefix.action_history]
            logits = model.output_head(model.commit(prefix.h_observed))
            expected = [[("ADVANCE", d) for d in range(8)] for _ in range(8)]
            if intervention:
                expected[7][4] = ("REPEAT", 3) if intervention == 8 else ("RESTART", None)
            assert histories == expected
            results.append(dict(token=7, decision=4, action=LABELS[4 if intervention is None else intervention],
                                prefix_readout_ce=F.cross_entropy(logits, target).item(),
                                prefix_readout_prediction=int(logits.argmax(-1)),
                                prefix_readout_logits=logits.tolist(),
                                prefix_decision_count=prefix.decision_count,
                                prefix_execution_count=prefix.execution_count, prefix_committed=prefix.committed,
                                termination_reason=prefix.termination_reason, prefix_rows=prefix.rows,
                                completed_reference_tokens=7,
                                histories=histories, block_calls=sum(kind != "RESTART" for h in histories for kind,_ in h),
                                pointer_after=4 if intervention is None else (3 if intervention == 8 else 0),
                                just_restarted_after=intervention == 9, additional_blocks_in_token=3))
    finally:
        model.executor.controller = original
    for result in results:
        result["delta_prefix_readout_ce"] = result["prefix_readout_ce"] - results[0]["prefix_readout_ce"]
    return results


def matrix(rows, field):
    tokens = sorted({r["token"] for r in rows})
    width = max((r["decision"] for r in rows), default=-1) + 1
    values = []
    for token in tokens:
        token_row = []
        for decision in range(width):
            observations = [r[field] for r in rows if r["token"] == token and r["decision"] == decision]
            token_row.append(statistics.mean(observations) if observations else None)
        values.append(token_row)
    return values


def aggregate(rows):
    distributions = []
    for decision in sorted({r["decision"] for r in rows}):
        subset = [r for r in rows if r["decision"] == decision]
        counts = Counter(r["selected_action"] for r in subset)
        by_action = []
        for action in range(11):
            legal = [r for r in subset if action in r["legal_successor_actions"]]
            by_action.append(dict(action=LABELS[action], mean_abs_gradient=statistics.mean(abs(r["gradient"][action]) for r in subset),
                                  legal_observations=len(legal), mean_abs_gradient_when_legal=
                                  statistics.mean(abs(r["gradient"][action]) for r in legal) if legal else None))
        distributions.append(dict(decision=decision, observations=len(subset),
                                  counts={LABELS[i]:counts[i] for i in range(11)},
                                  frequencies={LABELS[i]:counts[i]/len(subset) for i in range(11)},
                                  mean_entropy=statistics.mean(r["entropy"] for r in subset),
                                  mean_max_legal_probability=statistics.mean(r["max_legal_probability"] for r in subset),
                                  mean_probabilities=[statistics.mean(r["probabilities"][i] for r in subset) for i in range(10)],
                                  mean_pre_exclusion_halt_probability=statistics.mean(r["pre_exclusion_halt_probability"] for r in subset),
                                  mean_gradient_l2=statistics.mean(r["gradient_l2"] for r in subset), by_action=by_action))
    counts = Counter(r["selected_action"] for r in rows)
    return dict(by_decision=distributions, total_counts={LABELS[i]:counts[i] for i in range(11)},
                halt_max_abs_gradient=max(abs(r["halt_gradient"]) for r in rows),
                halt_exact_zero_count=sum(r["halt_gradient"] == 0 for r in rows),
                present_gradient_count=len(rows), nonzero_gradient_count=sum(r["gradient_l2"] > 0 for r in rows),
                decision7_nonzero_count=sum(r["gradient_l2"] > 0 for r in rows if r["decision"] == 7),
                decision7_observations=sum(r["decision"] == 7 for r in rows),
                terminal_decision_observations=sum(r["terminal_decision"] for r in rows),
                terminal_decision_nonzero_count=sum(r["gradient_l2"] > 0 for r in rows if r["terminal_decision"]),
                metadata_all_non_differentiable=all(not flag for r in rows for flag in r["metadata"].values()))


def markdown(report):
    a = report["aggregate"]
    text = ["# Task-A retained-logit diagnostic — seed3401", "",
            "Measurement only; no optimizer, parameter updates, or core changes.",
            f"Eight train examples of length eight, fresh D64 controller, frozen Stage-1 seed3305 backbone; tau={TAU}, zero budget, eight block executions/token with variable decision counts, training=True. Data uses Task-A's fixed FEATURE_SEED=3002 and train generator seed3001; controller and routing seed3401. Routing RNG is reset once before the eight-example stream, not per example.", "",
            f"Retained gradients: {a['present_gradient_count']} present, {a['nonzero_gradient_count']} nonzero. Decision 7: {a['decision7_nonzero_count']}/{a['decision7_observations']} nonzero. Final observed decisions: {a['terminal_decision_nonzero_count']}/{a['terminal_decision_observations']} nonzero. HALT: {a['halt_exact_zero_count']}/{a['present_gradient_count']} exactly zero; maximum absolute gradient {a['halt_max_abs_gradient']:.8g}.",
            f"All captured selection/pointer/mask metadata lacks requires_grad/grad_fn: {a['metadata_all_non_differentiable']}.", ""]
    for title, values in report["matrices"].items():
        width = max(map(len, values), default=0)
        text.extend([f"## {title} (means across observed examples; blank cells have no observations)", "", "| token | " + " | ".join(map(str, range(width))) + " |",
                     "|---|" + "---|"*width])
        for t, row in enumerate(values):
            text.append(f"| {t} | " + " | ".join(f"{value:.5g}" if value is not None else "" for value in row) + " |")
        text.append("")
    text.extend(["## Decision summaries", "", "| Decision | Mean logit L2 | Mean entropy | REPEAT count | RESTART count | Mean abs REPEAT grad | Mean abs RESTART grad |", "|---|---|---|---|---|---|---|"])
    for d in a["by_decision"]:
        text.append(f"| {d['decision']} | {d['mean_gradient_l2']:.6g} | {d['mean_entropy']:.6g} | {d['counts']['REPEAT']} | {d['counts']['RESTART']} | {d['by_action'][8]['mean_abs_gradient']:.6g} | {d['by_action'][9]['mean_abs_gradient']:.6g} |")
    text.extend(["", "Total selected actions: " + json.dumps(a["total_counts"]), "",
                 "HALT is never sampled before exclusion in this path. Its full-mask probability is recorded as a hypothetical categorical diagnostic; it is not an observed HALT frequency.", "",
                 "## Eight-decision-prefix sensitivity", "", "Only token7/decision4 changes; the three supplied continuation destinations remain canonical and legal. Seven preceding canonical tokens are completed normally. The observed token7 prefix stops at eight decisions without terminal commit: canonical/REPEAT have P=8 and RESTART has P=7. CE applies the unchanged final norm/output head to observed h only; this is a prefix readout, not completed counterfactual-token performance. Successor and metadata effects are not isolated.", "", "| Action | Prefix readout CE | Delta readout CE | Readout prediction | Block calls including reference tokens |", "|---|---|---|---|---|"])
    for r in report["counterfactuals"]:
        text.append(f"| {r['action']} | {r['prefix_readout_ce']:.9g} | {r['delta_prefix_readout_ce']:.9g} | {r['prefix_readout_prediction']} | {r['block_calls']} |")
    text.extend(["", "## Interpretation boundary", "", "Frozen parameters still pass gradients through activations and persistent token states. Retained logits have a rigorous loss cotangent; shared parameter gradients are reported per example, not attributed to individual decisions. Magnitudes do not imply an optimizer preference. No optimization direction was applied.", "", "This single-seed, eight-example measurement does not establish the cause of the earlier training failure, estimator bias magnitude, or an exact hidden-state/metadata decomposition. Counterfactual prefix-readout CE is a forward observable, not a gradient or a completed-token result. Non-differentiable outgoing metadata at the final observed decision of each token is unused by terminal commit; its action logits can nevertheless affect terminal h and receive gradient. No fix is proposed.", ""])
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/results")
    args = parser.parse_args()
    output = args.output_dir / "stage3_gradient_diagnostic_seed3401.json"
    md = output.with_suffix(".md")
    if output.exists() or md.exists():
        parser.error("Diagnostic output already exists; use a new output directory")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = load_model(SEED, device)
    assert model.dim == 64 and len(model.blocks.blocks) == 8 and model.executor.max_execution_depth == 8
    assert model.memory.initial_state(batch_size=1,device=device).shape[1] == 16
    snapshot = {name:value.clone() for name,value in model.state_dict().items()}
    task = TaskAParity(64, device)
    batch = task.make_fixed_set("train", 8, 8)
    bits = torch.cdist(batch.x.reshape(-1,64), task.token_features).argmin(-1).reshape(8,8)
    assert torch.equal(task.target_from_bits(bits), batch.target)
    set_seed(SEED)
    examples = [measured_example(model, batch.x[i:i+1], batch.target[i:i+1], i) for i in range(8)]
    rows = [row for example in examples for row in example["decisions"]]
    report = dict(seed=SEED, checkpoint=str(CHECKPOINT), checkpoint_sha256=file_sha256(CHECKPOINT),
                  script_sha256=file_sha256(__file__), dim=64, num_blocks=8, memory_slots=16,
                  execution_budget=DEPTH, protocol="fixed-executions-variable-decisions",
                  task_length=8, example_count=8, split="train", tau=TAU, budget="zeros",
                  data_bits=bits.tolist(), targets=batch.target.tolist(), examples=examples,
                  matrices={key:matrix(rows,key) for key in ["gradient_l2","legal_gradient_l2","max_abs_gradient","entropy"]},
                  aggregate=aggregate(rows), counterfactuals=route_sensitivity(model,batch.x[:1],batch.target[:1]))
    report["all_weights_unchanged"] = all(torch.equal(value,model.state_dict()[name]) for name,value in snapshot.items())
    if not report["all_weights_unchanged"]:
        raise RuntimeError("Weights changed during diagnostic")
    encoded = json.dumps(report, indent=2, allow_nan=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(encoded + "\n")
    with md.open("x") as handle:
        handle.write(markdown(report))
    print(json.dumps(report["aggregate"]["total_counts"]))
    print("Wrote", output, md, flush=True)


if __name__ == "__main__":
    main()
