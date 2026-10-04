"""Paired frozen-backbone Gate-C diagnostic; no architecture/objective changes."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_stage3_taskA_controller import (
    CHECKPOINT, AGTv3, TaskAParity, controller_gradient_coverage,
    run_diff_sequence, run_fixed_sequence, set_seed,
)


@torch.no_grad()
def evaluate_cases(model, cases):
    result = {}
    expected = [("ADVANCE", i) for i in range(8)]
    for split, batches in cases.items():
        losses, fixed_losses = [], []
        correct = fixed_correct = tokens = exact = 0
        decisions = [0] * 8
        actions = Counter()
        max_delta = 0.0
        for batch in batches:
            for i in range(batch.target.numel()):
                x, target = batch.x[i:i+1], batch.target[i:i+1]
                fixed = run_fixed_sequence(model, x)
                learned, histories = run_diff_sequence(model, x, 2.0, False)
                losses.append(F.cross_entropy(learned, target).item())
                fixed_losses.append(F.cross_entropy(fixed, target).item())
                correct += int((learned.argmax(-1) == target).sum())
                fixed_correct += int((fixed.argmax(-1) == target).sum())
                max_delta = max(max_delta, (fixed - learned).abs().max().item())
                for history in histories:
                    assert len(history) == 8
                    tokens += 1
                    exact += int(history == expected)
                    for j, action in enumerate(history):
                        actions[str(action)] += 1
                        decisions[j] += int(action == expected[j])
        result[split] = dict(
            examples=len(losses), accuracy=correct/len(losses),
            ce=sum(losses)/len(losses), fixed_accuracy=fixed_correct/len(losses),
            fixed_ce=sum(fixed_losses)/len(fixed_losses),
            max_logit_delta=max_delta, tokens=tokens, exact_route_tokens=exact,
            decision_accuracy=[n/tokens for n in decisions], actions=dict(actions),
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[3401, 3402, 3403])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--per-length", type=int, default=4)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = torch.load(CHECKPOINT, map_location="cpu")
    weights = {k: v for k, v in backbone["state_dict"].items()
               if not k.startswith("controller.")}
    warmstart = torch.load(args.warmstart, map_location="cpu")
    task = TaskAParity(int(backbone["dim"]), device)
    cases = {split: [task.make_fixed_set(split, length, args.per_length)
                     for length in lengths]
             for split, lengths in (("val", range(4, 17)), ("test", range(17, 33)))}
    report = dict(scope="paired diagnostic; warmstart is supervised route imitation",
                  backbone=str(CHECKPOINT), warmstart=str(args.warmstart),
                  steps=args.steps, lr=1e-4, tau=2.0, budget="zeros",
                  routing_decisions=8, per_length=args.per_length, runs=[])
    for seed in args.seeds:
        # Generate once, then reuse exact token sequences across both arms.
        set_seed(seed)
        train_task = TaskAParity(int(backbone["dim"]), device)
        batches = [train_task.make_batch(1, int(torch.randint(4, 17, (1,))), "train")
                   for _ in range(args.steps)]
        for initialization in ("random", "imitation"):
            set_seed(seed)
            model = AGTv3(dim=int(backbone["dim"]), num_memory_slots=16, vocab_size=2)
            missing, unexpected = model.load_state_dict(weights, strict=False)
            assert set(missing) == {k for k in model.state_dict() if k.startswith("controller.")}
            assert not unexpected
            if initialization == "imitation":
                model.controller.load_state_dict(warmstart["controller_state_dict"], strict=True)
            model.to(device).eval()
            model.executor.max_execution_depth = 8
            for p in model.parameters():
                p.requires_grad_(False)
            for p in model.controller.parameters():
                p.requires_grad_(True)
            before = evaluate_cases(model, cases)
            optimizer = torch.optim.AdamW(model.controller.parameters(), lr=1e-4, weight_decay=0)
            set_seed(seed + 10000)  # Paired Gumbel draws; evaluation consumes no RNG.
            training = []
            model.train()
            for step, batch in enumerate(batches, 1):
                optimizer.zero_grad(set_to_none=True)
                logits, history = run_diff_sequence(model, batch.x, 2.0, True)
                loss = F.cross_entropy(logits, batch.target)
                loss.backward()
                coverage = controller_gradient_coverage(model)
                assert coverage["parameters_with_grad"] == len(list(model.controller.parameters()))
                assert coverage["finite_grad_parameters"] == coverage["parameters_with_grad"]
                training.append(dict(step=step, length=batch.length, ce=loss.item(),
                                     gradient_coverage=coverage,
                                     actions=dict(Counter(str(a) for h in history for a in h))))
                optimizer.step()
            model.eval()
            after = evaluate_cases(model, cases)
            run = dict(seed=seed, initialization=initialization, before=before,
                       after=after, training=training)
            report["runs"].append(run)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: v for k, v in run.items() if k != "training"}), flush=True)


if __name__ == "__main__":
    main()
