"""Matched fresh/warm-start Task-A replication, including no-update controls."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.task_a import TaskAParity
from experiments.run_stage3_controlled_audit import evaluate_cases
from experiments.run_stage3_taskA_controller import run_diff_sequence, set_seed
from experiments.stage3_measurements import (
    CHECKPOINT, count_block_calls, file_sha256, load_model, parameter_gradients, route_metrics,
)


def save_report(path, report):
    temporary = path.with_suffix(".inprogress")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def bits_for_batches(batches, task):
    return [dict(bits=(batch.x == task.token_features[1]).all(-1).long().cpu().tolist(),
                 targets=batch.target.cpu().tolist()) for batch in batches]


def measured_evaluation(model, cases):
    result = {}
    for split, batches in cases.items():
        with count_block_calls(model) as counter:
            metrics = evaluate_cases(model, {split: batches})[split]
        metrics["actual_executor_block_calls"] = counter["block_calls"]
        if counter["block_calls"] != metrics["routing"]["selected_block_calls"]:
            raise RuntimeError("Inference block calls differ from selected-path computation")
        result[split] = metrics
    return result


def summarize(runs):
    summary = {}
    for arm in ("random", "imitation", "imitation_no_update"):
        arm_runs = [r for r in runs if r["initialization"] == arm and r["numerically_stable"]]
        if not arm_runs:
            continue
        summary[arm] = {}
        for split in arm_runs[0]["after"]:
            result = {}
            for metric in ("accuracy", "mean_ce", "exact_route_fraction", "repeat_rate", "restart_rate", "unique_blocks_per_token", "selected_block_calls_per_token"):
                values = [r["after"][split]["ce" if metric == "mean_ce" else metric]
                          if metric in ("accuracy", "mean_ce") else r["after"][split]["routing"][metric]
                          for r in arm_runs]
                result[metric] = dict(mean=statistics.mean(values),
                                      sample_sd=statistics.stdev(values) if len(values) > 1 else None,
                                      by_seed={str(r["seed"]): value for r, value in zip(arm_runs, values)})
            result["perfect_task_seed_count"] = sum(r["after"][split]["accuracy"] == 1 for r in arm_runs)
            result["seeds"] = len(arm_runs)
            summary[arm][split] = result
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[3305,3306,3307])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--per-length", type=int, default=4)
    parser.add_argument("--warmstart-template", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".inprogress").exists():
        parser.error("Output exists; use a new run path")
    warmstarts = {seed: Path(args.warmstart_template.format(seed=seed)) for seed in args.seeds}
    for seed, path in warmstarts.items():
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint.get("seed") != seed or not checkpoint.get("route_recovery"):
            parser.error(f"Unverified imitation checkpoint for seed {seed}")
        if checkpoint.get("backbone_sha256") != file_sha256(CHECKPOINT):
            parser.error("Imitation backbone fingerprint differs")
        for arm in ("random", "imitation"):
            if args.output.with_name(f"{args.output.stem}_{seed}_{arm}.pt").exists():
                parser.error("Output checkpoint already exists")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = TaskAParity(64, device)
    cases = {"val": [task.make_fixed_set("val", n, args.per_length) for n in range(4,17)],
             "in_distribution_test": [task.make_fixed_set("val", n, args.per_length) for n in range(4,17)],
             "ood_test": [task.make_fixed_set("test", n, args.per_length) for n in range(17,33)]}
    report = dict(hypothesis="supervised route warm start versus fresh task-loss route discovery",
                  scope="D64 frozen backbone; supervised warm start is a distinct methodology",
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  torch_version=torch.__version__, device=str(device),
                  backbone=str(CHECKPOINT), backbone_sha256=file_sha256(CHECKPOINT),
                  model_dim=64, routing_depth=8, budget="zeros", temperature=2.0,
                  controller_lr=1e-4, weight_decay=0.0, batch_size=1, steps=args.steps,
                  seeds=args.seeds, per_length=args.per_length, runs=[],
                  source_sha256={str(path): file_sha256(path) for path in
                                 (Path("src/routing.py"), Path("src/executor.py"), Path("src/agt.py"),
                                  Path("experiments/run_stage3_replication.py"))},
                  evaluation_data={split: bits_for_batches(batches, task) for split,batches in cases.items()},
                  training_data={})
    # Reserve the new run identity before starting; subsequent writes only
    # update this run's own progress record.
    with args.output.open("x") as handle:
        json.dump(report, handle)
    for seed in args.seeds:
        set_seed(seed)
        train_task = TaskAParity(64, device)
        batches = [train_task.make_batch(1, int(torch.randint(4,17,(1,))), "train")
                   for _ in range(args.steps)]
        seed_cases = {"train": batches[-16:], **cases}
        report["training_data"][str(seed)] = bits_for_batches(batches, train_task)
        imitation_before = None
        for arm in ("random", "imitation"):
            warmstart = warmstarts[seed] if arm == "imitation" else None
            model = load_model(seed, device, warmstart)
            frozen = {name: value.detach().clone() for name,value in model.state_dict().items()
                      if not name.startswith("controller.")}
            before = measured_evaluation(model, seed_cases)
            if arm == "imitation":
                imitation_before = before
            run = dict(seed=seed, initialization=arm, before=before, training=[],
                       numerically_stable=True, warmstart=str(warmstart) if warmstart else None,
                       warmstart_sha256=file_sha256(warmstart) if warmstart else None,
                       trainable_parameters=[dict(name=name, numel=p.numel())
                                             for name,p in model.named_parameters() if p.requires_grad])
            optimizer = torch.optim.AdamW(model.controller.parameters(), lr=1e-4, weight_decay=0)
            set_seed(seed+10000)
            model.train()
            for step, batch in enumerate(batches, 1):
                optimizer.zero_grad(set_to_none=True)
                with count_block_calls(model) as counter:
                    logits, histories = run_diff_sequence(model, batch.x, 2.0, True)
                loss = F.cross_entropy(logits, batch.target)
                loss.backward()
                gradients = parameter_gradients(model)
                finite = bool(torch.isfinite(loss)) and all(g["present"] and g["finite"] for g in gradients.values())
                row = dict(step=step, length=batch.length, ce=loss.item(), gradients=gradients,
                           actual_candidate_block_calls=counter["block_calls"],
                           routing=route_metrics(histories))
                run["training"].append(row)
                if not finite:
                    run["numerically_stable"] = False
                    run["failure"] = "Non-finite loss or missing/non-finite controller gradient"
                    report["runs"].append(run)
                    save_report(args.output, report)
                    raise RuntimeError(run["failure"])
                optimizer.step()
                if step % 25 == 0 or step == 1:
                    print(seed, arm, step, "ce", row["ce"], flush=True)
            model.eval()
            run["after"] = measured_evaluation(model, seed_cases)
            run["backbone_unchanged"] = all(torch.equal(value, model.state_dict()[name])
                                            for name,value in frozen.items())
            if not run["backbone_unchanged"]:
                raise RuntimeError("Frozen backbone mutated")
            path = args.output.with_name(f"{args.output.stem}_{seed}_{arm}.pt")
            checkpoint = dict(seed=seed, dim=64, routing_decisions=8, budget="zeros", steps=args.steps,
                              controller_state_dict={k:v.detach().cpu() for k,v in model.controller.state_dict().items()},
                              backbone_sha256=report["backbone_sha256"], initialization=arm)
            with path.open("xb") as handle:
                torch.save(checkpoint, handle)
            run["checkpoint"] = str(path)
            run["checkpoint_sha256"] = file_sha256(path)
            report["runs"].append(run)
            report["summary"] = summarize(report["runs"])
            save_report(args.output, report)
            print("completed", seed, arm, {s:r["accuracy"] for s,r in run["after"].items()}, flush=True)
        # Independently reload the supervised initialization, run no optimizer,
        # and repeat evaluation after both optimized arms. This is a genuine
        # no-update control using the same token sequences and persistent resets.
        control = load_model(seed, device, warmstarts[seed])
        after = measured_evaluation(control, seed_cases)
        report["runs"].append(dict(seed=seed, initialization="imitation_no_update", before=imitation_before,
                                   after=after, numerically_stable=True, optimizer_updates=0,
                                   repeated_metrics_identical=after == imitation_before))
        report["summary"] = summarize(report["runs"])
        save_report(args.output, report)
    report["completed"] = True
    save_report(args.output, report)
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
