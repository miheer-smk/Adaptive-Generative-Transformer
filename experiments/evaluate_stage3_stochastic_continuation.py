"""Paired forward-policy evaluation; uses the real ST-Gumbel executor, no optimizer."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.task_a import TaskAParity
from experiments.diagnose_stage3_gradients import check_observation_alignment
from experiments.run_stage3_taskA_controller import run_diff_sequence, set_seed
from experiments.stage3_measurements import count_block_calls, file_sha256, load_model, route_metrics


@torch.no_grad()
def assess(model, cases, noise_seed, repeats=4):
    model.eval()
    rows, histories = [], []
    candidate_calls = 0
    for index, batch in enumerate(cases):
        for repeat in range(repeats):
            draw_seed = noise_seed + index*repeats + repeat
            set_seed(draw_seed)
            with count_block_calls(model) as counter:
                # training=True selects the real stochastic ST-Gumbel path;
                # no_grad disables autograd, not its candidate construction.
                logits, history = run_diff_sequence(model, batch.x, 2.0, True)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite stochastic output")
            execution_counts = [sum(kind in ("ADVANCE", "REPEAT") for kind, _ in token)
                                for token in history]
            if any(count != model.executor.max_execution_depth for count in execution_counts):
                raise RuntimeError("Stochastic assessment received an incomplete execution-depth token")
            candidate_calls += counter["block_calls"]
            rows.append(dict(length=batch.length, case=index, noise_seed=draw_seed,
                             ce=F.cross_entropy(logits, batch.target).item(),
                             correct=int((logits.argmax(-1) == batch.target).sum()),
                             exact_tokens=route_metrics(history)["exact_tokens"],
                             histories=history,
                             token_decision_counts=[len(token) for token in history],
                             token_execution_counts=execution_counts,
                             actual_candidate_block_calls=counter["block_calls"]))
            histories.extend(history)
    routes = route_metrics(histories)
    if routes["illegal_decisions"]:
        raise RuntimeError("Illegal stochastic forward action")
    return dict(accuracy=sum(row["correct"] for row in rows)/len(rows),
                ce=statistics.mean(row["ce"] for row in rows), trajectories=len(rows),
                actual_candidate_block_calls=candidate_calls,
                routing=routes, rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--short-imitation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new run path")
    primary = json.loads(args.primary.read_text())
    short = json.loads(args.short_imitation.read_text())
    if not primary.get("completed") or not short.get("completed"):
        parser.error("Both replication reports must be complete")
    if primary["evaluation_data"] != short["evaluation_data"] or primary["training_data"] != short["training_data"]:
        parser.error("Paired replication data differ")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = TaskAParity(64, device)
    cases = [task.make_fixed_set(split, length, 1)
             for split,length in (("val",4),("val",8),("val",12),("val",16),("test",20),("test",32))]
    report = dict(scope="stochastic forward robustness, no optimizer; six unique sequences with repeated Gumbel draws",
                  primary=str(args.primary), primary_sha256=file_sha256(args.primary),
                  short_imitation=str(args.short_imitation), short_imitation_sha256=file_sha256(args.short_imitation),
                  repeats=args.repeats, tau=2.0, routing_depth=8, budget="zeros", runs=[],
                  cases=[dict(length=batch.length,
                              bits=(batch.x == task.token_features[1]).all(-1).long().cpu().tolist(),
                              targets=batch.target.cpu().tolist()) for batch in cases])
    with args.output.open("x") as handle:
        json.dump(report,handle)
    for seed in primary["seeds"]:
        strong_run = next(r for r in primary["runs"] if r["seed"] == seed and r["initialization"] == "imitation")
        short_run = next(r for r in short["runs"] if r["seed"] == seed and r["initialization"] == "imitation")
        paths = {"short_before": short_run["warmstart"], "short_after": short_run["checkpoint"],
                 "strong_no_update": strong_run["warmstart"]}
        runs = {}
        for arm,path in paths.items():
            model = load_model(seed, device, path)
            metrics = assess(model, cases, noise_seed=53000+seed, repeats=args.repeats)
            metrics["checkpoint"] = str(path)
            metrics["checkpoint_sha256"] = file_sha256(path)
            metrics["nominal_teacher_trajectory_probabilities"] = [
                check_observation_alignment(model,batch.x)["nominal_teacher_trajectory_probability"] for batch in cases]
            runs[arm] = metrics
            print(seed, arm, "accuracy",metrics["accuracy"],"ce",metrics["ce"],flush=True)
        paired = list(zip(runs["short_before"]["rows"],runs["short_after"]["rows"]))
        report["runs"].append(dict(seed=seed, arms=runs,
                                   accuracy_difference=runs["short_after"]["accuracy"]-runs["short_before"]["accuracy"],
                                   ce_difference=runs["short_after"]["ce"]-runs["short_before"]["ce"],
                                   corrected_trajectories=sum(a["correct"] == 0 and b["correct"] == 1 for a,b in paired),
                                   worsened_trajectories=sum(a["correct"] == 1 and b["correct"] == 0 for a,b in paired)))
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    report["summary"] = {}
    for arm in ("short_before","short_after","strong_no_update"):
        report["summary"][arm] = {}
        for metric in ("accuracy","ce"):
            values = [r["arms"][arm][metric] for r in report["runs"]]
            report["summary"][arm][metric] = dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values))
    report["completed"] = True
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
