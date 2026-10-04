"""Average existing ST gradients; no new estimator, loss, or architecture."""
from __future__ import annotations

import argparse
import traceback
import gzip
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from data.task_a import TaskAParity, TaskABatch
from experiments.run_stage3_replication import measured_evaluation, save_report
from experiments.run_stage3_taskA_controller import run_diff_sequence, set_seed
from experiments.stage3_measurements import CHECKPOINT, count_block_calls, file_sha256, load_model, parameter_gradients, route_metrics


def averaged_gradients(model, batch, noise_seeds):
    vectors, samples = [], []
    parameters = list(model.controller.parameters())
    for seed in noise_seeds:
        model.zero_grad(set_to_none=True)
        set_seed(seed)
        with count_block_calls(model) as counter:
            logits, histories = run_diff_sequence(model,batch.x,2.0,True)
        loss = F.cross_entropy(logits,batch.target)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise RuntimeError("Non-finite loss or missing/non-finite sample gradient")
        vectors.append(torch.cat([p.grad.detach().reshape(-1).cpu() for p in parameters]))
        samples.append(dict(noise_seed=seed,ce=loss.item(),routing=route_metrics(histories),
                            actual_candidate_block_calls=counter["block_calls"]))
    matrix = torch.stack(vectors)
    mean = matrix.mean(0)
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        parameter.grad = mean[offset:offset+size].reshape_as(parameter).to(parameter.device)
        offset += size
    return mean,dict(samples=samples,mean_ce=statistics.mean(row["ce"] for row in samples),
                     averaged_gradient_l2=mean.norm().item(),
                     sample_gradient_l2=matrix.norm(dim=1).tolist(),
                     sample_gradient_dispersion_l2=(matrix-mean).square().mean(0).sum().sqrt().item(),
                     gradients=parameter_gradients(model))


def reconstruct(records,task):
    batches = []
    for record in records:
        bits = torch.tensor(record["bits"],device=task.device,dtype=torch.long)
        targets = torch.tensor(record["targets"],device=task.device,dtype=torch.long)
        assert torch.equal(task.target_from_bits(bits),targets)
        batches.append(TaskABatch(x=task.token_features[bits],target=targets,length=bits.shape[1]))
    return batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=20)
    parser.add_argument("--samples",type=int,nargs="+",default=[1,8])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    if args.steps < 1 or any(k < 1 for k in args.samples):
        parser.error("Steps and sample counts must be positive")
    raw = args.reference.read_bytes()
    reference = json.loads(gzip.decompress(raw) if args.reference.suffix == ".gz" else raw)
    if any(len(reference["training_data"][str(seed)]) < args.steps for seed in reference["seeds"]):
        parser.error("Requested updates exceed saved reference data")
    if not reference.get("completed"):
        parser.error("Reference must be complete")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = TaskAParity(64,device)
    cases = {split:reconstruct(records,task) for split,records in reference["evaluation_data"].items()}
    report = dict(hypothesis="sampling variance versus h-only surrogate credit limitation",
                  scope="fresh frozen-backbone controllers; update/data matched, not compute matched",
                  backbone=str(CHECKPOINT),backbone_sha256=file_sha256(CHECKPOINT),
                  reference=str(args.reference),reference_sha256=file_sha256(args.reference),
                  seeds=reference["seeds"],steps=args.steps,samples=args.samples,runs=[],
                  dim=64,routing_depth=8,budget="zeros",tau=2.0,lr=1e-4,weight_decay=0.0,
                  noise_rule="100000000 + seed*1000000 + step*100 + sample_index; paired first draw per update")
    with args.output.open("x") as handle:
        json.dump(report,handle)
    for seed in reference["seeds"]:
        batches = reconstruct(reference["training_data"][str(seed)][:args.steps],task)
        for count in args.samples:
            model = load_model(seed,device)
            frozen = {name:value.clone() for name,value in model.state_dict().items() if not name.startswith("controller.")}
            before = measured_evaluation(model,cases)
            optimizer = torch.optim.AdamW(model.controller.parameters(),lr=1e-4,weight_decay=0)
            run = dict(seed=seed,samples_per_update=count,before=before,training=[],numerically_stable=True)
            report["runs"].append(run)
            save_report(args.output,report)
            model.train()
            for step,batch in enumerate(batches,1):
                noise = [100000000+seed*1000000+step*100+sample for sample in range(count)]
                try:
                    _,row = averaged_gradients(model,batch,noise)
                except Exception:
                    run["numerically_stable"] = False
                    run["failure"] = traceback.format_exc()
                    save_report(args.output,report)
                    raise
                row.update(step=step,length=batch.length)
                run["training"].append(row)
                optimizer.step()
                if step == 1 or step % 10 == 0:
                    print(seed,count,step,row["mean_ce"],flush=True)
            model.eval()
            run["after"] = measured_evaluation(model,cases)
            run["backbone_unchanged"] = all(torch.equal(value,model.state_dict()[name]) for name,value in frozen.items())
            if not run["backbone_unchanged"]:
                raise RuntimeError("Frozen backbone mutated")
            save_report(args.output,report)
            print("completed",seed,count,{split:m["accuracy"] for split,m in run["after"].items()},flush=True)
    report["summary"] = {}
    for count in args.samples:
        runs = [r for r in report["runs"] if r["samples_per_update"] == count]
        report["summary"][str(count)] = {}
        for split in cases:
            values = [run["after"][split]["accuracy"] for run in runs]
            report["summary"][str(count)][split] = dict(mean_accuracy=statistics.mean(values),
                                                       sample_sd=statistics.stdev(values) if len(values)>1 else None)
    report["completed"] = True
    save_report(args.output,report)


if __name__ == "__main__":
    main()
