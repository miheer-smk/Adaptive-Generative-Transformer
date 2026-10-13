"""One bounded prerequisite attempt. Failures are retained; never invokes training."""
from __future__ import annotations
import os
import sys
from pathlib import Path
import hashlib
import json
import subprocess
import time
import traceback

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.dont_write_bytecode = True
sys.path[:0] = [str(OUT), str(ROOT)]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
os.environ["CUDA_CACHE_PATH"] = str(OUT / "runtime_cache")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(OUT / "runtime_cache/inductor")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def verify_integrity(before):
    differences = []
    for relative, expected in before["protected_files"].items():
        p = ROOT / relative
        if not p.is_file() or sha(p) != expected:
            differences.append(relative)
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    tracked = hashlib.sha256(diff).hexdigest()
    extra = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()
             and ".git" not in p.relative_to(ROOT).parts
             and not p.is_relative_to(OUT) and str(p.relative_to(ROOT)) not in before["protected_files"]]
    return dict(protected_count=len(before["protected_files"]), changed_or_missing=differences,
                unexpected_files_outside_attempt=extra, tracked_diff_sha256=tracked,
                tracked_diff_unchanged=tracked == before["tracked_diff_sha256"],
                historical_exception=before["historical_exception"],
                success=not differences and not extra and tracked == before["tracked_diff_sha256"])


def main():
    if (OUT / "execution_started_v3.json").exists():
        raise SystemExit("STOP: refusing to rerun or resume this prerequisite attempt")
    before = json.loads((OUT / "integrity_before.json").read_text())
    initial = verify_integrity(before)
    if not initial["success"]:
        dump("failure_v3.json", dict(boundary="before imports/tests", integrity=initial))
        raise SystemExit("STOP: initial integrity failure")
    bindings = dict(authority="full prospective protocol and final clarifications supplied in conversation; no local original-file hash claimed",
        experiment="receiver-anchored continuation training; this attempt is prerequisites only",
        programs={"research/P0":dict(actions=list(range(8)),blocks=list(range(8))),
                  "research/P1":dict(actions=[0,1,2,3,8,5,6,7],blocks=[0,1,2,3,3,5,6,7])},
        historical_v4_P1_unchanged=True, configuration=dict(dim=64,memory_slots=16,batch=1,dtype="float32",P=8,D=8,
            budget=[0,0],compatibility_input="zeros [1,64]",HALT=False,convergence=False,Gumbel=False,selector=False),
        ownership="all existing non-controller parameters shared and trainable in disposable models; controller frozen and unused",
        receiver="current values detached and cloned per episode/branch; input committed state cloned with graph preserved",
        native_objective="mean of both programs' per-token parity CE",
        arm_objective="0.5 native + 0.25 each of two continuation means; T cross-program, C same-program",
        training_not_executed=dict(common_updates=512,paired_updates_per_arm=512,AdamW=dict(lr=.0001,weight_decay=0,betas=[.9,.999],eps=1e-8),
            clip_global_norm=1,effective_batch=2,physical_batch=1,selection="paired snapshots256/512, native-only minimax CE; tie512"),
        study_seeds_not_used=dict(training_inputs=6000,common_order=6001,development_inputs=6002,development_suffixes=6003,
            selection_inputs=6004,selection_suffixes=6005,confirmation_inputs=6006,confirmation_suffixes=6007,paired_order=6008,
            bootstrap=6010,profiling_order=6012),
        prerequisite_limits=dict(attempted_tokens=512,actual_selected_bank_calls=4096,optimizer_updates=0),
        profiling_not_executed=dict(suffix="z0",paths=5,cases=32,warmups=1,timed_repetitions=5,tokens=26880,
            starts="independent original initial state for every full trajectory"),
        uncertainty=dict(replicates=20000,seed=6010,ablation_lower_quantile=.0125,latency_upper_quantile=.9875,
            interpolation="linear",families="two separate families of four",latency_estimator="ratio of arithmetic case means"),
        data="only retained archival length8 control and fixed disposable synthetic bit fixtures; no panels generated",
        historical_exception=before["historical_exception"])
    dump("protocol_binding_v3.json", bindings)
    started = time.time()
    dump("execution_started_v3.json", dict(started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         no_restart=True, optimizer_steps_authorized=0))
    status = 1
    runtime = {}
    try:
        import torch
        import pytest
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        # Preserve historical archival-control cuDNN flag; all tensors remain FP32.
        torch.backends.cudnn.allow_tf32 = True
        runtime = dict(python=sys.version,torch=torch.__version__,numpy="not installed; statistical fixtures use torch CPU",pytest=pytest.__version__,
             cuda_available=torch.cuda.is_available(),cuda_runtime=torch.version.cuda,CPU_threads=1,
             device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
             TF32_matmul=False,TF32_cudnn=True,default_dtype=str(torch.get_default_dtype()),
             deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
             bytecode_writing_disabled=sys.dont_write_bytecode)
        for p in OUT.glob("*.py"):
            compile(p.read_text(), str(p), "exec")  # syntax check without creating bytecode
        dump("sealed_implementation_v3.json", dict(source_hashes={p.name:sha(p) for p in sorted(OUT.glob("*.py"))},
             document_hashes={p.name:sha(p) for p in [OUT/"protocol_clarifications_v1.md",OUT/"protocol_binding_v3.json"]},
             runtime=runtime,fixture_budget=512,sealed_before_any_model_forward=True))
        args = [str(OUT/"test_receiver_anchored_v3.py"),"-q","-x","-p","no:cacheprovider",
                "--confcutdir="+str(OUT),"--junitxml="+str(OUT/"pytest_results_v3.xml")]
        dump("test_selection_v3.json",dict(arguments=args,full_repository_suite_not_run="outside bounded token-accounted authorization"))
        status = int(pytest.main(args))
        if status:
            dump("failure_v3.json",dict(boundary="prerequisite tests",pytest_exit_code=status,
                 no_retry=True,evidence="pytest_results_v3.xml and retained ledger/traces"))
    except BaseException as error:
        dump("failure_v3.json", dict(boundary="runner",error=str(error),traceback=traceback.format_exc(),no_retry=True))
        status = 1
    finally:
        final = verify_integrity(before)
        dump("integrity_after_v3.json", final)
        manifest = dict(runtime=runtime,pytest_exit_code=status,elapsed_seconds=time.time()-started,
            source_hashes={p.name:sha(p) for p in sorted(OUT.glob("*.py"))},
            protected_identity_result=final["success"],training_executed=False,optimizer_updates=0,
            production_edits=False,study_checkpoints_created=False,git_commit_created=False,
            original_diff_sha256=before["tracked_diff_sha256"])
        dump("final_manifest_v3.json", manifest)
        if not final["success"]:
            status=1
        facts = json.loads((OUT/"test_measurements.json").read_text()) if (OUT/"test_measurements.json").exists() else {}
        ledger = json.loads((OUT/"resource_ledger.json").read_text()) if (OUT/"resource_ledger.json").exists() else {}
        probes = {k:v for k,v in facts.items() if k.startswith("resource_probe/")}
        rates = [v["tokens_per_second"] for v in probes.values()]
        estimate = dict(required_overall_tokens_per_second=534288/(12*3600),
             measured_fixture_rates=rates,
             conservative_all_tokens_at_slowest_fixture_rate_hours=534288/min(rates)/3600 if rates else None,
             not_a_guarantee="fixture time includes observation/ledger overhead but excludes optimizer work, complete study I/O and statistical analysis; no deployment latency measured")
        dump("resource_estimate_v3.json", estimate)
        report = ["# Receiver-anchored prerequisites — one bounded attempt", "",
             "Status: "+("PREREQUISITES VERIFIED" if status==0 and runtime.get("cuda_available") else "STOPPED / PARTIAL"),
             "", "No training, optimizer step, parameter update, study checkpoint, study profiling, ablation, confirmation or commit occurred.",
             "The authoritative protocol is conversation text. The separately hashed clarification and binding documents do not claim a local original-protocol identity.",
             "", "## Executed tests", "", f"Pytest exit code: {status}. Exact tests/results: pytest_results_v3.xml; selection: test_selection_v3.json.",
             "CPU and CUDA results are separately keyed in test_measurements.json. CPU skips for accelerator-only probes do not count as CUDA evidence.",
             "", "## Forward and gradient evidence", "",
             "P0 comparisons cover each of eight control tokens, complete typed returned states, exact metadata, macro tensors and commits against fresh same-parameter canonical execution. Archived control CE is checked separately; historical full-tensor reproduction is not claimed.",
             "P1 checks actual blocks [0,1,2,3,3,5,6,7], including the unselected GDN4 terminal entry. Exhaustive candidate construction is forbidden.",
             "Gradient measurements compare original continuation gradients with the complete cloned-state VJP. Native gradients and a genuinely disconnected explicit detached-state negative control are separate. Individual state fields may have connected zero or disconnected gradients.",
             "Receiver parameter values have requires_grad=False; native/writer parameters remain differentiable. Model/parameter/buffer identities are checked after fixtures. Optimizer ownership uses synthetic moments without any optimizer step.",
             "", "## Accounting", "", "```json",json.dumps(ledger,indent=2),"```",
             "", "## Resource estimate", "", "```json",json.dumps(estimate,indent=2),"```",
             "A representative effective batch for each arm uses 112 model tokens and full graphs at lengths16/24. This is disposable prerequisite work, not an acquisition update. No empirical learning or study-cost verdict follows.",
             "", "## Integrity and execution boundary", "",
             f"All {final['protected_count']} pre-existing files rehashed: {final['success']}. Original tracked diff unchanged: {final['tracked_diff_unchanged']}.",
             "Historical bytecode exception retained: "+before["historical_exception"],
             "Only this new evidence directory was permitted for writes. No automatic retry or training continuation is authorized."]
        (OUT/"report_v3.md").write_text("\n".join(report)+"\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
