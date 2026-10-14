# CLRS insertion-sort fixed-program learning run

Status: **COMPLETED_NON_SUCCESS**.

Exactly one fresh seed6200 model, fixed blocks0–7, full BPTT, one seed6201 shuffled pass. No hints, schedule tuning, intermediate selection or retry.

The engineering milestone and its 75 published files remain separate, unchanged evidence. The approved remaining runtime forecast used a 2× margin under observed accelerator contention; no engineering model calls were repeated.

## Evaluation

|Split|n|Arrays|Predecessor accuracy|Exact array accuracy|Mean CE|Invalid chain frequency|
|---|---:|---:|---:|---:|---:|---:|
|training_subset|16|128|0.033203125|0.000000000|2.772870822|1.000000000|
|development|16|32|0.031250000|0.000000000|2.772864416|1.000000000|
|size_transfer|64|32|0.011718750|0.000000000|4.158960268|1.000000000|

Complete-array bootstrap intervals and target-only controls: `analysis_v1.json`. Official evaluator cross-checks: `official_metric_verification_v1.json`.

## Locked gates
```json
{
  "development_pointer_accuracy": {
    "value": 0.03125,
    "threshold": 0.95,
    "passed": false
  },
  "development_exact_array_accuracy": {
    "value": 0.0,
    "threshold": 0.8,
    "passed": false
  },
  "size_transfer_pointer_accuracy": {
    "value": 0.01171875,
    "threshold": 0.9,
    "passed": false
  },
  "size_transfer_exact_array_accuracy": {
    "value": 0.0,
    "threshold": 0.5,
    "passed": false
  },
  "mechanical_numerical_integrity": {
    "passed": true
  }
}
```

## Resources
```json
{
  "attempted_tokens": 41472,
  "completed_tokens": 41472,
  "failed_tokens": 0,
  "selected_bank_attempted": 331776,
  "selected_bank_completed": 331776,
  "commits": 41472,
  "backward_attempted": 1000,
  "backward_completed": 1000,
  "optimizer_objects": 1,
  "optimizer_attempted": 1000,
  "optimizer_updates": 1000,
  "optimizer_failed": 0,
  "array_attempted": 1200,
  "array_completed": 1200,
  "array_failed": 0,
  "phase_counts": {
    "initial_diagnostic": {
      "attempted": 256,
      "completed": 256,
      "failed": 0,
      "selected_attempted": 2048,
      "selected_completed": 2048,
      "commits": 256
    },
    "training": {
      "attempted": 32000,
      "completed": 32000,
      "failed": 0,
      "selected_attempted": 256000,
      "selected_completed": 256000,
      "commits": 32000
    },
    "training_subset": {
      "attempted": 4096,
      "completed": 4096,
      "failed": 0,
      "selected_attempted": 32768,
      "selected_completed": 32768,
      "commits": 4096
    },
    "development": {
      "attempted": 1024,
      "completed": 1024,
      "failed": 0,
      "selected_attempted": 8192,
      "selected_completed": 8192,
      "commits": 1024
    },
    "size_transfer": {
      "attempted": 4096,
      "completed": 4096,
      "failed": 0,
      "selected_attempted": 32768,
      "selected_completed": 32768,
      "commits": 4096
    }
  },
  "engineering_completed": {
    "attempted_tokens": 512,
    "completed_tokens": 512,
    "failed_tokens": 0,
    "selected_bank_calls": 4096,
    "commits": 512,
    "backwards": 6,
    "optimizer_updates": 0
  },
  "combined_attempted_tokens": 41984,
  "combined_completed_tokens": 41984,
  "combined_selected_bank_calls": 335872,
  "combined_commits": 41984,
  "combined_backwards": 1006,
  "contextual_commit_entries": 290304,
  "elapsed_seconds": 5171.713088182267,
  "peak_CUDA_allocated": 169708544,
  "peak_CUDA_reserved": 178257920
}
```

The 512-token/6-backward engineering work is included only in combined totals; it is not pooled into task outcomes. Contextual entries describe actual commit-plan entries, not unsupported FLOP estimates.

## Checkpoint and integrity
```json
{
  "checkpoint": {
    "path": "model_update1000_v1.pt",
    "sha256": "2c0e3d6ad3711d8f0ee209eb3e3b0f8db8964e6eeaf2be77b7bd56120f29e6c8",
    "size": 22507575,
    "optimizer_updates": 1000,
    "selection": "Predetermined final update1000 only"
  },
  "integrity": {
    "protected_count": 2365,
    "changed_or_missing": [],
    "HEAD_unchanged": true,
    "tracked_diff_unchanged": true,
    "historical_exception_preserved": true,
    "no_new_exception": true,
    "success": true
  },
  "new_implementation_seal_unchanged": true
}
```

No learned selector, route discovery, compatibility objective, new task, new seed, Enwik8 experiment or follow-up was executed. Publication will use a new remote branch with the verified engineering commit as parent; local Git state remains untouched.
