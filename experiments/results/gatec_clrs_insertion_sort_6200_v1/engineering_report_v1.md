# CLRS insertion-sort engineering milestone

Status: **ENGINEERING_PASSED**.

Official source: `6d5728b9efadf8648413c8c473929a733caacced`. Unchanged official sampler, algorithm, probing and evaluator; isolated package namespaces avoid unrelated convenience imports. TensorFlow is required by probing; TFDS was not installed.

Pools: 1,000 size-16 train, 32 size-16 development, 32 size-64 transfer. Source-generated official pools; no byte-equivalence claim for released TFRecords. Stable ties, pointer bounds, source targets, FP32 ordering and split overlap checked before AGT execution.

Adapter: fresh seed-6200 D64/16-slot core, canonical blocks 0–7, trainable 4→64 encoder and size-independent query/key readout. Controller and old two-class head frozen/unused. Readout candidate representations remain autograd-connected. This is serialized output-only AGT, not the official GNN.

## Scope and accounting

```json
{
  "attempted_tokens": 512,
  "completed_tokens": 512,
  "failed_tokens": 0,
  "selected_bank_attempted": 4096,
  "selected_bank_completed": 4096,
  "commits": 512,
  "backward_attempted": 6,
  "backward_completed": 6,
  "optimizer_objects": 0,
  "optimizer_updates": 0,
  "contextual_commit_entries": 3584,
  "elapsed_seconds": 63.60001838719472,
  "peak_CUDA_allocated": 147773952,
  "peak_CUDA_reserved": 157286400
}
```

Native CPU/CUDA fixtures, canonical executor comparison and gradient records: `smoke_results_v1.json`. Complete retained fixture states: device-specific `.pt` files. Trace reconciliation: {'trace_attempts': True, 'completed': True, 'failed': True, 'selected_completions': True, 'commits': True, 'completed_paths': True}

## Runtime forecast

```json
{
  "assessable": true,
  "training_authorized": false,
  "measured_three_timed_updates_seconds": [
    4.122069245669991,
    4.131428122520447,
    4.235346626024693
  ],
  "measured_forward_seconds": [
    3.596380884759128,
    3.619867077562958,
    3.6997534641996026
  ],
  "measured_backward_seconds": [
    0.5256883609108627,
    0.5115610449574888,
    0.5355931618250906
  ],
  "measured_warmup_seconds": 4.030905099585652,
  "measured_CUDA_N16_fixture_forward_seconds": 3.6783898980356753,
  "N16_fixture_autograd_enabled": true,
  "measured_CUDA_N64_fixture_inference_seconds": 14.388475536834449,
  "measured_smoke_elapsed_seconds": 63.60001838719472,
  "margin_multiplier": 1.25,
  "unmeasured_optimizer_clipping_logging_allowance_per_update": 1.0,
  "setup_checkpoint_analysis_reporting_reserve_seconds": 600.0,
  "estimated_training_seconds": 6294.183282530867,
  "estimated_initial_diagnostic_seconds": 36.78389898035675,
  "estimated_final_training_subset_seconds": 588.542383685708,
  "estimated_development_seconds": 147.135595921427,
  "estimated_size_transfer_seconds": 575.5390214733779,
  "conservative_complete_seconds": 8305.784200978931,
  "ceiling_seconds": 21600,
  "eligibility_limit_seconds": 17280,
  "resource_gate_pass": true,
  "limitations": [
    "Three synthetic calibration runs are not a runtime guarantee.",
    "Native N16 fixture timing includes autograd; used as conservative inference proxy.",
    "Optimizer steps and checkpoint/analysis times were not executed.",
    "No official train/development/size-transfer arrays were model-evaluated.",
    "Common filesystem/accelerator contention can change runtime."
  ]
}
```

Separate training authorization may be considered; not granted by this milestone.

No training, eight-array diagnostic, full development, held-out model evaluation, profiling study, optimizer creation/step, parameter update, checkpoint creation, retry or follow-up occurred. Synthetic size-64 fixtures are engineering evidence, not larger-problem generalization results.

Protected integrity: {'protected_count': 2290, 'changed_or_missing': [], 'HEAD_unchanged': True, 'tracked_diff_unchanged': True, 'historical_exception_preserved': True, 'success': True}
