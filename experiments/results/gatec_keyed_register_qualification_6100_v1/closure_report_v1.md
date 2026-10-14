# Qualification pilot closure

**Classification: engineering/integrity/resource stop.**

Protocol SHA-256: `cc9c1d6aac44f7949070a51225363d000efa34905c1324bde91a70f0fab42d26`.

Receiver-anchored study remains COMPLETED_NON_SUCCESS; no prior scientific outcome was changed.

## Execution boundary

Resource-time ceiling enforced; training incomplete. Final development, solvability, headroom, complementarity and preliminary ablations remain unverified.

## Training metrics

These are online results from completed optimizer updates, not final-model training/development accuracy.

| Length | Program | Episodes | Complete accuracy | Query accuracy | Mean CE |
|---|---|---:|---:|---:|---:|
| 64 | B | 39 | 0.0 | 0.46474358974358976 | 0.9667298656243545 |
| 64 | A | 39 | 0.0 | 0.47115384615384615 | 0.9887665571310581 |
| 64 | R | 39 | 0.0 | 0.4807692307692308 | 0.9454463131916828 |
| 96 | B | 39 | 0.0 | 0.5192307692307693 | 0.9853412868120731 |
| 96 | A | 39 | 0.0 | 0.5160256410256411 | 0.9852398236592611 |
| 96 | R | 39 | 0.0 | 0.5128205128205128 | 0.9802536391294919 |

## Gates

- Symbolic, encoding, counterfactual, separation and shortcut checks: see sealed validation records.
- CPU and genuinely executed CUDA fixture menus: passed; canonical equivalence and connected native gradients recorded.
- Training/final development: 78/128 completed updates; final development executed: False.
- Qualification statistics executed: False.
- Preliminary ablation executed: False. Any such results are descriptive, never inferential.

## Accounting

```json
{
  "attempted_tokens": 22935,
  "completed_tokens": 22934,
  "failed_tokens": 1,
  "selected_bank_attempts": 183480,
  "selected_bank_completed": 183480,
  "commit_attempts": 22935,
  "commits": 22935,
  "backward_VJP_evaluations": 248,
  "optimizer_updates": 78,
  "phase_counts": {
    "fixtures_cpu/completed": 57,
    "fixtures_cuda/completed": 57,
    "initial_diagnostic/completed": 3840,
    "training/completed": 18980,
    "training/failed": 1
  },
  "selected_block_counts": {
    "0": 22935,
    "1": 22935,
    "2": 22935,
    "3": 22962,
    "4": 22908,
    "5": 22935,
    "6": 22935,
    "7": 22935
  },
  "contextual_commit_payload_calls": {
    "SWA_bank": 45870,
    "GQA_bank": 45870,
    "GDN_bank": 45870,
    "SWA_prelude": 22935
  },
  "elapsed_accelerator_seconds": 7193.761162369046,
  "peak_memory_allocated": 450295296,
  "peak_memory_reserved": 457179136,
  "ceilings": {
    "attempted_tokens": 100000,
    "bank_calls": 800000,
    "accelerator_seconds": 7200
  },
  "planned_max_tokens": 99954,
  "no_warmups_or_profiling": true
}
```

Trace reconciliation: True. Protected-file verification: True.

No eligible final checkpoint was created unless the predetermined update 128 completed. No partial checkpoint is selected or substituted. No selector, compatibility training, profiling, transfer confirmation, retry or extension occurred.

Resource-limited noncompletion does not establish task unsolvability, absence of program complementarity, or useful learned selection.
