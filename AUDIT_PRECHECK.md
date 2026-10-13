# Canonical replay precheck

HEAD: `75f138e6b909c5ca9884e9a7755de63489d3d6f2`
Branch: `research/stage3-routing-gate-c-audit` (differs from the supplied branch label; retained actual worktree).

No files cleaned/reset, no optimizer updates, no core correction.

## Current source semantics

- `execute_differentiable`: fixed routing-decision loop; returns its configured depth as execution_count, including RESTART decisions. ADVANCE/REPEAT execute blocks; RESTART preserves h, resets pointer0 and just_restarted=True.
- Pointer is hard selected-action metadata. Initial/post-restart masks allow all ADVANCE destinations; other ADVANCE targets are strictly greater than current pointer. HALT is source-legal but excluded from successor mixing and fixed-depth inference.
- Candidate construction evaluates all eight ADVANCE states and current-block REPEAT from the same h/h0/R/context, then adds RESTART identity. It additionally evaluates masked candidates; no singleton legality mask is introduced by this audit.
- Memory is read once per token from previous stream state and frozen slots, before Prelude; R is reused by every routing decision. Both existing fixed and differentiable paths do this. Per-step Read(h_p,M) is not implemented. Whether that alternative is required cannot be inferred merely from query dependence.
- Controller normalizes h and R; receives slot mean, one-hot pointer and supplied two-component budget. This controlled pilot supplies zero budget (optional conditioning absent).
- Terminal commit computes all context-bearing bank entries plus Prelude from pre-commit state, stream normalization and memory write; applies once. It accepts terminal h, h0 and R, not the last GDN ephemeral state or route history.
- GDN forward starts from committed recurrent state/convolution buffer, computes ephemeral output/state, and returns h plus transformed output. Ephemeral state is discarded by ordinary executor calls. Repeats reconstruct from committed context; purity requires runtime verification.

## Checkpoints

Stage-1 checkpoint SHA256: `53b56e7e90f7eb7a6570107b6dfe3a297caac48d637fe27e5f00fbe94c6f09c3`.
Requested `experiments/runs/stage3_controller_route_imitation_seed3401.pt`: **missing**. Path D will be NOT TESTED unless that exact artifact is available; other old/new controller files are not substituted.

## Planned minimal diagnostic

One train sequence, length8, batch1, FP32, one device; A existing step, B exhaustive real candidate construction with canonical successor forced after construction, C actual selected-only inference forced canonical. Preserve full legality masks. Compare live tensors and all persistent/cache tensors with atol1e-6/rtol1e-5; exact discrete metadata. Then demonstrate restart counter discrepancy using a scripted legal alternating ADVANCE/RESTART control, with separate decision/execution/restart/repeat counts. No production correction before evidence.

## Complete starting worktree status

```text
 M experiments/runs/stage1_taskA_tiny.json
 M src/__pycache__/agt.cpython-312.pyc
 M src/__pycache__/blocks.cpython-312.pyc
 M src/__pycache__/controller.cpython-312.pyc
 M src/__pycache__/executor.cpython-312.pyc
 M src/__pycache__/masks.cpython-312.pyc
 M src/executor.py
 M tests/test_stage3_successor_mix.py
?? GATE_B_STAGE3_SPEC.before_source_reconciliation.md
?? GATE_B_STAGE3_SPEC.md.pre_inspection
?? GATE_B_STAGE3_SPEC.md.pre_reconcile
?? GATE_B_STAGE3_SPEC.md.pre_source_lock
?? HALT
?? RESTART
?? data/__pycache__/
?? experiments/__pycache__/
?? experiments/check_stage3_equivalence_batch1.py
?? experiments/diagnose_stage1_taskA.py
?? experiments/diagnose_stage3_closed_loop_parity.py
?? experiments/diagnose_stage3_route_equivalence.py
?? experiments/diagnose_stage3_taskA_gradient_flow.py
?? experiments/evaluate_stage3_closed_loop_equivalence.py
?? experiments/evaluate_stage3_route_controller.py
?? experiments/evaluate_stage3_warmstart.py
?? experiments/evaluate_stage3_warmstart_parity.py
?? experiments/evaluate_stage3_warmstart_taskA.py
?? experiments/gateA_a3_state_vs_output.py
?? experiments/gateA_a4_state_ablation.py
?? experiments/gateA_a5_state_shuffle.py
?? experiments/gateA_a6_state_patching.py
?? experiments/gateA_a7_controls.py
?? experiments/results/
?? experiments/run_stage1_taskA_overfit.py
?? experiments/run_stage3_masked_route_imitation.py
?? experiments/run_stage3_taskA_warmstart.py
?? experiments/run_stage3_warmstart_taskA_train.py
?? experiments/runs/stage3_controller_masked_route_imitation_seed3401.pt
?? experiments/runs/stage3_controller_route_imitation_audit_seed3401.pt
?? experiments/runs/stage3_controller_route_imitation_audit_seed3401_reproduced.pt
?? experiments/runs/stage3_controller_route_imitation_audit_seed3402.pt
?? experiments/runs/stage3_controller_route_imitation_audit_seed3403.pt
?? experiments/runs/stage3_controller_route_imitation_v2_seed3401.pt
?? experiments/runs/stage3_gradient_diagnostics_3305_v1.json
?? experiments/runs/stage3_gradient_short_imitation_3305_v1.json
?? experiments/runs/stage3_imitation_partial_3305_100_v1.pt
?? experiments/runs/stage3_imitation_partial_3306_100_v1.pt
?? experiments/runs/stage3_imitation_partial_3307_100_v1.pt
?? experiments/runs/stage3_imitation_replication_3305_v1.pt
?? experiments/runs/stage3_imitation_replication_3306_v1.pt
?? experiments/runs/stage3_imitation_replication_3307_v1.pt
?? experiments/runs/stage3_metadata_credit_probe_v1.json
?? experiments/runs/stage3_replication_3305_3307_100_v1.json
?? experiments/runs/stage3_replication_3305_3307_100_v1_3305_imitation.pt
?? experiments/runs/stage3_replication_3305_3307_100_v1_3305_random.pt
?? experiments/runs/stage3_replication_3305_3307_100_v1_3306_imitation.pt
?? experiments/runs/stage3_replication_3305_3307_100_v1_3306_random.pt
?? experiments/runs/stage3_replication_3305_3307_100_v1_3307_imitation.pt
?? experiments/runs/stage3_replication_3305_3307_100_v1_3307_random.pt
?? experiments/runs/stage3_replication_short_imitation_3305_3307_100_v1.json
?? experiments/runs/stage3_replication_short_imitation_3305_3307_100_v1_3305_imitation.pt
?? experiments/runs/stage3_replication_short_imitation_3305_3307_100_v1_3306_imitation.pt
?? experiments/runs/stage3_replication_short_imitation_3305_3307_100_v1_3307_imitation.pt
?? experiments/runs/stage3_variance_ablation_3305_3307_20_v1.json
?? experiments/train_stage1_taskA_len8.py
?? "n' '===== BRANCH =====' && "
?? "n' '===== CURRENT WORKTREE =====' && "
?? src/__pycache__/routing.cpython-312.pyc
?? src/agt.py.pre_adaptive_step.bak
?? src/executor.py.pre_adaptive_integration.bak
?? tests/__pycache__/test_stage3_audit_metrics.cpython-312-pytest-9.1.1.pyc
?? tests/__pycache__/test_stage3_controller.cpython-312-pytest-9.1.1.pyc
?? tests/__pycache__/test_stage3_executor_controller_inputs.cpython-312-pytest-9.1.1.pyc
?? tests/__pycache__/test_stage3_measurements.cpython-312-pytest-9.1.1.pyc
?? tests/__pycache__/test_stage3_replication.cpython-312-pytest-9.1.1.pyc
?? tests/__pycache__/test_stage3_routing.cpython-312-pytest-9.1.1.pyc
?? tests/__pycache__/test_stage3_stochastic_evaluation.cpython-312-pytest-9.1.1.pyc
?? tests/__pycache__/test_stage3_successor_mix.cpython-312-pytest-9.1.1.pyc
?? tests/__pycache__/test_stage3_taskA_gradient_diagnostic.cpython-312.pyc
?? tests/__pycache__/test_stage3_transition.cpython-312-pytest-9.1.1.pyc
?? tests/__pycache__/test_stage3_variance_ablation.cpython-312-pytest-9.1.1.pyc
?? tests/__pycache__/test_stage4_halt.cpython-312-pytest-9.1.1.pyc
?? tests/__pycache__/test_stage4_halting.cpython-312-pytest-9.1.1.pyc
?? tests/__pycache__/test_task_a.cpython-312-pytest-9.1.1.pyc
?? tests/test_stage3_taskA_gradient_diagnostic.py
?? tests/test_stage4_halt.py
?? tests/test_stage4_halting.py
```
