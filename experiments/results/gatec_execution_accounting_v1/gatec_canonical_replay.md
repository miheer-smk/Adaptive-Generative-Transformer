# Gate-C zero-update canonical replay

HEAD `75f138e6b909c5ca9884e9a7755de63489d3d6f2`; actual branch `research/stage3-routing-gate-c-audit`. Starting worktree: [AUDIT_PRECHECK.md](../../../AUDIT_PRECHECK.md). Stage-1 `experiments/runs/stage1_parity_seed3305_best.pt` SHA256 `53b56e7e90f7eb7a6570107b6dfe3a297caac48d637fe27e5f00fbe94c6f09c3`. Exact provisional controller replay status is NOT TESTED; no substitute.

- Path A: existing AGTv3.step canonical reference.
- Path B: real differentiable_route_step and all candidates; forced canonical selection only after construction, unchanged legality mask.
- Path C: actual production selected-only deterministic path with canonical policy.
- Path D: Explicitly excluded from execution-accounting validation; no controller checkpoint loaded.

## First divergence

| Pair | State | Metadata | Commit | Logits |
|---|---|---|---|---|
| A/B | None | None | None | None |
| A/C | None | None | None | None |

**VERIFIED:** all 64 canonical transitions agree within atol1e-6/rtol1e-5, discrete metadata exact; final predictions agree.
Path A: final CE 0.0014546061866, prediction 0; 8 executions/8 decisions per token.
Path B: final CE 0.0014546061866, prediction 0; 8 executions/8 decisions per token.
Path C: final CE 0.0014546061866, prediction 0; 8 executions/8 decisions per token.

**VERIFIED GDN:** both blocks use ephemeral recurrence for output; two visits recompute from unchanged committed state/buffer and match cloned-context reconstruction. Convolution buffers remain frozen. Discarding ephemeral state is not a bug.

**VERIFIED candidate purity:** 576 exhaustive block calls leave supplied h/h0/R/context unchanged. The production builder evaluates masked as well as legal candidates.

**VERIFIED terminal commit:** one per token, all six context-bearing bank blocks plus Prelude updated; terminal-derived payloads recomputed from pre-commit state agree with actual committed values.

**VERIFIED memory behavior:** both paths perform Read(s_previous,M) once per token and reuse R; not Read(h_p,M) per decision. Whether a different read policy is required is not resolved by this experiment.

Learned-policy status is recorded above; this accounting validation uses --skip-learned-replay. No closed-loop learned-policy claim.

## Counter discrepancy and diagnostic correction

Read-only h-query probe: {"status": "VERIFIED", "read_calls_per_token": 1, "query": "s_(t-1)", "R_reused_within_token": true, "alternate_h_query_scope": "read-only probe on the same frozen slots; never fed to execution", "alternate_h_query_max_abs": 0.0, "alternate_h_query_equal_count": 64, "per_step_h_query": "NOT IMPLEMENTED; no source correction inferred from replay equivalence"}

{
  "status": "VERIFIED",
  "production": {
    "reported_execution_count": 8,
    "decision_count": 15,
    "actual_execution_count": 8,
    "restart_count": 7,
    "repeat_count": 0,
    "history": [
      [
        "ADVANCE",
        0
      ],
      [
        "RESTART",
        null
      ],
      [
        "ADVANCE",
        0
      ],
      [
        "RESTART",
        null
      ],
      [
        "ADVANCE",
        0
      ],
      [
        "RESTART",
        null
      ],
      [
        "ADVANCE",
        0
      ],
      [
        "RESTART",
        null
      ],
      [
        "ADVANCE",
        0
      ],
      [
        "RESTART",
        null
      ],
      [
        "ADVANCE",
        0
      ],
      [
        "RESTART",
        null
      ],
      [
        "ADVANCE",
        0
      ],
      [
        "RESTART",
        null
      ],
      [
        "ADVANCE",
        0
      ]
    ]
  },
  "diagnostic_source_budget": {
    "reported_execution_count": 8,
    "decision_count": 15,
    "actual_execution_count": 8,
    "restart_count": 7,
    "repeat_count": 0,
    "history": [
      [
        "ADVANCE",
        0
      ],
      [
        "RESTART",
        null
      ],
      [
        "ADVANCE",
        0
      ],
      [
        "RESTART",
        null
      ],
      [
        "ADVANCE",
        0
      ],
      [
        "RESTART",
        null
      ],
      [
        "ADVANCE",
        0
      ],
      [
        "RESTART",
        null
      ],
      [
        "ADVANCE",
        0
      ],
      [
        "RESTART",
        null
      ],
      [
        "ADVANCE",
        0
      ],
      [
        "RESTART",
        null
      ],
      [
        "ADVANCE",
        0
      ],
      [
        "RESTART",
        null
      ],
      [
        "ADVANCE",
        0
      ]
    ]
  },
  "production_corrected": true,
  "protocol": "fixed-eight-block-executions-v1",
  "historical_boundary": "Earlier fixed-eight-decision experiments retain their original protocol; they are not reinterpreted as eight executions."
}

**VERIFIED production execution accounting:** alternating ADVANCE/RESTART routing now performs and reports eight executions in fifteen decisions with seven restarts; history and terminal state match the independent diagnostic replay.

Protocol: fixed-eight-block-executions-v1. Earlier fixed-eight-decision experiments retain their historical meaning. The only production correction separates decision/execution accounting and terminates on actual block executions.

Canonical numerical/state/commit equivalence is established on one sequence. Learned-policy replay is excluded when --skip-learned-replay is supplied. These results do not uniquely explain task-loss collapse or establish learned-routing success. All weights unchanged; no optimizer updates or commit.

## Correction validation and compatibility boundary

The pre-correction control was reproduced before editing: eight decisions, four actual executions, eight reported executions. The new targeted tests initially had four failures (both training/inference RESTART controls) and four passes; after the correction all eight passed, and the independent diagnostic replay comparison also passed.

Only `execute_differentiable()` changed in production: explicit decision/execution counters; ADVANCE and REPEAT increment executions, RESTART does not; termination and the returned execution count use actual executions. No controller, mask, ST-Gumbel, block, memory, GDN, AGTv3 integration or terminal-commit source changes were made. The pre-existing Stage-4 addition in executor.py was preserved.

| Validation | Result |
|---|---|
| Targeted accounting tests | 9 passed |
| Relevant Stage-3/audit suite | 49 passed, 7 failed |
| Full worktree regression | 141 passed, 1 skipped, 9 failed |
| Compilation / finite output / diff whitespace | VERIFIED / VERIFIED / clean |
| Canonical traces compared to prior audit JSON | Exactly identical |

**Regression is not green.** Seven historical diagnostic tests depend on the prior fixed-eight-decision protocol: route metrics reject longer histories; scripted policies exhaust their eight-decision schedule or produce an additional continuation; gradient capture indexes tokens with call_count // 8. These dependencies must be explicitly migrated and versioned before reusing those diagnostics. Changing their counterfactual protocols or scientific report interpretation is outside this single production correction, so they were left unchanged. Two Stage-4 failures are pre-existing and untouched.

The complete failures are:

- `tests/test_stage3_measurements.py::test_real_gradient_diagnostic_preserves_forward_and_mask_contract`
- `tests/test_stage3_measurements.py::test_metadata_credit_blindspot_has_causal_forward_effect_and_zero_gradient`
- `tests/test_stage3_replication.py::test_compute_measurement_counts_only_inference_selected_blocks`
- `tests/test_stage3_stochastic_evaluation.py::test_paired_stochastic_assessment_replays_without_parameter_updates`
- `tests/test_stage3_taskA_gradient_diagnostic.py::test_proxy_preserves_real_forward_noise_and_exposes_boundary`
- `tests/test_stage3_taskA_gradient_diagnostic.py::test_scripted_replacements_have_one_difference_and_legal_continuation`
- `tests/test_stage3_variance_ablation.py::test_gradient_averaging_matches_independent_draws_without_updating_weights`
- `tests/test_stage4_halting.py::test_stage4_halt_terminates_before_any_block_execution`
- `tests/test_stage4_halting.py::test_stage4_halt_after_execution_preserves_current_live_state`

No optimization updates, checkpoint writes, learned-policy replay or commits. The execution-accounting correction does not establish the cause of earlier task-loss collapse or Gate-C success/failure.
