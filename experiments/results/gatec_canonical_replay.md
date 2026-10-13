# Gate-C zero-update canonical replay

HEAD `75f138e6b909c5ca9884e9a7755de63489d3d6f2`; actual branch `research/stage3-routing-gate-c-audit`. Starting worktree: [AUDIT_PRECHECK.md](../../AUDIT_PRECHECK.md). Stage-1 `experiments/runs/stage1_parity_seed3305_best.pt` SHA256 `53b56e7e90f7eb7a6570107b6dfe3a297caac48d637fe27e5f00fbe94c6f09c3`. Exact provisional controller replay status is NOT TESTED; no substitute.

- Path A: existing AGTv3.step canonical reference.
- Path B: real differentiable_route_step and all candidates; forced canonical selection only after construction, unchanged legality mask.
- Path C: actual production selected-only deterministic path with canonical policy.
- Path D: prescribed provisional controller; source-budget diagnostic planned, checkpoint missing.

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

**NOT TESTED learned first mismatch:** prescribed provisional checkpoint absent. No closed-loop learned-policy claim.

## Counter discrepancy and diagnostic correction

Read-only h-query probe: {"status": "VERIFIED", "read_calls_per_token": 1, "query": "s_(t-1)", "R_reused_within_token": true, "alternate_h_query_scope": "read-only probe on the same frozen slots; never fed to execution", "alternate_h_query_max_abs": 0.0, "alternate_h_query_equal_count": 64, "per_step_h_query": "NOT IMPLEMENTED; no source correction inferred from replay equivalence"}

| Path | Reported executions | Actual executions | Decisions | RESTART | REPEAT |
|---|---:|---:|---:|---:|---:|
| Current production | 8 | 4 | 8 | 4 | 0 |
| Diagnostic source-budget replay | 8 | 8 | 15 | 7 | 0 |


**FAILED production execution accounting:** eight alternating ADVANCE/RESTART decisions report eight executions but perform four. Separate source counters in the diagnostic replay perform eight executions in fifteen decisions with seven restarts. This diagnostic-only correction does not modify production.

The smallest proposed production correction is to separate counters and terminate on eight executions. Applying it would change the historical fixed-eight-decision protocol and needs explicit experiment versioning; no silent production migration was made.

**PARTIAL final verdict:** canonical numerical/state/commit equivalence is established on one sequence. A precise execution-budget invariant fails. Learned checkpoint replay remains NOT TESTED. These results do not establish whether the controller can learn routing or explain task-loss collapse uniquely. All weights unchanged; no optimizer updates, no commit. Stop at diagnostic boundary.

## Validation

**VERIFIED:** compilation; nine focused audit/GDN tests; all core source hashes unchanged; all model weights unchanged; canonical traces exactly reproduced after strengthening purity assertions to bitwise equality. No production correction or commit. Detailed per-step snapshots and the complete starting worktree status are retained in JSON.

Final regression: **48 relevant Stage-3/audit tests passed**. Full worktree: **140 passed, 1 skipped, 2 pre-existing Stage-4 failures**, in `test_stage4_halting.py` (HALT-before-execution and HALT-after-execution accounting). JSON contains no NaN/Inf. `git diff --check` is clean. Source and harness SHA-256 values rechecked against the recorded values.
