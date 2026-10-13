# Gate-C first no-update learned-policy replay

A: canonical route recovered and parity preserved

Controller `stage3_imitation_replication_3305_v1.pt` SHA-256 `259e2b86786c6cd037e3ba0db71d91fd05cdb76753bdb462be414bd1b9400df3`; frozen Stage-1 backbone SHA-256 `53b56e7e90f7eb7a6570107b6dfe3a297caac48d637fe27e5f00fbe94c6f09c3`.
HEAD `75f138e6b909c5ca9884e9a7755de63489d3d6f2`; branch `research/stage3-routing-gate-c-audit`. Full worktree identity and source hashes are in JSON.

One train control of length eight; FP32, batch one, zero budget, deterministic production inference, P=8. No forced actions, fallback, optimizer, or decision cap.

Bits: `[[0, 1, 1, 0, 1, 1, 0, 0]]`; final target `0`.
Teacher-trajectory decisions: 64/64 correct. This is this control's teacher-trajectory accuracy, not a new offline-dataset measurement.

| Path | Final prediction | Final-token CE | Sequence parity accuracy | Canonical tokens | Decisions | Executions | RESTART | REPEAT |
|---|---|---|---|---|---|---|---|---|
| Fixed | 0 | 0.0014546061865985394 | 1.0 | 1.0 | 64 | 64 | 0 | 0 |
| Learned | 0 | 0.0014546061865985394 | 1.0 | 1.0 | 64 | 64 | 0 | 0 |

| Token | D | P | Commits | Route | Prediction | Prefix CE |
|---|---|---|---|---|---|---|
| 0 | 8 | 8 | 1 | [('ADVANCE', 0), ('ADVANCE', 1), ('ADVANCE', 2), ('ADVANCE', 3), ('ADVANCE', 4), ('ADVANCE', 5), ('ADVANCE', 6), ('ADVANCE', 7)] | 0 | 0.0329677835 |
| 1 | 8 | 8 | 1 | [('ADVANCE', 0), ('ADVANCE', 1), ('ADVANCE', 2), ('ADVANCE', 3), ('ADVANCE', 4), ('ADVANCE', 5), ('ADVANCE', 6), ('ADVANCE', 7)] | 1 | 0.0019838193 |
| 2 | 8 | 8 | 1 | [('ADVANCE', 0), ('ADVANCE', 1), ('ADVANCE', 2), ('ADVANCE', 3), ('ADVANCE', 4), ('ADVANCE', 5), ('ADVANCE', 6), ('ADVANCE', 7)] | 0 | 0.0370367616 |
| 3 | 8 | 8 | 1 | [('ADVANCE', 0), ('ADVANCE', 1), ('ADVANCE', 2), ('ADVANCE', 3), ('ADVANCE', 4), ('ADVANCE', 5), ('ADVANCE', 6), ('ADVANCE', 7)] | 0 | 0.00125984428 |
| 4 | 8 | 8 | 1 | [('ADVANCE', 0), ('ADVANCE', 1), ('ADVANCE', 2), ('ADVANCE', 3), ('ADVANCE', 4), ('ADVANCE', 5), ('ADVANCE', 6), ('ADVANCE', 7)] | 1 | 0.00200095121 |
| 5 | 8 | 8 | 1 | [('ADVANCE', 0), ('ADVANCE', 1), ('ADVANCE', 2), ('ADVANCE', 3), ('ADVANCE', 4), ('ADVANCE', 5), ('ADVANCE', 6), ('ADVANCE', 7)] | 0 | 0.00130675256 |
| 6 | 8 | 8 | 1 | [('ADVANCE', 0), ('ADVANCE', 1), ('ADVANCE', 2), ('ADVANCE', 3), ('ADVANCE', 4), ('ADVANCE', 5), ('ADVANCE', 6), ('ADVANCE', 7)] | 0 | 0.00137234887 |
| 7 | 8 | 8 | 1 | [('ADVANCE', 0), ('ADVANCE', 1), ('ADVANCE', 2), ('ADVANCE', 3), ('ADVANCE', 4), ('ADVANCE', 5), ('ADVANCE', 6), ('ADVANCE', 7)] | 0 | 0.00145460619 |

Learned D mean/max: 8.0/8; P distribution: {8: 8}; unique blocks: [0, 1, 2, 3, 4, 5, 6, 7].
Prefix readout accuracy/mean CE (secondary): 1.0/0.00992285844. Primary task CE is final-token only.

First divergences: `{"first_action_divergence": null, "first_execution_divergence": null, "first_state_divergence": null, "first_execution_state_divergence": null, "first_persistent_divergence": null, "first_output_divergence": null}`.

VERIFIED: legal selected actions match actual bank execution, P=8 accounting, zero-compute RESTART semantics, frozen in-flight committed state, seven terminal-derived context entries, and one commit per completed token.
Selected-block purity checks: fixed 64, learned 64; GDN checks: {4: 8, 5: 8}. Candidate construction was not invoked; no new exhaustive-candidate purity result is claimed.
VERIFIED: reference live states/logits/CE exactly match the stored canonical control; weights and protected files unchanged; git diff --check clean.

This establishes behavior of this supervised controller on one control only. It does not establish task-loss route discovery, general parity competence, generalization, useful diversity, adaptive depth, learned HALT, or superiority.
