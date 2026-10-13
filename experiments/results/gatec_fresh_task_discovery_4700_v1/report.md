# Locked fresh-controller task-loss experiment, seed 4700

**Result: strong reference-level discovery was not established.** Exactly 1,040 predetermined controller-only task-CE updates and all four final stochastic tapes completed. T changed its parameters and routing, with partial deterministic task improvement over U, but remained far below the canonical reference. This is one controller initialization and a small fixed panel; it does not settle Gate C success or failure.

**Artifact-integrity exception:** a post-training inspection import regenerated the tracked executor bytecode cache and eight experiment bytecode caches. The original end-of-run protected-file assertion failed after all evaluation tapes completed. `failure.json` is preserved. Production Python sources and historical checkpoints are independently hash-verified unchanged. The complete existing records were aggregated and verified without another model evaluation, optimizer step, checkpoint selection or training retry. The full file-identity invariant is FAILED; measured execution/optimization invariants are VERIFIED.

## Locked setup and provenance

- Git revision `75f138e6b909c5ca9884e9a7755de63489d3d6f2`, branch `research/stage3-routing-gate-c-audit`; pre-existing dirty worktree preserved.
- Manifest SHA-256: `5c891e9b6939822cac7e937353c4077aae909d40dc405c01c5ab1a72d769bf38`.
- Stage-1 non-controller backbone: `experiments/runs/stage1_parity_seed3305_best.pt`, SHA-256 `53b56e7e90f7eb7a6570107b6dfe3a297caac48d637fe27e5f00fbe94c6f09c3`.
- Original AGTv3 constructor seeded 4700, followed by non-controller backbone loading; fresh unit controller RMSNorm gains, no trained controller transfer or canonical-action bias. The fresh controller was cloned into U and T. U received zero updates.
- Ten fixed permutations from seed 4701; 104 raw sequences per pass, lengths 4–16. AdamW LR 0.0001, weight decay 0, betas (0.9, 0.999), epsilon 1e-8. Final-sequence parity CE only; tau=2.0. No scheduler, clipping, auxiliary objective or teacher-forced routing.
- Batch 1, FP32 inputs/parameters, CUDA; zero controller budget fraction and presence flag `[0,0]`. Original immutable Prelude anchor and source masks retained. HALT/convergence disabled. Each completed token has P=8 selected block executions and D=P+RESTART, 8≤D≤15; no decision cap.
- Full-sequence BPTT, horizon 4–16 training tokens; independent reset between sequences. No token/state/cache truncation or extra detach. Frozen backbone computations remain differentiable in T. Existing ST relaxed stop-gradient and hard metadata `action.detach().argmax` remain unchanged. Existing GDN convolution history has receptive field 3; GQA retains all and SWA retains 255 committed rows, exceeding these sequence lengths.
- Training seed 4702 supplies sealed uniforms indexed by epoch/sequence/token/decision/action slot. Each visit has fresh predetermined noise, paired between U/T regardless of trajectory lengths. All sampled actions reconstructed from recorded logits, source masks and uniforms. Fifteen tape positions cover the legal maximum, without a stopping cap.
- Final tapes derived from seed 4704: `[924088217, 911787993, 126797460, 757739963]`. All four are reported, pooled with equal weight. They are sampling replication, not additional controller-training seeds.
- Final panel sealed before outcomes with seed 4703: 32 distinct balanced length-16 inputs and four distinct balanced inputs at each length 17–32 (64 total). Exact recorded training/prior matches and panel duplicates were rejected by rule; generation order and parity-quota rejections are retained. No final outcomes were used for selection.
- Known exact training/prior input overlap: 0. Rejection counts `{'accepted': 96, 'parity_quota_full': 31}`. Prior inventory contains recoverable records and explicit unresolved records. Stage-1 training membership and unretained historical probes remain unknown; “fresh” refers to the recoverable comparison inventory.
- Exposed development panel was assessed only at updates 0/104/520/1040. Final selection is update 1040 regardless of its performance.

## Final task results

| Evaluation | Split | U accuracy / CE | T accuracy / CE |
|---|---|---|---|
| deterministic | fresh L16 | 46.8750% / 1.902284461 | 56.2500% / 1.125961793 |
| deterministic | L17–32 | 43.7500% / 1.919968336 | 46.8750% / 1.358376613 |
| final_tape_1 | fresh L16 | 43.7500% / 3.512042129 | 43.7500% / 3.207853988 |
| final_tape_1 | L17–32 | 57.8125% / 2.589069031 | 59.3750% / 2.355396621 |
| final_tape_2 | fresh L16 | 43.7500% / 3.505502858 | 46.8750% / 3.042234393 |
| final_tape_2 | L17–32 | 53.1250% / 2.847577622 | 51.5625% / 2.779358135 |
| final_tape_3 | fresh L16 | 53.1250% / 2.952675610 | 53.1250% / 2.927483002 |
| final_tape_3 | L17–32 | 40.6250% / 3.571203623 | 46.8750% / 3.039045603 |
| final_tape_4 | fresh L16 | 59.3750% / 2.501573376 | 53.1250% / 2.559891268 |
| final_tape_4 | L17–32 | 48.4375% / 3.173922620 | 46.8750% / 3.113958401 |
| all four tapes | fresh L16 | 50.0000% / 3.117948493 | 49.2188% / 2.934365663 |
| all four tapes | L17–32 | 50.0000% / 3.045443224 | 51.1719% / 2.821939690 |
| F canonical | fresh L16 | 100.0000% / 0.001803286 | same reference |
| F canonical | L17–32 | 100.0000% / 0.001785499 | same reference |

T gained three correct length-16 predictions (15/32→18/32) and two OOD predictions (28/64→30/64) deterministically. This is partial evidence of task improvement over the unchanged controller, not solved parity or a reliable discovery claim. The sampled accuracy/CE differences above must be interpreted separately; the same inputs are repeated across tapes, so pooled observations are not 384 independent novel sequences.

## Training and optimization

All 1,040 updates were finite and nonzero; all 109,697 retained task-logit gradients were finite/nonzero. Controller gradient L2 mean 5.19881786, minimum 1.83283018e-05, maximum 1467.42784. HALT and illegal-action gradients were exactly zero. Final controller displacement L2 0.746575832; 103,306 scalar parameters changed. The unused HALT output row/bias remained unchanged. Magnitude/coverage establish an active optimization path, not useful gradient credit.

| Epoch | U accuracy / CE | T accuracy / CE |
|---|---|---|
| 1 | 53.8462% / 2.832779240 | 53.8462% / 2.790022596 |
| 2 | 53.8462% / 2.805179172 | 52.8846% / 2.900845796 |
| 3 | 50.9615% / 3.047518476 | 52.8846% / 2.889206646 |
| 4 | 50.0000% / 3.006887151 | 49.0385% / 3.136523620 |
| 5 | 50.0000% / 3.028202039 | 48.0769% / 3.174477038 |
| 6 | 50.9615% / 2.993692554 | 47.1154% / 3.273104217 |
| 7 | 51.9231% / 2.844606584 | 46.1538% / 3.325209044 |
| 8 | 50.9615% / 3.022204775 | 49.0385% / 3.054558251 |
| 9 | 50.9615% / 3.044164758 | 46.1538% / 3.296910713 |
| 10 | 51.9231% / 2.960775173 | 52.8846% / 2.830887943 |

The online training trajectories are recomputed under each current policy. There is no consistent paired training-set accuracy improvement across epochs; these sampled online losses are not an independent final deterministic training-set evaluation.

## Predetermined development snapshots

| Update | Split | U accuracy / CE | T accuracy / CE |
|---|---|---|
| 0 | val | 59.6154% / 1.287238048 | 59.6154% / 1.287238048 |
| 0 | ood_test | 56.2500% / 1.399210833 | 56.2500% / 1.399210833 |
| 104 | val | 59.6154% / 1.287238048 | 55.7692% / 2.416845617 |
| 104 | ood_test | 56.2500% / 1.399210833 | 39.0625% / 3.041051826 |
| 520 | val | 59.6154% / 1.287238048 | 50.0000% / 3.153190978 |
| 520 | ood_test | 56.2500% / 1.399210833 | 50.0000% / 3.153935688 |
| 1040 | val | 59.6154% / 1.287238048 | 48.0769% / 1.759611809 |
| 1040 | ood_test | 56.2500% / 1.399210833 | 48.4375% / 1.414575940 |

## Routing and computational accounting

| Evaluation | Arm/split | Mean/max D | RESTART / REPEAT | Mean unique blocks | Canonical action / block / sequence fraction | Entropy / selected raw margin |
|---|---|---|---|---|---|---|
| deterministic | U/val | 11.722656/15 | 1906/261 | 2.318359 | 0/0/0 | 1.796166/0.052605 |
| deterministic | U/ood_test | 11.635204/15 | 5700/709 | 2.419643 | 0/0/0 | 1.792668/0.052291 |
| deterministic | T/val | 14.914062/15 | 3540/20 | 1.542969 | 0/0/0 | 1.975940/0.309464 |
| deterministic | T/ood_test | 14.798469/15 | 10660/207 | 1.502551 | 0/0/0 | 1.987889/0.313395 |
| pooled tapes | U/val | 10.274414/14 | 4658/4152 | 4.385254 | 0/0/0 | 1.551432/-0.091234 |
| pooled tapes | U/ood_test | 10.285555/15 | 14335/12454 | 4.422832 | 0/0/0 | 1.551610/-0.092150 |
| pooled tapes | T/val | 10.937988/15 | 6017/3099 | 4.737305 | 0/0/0 | 1.594638/-0.144671 |
| pooled tapes | T/ood_test | 11.041454/15 | 19076/8933 | 4.771684 | 0/0/0 | 1.601704/-0.142112 |

Every observed completed token had P=8. Decision/action and RESTART frequencies use observed D; block utilization uses selected executions only. Stochastic routing constructs all ten successor tensors using nine speculative bank calls per decision (eight ADVANCE candidates and one REPEAT); RESTART itself is identity. Speculative work is reported separately from selected P. Deterministic execution constructed no candidates and executed selected blocks only. Noncanonical routing alone is not a failure, but its task usefulness here is limited by measured accuracy.

## First divergences

- First paired training `first_action`: update 3, `train/L11/row7`, `{'token': 7, 'decision': 6, 'U': 9, 'T': 8}`.
- First paired training `first_raw_state`: update 3, `train/L11/row7`, `{'token': 7, 'decision': 6, 'scope': 'successors of different hard actions', 'field': '', 'max_abs': 11.474899291992188}`.
- First paired training `first_execution`: update 3, `train/L11/row7`, `{'token': 7, 'execution_position': 5, 'U': 3, 'T': 7}`.
- First paired training `first_task`: update 23, `train/L7/row2`, `{'token': 6, 'U': 1, 'T': 0}`.
- Final U, first final input `val/L16/row0`: token 0 action route `7,9,5,7,9,5,9,5,9,5,9,5,9,5`; executed blocks `7,5,7,5,5,5,5,5`. The first action differs from F at token 0/decision 0; later state differences follow legitimate different actions, not a selection/execution mismatch.
- Final T, first final input `val/L16/row0`: token 0 action route `6,7,9,7,9,7,9,7,9,7,9,7,9,7`; executed blocks `6,7,7,7,7,7,7,7`. The first action differs from F at token 0/decision 0; later state differences follow legitimate different actions, not a selection/execution mismatch.

Per-sequence first action/state/execution/output/task details, raw features/logits, routes and counts are retained in JSON/JSONL; equivalence is compared only along common action prefixes. No repair or fallback was used.

## Verification and limitations

- Observed 60,384 completed tokens and 679,555 decisions; selected routed executions 483,072. All 393,500 sampled actions independently reconstructed. One terminal commit per token and 845,376 cache/version comparisons verified. No extra model evaluation or optimizer step was used for this post-run verification.
- Legality, selected-action/block correspondence, P/D/RESTART accounting, selected hard-forward successor equality, candidate/GDN/context/memory purity, immutable anchor and terminal commit assertions ran during the original paths. The final frozen-weight check ran before the failed file-identity assertion and passed.
- Focused tests: 24 passed. Full regression: 180 passed, 1 skipped, the same two pre-existing Stage-4 failures: `test_stage4_halt_terminates_before_any_block_execution` (count 1 vs expected 0) and `test_stage4_halt_after_execution_preserves_current_live_state` (count 2 vs expected 1). Those tests/source were not repaired.
- `git diff --check` is clean. Production Python sources, encoding/helpers and historical checkpoints are unchanged; the tracked binary diff changed due to regenerated bytecode. The cache exception is disclosed, not silently removed or treated as a successful full-file-identity check.
- One controller initialization; small balanced final panels; four tapes are fixed-noise sampling replications. Unknown backbone training membership and unretained historical evaluation membership limit unseen-data claims. No useful route-diversity, task-loss population trainability, adaptive-depth/HALT or superiority claim follows.
- Mechanical execution and an active full-BPTT optimization path are established for the measured trajectories. Strong reference-level discovery is not established. Partial deterministic improvement does not identify which credit-assignment/exploration limitation prevents a solved route. No next optimization step was attempted.

## Checkpoints and artifacts

- Update 0: `experiments/runs/stage3_fresh_task_4700_update0000_locked_v1.pt`, SHA-256 `ae8ff9256afd573331719cf15a3f82d21f70181891400640739a0415c82d6279`.
- Update 104: `experiments/runs/stage3_fresh_task_4700_update0104_locked_v1.pt`, SHA-256 `65821b572854946379dcc44ab96e5b96d1a7f88f597022c9de23d96778d5172b`.
- Update 520: `experiments/runs/stage3_fresh_task_4700_update0520_locked_v1.pt`, SHA-256 `8e3c3ad0ec39c7a79657e3254224530215382da0ce9c62f1b9ab1c94bcc17491`.
- Update 1040: `experiments/runs/stage3_fresh_task_4700_update1040_locked_v1.pt`, SHA-256 `37537106f47e3a003e4b5d2ff98ded93a8f98e3a2b60ee7814e502d28ddaa6d4`.

`manifest.json`, `encoding_attestation.json`, `training_sequences.json`, `training_schedule.json`, `final_panel.json`, `prior_input_inventory.json` and sealed `noise_*.pt` preserve preparation. `final_*.json`, `results.json`, `verification.json`, and compressed decisions/tokens/sequences/updates/snapshots/paired-divergence JSONL retain every run. Original `failure.json` is preserved. Diagnostic aggregation/verification scripts contain no model evaluation or optimizer operation. No historical artifact was overwritten, no commit created, and no subsequent experiment started.
