# Locked warm-started task-loss experiment: measured interpretation

Exactly one bounded experiment completed: T received 104 controller-only final-sequence parity CE updates; U received zero updates. The preregistered bounded-maintenance criterion is satisfied on the two fixed-noise panels. This establishes maintenance of this working imitation policy under this update procedure, with no convincing evidence of improved routing or route discovery.

Effective manifest SHA-256: `10ebe3df594899975212b731b1eaaf0bdf66daebc1b95b85f2345569c1976430`. Starting controller: `experiments/runs/stage3_imitation_replication_4401_locked_v1.pt`, SHA-256 `4e14f56d3762aac1442a8dcc6fd86eb950e0013986409ff31b48d514fd972fcd`. Frozen Stage-1 backbone: `experiments/runs/stage1_parity_seed3305_best.pt`, SHA-256 `53b56e7e90f7eb7a6570107b6dfe3a297caac48d637fe27e5f00fbe94c6f09c3`.

Final checkpoint: `experiments/runs/stage3_task_maintenance_4401_104_locked_v1.pt`, SHA-256 `c2dcee934edbc60e5e3faf0167ad202f9e25bf76ef47029a723d3fdcc4e39504`. Selection was final update 104 only. No additional updates, seed replacements, tuning, or evaluation-driven extension occurred.

The original 104 materialized raw sequences (eight per length 4–16) were used once in order seeded 4600. Both arms recomputed their own trajectories. T used fresh AdamW (LR 0.0001, weight decay 0, betas 0.9/0.999, epsilon 1e-8), batch 1/FP32, fixed tau 2.0, and final-sequence CE only. Full sequence BPTT through stream/memory/committed caches was retained, with the existing hard metadata and ST stop-gradient boundaries. No additional detach, clipping, scheduler, cached imitation features, or auxiliary objective was introduced. Budget fraction and presence flag both stayed zero.

Recorded uniform tables used seeds 4601/4602/4603 for training/validation/OOD, indexed by sequence/update, token, decision and ten successor action slots. The production ST-Gumbel primitive performed its original clamp, Gumbel transform, masked normalization, temperature and hard-forward/relaxed-backward calculation. Pairing did not alter the estimator. Validation uniforms were reused exactly before and after updates. Fifteen stored decision positions cover the legal P=8 trajectory bound; they are not a decision cap.

| Locked final evaluation | U accuracy | T accuracy | U mean CE | T mean CE |
|---|---:|---:|---:|---:|
| Validation: 52 sequences, lengths 4–16; deterministic | 100% | 100% | 0.00172144223828442 | 0.00172144223828442 |
| Validation: 52 sequences, lengths 4–16; stochastic | 100% | 100% | 0.00172144223828442 | 0.00172144223828442 |
| OOD: 64 sequences, lengths 17–32; deterministic | 100% | 100% | 0.00176086770625261 | 0.00176086770625261 |
| OOD: 64 sequences, lengths 17–32; stochastic | 100% | 100% | 0.00176086770625261 | 0.00176086398823827 |

Starting deterministic validation reproduced the previous result exactly: accuracy 100%, CE 0.0017214422382844181. Starting U stochastic validation had identical canonical routes, outputs and CE. Sampling therefore caused no observed degradation on that starting validation noise panel. Unchanged U reproduced exactly on post-update validation.

Both sampled training arms had accuracy 100% and mean CE 0.0017399457440030975. Their per-sequence losses and routes were identical throughout all 104 updates. The fraction of sampled training sequences and tokens departing from canonical was zero in both arms.

All 104 T updates had finite gradients and nonzero parameter changes. Gradient L2 mean/min/max: 8.9519981357e-05 / 6.26384031708e-07 / 0.00140466461934. Total controller displacement L2: 0.362365156412; 103048 scalar values changed. HALT retained-logit gradient was exactly zero in all 8,320 T training decisions; the HALT output weight row and bias remained unchanged. Shared controller features can nevertheless affect raw HALT logits.

The first paired controller-logit change was update 2, train/L12/row5, token 0, decision 0 (max absolute difference 0.09564971923828125). Training had no action, raw-state, execution, output or task divergence. Finite surrogate gradients and changed policy parameters are established; informative route discovery or a meaningful task improvement is not established.

All deterministic evaluations and U sampled evaluations recovered canonical actions and executed blocks on every token and sequence. Final T sampled validation was also fully canonical. Final T sampled OOD recovered canonical actions/executed blocks on 1,567/1,568 tokens (99.9362244898%) and exact canonical routes on 63/64 sequences (98.4375%). Its single noncanonical route was legal and preserved the final prediction.

The first autonomous route divergence was final stochastic OOD, ood_test/L20/row3, token 7, decision 6, execution count 6, pointer 5. U selected ADVANCE(6); T selected ADVANCE(7) using the identical indexed Gumbel draw. T then selected REPEAT at block 7. Its executed blocks were [0,1,2,3,4,5,7,7]. P=D=8, RESTART=0, REPEAT=1, unique blocks=7. Pre-action raw state was identical; the successor hidden-state max absolute difference was 6.216742038726807, and the first token-output logit difference was 0.026627063751220703. This is a legitimate route-induced difference, not an execution mismatch.

That length-20 sequence retained prediction/label 1. Final CE changed from 0.001967995660379529 to 0.0019677577074617147 (delta −2.3795291781425476e-7). The overall OOD sampled mean CE decreased by only 3.7180143408477306e-9. This tiny one-sequence, one-noise-panel effect is not convincing evidence of useful learned route diversity or stochastic task improvement.

| Canonical reference-state diagnostic | U | Final T |
|---|---:|---:|
| Validation: mean token full-route log probability | -1.58138857034e-05 | -0.000279088141474 |
| Validation: minimum teacher-action probability | 0.999856948853 | 0.994924902916 |
| OOD: mean token full-route log probability | -5.0188760963e-05 | -0.000497519035344 |
| OOD: minimum teacher-action probability | 0.996348202229 | 0.992287397385 |

These nominal full-route probabilities are diagnostics only, evaluated on independent canonical states and never used as a loss. Canonical confidence decreased after updates despite preserved deterministic behavior and nearly unchanged sampled routes. No inference about stochastic population performance follows from this single fixed-noise panel.

Every observed token had P=8, D=8, RESTART=0 and termination by execution budget; final T sampled OOD contained the sole REPEAT. All other tokens visited eight distinct blocks. Legality, hard successor selection, selected-block accounting, frozen in-flight cache/GDN/memory purity and seven terminal-derived context updates were asserted during execution. One commit per completed token was verified. The independent artifact audit checked 11,472 recorded tokens, 91,776 decisions, state continuity and all cache versions. It reconstructed all 54,208 sampled hard actions from the recorded logits and uniforms. Stochastic paths made 54,208 candidate constructions and 487,872 speculative bank calls (nine per decision); realized execution depth remained P=8. Deterministic paths constructed zero candidates.

The controller-only T checkpoint is new. Source/backbone/historical checkpoint hashes and the tracked worktree diff identity remained unchanged, as asserted at experiment completion. U remained unchanged. No seed4402 training or replay, additional model runs, auxiliary losses, production changes or commits occurred.

Validation has six samples overlapping the reconstructed training set and two duplicate sequence occurrences. OOD full-sequence training overlap is zero by both stored inputs and length ranges. The entire panel was already exposed in the prior replication evaluation. Exact Stage-1 training exposure and unretained older probe membership remain unknown. These results are not an untouched generalization estimate. The inherited manifest limitation saying the parent tested imitation only describes the parent replication; this current experiment explicitly tested bounded warm-started task-loss updates.

One preparation failure is preserved transparently: the reporting helper raised a KeyError for selected_probability during the first deterministic validation sequence, before any optimizer update. Original traces were empty. The diagnostic reporting correction and its regression test were completed, a linked amendment manifest was sealed, and the exact existing data order and uniform tables were reused. Original manifest/failure artifacts were preserved. No scientific setting changed. The subsequent experiment completed without mechanical or nonfinite failure.

Focused validation: 20 tests passed (three new indexed-noise/reporting tests and existing routing/successor tests). git diff --check is clean. verification.json records the independent post-run checks; results.json contains all per-length metrics, and the verified JSONL files contain per-decision, token, sequence and update records. No broad regression rerun was needed for unchanged production code.

The result supports the specified bounded-maintenance criterion on this one exposed panel: real finite controller updates, valid execution, preserved deterministic parity and no worse sampled accuracy/CE than U in either split. It does not establish fresh-initialization task-loss route discovery, recovery from the historical collapse, useful diversity, state-conditioned topology, adaptive depth, HALT benefit, or superiority over fixed routing. No further experiment was started.
