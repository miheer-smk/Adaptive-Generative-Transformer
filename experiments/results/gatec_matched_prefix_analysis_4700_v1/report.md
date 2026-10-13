# Gate-C matched final-token prefix analysis

The primary hypothesis is **UNRESOLVED** across the six snapshot-1040 pairs. Holding the checkpoint’s B-state root probabilities fixed, canonical prefixes increase estimated parity correctness by **0.0165821 (1.65821 percentage points)**; the approximate conditional 95% bootstrap interval is **[-0.00466511, 0.0394748]**. Both fixed halves are positive (0.0261934 and 0.00697084), but the interval includes zero. One length-16 evaluation pair supports the hypothesis; the remaining five pairs do not resolve it. This is a fixed-panel, saved-prefix comparison, not a causal estimate over training or all prefix draws.

The two training-input pairs do not establish reduced parity correctness from controller-generated prefixes: their equal-weight contrast is **-0.0110899**, interval **[-0.0644520, 0.0391206]**. No explicit training-state exposure gate was found in the supplied protocol or repository records; clarification was requested. The requested gate therefore cannot be certified, and these training contrasts do not meet the existing resolved-positive direction rule. No exposure experiment is designed or run.

## Locked inputs and definitions

Manifest SHA-256: `c562cbdc54d3900d2b044bb91f3e7dfd8111a539a4ccd9a7f7fe440302dd0aba`. Revision: `75f138e6b909c5ca9884e9a7755de63489d3d6f2`; branch: `research/stage3-routing-gate-c-audit`; tracked-diff SHA-256: `45b3d093bd7fd7976019bf4572dc574784ed271b3294b86194f80c05826a8eac`. The manifest was written and hash-sealed before deriving new results.

Parent action-value manifest: `11dac7bfc74b883f7b4ea89743816cb61ea21d502447a4ab737672e435276bfa`. Temporal diagnostic manifest: `6c549c09420060e8ee5c64e0c14abecbc7a126d7ed7c9532d1fe311da364bf03`. Every indexed historical artifact, relevant source/checkpoint and recorded post-exception identity was verified before and after arithmetic. The historical bytecode-only identity exception, temporal offline verification failure and pre-execution fixture failures remain preserved.

Only B/C at human token L / internal index L-1 are included: 24 states, 12 pairs, 1,536 retained action/tape outcomes and 192 retained ST roots. B uses each checkpoint’s sealed sampled prefix; C uses the sealed canonical prefix. In both cases the same checkpoint controls stochastic continuation. C is not a fully canonical trajectory. Incoming prefix states are fixed, not replayed here.

Hard probabilities are nominal softmax of the eight legal raw ADVANCE logits without temperature division. Log probabilities and tiny probabilities are retained at full precision. HALT/REPEAT/RESTART are excluded at these first-decision successor roots; all B/C roots have exactly ADVANCE(0..7). Tau=2 applies to relaxed weights, reconstructed with original FP32 clamped-uniform sampler arithmetic on CPU. Selected actions match retained records. Original CUDA soft weights were not retained; bitwise CUDA relaxed-weight identity is unavailable.

Eight CF tapes (4800) and eight independent ST tapes (4801) are reused. No model, autograd or random-generation calls were allowed in this analysis. Analytic value mixtures do not execute reweighted policies. Canonical actions are not used as action-value labels.

The six sequence selections are unchanged:

| Sequence ID | Length | Label | Input/label SHA-256 | Exposure |
|---|---:|---:|---|---|
| train/L16/row0 | 16 | 0 | `57e0cd524f7c634db3e38035a93b713783f2bce5633fb130acbd4e2c7a973cb6` | training: ten recorded seed4700 task-loss passes |
| train/L16/row1 | 16 | 1 | `6d188347b8022adc1e3c9b5524af34d7e6709bef5b991624f97909fa97d8ee95` | training: ten recorded seed4700 task-loss passes |
| val/L16/row0 | 16 | 0 | `6d471b259a568bf4109f582f63a699a19bf5d7c16d535d62a73350cafcdea2d8` | locked final panel already evaluated in the recorded seed4700 experiment; not untouched data |
| val/L16/row2 | 16 | 1 | `01ebaa78ce38d3069a6c4947369d5d48349967bedc65d6c4b171abbdd1171e58` | locked final panel already evaluated in the recorded seed4700 experiment; not untouched data |
| ood_test/L32/row2 | 32 | 0 | `d3ce5146b9dce17652e41caa86e7748b4ad918b9ca543c3b88080eb373536394` | locked final panel already evaluated in the recorded seed4700 experiment; not untouched data |
| ood_test/L32/row0 | 32 | 1 | `90e1f6f864796f269631ac4829189c5936c1e53c691c6a0285a5bd95a6d3464f` | locked final panel already evaluated in the recorded seed4700 experiment; not untouched data |

The two training inputs are recorded training members; four evaluation inputs are outside the materialized 104 inputs but were previously evaluated. Stage-1 training membership remains unknown. Snapshot 0 is historical context and is not pooled as an independent replication.

## Fixed-probability prefix-state effects

Delta parity = pi_B dot (R_C-R_B); positive favors C. Delta CE = pi_B dot (Q_C-Q_B); negative favors C. Probabilities are held fixed in each contrast. All parity values below are proportions, not percentages.

| Snapshot | Sequence | B parity under pi_B | C parity under pi_B | Delta parity | 95% interval | Direction | Delta CE | CE direction | B/C binary discordance /64 |
|---:|---|---:|---:|---:|---|---|---:|---|---:|
| 0 | train/L16/row0 | 0.535466 | 0.648904 | 0.113438 | [0, 0.274563] | unresolved | -0.732878 | negative | 7 |
| 0 | train/L16/row1 | 1.000000 | 1.000000 | 0.000000 | [0, 0] | unresolved | 0.00479741 | positive | 0 |
| 0 | val/L16/row0 | 0.483622 | 0.452346 | -0.031277 | [-0.0722138, 0] | unresolved | 0.212259 | positive | 2 |
| 0 | val/L16/row2 | 0.711149 | 0.983538 | 0.272388 | [0.130398, 0.449393] | positive | -1.70206 | negative | 19 |
| 0 | ood_test/L32/row2 | 1.000000 | 1.000000 | 0.000000 | [0, 0] | unresolved | 0.000395077 | unresolved | 0 |
| 0 | ood_test/L32/row0 | 0.358728 | 1.000000 | 0.641272 | [0.451207, 0.785996] | positive | -3.45352 | negative | 42 |
| 1040 | train/L16/row0 | 0.535803 | 0.513623 | -0.022180 | [-0.128904, 0.0782412] | unresolved | 0.0939635 | unresolved | 7 |
| 1040 | train/L16/row1 | 1.000000 | 1.000000 | 0.000000 | [0, 0] | unresolved | 0.00623953 | positive | 0 |
| 1040 | val/L16/row0 | 0.287903 | 0.237579 | -0.050324 | [-0.118054, 0] | unresolved | 0.400884 | positive | 3 |
| 1040 | val/L16/row2 | 0.813223 | 0.985219 | 0.171996 | [0.084486, 0.276314] | positive | -1.05504 | negative | 14 |
| 1040 | ood_test/L32/row2 | 1.000000 | 1.000000 | 0.000000 | [0, 0] | unresolved | -0.000651909 | negative | 0 |
| 1040 | ood_test/L32/row0 | 1.000000 | 1.000000 | 0.000000 | [0, 0] | unresolved | -0.000558664 | unresolved | 0 |

Equal-pair summaries (two pairs per source; six in each snapshot summary):

| Snapshot/source | Delta parity | 95% interval | Direction | Delta CE | 95% interval | Direction |
|---|---:|---|---|---:|---|---|
| 0/all | 0.16597 | [0.110339, 0.219054] | positive | -0.945168 | [-1.21779, -0.643887] | negative |
| 0/training | 0.056719 | [0, 0.137281] | unresolved | -0.36404 | [-0.863535, -0.0202922] | negative |
| 0/length16_eval | 0.120556 | [0.0405339, 0.215427] | positive | -0.744899 | [-1.32519, -0.305116] | negative |
| 0/length32_ood | 0.320636 | [0.225604, 0.392998] | positive | -1.72656 | [-2.3234, -1.06343] | negative |
| 1040/all | 0.0165821 | [-0.00466511, 0.0394748] | unresolved | -0.092528 | [-0.193539, 0.00606914] | unresolved |
| 1040/training | -0.0110899 | [-0.064452, 0.0391206] | unresolved | 0.0501015 | [-0.150277, 0.278654] | unresolved |
| 1040/length16_eval | 0.0608364 | [-0.00265576, 0.125565] | unresolved | -0.32708 | [-0.637679, -0.00911427] | negative |
| 1040/length32_ood | 0 | [0, 0] | unresolved | -0.000605286 | [-0.00148642, -7.34111e-05] | negative |

Snapshot1040 val/L16/row2 supports the hypothesis (parity delta +0.171996; interval [0.0844860,0.276314]). The training parity-0 and evaluation parity-0 pairs have negative point estimates but unresolved directions. The other three pairs have identical observed B/C binary outcomes, yielding zero contrasts and degenerate intervals. This is not proof of zero population effect. Snapshot0’s six-pair contrast is +0.165970, conditionally resolved; its different states/checkpoint do not establish an optimizer causal effect.

## Root probability allocation

Within-B contrasts change only root weights; B values and the downstream policy stay fixed. pi_C is the checkpoint distribution at the C state, not teacher forcing. Complete B/C expectations under pi_B, pi_C and uniform, every pair/contrast and all uncertainty are in results.json and paired_contrasts.csv.

| Snapshot/source | Parity: pi_C minus pi_B within B | Direction | Parity: uniform minus pi_B within B | Direction | CE: pi_C minus pi_B within B | Direction |
|---|---:|---|---:|---|---:|---|
| 0/all | -0.00170392 | negative | -0.00441105 | negative | 0.00480335 | unresolved |
| 0/training | 0.00297219 | positive | -0.00210822 | unresolved | -0.0178975 | negative |
| 0/length16_eval | 0.000940682 | unresolved | -0.00363588 | unresolved | -0.00735593 | unresolved |
| 0/length32_ood | -0.00902462 | negative | -0.00748904 | negative | 0.0396635 | positive |
| 1040/all | -0.00297329 | negative | 0.00061604 | unresolved | 0.0216468 | positive |
| 1040/training | -0.00342765 | negative | 0.00553594 | unresolved | 0.0222493 | positive |
| 1040/length16_eval | -0.00549221 | unresolved | -0.00368782 | unresolved | 0.0432188 | positive |
| 1040/length32_ood | 0 | unresolved | 0 | unresolved | -0.000527843 | unresolved |

At snapshot1040, replacing pi_B with pi_C on B values reduces parity by 0.00297329 (interval [-0.00484036,-0.000857538]) and increases CE by 0.0216468 (interval [0.0105696,0.0322147]), both resolved. Uniform weighting changes parity by +0.000616040, unresolved. Fixed-probability state contrasts remain unresolved under pi_C and uniform as well. Root allocation and prefix-state effects are distinct; changing allocation cannot be inferred to repair the entire trajectory.

## Predefined low-hard-mass coverage control

All 12 B groups are empty: each legal action individually exceeds the 1% threshold. Across all 24 B/C states, individual hard probabilities range from 0.0879200 to 0.161846. Thus no nonempty rare-action group exists under the declared rule, and group-uniform opportunity, group relaxed mass and group descent predictions are not applicable. No group was widened after observing results. All action-level hard/relaxed weights remain in the complete tables.

This offers no evidence for 1%-rare root-action undercoverage at these measured states. It does not estimate training-wide visits or later-route coverage. Exhaustive successor construction and relaxed backward paths preclude inferring gradient starvation from hard probability alone.

## CE versus parity credit

Gamma_par = -[pi*(R-pi dot R)] dot mean full ST gradient. Positive predicts improved parity under an independent local-logit descent direction; positive CE alignment predicts reduced CE. Neither predicts the shared-parameter optimizer step. CE and parity action values come from the independent CF bank, gradients from the ST bank.

| Snapshot | Sequence/prefix | Gamma_par | 95% interval | Direction | CE alignment | 95% interval | Direction | Binary table constant |
|---:|---|---:|---|---|---:|---|---|---|
| 0 | train/L16/row0/B | 0.0240628 | [3.04396e-05, 0.0818694] | positive | 0.14605 | [0.000185047, 0.494414] | positive | False |
| 0 | train/L16/row0/C | 0.00710304 | [3.60266e-05, 0.0231991] | positive | 0.0434794 | [0.000222987, 0.141791] | positive | False |
| 0 | train/L16/row1/B | -0 | [-0, 0] | unresolved | 4.07781e-09 | [-9.77236e-10, 1.78045e-08] | unresolved | True |
| 0 | train/L16/row1/C | -0 | [-0, 0] | unresolved | 4.84187e-07 | [1.12461e-09, 2.00213e-06] | positive | True |
| 0 | val/L16/row0/B | 0.117868 | [7.42762e-05, 0.39613] | positive | 0.720167 | [0.00046751, 2.40056] | positive | False |
| 0 | val/L16/row0/C | 0.0272318 | [0.00018276, 0.0909296] | positive | 0.168665 | [0.00113452, 0.556978] | positive | False |
| 0 | val/L16/row2/B | 0.00106341 | [-1.25852e-05, 0.00335552] | unresolved | 0.00665769 | [-8.76782e-05, 0.0208914] | unresolved | False |
| 0 | val/L16/row2/C | 8.241e-08 | [-0, 3.70549e-07] | unresolved | 3.45875e-07 | [1.65033e-09, 1.31584e-06] | positive | False |
| 0 | ood_test/L32/row2/B | -0 | [-0, 0] | unresolved | 3.07794e-11 | [5.2217e-12, 7.771e-11] | positive | True |
| 0 | ood_test/L32/row2/C | -0 | [-0, 0] | unresolved | 8.59066e-12 | [-8.72671e-11, 1.22098e-10] | unresolved | True |
| 0 | ood_test/L32/row0/B | 0.0680824 | [8.21487e-05, 0.213842] | positive | 0.377746 | [0.0004957, 1.21195] | positive | False |
| 0 | ood_test/L32/row0/C | -0 | [-0, 0] | unresolved | 5.62081e-08 | [-4.01208e-11, 3.20914e-07] | unresolved | True |
| 1040 | train/L16/row0/B | 0.00400029 | [2.16056e-05, 0.0133312] | positive | 0.0249556 | [0.0001362, 0.0826574] | positive | False |
| 1040 | train/L16/row0/C | 0.0461639 | [0.000164605, 0.15374] | positive | 0.28141 | [0.00102697, 0.918088] | positive | False |
| 1040 | train/L16/row1/B | -0 | [-0, 0] | unresolved | 2.35993e-07 | [-2.22132e-08, 1.21042e-06] | unresolved | True |
| 1040 | train/L16/row1/C | -0 | [-0, 0] | unresolved | 4.23327e-06 | [-4.80064e-08, 2.29889e-05] | unresolved | True |
| 1040 | val/L16/row0/B | 0.000373174 | [5.02049e-05, 0.000950823] | positive | 0.00246048 | [0.00034475, 0.0062048] | positive | False |
| 1040 | val/L16/row0/C | 0.352567 | [0.000194902, 1.14027] | positive | 2.13912 | [0.0013234, 6.91486] | positive | False |
| 1040 | val/L16/row2/B | 0.000159787 | [-3.22933e-05, 0.000534035] | unresolved | 0.000955252 | [-0.000212429, 0.00305647] | unresolved | False |
| 1040 | val/L16/row2/C | 4.87382e-08 | [-0, 1.82762e-07] | unresolved | 1.7246e-07 | [6.69146e-10, 6.18944e-07] | positive | False |
| 1040 | ood_test/L32/row2/B | -0 | [-0, 0] | unresolved | 4.63272e-11 | [1.40757e-11, 9.59453e-11] | positive | True |
| 1040 | ood_test/L32/row2/C | -0 | [-0, 0] | unresolved | -6.10494e-13 | [-3.43755e-12, 1.15588e-12] | unresolved | True |
| 1040 | ood_test/L32/row0/B | -0 | [-0, 0] | unresolved | 3.22026e-06 | [2.09457e-10, 1.86811e-05] | positive | True |
| 1040 | ood_test/L32/row0/C | -0 | [-0, 0] | unresolved | 1.14757e-07 | [1.23461e-09, 5.00907e-07] | positive | True |

Snapshot1040: four states resolve positive parity credit, eight are unresolved; none resolves negative parity credit. Eight states resolve positive CE alignment. Three of those have constant observed correctness (both B OOD states and C ood_test/L32/row0), so their measured credit concerns CE/calibration alone. C val/L16/row2 has resolved CE alignment with unresolved parity credit. These observations do not establish opposed CE/parity credit as the failure mechanism. Constant outcomes can conceal opportunities outside the eight sampled tapes.

Snapshot0: five states resolve positive parity credit and seven are unresolved; eight resolve positive CE alignment. No snapshot is treated as another independent controller-training seed.

## Uncertainty, limitations and integrity

Intervals are approximate conditional Monte Carlo intervals, unadjusted for multiple comparisons. The first-four and last-four fixed halves must agree in sign and the full 95% interval must exclude zero to resolve a direction. Existing 2,000-replicate bootstrap indices from seed4802 are reused: CF[1] and ST[1], the first locked B in manifest order, chosen before new outcomes. Whole tape coordinates are jointly resampled across actions, B/C and fixed-panel sequences; retained gradient/noise associations travel together. CF and ST banks are independently resampled. No sequence identities or controller seeds are resampled. Original per-state CE intervals are retained separately; the newly paired intervals use the sealed joint coupling and can differ.

Small discordance (at most four of 64 paired binary entries), zero discordance, 0/8, 8/8 and constant per-action binary outcomes are explicitly flagged in results.json. The snapshot1040 evaluation parity-0 pair has only 3/64 discordances. Degenerate zero intervals do not establish no population opportunity. B/C values depend on the frozen checkpoint’s stochastic continuation, not a fully canonical suffix. One saved prefix realization per condition and eight continuation tapes limit conclusions.

Historical bytecode-only identity exception and prior failure records remain unchanged. The array-only focused suite passed 7 tests; a pre-analysis mixed-dtype synthetic fixture failure was corrected and preserved in the manifest. Independent offline checks verified every per-action mean, nominal probability, original projection identity, paired mixture, bootstrap interval, fixed-half decision and equal-pair summary. No full model test suite was run because this task forbids new model executions.

No model forwards, backwards, counterfactual continuations, experimental noise, optimizer steps, parameter updates or production edits occurred. No training-state exposure protocol or new training run is proposed. Checkpoints, historical artifacts and protected sources remained byte-identical, and git diff --check is clean.

Artifacts: manifest.json; results.json (complete states, root/noise/gradient records, per-tape outcomes, mixtures and uncertainty); action_tables.csv; paired_contrasts.csv; verification.json; offline_verification.py; artifact_index.json. All files were created under a new analysis directory without overwriting historical artifacts.
