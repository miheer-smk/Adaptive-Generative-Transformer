# Seed4700 snapshot1040: temporal controller-output VJP diagnostic

The full unreliability hypothesis is unresolved at the tested states and is not supported by a resolved execution-credit-to-full-credit failure. Opposing components, if resolved, are separate from loss of resolved full alignment.

Completed **144 identical forward replays and 384 original-task-loss VJPs** at all 18 predeclared states. No new counterfactual continuations or noise were generated. Resolved aligned execution-only CE credit: 8/18. Resolved opposing remaining projection: 6/18. Resolved added vector variance: 2/18; resolved added projection variance: 2/18. Resolved reduced vector variance: 10/18. Strong state-level support under the sealed criterion: 0/18.

## Identity, controls and graph definition

Manifest SHA-256 `6c549c09420060e8ee5c64e0c14abecbc7a126d7ed7c9532d1fe311da364bf03`; HEAD `75f138e6b909c5ca9884e9a7755de63489d3d6f2`; tracked-diff SHA-256 `45b3d093bd7fd7976019bf4572dc574784ed271b3294b86194f80c05826a8eac`. Snapshot1040 `experiments/runs/stage3_fresh_task_4700_update1040_locked_v1.pt`, SHA-256 `37537106f47e3a003e4b5d2ff98ded93a8f98e3a2b60ee7814e502d28ddaa6d4`. Frozen backbone `experiments/runs/stage1_parity_seed3305_best.pt`, SHA-256 `53b56e7e90f7eb7a6570107b6dfe3a297caac48d637fe27e5f00fbe94c6f09c3`.

All completed action-value artifact hashes, selected weights, input states and historical parent identities were checked. The historical bytecode-only exception remains disclosed and preserved. Bytecode writing was disabled. No optimizer, training, parameter/buffer update, checkpoint overwrite, production edit or commit occurred.

A starts at human ordinal1/internal0, with zero prior commits. B/C start at human ordinal L/internal L−1. Incoming states were restored from the full retained states, including memory, all caches, GDN recurrence/convolution buffers, versions and commit counters. B preserves its sealed checkpoint-owned sampled prefix; C its canonical prefix. No prefix was rerun, and C forcing does not extend into the suffix. All eight ST tapes from seed4801 and all retained CF outcomes from seed4800 were reused. Snapshot0 supplied recorded context only; it was never replayed.

Every forward uses the existing hard-forward exhaustive-candidate ST path, tau2, FP32/batch1, zero budget/presence, P8, original masks, frozen in-flight state and one terminal commit/token. Incoming state is fixed conditioning; the historical full-sequence BPTT policy is preserved within the suffix. Model/runtime flags and exact state/noise sources are in manifest.json.

A tensor hook on each original unified controller-logit output gates only its VJP. It never changes a forward value or an action. The probe output is always allowed. Execution-only excludes every later controller-output path. Token includes later calls in the probe token. Next additionally includes calls in the immediately following token. Full suppresses no output VJP. Direct execution, candidate/state, memory and commit derivatives remain available in every mode. Existing ST relaxed.detach and hard action/pointer/count/mask boundaries are unchanged; none was reconnected.

Each A graph is replayed once with four task-loss VJPs. Each B/C graph is replayed once with execution-only and full VJPs: its suffix has no later token, so token=next=full was verified structurally and reused without redundant derivatives. Full VJPs also retain original gradients at every controller call and available live-state/anchor/read/slot-mean adjoints. These nonprobe calls receive no new action values or action-value alignment labels.

Execution-state adjoints refer to the actual graph tensors. Shared anchors, memory reads and slot means are deduplicated VJP targets; their repeated call records contain the total adjoint of the shared tensor, not separate per-use gradient attributions.

Components are c_exec=g_exec, c_within=g_token−g_exec, c_next=g_next−g_token and c_distant=g_full−g_next. This is an exact first-order telescoping path partition within numerical tolerance, not an additive decomposition of task losses or historical optimization effects. Logit-hook boundary indices are recorded per tape.

## Previous classifications

| Checkpoint | Source | Probe | Previous classifications (two parity inputs) |
|---|---|---|---|
| 0 | length16_eval | A | {'informative_unresolved_direction': 2} |
| 0 | length16_eval | B | {'informative_aligned_ST': 1, 'informative_unresolved_direction': 1} |
| 0 | length16_eval | C | {'informative_aligned_ST': 2} |
| 0 | length32_ood | A | {'informative_unresolved_direction': 1, 'unresolved_action_values': 1} |
| 0 | length32_ood | B | {'informative_aligned_ST': 2} |
| 0 | length32_ood | C | {'informative_unresolved_direction': 2} |
| 0 | training | A | {'informative_aligned_ST': 1, 'informative_unresolved_direction': 1} |
| 0 | training | B | {'informative_aligned_ST': 1, 'informative_unresolved_direction': 1} |
| 0 | training | C | {'informative_aligned_ST': 2} |
| 1040 | length16_eval | A | {'informative_unresolved_direction': 2} |
| 1040 | length16_eval | B | {'informative_aligned_ST': 1, 'informative_unresolved_direction': 1} |
| 1040 | length16_eval | C | {'informative_aligned_ST': 2} |
| 1040 | length32_ood | A | {'informative_unresolved_direction': 2} |
| 1040 | length32_ood | B | {'informative_aligned_ST': 2} |
| 1040 | length32_ood | C | {'informative_unresolved_direction': 1, 'informative_aligned_ST': 1} |
| 1040 | training | A | {'informative_unresolved_direction': 2} |
| 1040 | training | B | {'informative_aligned_ST': 1, 'informative_unresolved_direction': 1} |
| 1040 | training | C | {'informative_aligned_ST': 1, 'informative_unresolved_direction': 1} |

Full prior per-input classifications are retained in the manifest; aggregate alignment counts were not used to select states.

## All 18 states

Directions below use the original four-tape-half and 95% interval rule. Variance ratio is descriptive; the paired variance-difference interval is the inferential quantity. Conditional correctness range uses the retained hard action continuations.

| Sequence / probe | Exec / token / next / full | Remaining CE projection [95% CI] | Full/exec variance | Variance difference [95% CI] | Correctness range | Classification |
|---|---|---|---:|---|---|---|
| train/L16/row0 / A | unresolved / unresolved / unresolved / unresolved | -0.000112828 [-0.0006356860387448849, 5.304635606986968e-07] | 0.0370492 | -1.32074e-05 [-2.8459785241149982e-05, -2.537358802515069e-09] | 0.375…0.5 | unresolved_execution_only_credit |
| train/L16/row0 / B | aligned / aligned / aligned / aligned | -0.0293826 [-0.09963667951238185, -1.79528686560462e-05] | 0.203663 | -0.0756245 [-0.1623854683699592, -8.876967679995738e-08] | 0…1 | component_effect_with_resolved_full_alignment |
| train/L16/row0 / C | aligned / aligned / aligned / aligned | -0.156887 [-0.4780150980790188, 0.00018393859678792103] | 0.432459 | -3.68676 [-8.020612424263994, 7.968197480965827e-06] | 0…1 | no_resolved_adverse_temporal_effect |
| train/L16/row1 / A | unresolved / unresolved / unresolved / unresolved | 2.28236e-06 [-6.571008891535736e-08, 1.1306296828655058e-05] | 0.215949 | -4.26146e-09 [-8.466418408764446e-09, 1.7651487018160077e-13] | 0.375…0.5 | unresolved_execution_only_credit |
| train/L16/row1 / B | unresolved / unresolved / unresolved / unresolved | -5.27717e-07 [-2.936178755601674e-06, 4.659105067071673e-08] | 0.080819 | -0.000113129 [-0.0002457646591826153, -1.775237025191127e-08] | 1…1 | unresolved_execution_only_credit |
| train/L16/row1 / C | unresolved / unresolved / unresolved / unresolved | -1.33585e-05 [-7.52250451752846e-05, 1.759965900120105e-07] | 0.0524154 | -0.00252953 [-0.005435698637612456, -8.185479698408138e-11] | 1…1 | unresolved_execution_only_credit |
| val/L16/row0 / A | unresolved / unresolved / unresolved / unresolved | -1.87022e-09 [-8.418190984956583e-09, 1.48847325045826e-10] | 1.01807 | 1.84311e-12 [-1.358922071044748e-12, 7.294482209844673e-12] | 0…0 | unresolved_execution_only_credit |
| val/L16/row0 / B | aligned / aligned / aligned / aligned | -0.000317636 [-0.0008110683500387455, -5.105934493391543e-05] | 0.860098 | -2.03218e-05 [-4.1152924228583566e-05, -3.5044848671032837e-07] | 0…0.625 | component_effect_with_resolved_full_alignment |
| val/L16/row0 / C | aligned / aligned / aligned / aligned | -1.52231 [-5.065565873629262, -0.00019223610139037877] | 0.326613 | -431.094 [-941.021101860522, -4.27522552092667e-06] | 0…0.625 | component_effect_with_resolved_full_alignment |
| val/L16/row2 / A | unresolved / unresolved / unresolved / unresolved | 5.96877e-11 [-1.4399540892804688e-13, 3.066014362761456e-10] | 0.256449 | -6.5568e-10 [-1.4150801634156146e-09, 1.422144275958057e-13] | 0.875…0.875 | unresolved_execution_only_credit |
| val/L16/row2 / B | unresolved / unresolved / unresolved / unresolved | -0.00204405 [-0.006589019931210524, 1.1858903612953713e-05] | 0.10072 | -0.00196844 [-0.004229368439544376, -4.6978017177983184e-09] | 0…1 | unresolved_execution_only_credit |
| val/L16/row2 / C | aligned / aligned / aligned / aligned | -1.50032e-07 [-7.259600209590748e-07, -4.434759005548692e-10] | 0.54638 | -3.20827e-09 [-5.685699214781252e-09, -1.0926643757069275e-10] | 0.875…1 | component_effect_with_resolved_full_alignment |
| ood_test/L32/row2 / A | unresolved / unresolved / unresolved / unresolved | -1.28515e-09 [-5.2998264749370135e-09, 3.191758546129385e-11] | 1.19808 | 9.92257e-13 [-3.758672350029767e-12, 6.2361315358740506e-12] | 0.125…0.125 | unresolved_execution_only_credit |
| ood_test/L32/row2 / B | aligned / aligned / aligned / aligned | 4.48367e-12 [1.535244261624806e-12, 9.058807676767685e-12] | 1.14313 | 2.98422e-12 [8.314680445203515e-13, 4.6036748517125126e-12] | 1…1 | component_effect_with_resolved_full_alignment |
| ood_test/L32/row2 / C | unresolved / unresolved / unresolved / unresolved | -5.95881e-14 [-2.8762112609009246e-13, 6.481246156946692e-14] | 1.13971 | 1.14122e-13 [1.4796435001646863e-14, 1.9724170634451594e-13] | 1…1 | unresolved_execution_only_credit |
| ood_test/L32/row0 / A | unresolved / unresolved / unresolved / unresolved | -2.67323e-09 [-1.085101918263733e-08, 2.133910451017274e-12] | 0.768794 | -9.54806e-10 [-4.737619536090547e-09, 3.3737807355998374e-09] | 0.875…0.875 | unresolved_execution_only_credit |
| ood_test/L32/row0 / B | aligned / aligned / aligned / aligned | -5.46577e-06 [-3.207693236533725e-05, -1.1389125475783441e-10] | 0.134867 | -0.000429543 [-0.0009236135459401505, -4.775631827365336e-10] | 1…1 | component_effect_with_resolved_full_alignment |
| ood_test/L32/row0 / C | aligned / aligned / aligned / aligned | -8.22725e-08 [-3.075670941016956e-07, -9.45553944271774e-10] | 0.428115 | -6.41575e-07 [-1.3923036026230282e-06, -2.660253979994063e-10] | 1…1 | component_effect_with_resolved_full_alignment |

## Component projections

| Sequence / probe | Exec | Within token | Next token | Distant tokens | Full |
|---|---:|---:|---:|---:|---:|
| train/L16/row0 / A | 0.000100294 | -9.66316e-06 | -1.67905e-07 | -0.000102997 | -1.2534e-05 |
| train/L16/row0 / B | 0.0543382 | -0.0293826 | 0 | 0 | 0.0249556 |
| train/L16/row0 / C | 0.438297 | -0.156887 | 0 | 0 | 0.28141 |
| train/L16/row1 / A | -3.93565e-06 | 1.66541e-06 | 5.75166e-09 | 6.11202e-07 | -1.65329e-06 |
| train/L16/row1 / B | 7.6371e-07 | -5.27717e-07 | 0 | 0 | 2.35993e-07 |
| train/L16/row1 / C | 1.75918e-05 | -1.33585e-05 | 0 | 0 | 4.23327e-06 |
| val/L16/row0 / A | -4.80247e-08 | -2.72875e-10 | 7.62568e-10 | -2.35991e-09 | -4.98949e-08 |
| val/L16/row0 / B | 0.00277811 | -0.000317636 | 0 | 0 | 0.00246048 |
| val/L16/row0 / C | 3.66143 | -1.52231 | 0 | 0 | 2.13912 |
| val/L16/row2 / A | -7.62986e-11 | 4.87804e-12 | -1.42844e-12 | 5.62381e-11 | -1.66109e-11 |
| val/L16/row2 / B | 0.00299931 | -0.00204405 | 0 | 0 | 0.000955252 |
| val/L16/row2 / C | 3.22491e-07 | -1.50032e-07 | 0 | 0 | 1.7246e-07 |
| ood_test/L32/row2 / A | -2.3901e-09 | 3.39893e-11 | -1.07937e-10 | -1.2112e-09 | -3.67525e-09 |
| ood_test/L32/row2 / B | 4.18435e-11 | 4.48367e-12 | 0 | 0 | 4.63272e-11 |
| ood_test/L32/row2 / C | -5.50906e-13 | -5.95881e-14 | 0 | 0 | -6.10494e-13 |
| ood_test/L32/row0 / A | 6.27677e-10 | -2.18916e-10 | -1.58347e-11 | -2.43848e-09 | -2.04555e-09 |
| ood_test/L32/row0 / B | 8.68603e-06 | -5.46577e-06 | 0 | 0 | 3.22026e-06 |
| ood_test/L32/row0 / C | 1.9703e-07 | -8.22725e-08 | 0 | 0 | 1.14757e-07 |

## Stratification and matched horizons

| Source / probe | State classifications |
|---|---|
| length16_eval/A | {'unresolved_execution_only_credit': 2} |
| length16_eval/B | {'component_effect_with_resolved_full_alignment': 1, 'unresolved_execution_only_credit': 1} |
| length16_eval/C | {'component_effect_with_resolved_full_alignment': 2} |
| length32_ood/A | {'unresolved_execution_only_credit': 2} |
| length32_ood/B | {'component_effect_with_resolved_full_alignment': 2} |
| length32_ood/C | {'unresolved_execution_only_credit': 1, 'component_effect_with_resolved_full_alignment': 1} |
| training/A | {'unresolved_execution_only_credit': 2} |
| training/B | {'component_effect_with_resolved_full_alignment': 1, 'unresolved_execution_only_credit': 1} |
| training/C | {'no_resolved_adverse_temporal_effect': 1, 'unresolved_execution_only_credit': 1} |

| Sequence | B exec/full | C exec/full | B class | C class |
|---|---|---|---|---|
| train/L16/row0 | aligned/aligned | aligned/aligned | component_effect_with_resolved_full_alignment | no_resolved_adverse_temporal_effect |
| train/L16/row1 | unresolved/unresolved | unresolved/unresolved | unresolved_execution_only_credit | unresolved_execution_only_credit |
| val/L16/row0 | aligned/aligned | aligned/aligned | component_effect_with_resolved_full_alignment | component_effect_with_resolved_full_alignment |
| val/L16/row2 | unresolved/unresolved | aligned/aligned | unresolved_execution_only_credit | component_effect_with_resolved_full_alignment |
| ood_test/L32/row2 | aligned/aligned | unresolved/unresolved | component_effect_with_resolved_full_alignment | unresolved_execution_only_credit |
| ood_test/L32/row0 | aligned/aligned | aligned/aligned | component_effect_with_resolved_full_alignment | component_effect_with_resolved_full_alignment |

B versus C holds the remaining one-token horizon fixed while changing prefix-conditioned incoming state. These are state-conditioned comparisons, not random-assignment causal prefix interventions. A versus B additionally changes horizon, token input and incoming state; no causal horizon claim is inferred from that comparison.

## Uncertainty, cancellation and variance

The original 2,000 bootstrap index rows (seed4802, original36-state indexing) were reused. All components of an ST tape are resampled together. CF action outcomes retain their pairing, and the CF and ST banks are independently resampled. Projections use the retained g_CF, with its CF uncertainty propagated during bootstrap. Positive alignment concerns descent of expected final CE with incoming state/downstream policy fixed.

A resolved cancellation claim requires a negative remaining projection in both halves and a95% upper bound below zero. A resolved added-variance claim requires positive paired differences in both halves and a95% lower bound above zero. Norm shrinkage alone is not cancellation evidence. Variance/covariance is sample trace covariance, ddof1: Var(full)=Var(exec)+Var(rest)+2Cov(exec,rest). Both vector and g_CF-projection decompositions, covariance matrices and intervals are retained.

Strong hypothesis support requires informative values and resolved aligned execution-only CE credit, followed by resolved cancellation or added variance and loss of resolved full alignment. An adverse component with still-resolved full alignment is reported separately. Weak/unresolved execution-only credit is not attributed to later-controller cancellation. All intervals are conditional Monte Carlo intervals from eight tapes per bank, unadjusted for multiplicity; they are not training-seed/population estimates.

## Parity and detach evidence

Conditional correctness differs across first-action branches at 8/18 states. The other 10/18 show only CE/calibration differences in this bank. No policy update was tested, and CE-gradient alignment is not labeled useful parity learning.

Original full routing-gradient statuses across all calls: `{'connected_nonzero': 10770, 'connected_small': 1435}`. Available execution-state adjoint statuses: `{'h/fixed_tensor_no_autograd_input': 144, 'memory_read/fixed_tensor_no_autograd_input': 1546, 'memory_mean/fixed_tensor_no_autograd_input': 1546, 'h_next/connected_nonzero': 12205, 'h0/fixed_tensor_no_autograd_input': 1546, 'h/connected_nonzero': 12061, 'memory_read/connected_nonzero': 10659, 'memory_mean/connected_nonzero': 9519, 'h0/connected_nonzero': 10659, 'memory_mean/connected_small': 1140}`. Every record includes actual token ID, within-token decision index/count, execution count and remaining-token/execution horizon. Fixed incoming tensors are labeled separately from disconnected gradients, connected exact zeros and connected small derivatives.

Existing detach boundaries are the ST relaxed stop-gradient and hard routing metadata. The incoming state is deliberately conditioned as fixed. No extra suffix detach was added. Magnitude versus position alone is not treated as temporal causation; causal path exclusions occur only at the explicitly recorded controller-output VJP boundaries.

## Validation and limits

All144 forward trajectories reproduce retained incoming identities, sealed noise/actions, selected blocks, logits/CE and exact per-token incoming/outgoing commit-state hashes. Full probe gradients reproduce their retained values at atol1e−6/rtol1e−5. Every component sum reconstructs full; B/C next/distant components are exact zeros. Accounting, legality, candidate/GDN/anchor purity, cache versions and one seven-entry commit/token passed. Parameters, production source and historical files remained hash-identical.

Focused tests:32 passed. A pre-execution synthetic zero-loss fixture failure was corrected and is preserved in manifest.json; no AGT replay occurred during that failed test. Model derivative budget:144 forwards/384 task-loss VJPs. The reused comparator additionally performed216 small categorical-formula autograd identity checks; these never execute AGT or measure another trajectory. Offline analysis performs no model call or VJP. No full regression or training was launched.

An initial offline bootstrap check used float64 sums of serialized FP32 component differences instead of the original cumulative vectors. Its remaining-projection interval differed by2.45e−11. The offline input selection was corrected without changing gradients, tolerances or measurements. The failure is preserved in offline_verification_failure_1.json. Component covariance matrices retain the FP32 subtraction rounding; their small reconstruction residuals are reported explicitly.

The full unreliability hypothesis is unresolved at the tested states and is not supported by a resolved execution-credit-to-full-credit failure. Opposing components, if resolved, are separate from loss of resolved full alignment.

This isolates first-order derivative paths at18 selected states of one checkpoint and one controller-training seed. It does not establish the historical cause of training failure, a replacement estimator, task-loss route discovery, general trainability or parity improvement. No isolated component is used for training or recommended as a replacement. Sampling was not expanded.

## Artifacts

- manifest.json: sealed graph boundaries, identities, prior classifications and budget.
- results.json and probe18.json…probe35.json: complete per-tape cumulative/component vectors, alignments, conditional outcomes and uncertainty.
- runs/decisions/tokens.jsonl.gz: replay outcomes, logits/actions/masks/noise coordinates, state summaries and commit identities.
- adjoints.jsonl.gz: original gradients at every controller call and available execution-state adjoints.
- interpretation.json: covariance matrices/intervals, signal-to-noise summaries, all strata and matched B/C comparisons.
- verification.json: independent offline checks; no model execution.
