# AGT-v3 seed4700: local ST credit versus hard closed-loop action values

The proposed directional-failure mechanism is not supported at the tested states. Resolved comparisons favor alignment; the remaining comparisons are unresolved. This contradicts a blanket claim of consistently opposed local ST credit at these probes, while leaving a broader or rarer failure mechanism unresolved.

Overall hypothesis verdict: **unresolved across the full tested set**. Seventeen resolved comparisons contradict local directional opposition; none support it. The other nineteen states cannot resolve that hypothesis under the sealed uncertainty rule.

Completed the sealed budget exactly: **36 states, 2,304 counterfactual continuations, 288 unforced ST gradient probes**. 17 informative probes have resolved alignment and 0 have resolved opposition. Classifications: `{'informative_aligned_ST': 17, 'informative_unresolved_direction': 18, 'unresolved_action_values': 1}`.

## Protocol and identity

Git HEAD: `75f138e6b909c5ca9884e9a7755de63489d3d6f2`; branch `research/stage3-routing-gate-c-audit`. Manifest SHA-256: `11dac7bfc74b883f7b4ea89743816cb61ea21d502447a4ab737672e435276bfa`. Tracked-diff SHA-256: `45b3d093bd7fd7976019bf4572dc574784ed271b3294b86194f80c05826a8eac`. The manifest was saved before any counterfactual or ST probe.

Backbone: `experiments/runs/stage1_parity_seed3305_best.pt`, SHA-256 `53b56e7e90f7eb7a6570107b6dfe3a297caac48d637fe27e5f00fbe94c6f09c3`. Only the controller state from the two fixed snapshots was loaded; all non-controller weights remained identical and frozen.

| Snapshot | Checkpoint | SHA-256 |
|---|---|---|
| 0 | experiments/runs/stage3_fresh_task_4700_update0000_locked_v1.pt | ae8ff9256afd573331719cf15a3f82d21f70181891400640739a0415c82d6279 |
| 1040 | experiments/runs/stage3_fresh_task_4700_update1040_locked_v1.pt | 37537106f47e3a003e4b5d2ff98ded93a8f98e3a2b60ee7814e502d28ddaa6d4 |

No optimizer was constructed. No parameters or buffers changed. No training, checkpoint overwrite, production edit, additional samples or commit occurred. Bytecode writing was disabled.

The parent experiment preserved a bytecode-only file-identity exception. Its original failure record remains intact. This diagnostic verified the recorded post-exception cache/diff identities and all experiment-relevant sources, inputs, noise, backbone and selected checkpoints. This exception was disclosed, not repaired or hidden.

| Locked sequence | Source | Parity | Bits |
|---|---|---|---|
| train/L16/row0 | training | 0 | 1110101111001111 |
| train/L16/row1 | training | 1 | 1110001110011010 |
| val/L16/row0 | length16_eval | 0 | 0101000100011111 |
| val/L16/row2 | length16_eval | 1 | 0110011110101001 |
| ood_test/L32/row2 | length32_ood | 0 | 00000001011000111101011000110000 |
| ood_test/L32/row0 | length32_ood | 1 | 11000000100101100010010111101110 |

These sequences were selected by manifest order, not outcomes. Training inputs were used during the historical ten-pass run. The four evaluation inputs exclude exact training membership but were previously evaluated. Stage-1 training membership is unknown; this is not an untouched generalization assessment. Sequence/label hashes and row identities are in the manifest.

## Prefixes and sealed randomness

A is human token ordinal 1, internal index 0, with zero preceding commits. B is the final token (internal L−1) after the checkpoint’s own sampled prefix. C is the final token after the fixed canonical prefix; canonical forcing ends at that boundary. Every probe starts after normal memory read/Prelude initialization, before decision zero, with P=D=0 and exactly A0…A7 legal successor choices. The source mask still contains its HALT slot; fixed-depth successor routing excludes it unchanged. C continuations follow the checkpoint, not a canonical suffix.

Historical tape 0 was reconstructed from CPU generator seed 924088217, in its original 96-sequence order. Every retained tensor and sequence-coordinate association was bitwise identical. The same generator was then continued to append train/L16/row0 and train/L16/row1 in training-manifest order. The extension is separate and supplies training B-prefix randomness only. Historical tensors were neither modified nor relabeled. All eight available historical evaluation B-prefix incoming hashes matched exactly.

The CF bank has eight tapes from seed 4800; the independent ST bank has eight tapes from seed 4801. Both were precomputed and sealed before measurement. Keys are bank/tape/raw-sequence/token/decision/action; checkpoint, branch, condition and outcomes are excluded. Differing route lengths therefore do not shift unrelated coordinates. Bootstrap indices (2,000 replicates/state, seed 4802) were also sealed. Tensor hashes and exact generation order are in manifest.json.

## Computation and comparison

Each incoming state was independently cloned for every branch. The CF arm forces only its first legal ADVANCE successor after the unchanged exhaustive candidate construction. Every subsequent action is sampled autonomously under the owner checkpoint until the original final-sequence parity CE. The ST arm is entirely unforced and retains the original probe logits for autograd.grad of that same final CE.

Runtime configuration: FP32, batch 1, CUDA GB10, tau=2, zero budget and zero presence, eight actual executions/token, HALT/convergence disabled. Persistent states evolve normally across suffix tokens with one commit/token. Frozen computations remain differentiable in ST. The fixed incoming prefix state is deliberate conditioning; there is no additional suffix detach. The historical full-sequence BPTT policy, existing ST stop-gradient and hard metadata transitions are unchanged. GQA/SWA/GDN context horizons are those recorded in the parent manifest.

The hard categorical law is pi=softmax(raw legal logits); tau affects the relaxed derivatives, not the argmax categorical law. The implementation uses finite-precision, epsilon-clamped uniforms; pi is the nominal Gumbel categorical law. Q(a) is mean final CE across the eight actual hard continuations. The independent local-logit benchmark is g_CF[a]=pi[a]*(Q(a)−sum_b pi[b]Q(b)). It was checked against differentiation of the explicit categorical expected loss with Q held constant at every state.

A positive inner product means the negative ST gradient locally reduces the benchmark expected CE; negative means opposition. Resolution requires both four-tape halves to share the sign and the independently resampled full-bank 95% interval to exclude zero, with meaningful norms >1e−8. CF action pairing is retained. ST and CF banks are independently resampled. Intervals describe conditional Monte Carlo uncertainty, not controller-seed or population uncertainty. Pairwise action intervals and per-probe directional intervals are unadjusted for multiplicity; they are not simultaneous confidence bands across all probes.

## All 36 probes

Accuracy range below is min/max conditional correctness across eight action means. All eight Q values, 28 paired action differences, per-tape outcomes, 11-component ST vectors, component variability, pi, entropy, half-bank signs and probability-shift estimates are retained in probeNN.json and results.json.

| Update / sequence / prefix | Q min…max | Correctness min…max | g_CF L2 | Mean ST L2 | Dot | Dot 95% interval | Direction |
|---|---:|---:|---:|---:|---:|---|---|
| 0 / train/L16/row0 / A | 3.77283…3.78366 | 0.375…0.375 | 0.00143329 | 0.000193395 | 1.59075e-07 | [2.591855752368156e-09, 5.576062673263588e-07] | aligned |
| 0 / train/L16/row0 / B | 0.00144049…6.1116 | 0…1 | 0.567112 | 0.313171 | 0.14605 | [0.0001850471869773498, 0.49441375542113747] | aligned |
| 0 / train/L16/row0 / C | 0.00140828…6.10976 | 0…1 | 0.611181 | 0.0845178 | 0.0434794 | [0.00019170058596668064, 0.1356079246951249] | aligned |
| 0 / train/L16/row1 / A | 2.37075…3.16973 | 0.5…0.625 | 0.100991 | 2.22638e-05 | -3.43834e-07 | [-9.798310914214227e-06, 7.797291340903696e-06] | unresolved |
| 0 / train/L16/row1 / B | 0.00189539…0.00212896 | 1…1 | 3.1076e-05 | 0.000176433 | 4.07781e-09 | [-9.640769764506425e-10, 1.7436643115323563e-08] | unresolved |
| 0 / train/L16/row1 / C | 0.00191444…0.00975989 | 1…1 | 0.00122897 | 0.00054527 | 4.84187e-07 | [1.0881622223715461e-09, 1.9987819318071847e-06] | aligned |
| 0 / val/L16/row0 / A | 4.47975…4.48004 | 0.25…0.25 | 3.99591e-05 | 2.80788e-06 | 7.72061e-11 | [-3.010052720606335e-11, 4.766021217589142e-10] | unresolved |
| 0 / val/L16/row0 / B | 1.55738…6.13892 | 0…0.75 | 0.549917 | 1.51134 | 0.720167 | [0.00045848006579731864, 2.384686211470198] | aligned |
| 0 / val/L16/row0 / C | 1.56124…6.13659 | 0…0.75 | 0.513177 | 0.416258 | 0.168665 | [0.0010721123437813804, 0.5368241294449233] | aligned |
| 0 / val/L16/row2 / A | 2.39529…2.39553 | 0.625…0.625 | 3.03015e-05 | 0.000620987 | -6.28157e-09 | [-4.4250003373998153e-08, 6.804675693160269e-09] | unresolved |
| 0 / val/L16/row2 / B | 0.00192576…6.59596 | 0…1 | 0.751873 | 0.0250212 | 0.00665769 | [-9.55822724515294e-05, 0.02158070868184932] | unresolved |
| 0 / val/L16/row2 / C | 0.0019318…0.409733 | 0.875…1 | 0.040138 | 2.53057e-05 | 3.45875e-07 | [1.927523339076717e-09, 1.2812212712932097e-06] | aligned |
| 0 / ood_test/L32/row2 / A | 4.48202…5.17476 | 0.125…0.25 | 0.0712174 | 1.06302e-06 | -7.31876e-09 | [-3.826386493488488e-08, 7.045740574469274e-09] | unresolved |
| 0 / ood_test/L32/row2 / B | 0.00176665…0.00197987 | 1…1 | 2.31668e-05 | 2.06969e-06 | 3.07794e-11 | [4.878666069486922e-12, 8.141929653712644e-11] | aligned |
| 0 / ood_test/L32/row2 / C | 0.00140174…0.00304204 | 1…1 | 0.000193065 | 2.36951e-07 | 8.59066e-12 | [-1.048451774957485e-10, 1.1052528518852888e-10] | unresolved |
| 0 / ood_test/L32/row0 / A | 1.27028…1.27456 | 0.625…0.625 | 0.000529924 | 0.000441316 | -2.2569e-07 | [-1.4308892209802805e-06, 1.112282328925722e-07] | unresolved |
| 0 / ood_test/L32/row0 / B | 0.00625397…6.21909 | 0…1 | 0.589742 | 0.695416 | 0.377746 | [0.0004789998007234063, 1.2295175953383717] | aligned |
| 0 / ood_test/L32/row0 / C | 0.00202534…0.0289683 | 1…1 | 0.00379216 | 6.13766e-05 | 5.62081e-08 | [-4.244431176224206e-12, 3.196993960039054e-07] | unresolved |
| 1040 / train/L16/row0 / A | 2.32514…3.13703 | 0.375…0.5 | 0.136852 | 0.00020402 | -1.2534e-05 | [-8.926020521885956e-05, 1.383054764014604e-05] | unresolved |
| 1040 / train/L16/row0 / B | 0.00158333…6.0804 | 0…1 | 0.627743 | 0.0505823 | 0.0249556 | [0.00016458621664597423, 0.08242242571251568] | aligned |
| 1040 / train/L16/row0 / C | 0.0119021…5.94016 | 0…1 | 0.632658 | 0.632665 | 0.28141 | [0.001214195155114399, 0.9118584687093346] | aligned |
| 1040 / train/L16/row1 / A | 2.71085…3.52957 | 0.375…0.5 | 0.110302 | 1.65415e-05 | -1.65329e-06 | [-7.724488255654792e-06, 5.6939301470621185e-08] | unresolved |
| 1040 / train/L16/row1 / B | 0.00206064…0.00386094 | 1…1 | 0.000203694 | 0.00137183 | 2.35993e-07 | [-2.0246859504278382e-08, 1.2042964578585306e-06] | unresolved |
| 1040 / train/L16/row1 / C | 0.00208202…0.0230586 | 1…1 | 0.00189861 | 0.00442107 | 4.23327e-06 | [-5.501254971022632e-08, 2.2945144837679473e-05] | unresolved |
| 1040 / val/L16/row0 / A | 5.42376…5.60349 | 0…0 | 0.0224983 | 3.82787e-06 | -4.98949e-08 | [-2.8311899966151413e-07, 4.861870466953341e-10] | unresolved |
| 1040 / val/L16/row0 / B | 2.23309…6.00602 | 0…0.625 | 0.46412 | 0.00714487 | 0.00246048 | [0.0003523287110260959, 0.006317076656298845] | aligned |
| 1040 / val/L16/row0 / C | 2.28081…6.00833 | 0…0.625 | 0.439049 | 5.94578 | 2.13912 | [0.001351810343560677, 6.862127041316262] | aligned |
| 1040 / val/L16/row2 / A | 0.811314…0.811419 | 0.875…0.875 | 1.451e-05 | 5.15463e-06 | -1.66109e-11 | [-9.722258691645466e-11, 1.9714846918188062e-11] | unresolved |
| 1040 / val/L16/row2 / B | 0.0019898…6.62258 | 0…1 | 0.669002 | 0.00541651 | 0.000955252 | [-0.00020711560601947474, 0.003004605134368153] | unresolved |
| 1040 / val/L16/row2 / C | 0.00203172…0.4098 | 0.875…1 | 0.0381 | 3.05564e-05 | 1.7246e-07 | [5.214576036382161e-10, 6.550681421396621e-07] | aligned |
| 1040 / ood_test/L32/row2 / A | 5.06254…5.13958 | 0.125…0.125 | 0.00781555 | 1.94459e-06 | -3.67525e-09 | [-1.3881366802518919e-08, 1.0675071154021197e-10] | unresolved |
| 1040 / ood_test/L32/row2 / B | 0.00200436…0.00210253 | 1…1 | 1.15031e-05 | 4.24734e-06 | 4.63272e-11 | [1.4287876206635418e-11, 9.638483607869321e-11] | aligned |
| 1040 / ood_test/L32/row2 / C | 0.00139998…0.001437 | 1…1 | 4.18101e-06 | 3.04054e-07 | -6.10494e-13 | [-3.550016114505173e-12, 1.2131181166461238e-12] | unresolved |
| 1040 / ood_test/L32/row0 / A | 0.506957…0.508758 | 0.875…0.875 | 0.000232635 | 1.49087e-05 | -2.04555e-09 | [-1.4521530285773683e-08, 2.24986356262288e-09] | unresolved |
| 1040 / ood_test/L32/row0 / B | 0.00230133…0.0269328 | 1…1 | 0.00399153 | 0.00303193 | 3.22026e-06 | [3.8425559509834257e-10, 1.8714641817313367e-05] | aligned |
| 1040 / ood_test/L32/row0 / C | 0.00238912…0.024249 | 1…1 | 0.00307605 | 0.000279513 | 1.14757e-07 | [1.0169389556616767e-09, 4.712030079677815e-07] | aligned |

## Stratified results

Each stratum contains the two preselected parity inputs. No stratum was selected after outcomes.

| Update | Source | Prefix | Aligned | Opposed | Other classifications |
|---|---|---|---:|---:|---|
| 0 | length16_eval | A | 0 | 0 | {'informative_unresolved_direction': 2} |
| 0 | length16_eval | B | 1 | 0 | {'informative_unresolved_direction': 1} |
| 0 | length16_eval | C | 2 | 0 | {} |
| 0 | length32_ood | A | 0 | 0 | {'informative_unresolved_direction': 1, 'unresolved_action_values': 1} |
| 0 | length32_ood | B | 2 | 0 | {} |
| 0 | length32_ood | C | 0 | 0 | {'informative_unresolved_direction': 2} |
| 0 | training | A | 1 | 0 | {'informative_unresolved_direction': 1} |
| 0 | training | B | 1 | 0 | {'informative_unresolved_direction': 1} |
| 0 | training | C | 2 | 0 | {} |
| 1040 | length16_eval | A | 0 | 0 | {'informative_unresolved_direction': 2} |
| 1040 | length16_eval | B | 1 | 0 | {'informative_unresolved_direction': 1} |
| 1040 | length16_eval | C | 2 | 0 | {} |
| 1040 | length32_ood | A | 0 | 0 | {'informative_unresolved_direction': 2} |
| 1040 | length32_ood | B | 2 | 0 | {} |
| 1040 | length32_ood | C | 1 | 0 | {'informative_unresolved_direction': 1} |
| 1040 | training | A | 0 | 0 | {'informative_unresolved_direction': 2} |
| 1040 | training | B | 1 | 0 | {'informative_unresolved_direction': 1} |
| 1040 | training | C | 1 | 0 | {'informative_unresolved_direction': 1} |

## Temporal credit, prefix sensitivity and task meaning

All 288 ST probes were connected; 0 were disconnected, 0 were exactly zero, and 16 had connected L2≤1e−8. HALT and the initially illegal REPEAT/RESTART slots were experimentally zero in every connected probe gradient.

| Update / prefix | Minimum ST norm | Median ST norm | Maximum ST norm |
|---|---:|---:|---:|
| 0/A | 1.06302e-06 | 0.000107829 | 0.000620987 |
| 0/B | 2.06969e-06 | 0.169096 | 1.51134 |
| 0/C | 2.36951e-07 | 0.000303323 | 0.416258 |
| 1040/A | 1.94459e-06 | 1.00317e-05 | 0.00020402 |
| 1040/B | 4.24734e-06 | 0.00422422 | 0.0505823 |
| 1040/C | 3.04054e-07 | 0.00235029 | 5.94578 |

Early and final-token gradients differ in magnitude, but connectivity alone cannot establish useful credit or explain optimizer failure. No early/late difference is labeled a truncation failure: the measured paths remain connected and the saved suffix graph has no additional token-state detach.

35/36 states have at least one resolved paired CE difference (468/1008 paired comparisons). 19/36 have identical action-conditional accuracy means; 19/36 have identical correctness across all eight actions on every paired tape. Thus measurable loss sensitivity can primarily reflect calibration, rather than a change in parity correctness. Such CE effects are not evidence of useful task discovery.

Probe entropy ranges 2.05894…2.07504; maximum action probability ranges 0.139074…0.161846. Saturation flags (max pi≥0.99): 0/36.

B-versus-C comparisons use different incoming histories. Their local values and gradients describe prefix-conditioned sensitivity; they are not a decomposition of historical optimizer effects. The complete paired comparisons and incoming-state identities are in interpretation.json.

At each identical frozen state, sum_a(pi1040−pi0)Q_owner was evaluated with downstream policy fixed to the indicated owner checkpoint. Conditional intervals classify {'higher_Q': 12, 'unresolved': 17, 'lower_Q': 7}. Negative means the local probability shift lowers that owner-policy Q, positive raises it. This omits changes in upstream state, downstream policy and shared parameters, so it cannot identify the complete historical update effect.

## Invariants and validation

Runtime assertions and independent offline trace checks verified 20160 completed suffix tokens, P=8, D=8+R∈[8,15], legal actions, selected-successor equality, zero-compute RESTART, all recorded continuations, state/cache versions and one seven-entry commit/token. There were 213118 continuation decisions, 2304 single CF interventions and 210814 independently reconstructed unforced decisions.

The original exhaustive training path makes nine physical speculative bank calls per decision (eight ADVANCE candidates plus the current-block REPEAT candidate). These total 1918062; they are separate from eight selected executions/token. GDN purity checks: {'4': 228724, '5': 232654}. Shared candidate inputs, committed GDN state/convolution buffers, memory/context and the Prelude anchor remained unchanged in flight; terminal updates were checked against the commit plan. No mechanical or nonfinite violation occurred.

Focused validation: **25 passed** (new artifact-independent diagnostic helper tests, paired-noise tests, routing tests and successor-mixing tests). Compilation wrote bytecode only to /tmp. No new full-regression claim is made. Offline checks reran no model: they verified all counts, states, hashes, logical noise/action reconstructions, categorical benchmark and sealed bootstrap calculations. git diff --check is clean; the pre-existing tracked diff is unchanged.

## Limits and conclusion

The proposed directional-failure mechanism is not supported at the tested states. Resolved comparisons favor alignment; the remaining comparisons are unresolved. This contradicts a blanket claim of consistently opposed local ST credit at these probes, while leaving a broader or rarer failure mechanism unresolved.

The experiment conditions on six previously selected inputs, two snapshots of one training seed, first-decision states and eight tapes/bank. It does not cover later within-token decisions or all reachable states. Unresolved values are not proof that useful gradients exist nowhere. Resolved alignment does not guarantee optimizer success; resolved opposition, if present, is local evidence rather than proof of the cause of historical collapse. No canonical action was used as an action-value correctness label.

This does not establish task-loss route discovery, useful topology/diversity, adaptive depth, learned HALT, general trainability or superiority. No fix or training run is proposed or launched. The diagnostic stops at measurement.

## Artifacts

- manifest.json: sealed definitions, identities, six inputs and randomness recipes.
- results.json and probe00.json…probe35.json: complete action-value and ST comparisons.
- decisions/tokens/runs/prefixes.jsonl.gz: routes, logits, masks, noise keys, state summaries, gradients and outcome records.
- states/probe00.pt…probe35.pt: full frozen incoming states and initialized controller features.
- prefix_noise_extension.pt, continuation_noise.pt, ST_probe_noise.pt, bootstrap_indices.pt: separately sealed randomness.
- verification.json and interpretation.json: independent offline checks and stratified interpretation.
- analyze_artifacts.py and write_report.py: offline processing only; no model execution.
