# Counterfactual pointer readout: retained raw, committed and direction-only states

Status: **COMPLETED_OFFLINE_COUNTERFACTUAL_READOUT_ANALYSIS**. Historical CLRS training remains **COMPLETED_NON_SUCCESS**. This is CPU tensor algebra on one retained training array and one checkpoint, not a trained model, AGT rollout, benchmark evaluation or intervention on persistent-state evolution.

## Main diagnosis

**Restoring raw radii does not recover query-conditioned predecessor selection under the existing weights on this array.** The raw readout improves from0/16 to1/16 correct pointers and mean target rank9.4375→7.75, but predicts candidate14 for every query. The apparent gain is completely reproduced by its query-independent candidate preference. Its query-dependent interaction improves86/240 margins, worsens154/240, changes no target-pair sign or target rank, and increases CE. Thus this result matches the predeclared outcome **radial variation survives but remains dominated by a global preference and target-misaligned interaction**, rather than useful query-dependent recovery.

Raw CE is1284.6773849 nats with near-certain wrong predictions. The fixed global scalar reduces it to2.77109080045 while preserving every ranking. The mean-rank gain over A remains because the candidate preference changed, not because scaling created information. Direction-only C has baseline-identical predictions and target ranks; its slightly lower CE than A is a calibration effect. Its much lower CE than unscaled B does not demonstrate better target discrimination.

## Evidence identity and observation boundary

- Checkpoint SHA-256: `2c0e3d6ad3711d8f0ee209eb3e3b0f8db8964e6eeaf2be77b7bd56120f29e6c8`.
- Array: `official/train/row0000`, exactly the prior locked localization input and first retained sealed training row. Original input SHA-256: `e147c8817d320036834f4318fe40485a7fa37e9fa243d2ef500b1a08017f1b7c`.
- Official target indices: `[8, 11, 2, 12, 5, 14, 4, 3, 7, 10, 0, 15, 6, 1, 2, 9]`. Compact-JSON target SHA-256: `bbc1487fc6c04edb476b98354b3a76416bc6a01f948ebaa2950e703a569d8a31`. No prediction was used as a target.
- The mHC directory named in the request is absent. The actual retained directory is `experiments/results/gatec_clrs_mhc_capture_6200_v1/`. All35 files match their Git blob identities in the supplied capture commit `d9d5eb0061b860b796f64fbc559420caff863509`. No directory or historical artifact was renamed.
- Raw H is the retained last-block live output. `src/agt.py:278-299` assigns `h_terminal=h` and passes it to terminal commit; `src/terminal_commit.py:215-218` applies `final_norm(h_terminal)` before memory write; `src/commit.py:28-47` calls `state_norm`; `src/norms.py:30-36` computes `h*rsqrt(mean(h²)+1e-6)*gain`. Thus these are actual pre-RMSNorm live tensors, not inferred core activations.
- The localization raw block7 outputs and committed streams are bitwise equal to the latest capture. Stored candidate rows are the first16 committed streams; query rows are the next16. Candidate/query index j refers to original input element j. Source-backed alignment: `clrs_adapter_v1.py:73-108`, and the retained localization capture.
- All checkpoint/input/tensor/source/adapter identities are in `manifest_v1.json` and checked before analysis. The norm reconstruction agrees with retained committed states with max error1.78232423e-07.

## Fixed analytical definitions

A uses exact saved committed representations. B uses actual raw pre-RMSNorm states. C uses unit-L2 versions of B. B_global multiplies **every** raw query/candidate coordinate by the same positive scalar. All four use the unchanged checkpoint matrices `query.weight` and `key.weight`, without bias or fitting: `Q=Xq Wqᵀ`, `K=Xk Wkᵀ`, `S=QKᵀ/8`.

`alpha = pooled_RMS(A) / pooled_RMS(B) = 0.0039787276785280398`. Both groups and all32×64 coordinates are pooled equally. Raw RMS is242.04910697062; committed RMS is0.96304748146699; scaled raw RMS is0.96304748146699. No targets or predictions choose the scalar. Its square is1.5830273939885125e-05; `S_Bglobal=alpha² S_B` within max numerical error8.881784197e-16. All argmaxes and full candidate rankings are unchanged.

Definitions and code were sealed before new outcomes. A is reconstructed in CPU FP64, with a separate literal FP32 matrix comparison to saved projections/scores. Maximum baseline score error is1.148484301e-07 in FP64 and3.576278687e-07 in FP32, within the original atol1e-6/rtol1e-5 requirements. Baseline CE is2.77286672127148, versus retained2.77286672592163; predictions and all target ranks agree exactly.

**Scope caveats:** B and C omit the learned coordinate gains present in A. B versus C isolates radial removal under the same unchanged projection weights; A comparisons additionally involve those gains and RMS epsilon. All stored raw states were produced under the original normal committed-state trajectory. This analysis does **not** simulate bypassing normalization during future state carry or memory/cache updates.

## Readout metrics

|Condition|Accuracy / exact array|Mean target rank|Mean target margin|CE|Mean max probability|
|---|---|---:|---:|---:|---:|
|A|0.00% / 0%|9.4375|-0.002549230195|2.77286672127|0.0626421132727|
|B|6.25% / 0%|7.75|-1282.649397|1284.6773849|1|
|B_global|6.25% / 0%|7.75|-0.02030469132|2.77109080045|0.0638796775325|
|C|0.00% / 0%|9.4375|-4.210048159e-05|2.77259330104|0.0625023451491|

Exact-array accuracy is0 for every condition. Target margin means target score minus the strongest incorrect candidate. `all_target_margins_v1.csv` retains all960 condition/query/incorrect-candidate margins; distributions including quantiles, min/max and signs are in `condition_summaries_v1.json`. Raw scores range −65258.4050952 to −56770.021243, with mean within-query span7934.48269947. CE uses finite float64 log-sum-exp, without exponentiating raw logits directly. B_global score range is −1.03305843 to −0.898684988; C is near uniform in confidence.

|Query|Official target|A rank / prediction|B rank / prediction|B_global rank / prediction|C rank / prediction|
|---:|---:|---|---|---|---|
|0|8|8 / 13|9 / 14|9 / 14|8 / 13|
|1|11|4 / 13|13 / 14|13 / 14|4 / 13|
|2|2|16 / 13|2 / 14|2 / 14|16 / 13|
|3|12|11 / 13|6 / 14|6 / 14|11 / 13|
|4|5|14 / 13|3 / 14|3 / 14|14 / 13|
|5|14|15 / 13|1 / 14|1 / 14|15 / 13|
|6|4|13 / 13|4 / 14|4 / 14|13 / 13|
|7|3|10 / 13|7 / 14|7 / 14|10 / 13|
|8|7|9 / 13|8 / 14|8 / 14|9 / 13|
|9|10|7 / 13|10 / 14|10 / 14|7 / 13|
|10|0|2 / 13|16 / 14|16 / 14|2 / 13|
|11|15|5 / 13|12 / 14|12 / 14|5 / 13|
|12|6|12 / 13|5 / 14|5 / 14|12 / 13|
|13|1|3 / 13|15 / 14|15 / 14|3 / 13|
|14|2|16 / 13|2 / 14|2 / 14|16 / 13|
|15|9|6 / 13|11 / 14|11 / 14|6 / 13|

B and B_global improve ranks on9 queries and worsen them on7 relative to A. There is no uniform target-ranking improvement across queries. All query/candidate indices and ties are retained in `per_query_v1.csv`.

## Target-conditioned common/interaction decomposition

For each condition, `b=Kc mean(Q)/8`, `I=Qc Kcᵀ/8` and `Srow=1 bᵀ+I`. Every one of the240 target-versus-incorrect comparisons is decomposed into its common and interaction contributions. Common norm is the Frobenius norm of the replicated matrix `1 bᵀ`, not merely the vector b. No comparison treats the240 dependent pairs as independent observations.

|Condition|Common preference norm|Interaction norm|Interaction/common|Interaction improves / worsens margins|Common / combined positive margins|CE effect of interaction|
|---|---:|---:|---:|---|---|---:|
|A|0.02282115057|8.19996619e-06|0.0003593143196|153 / 87|105 / 105|-1.437114382e-07|
|B|30094.95395|85.6040042|0.002844463705|86 / 154|132 / 132|1.436590879|
|B_global|0.4764113652|0.001355134837|0.002844463705|86 / 154|132 / 132|2.274494915e-05|
|C|0.0003781068594|1.360749328e-07|0.0003598848563|153 / 87|105 / 105|-2.385892373e-09|

Positive interaction margins improve the existing common-only target margin; negative ones worsen it. The raw interaction mean margin is−1.53236360429 (globally scaled−2.42577356315e-5), whereas A is+1.53292341056e-7. No interaction crosses a target-pair sign in **any** condition; every query's target rank equals its common-only rank.

The raw interaction/common norm ratio is0.00284446 (about0.2844%), versus0.00035931432 for A. More interaction magnitude does not imply better alignment. Raw CE increases by1.43659087903 when adding its interaction to common-only scores; B_global increases by2.27449491521e-5. A decreases CE by1.43711438250e-7. Algebraic CE score adjoints are also retained; these are local score-space diagnostics, not gradients obtained by backward or predictions of realizable parameter updates.

**Why mean rank and1/16 accuracy can improve without useful query conditioning:** the sealed target vector contains index2 twice, index13 zero times and every other index once. For any single fixed candidate ordering, the mean target rank is `(136−rank(13)+rank(2))/16`. A/C rank index13 first and index2 last, giving9.4375. B/B_global rank index13 fourteenth and index2 second, giving7.75. These formulae exactly reproduce the observed means without a query-dependent decision. Raw's preferred candidate14 is the target of only query5, accounting for its entire1/16 accuracy. `query_independence_accounting_v1.json` records the fixed orders and arithmetic. This is descriptive accounting of retained outcomes, not a fitted baseline or an oracle.

## Radial and angular geometry

|Condition/stage|Centered energy|Radial fraction|Mean pair cosine|Mean pair distance|
|---|---:|---:|---:|---:|
|A/query_representations|0.0001104254034|1.979142737e-05|0.999999875979|0.003236697882|
|A/candidate_representations|0.001890033523|2.03553352e-05|0.999997877244|0.01178844842|
|A/projected_queries|3.326266223e-05|0.01735400331|0.999999776217|0.001776522751|
|A/projected_keys|0.0009663824616|0.1836015399|0.999993897727|0.008933009853|
|B/query_representations|433.1973544|0.9846497384|0.999999876275|6.388932778|
|B/candidate_representations|69632.37139|0.9980996012|0.999997891873|62.32833225|
|B/projected_queries|73.36721399|0.9734402679|0.999999773895|2.630014891|
|B/projected_keys|9409.164974|0.9941687451|0.999993818523|22.57098949|
|B_global/query_representations|0.006857632791|0.9846497384|0.999999876275|0.02541982368|
|B_global/candidate_representations|1.102299514|0.9980996012|0.999997891873|0.2479874607|
|B_global/projected_queries|0.001161423096|0.9734402679|0.999999773895|0.01046411304|
|B_global/projected_keys|0.1489496591|0.9941687451|0.999993818523|0.08980382064|
|C/query_representations|1.855869669e-06|1.992481237e-26|0.999999876275|0.0004195870657|
|C/candidate_representations|3.162190634e-05|3.508123881e-27|0.999997891873|0.001521698803|
|C/projected_queries|5.537209597e-07|0.01783456283|0.999999773895|0.0002292093029|
|C/projected_keys|1.613363881e-05|0.1810362238|0.999993818523|0.001152477086|

Raw query/candidate squared-distance variation is98.46497%/99.80996% radial. Global scaling preserves those fractions and all angles. Unit-vector C has effectively zero radial variation (nonzero residuals of order1e-26 arise from floating-point normalization). A's surviving pairwise variation is predominantly angular but very small in absolute energy. Normalization does discard large magnitude differences; these magnitudes nevertheless fail to yield useful query-conditioned predecessor discrimination under the existing readout here.

Full row norms, cosine/distance distributions and all1920 pairwise representation/projection records are retained. No inference is made from a difference of norms as if it were a tensor distance. `analytical_readout_tensors_v1.pt` stores the actual analytical Q/K, score, probability, common and interaction arrays, and is **not a model checkpoint**.

## Evidence classification, limitations and recommendation

Direct observations are the saved raw/committed tensors, official targets and checkpoint weights. Analytical reconstructions are the four fixed readouts, scalar, geometries, scores and decompositions. No omitted activation is manufactured. No model or AGT execution module is imported. No development/transfer inputs or targets are used for fitting or selection.

This does not prove that radial information is intrinsically useless, that another trained readout could never exploit it, or that RMSNorm was harmless throughout training. Neither an alternative persistent trajectory nor parameter optimization is evaluated. One array/checkpoint provides no generalization estimate, independent replication or historical causal chronology. A small raw rank gain through a different global preference is not the predeclared favorable outcome of task-aligned query-dependent recovery. Direction-only's lower CE is not an accuracy gain.

**One recommended next step:** decline a raw-state bypass on this evidence; require demonstrated query-dependent target discrimination as the prerequisite for any separately authorized upstream representation/readout study. Do not restore raw activations merely because their norms are larger. No next experiment or architecture change is initiated here.

## Preservation and publication

The analysis runs on CPU only with CUDA hidden and autograd disabled. Execution counters are all0: model constructions, forwards, backwards, optimizer objects/updates, tests, profiling and GPU operations. The checkpoint is deserialized as CPU parameter tensors only, never instantiated as a model.

`integrity_initial_v1.json` fingerprints2516 pre-existing files, including the completed negative training result and every prior capture/audit. Final integrity and publication receipts are separate files. Original HEAD, tracked diff, source, checkpoint and historical bytecode exceptions are preserved. All new artifacts remain confined to this new analysis directory.
