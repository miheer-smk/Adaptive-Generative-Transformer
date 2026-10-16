# Residual geometry and target-conditioned score audit

Completed offline on `official/train/row0000`,16 input-element and16 query positions, initial/final identities from the prior localization. The historical training result remains **COMPLETED_NON_SUCCESS**.

**Execution boundary:** zero model construction, imports of AGT implementation modules, forward/backward calls, tests, profiling, optimizer operations or GPU execution. Tensor/checkpoint files are read with CPU placement. No model or scientific artifact is modified.

## Ranked diagnosis

1. **Strongest direct evidence: growth of a common component, predominantly radial surviving variation, followed by ordinary normalization.** Final query prelude centered energy is6.58953 and rises to433.19735 before commit; candidate energy rises from103.80763 to69632.37139. Thus the whole bank does not generally erase absolute centered energy. Final pre-commit vectors are almost parallel, with98.46497%/99.80996% of query/candidate pairwise squared-distance variation radial. This is a measured representation geometry, not proof that those magnitudes encode usable sorting information.

2. **Strongest immediate score explanation: query-independent candidate preference dominates a tiny surviving interaction.** Both models prefer candidate13 for every query on this array. Initial interaction is locally CE-misleading; final interaction is weakly CE-aligned but insufficient to alter argmax or target ranks. Near-uniform CE must not be interpreted as literal absence of task-aligned interaction.

3. **mHC susceptibility is pronounced, but causal attribution is unavailable.** Effective operators favor a uniform input direction and residual increments align with dominant output directions. Initial operators are already analytically rank1, so singularity alone does not explain the difference between checkpoints. Missing pre-mHC core outputs prevent separating already-aligned core computation from mixer-induced compression.

4. **Active cancellation exists locally, but is not a general explanation.** Final candidate block5 decreases centered energy7069.38539→5119.63895 (27.58% reduction). Query block7 decreases436.01366→433.19735 (0.65%). Other negative cross terms coexist with net centered growth; they cannot be called erasure of all positional signal.

5. **Terminal gains and pointer projections are weak explanations for the scale of the contraction.** Final coordinate gains are0.924289–0.998151. Gain-only centered-energy factors are0.929692 for queries and0.933901 for candidates. Pointer projections retain30.12%/51.13% of absolute centered query/candidate energy and increase relative pairwise variation by1.355/1.987. They scale distinctions but do not introduce the much larger upstream near-collinearity.

## Residual-energy decomposition: all observed blocks

Block0 X is retained h0; each subsequent X is the preceding captured block output. This is source-backed input recovery, not inversion or invented activation. Delta=Y−X. Rows are centered across positions per feature. Both coordinate and Frobenius energy identities are verified in float64 with explicit roundoff bounds.

|Model/positions|Block|E(Xc)|E(Delta c)|2<Xc,Delta c>|E(Yc)|Common energy change|
|---|---:|---:|---:|---:|---:|---:|
|initial/candidate|0|101.86172|0.018998337|0.083421069|101.96414|1.3720212|
|initial/candidate|1|101.96414|0.074887066|0.0099155383|102.04894|10.19691|
|initial/candidate|2|102.04894|0.11177405|-0.17179502|101.98892|1.670232|
|initial/candidate|3|101.98892|0.25598411|-0.0048719098|102.24003|21.823528|
|initial/candidate|4|102.24003|0.00052587385|0.0074586495|102.24802|-0.0098411236|
|initial/candidate|5|102.24802|8.4938866e-05|-0.0098048463|102.2383|-0.078060405|
|initial/candidate|6|102.2383|0.0089597727|0.071236766|102.31849|2.1624513|
|initial/candidate|7|102.31849|0.037268777|-0.24470997|102.11105|-6.0233127|
|initial/query|0|29.983089|0.021580724|0.014632891|30.019303|-0.43551512|
|initial/query|1|30.019303|0.0036040032|0.0056620721|30.028569|5.8387606|
|initial/query|2|30.028569|0.022594244|-0.053693401|29.99747|5.0132429|
|initial/query|3|29.99747|0.062716895|-0.0082001462|30.051986|16.586743|
|initial/query|4|30.051986|0.00025459023|0.0078265516|30.060068|0.10437722|
|initial/query|5|30.060068|0.00025065638|0.0074123621|30.067731|0.088421857|
|initial/query|6|30.067731|0.046084019|0.093021969|30.206837|-0.6034332|
|initial/query|7|30.206837|0.013152755|0.099415309|30.319405|-2.7125228|
|final/candidate|0|103.80763|80.210063|24.22779|208.24548|51.958486|
|final/candidate|1|208.24548|31.256703|114.61547|354.11765|-53.128082|
|final/candidate|2|354.11765|48.505892|179.32794|581.95149|0.74438951|
|final/candidate|3|581.95149|4.2992073|60.637301|646.888|1192.1632|
|final/candidate|4|646.888|4133.726|2288.7714|7069.3854|50460.769|
|final/candidate|5|7069.3854|585.5024|-2535.2488|5119.6389|45190.848|
|final/candidate|6|5119.6389|43231.699|21284.174|69635.512|31300666|
|final/candidate|7|69635.512|0.14163697|-3.2819558|69632.371|31172714|
|final/query|0|6.5895295|0.0036256688|-0.0036195654|6.5895356|108.46309|
|final/query|1|6.5895356|0.023996586|0.011980291|6.6255124|-63.751978|
|final/query|2|6.6255124|0.40195301|0.2429946|7.2704601|-41.116598|
|final/query|3|7.2704601|0.3204493|0.89078719|8.4816965|1393.5111|
|final/query|4|8.4816965|7.5865554|-7.1777622|8.8904898|54973.587|
|final/query|5|8.8904898|0.039322657|0.53603728|9.4658498|43411.263|
|final/query|6|9.4658498|497.94184|-71.394036|436.01366|27632912|
|final/query|7|436.01366|0.0060992709|-2.8224002|433.19735|29612787|

Common energy means n||mean feature vector||². Negative cross terms measure cancellation of the existing centered residual by the increment; net centered reduction is reported separately. All pairwise distances/cosines and absolute centered singular spectra are in `residual_energy_v1.json`. Added variance is not automatically task-useful; block-level target utility remains unresolved.

## Radial, angular and terminal-gain evidence

|Model/positions|Pre-commit radial fraction|Pre-commit centered energy|Pre-gain centered energy|Committed centered energy|Gain-only energy factor|
|---|---:|---:|---:|---:|---:|
|initial/candidate|0.0002101117205|102.1110509|99.29301454|99.29301499|1|
|initial/query|0.0002966654732|30.31940469|29.57555227|29.57555314|1|
|final/candidate|0.9980996012|69632.37139|0.002023802006|0.001890033523|0.933900794|
|final/query|0.9846497384|433.1973544|0.0001187756588|0.0001104254034|0.929691995|

Pairwise radial fractions use the ratio of summed components of squared distances, not averages of ratios. Every pair satisfies ||hi−hj||²=(ri−rj)²+ri rj||ui−uj||² within numeric precision. No zero vectors occur. Pre-gain normalization is reconstructed from stored H using the actual RMSNorm epsilon; gains come directly from the final checkpoint or deterministic initial ones. Both double and literal CPU-FP32 reconstructions agree with stored committed tensors under the prior atol1e-6/rtol1e-5 standard.

Normalization removes substantial radial magnitude variation and rescales the remaining tiny angles. The directional near-collinearity is already present before normalization: mean pre-commit pairwise cosine is0.9999998763 for final queries and0.9999978919 for candidates. Learned gains add only modest attenuation. Task-adjoint-weighted gain RMS is0.969964/0.974512 for query/candidate directions, so the recorded gains do not selectively annihilate those directions.

Final query radii correlate0.998787 with element position and0.262329 with original keys; candidate radii correlate0.460646 with keys. These are16-point descriptive correlations only. Numeric predecessor indices are arbitrary identifiers; correlations are not semantic correctness or evidence of usable magnitude coding. Raw pairs, all stages and spectra are retained in `radial_pairs_v1.csv`, `radial_summary_v1.json`, `normalization_v1.json`.

## Effective mHC operators

The audited operator is A=D(scale)P, not P alone. P is reconstructed with the exact source clamp/exp and five normalization iterations in CPU FP32; spectra use float64 SciPy gesvd. All nine registered mixers are included; the prelude mixer is not executed on this fixed bank path and has no residual-increment comparison.

Initial logits=zeros and scales=ones are deterministic source constants, not a retained initial checkpoint. This makes all initial operators analytically rank1: uniform gain≈0.999999046 and orthogonal gain0. Other initial parameter matrices are unavailable. No model or RNG-based initializer is constructed.

|Final mixer|Scale range|sigma max|sigma min|Condition2|Uniform gain|Orthogonal RMS gain|Uniform/RMS orthogonal|
|---|---:|---:|---:|---:|---:|---:|
|body.blocks.blocks.0.mhc|0.956661–1.03722|0.99857139|5.8340774e-06|171161.8|0.99857133|0.0026062237|383.1487|
|body.blocks.blocks.1.mhc|0.956297–1.05487|0.99932814|3.3520306e-06|298126.2|0.999328|0.0026303155|379.927|
|body.blocks.blocks.2.mhc|0.950729–1.02136|0.9942194|1.7636642e-06|563723.8|0.99421935|0.002815938|353.0686|
|body.blocks.blocks.3.mhc|0.944491–1.04456|0.99730258|3.5412501e-06|281624.4|0.99730247|0.0032535326|306.5291|
|body.blocks.blocks.4.mhc|0.949293–1.06524|0.99463549|1.7103512e-06|581538.7|0.99463534|0.0025234857|394.1514|
|body.blocks.blocks.5.mhc|0.9227–1.04429|0.99383327|1.6050977e-07|6191731|0.99383311|0.003385589|293.5481|
|body.blocks.blocks.6.core.mhc|0.939819–1.04721|1.0039304|5.5855373e-06|179737.5|1.0039303|0.0022841212|439.5259|
|body.blocks.blocks.7.core.mhc|0.919616–1.03276|0.99102207|2.1316687e-06|464904.4|0.99102193|0.0023181778|427.5004|
|body.prelude_block.mhc|1–1|0.99999905|-0|singular|0.99999905|0|undefined (zero denominator)|

Uniform-direction gains are near1, so these values do not establish huge absolute amplification by A. They establish a roughly294–440-fold preference relative to the RMS gain of isotropic unit inputs in the orthogonal complement. Condition estimates range1.71e5–6.19e6 for the active trained mixers.

All final common increments align with their operator’s dominant output direction at absolute cosine≥0.999824; centered increment energy in that direction is≥0.99951. Such alignment is expected under strongly anisotropic operators, but cannot identify whether omitted core outputs were already aligned or lost distinctions in the mixer. No inverse/pseudoinverse or manufactured pre-mHC activation was used. Complete singular spectra, scales and alignments: `mhc_operators_v1.json`, `mhc_increment_alignment_v1.json`.

## Target-conditioned scores

For stored Q,K, compute Qc,Kc, b=Kc qbar/8 and I=Qc KcT/8. The decomposition Srow=1 bT+I and all480 target/incorrect-candidate margin decompositions agree with retained FP32 scores within the original comparison tolerance. Exact algebraic identities apply to double QKT/8; retained GPU-FP32 score discrepancies up to≈1.01e-7 are explicitly recorded. Centering leaves ranks and target margins unchanged.

|Quantity|Initial|Final|
|---|---:|---:|
|Common preference matrix norm|1.47625727801|0.0228211766956|
|Interaction norm|0.194799234457|8.19995904898e-06|
|Interaction/common norm|0.131954800399|0.000359313595366|
|Positive target interaction margins /240|129|153|
|Negative target interaction margins /240|111|87|
|CE effect of adding the existing interaction|0.000821939173648|-1.43720483514e-07|
|<GS,I>|0.000886785262573|-1.43720355574e-07|

The common preference selects candidate13 (the input maximum and therefore never a target predecessor) for every query at both checkpoints. Initial interaction improves129 margins and worsens111, but weighted effects are misleading: CE increases0.000821939 when adding it, and <GS,I> is positive0.000886785. Counts alone do not establish aligned credit.

Final interaction improves153 margins and worsens87. Its negative <GS,I>=−1.43720356e−7 indicates that increasing the existing interaction infinitesimally would locally reduce CE at fixed scores. The saved-adjoint versus algebraic-adjoint uncertainty bound is≈7.01e−14, far smaller than this sign. This is a local score-space direction, not a shared-parameter optimizer prediction or evidence that training can realize that direction.

Its norm is only0.03593% of the common-preference matrix. It changes neither the common/actual105 positive pairwise target comparisons out of240 nor any argmax. Final mean target rank is9.4375 before and after centering; initial rank is9.5. The interaction-only rank averages6.4375 final versus7.9375 initial, but this algebraic component is not a deployed selector or a new model evaluation.

All queries, target ranks, margins and sign contributions are retained in `query_target_ranks_v1.csv`, `target_margins_v1.csv`, `target_score_decomposition_v1.json`. The retained exact original score adjoint was used; the CE derivative from saved probabilities/labels independently agrees. There is no substituted prediction-as-target.

## Evidence classification and next step

Direct observations: stored states, projections, scores, probabilities, original task adjoints, and final parameter values. Algebraic reconstructions: source-backed X and Delta, norm stages, effective operators, energy/radial identities, b/I and local score-space derivative. Hypotheses: common-direction dominance may prevent useful query-conditioned use of retained distinctions; anisotropic mHC may contribute. Unresolved: historical onset, optimizer causality, pre-mHC core geometry, task usability of radial information and broader input/seed prevalence.

**One recommended next step, requiring separate authorization:** one frozen final-checkpoint32-token forward on this same array, capturing actual core outputs immediately before each mHC and corresponding mixer outputs. No update or backward is needed. This would distinguish already-common core outputs from additional mixer compression without inverting any mixer. It would not establish the historical cause. No RMSNorm, residual, memory, mixer or schedule change is recommended by this audit.

## Provenance, numerical failures and preservation

`manifest_v1.json` hashes15 exact input/checkpoint/source artifacts. `integrity_initial_v1.json` fingerprints2,446 protected files. Original training checkpoint SHA256 remains `2c0e3d6ad3711d8f0ee209eb3e3b0f8db8964e6eeaf2be77b7bd56120f29e6c8`.

Two offline Torch LAPACK attempts stopped on a finite constant initial operator. Their code and logs are retained unchanged. Installed CPU SciPy gesvd completed the unmodified-matrix calculations; `numeric_method_addendum_v1.json` records the change. No packages were installed and no model execution was used to diagnose it. Initial null singular directions are nonunique and carry no causal interpretation.

Source boundaries:

- residual_inputs: `src/agt.py:233-279; src/blocks.py:137-150,376-438`.

- normalization: `src/terminal_commit.py:215-218; src/commit.py:28-47; src/norms.py:21-36`.

- mhc: `src/mhc.py:38-83 (clamp/exp, five row/column normalizations with eps1e-6, scale after matrix multiplication)`.

- pointer: `experiments/results/gatec_clrs_insertion_sort_6200_v1/clrs_adapter_v1.py:73-108`.

- capture: `experiments/results/gatec_clrs_pointer_localization_6200_v1/run_diagnostic_v1.py`.

- targets: `experiments/results/gatec_clrs_pointer_localization_6200_v1/locked_input_v1.json; official source probing.array(original-index order)`.

Numeric verification:
```json
{
  "success": true,
  "residual_decompositions": 32,
  "radial_pairs": 1920,
  "target_margin_rows": 480,
  "max_residual_energy_error": 7.275957614183426e-12,
  "max_residual_coordinate_error": 0.0,
  "score_max_errors": {
    "initial": 6.669843019904698e-08,
    "final": 1.011752148924721e-07
  },
  "normalization_max_errors": {
    "initial/candidate": 2.8979777200177637e-07,
    "initial/query": 1.9638048387449203e-07,
    "final/candidate": 1.6769446187225867e-07,
    "final/query": 1.7823242304260134e-07
  },
  "model_constructions": 0,
  "model_calls": 0,
  "backward_calls": 0,
  "optimizer_operations": 0,
  "tests": 0,
  "profiling": 0,
  "GPU_execution": 0,
  "numeric_precision": "CPU float64 audit; source-equivalent FP32 mHC/RMSNorm reconstructions identified separately",
  "numpy_version": "2.5.4",
  "scipy_version": "1.18.1",
  "SVD_driver": "SciPy LAPACK gesvd; no jitter, inverse or matrix alteration"
}
```

`evidence_limits_v1.json` lists unavailable evidence. All conclusions are conditional on one input and two model identities; these are not independent experimental replicates. No bootstrap, model call, test suite, backward, optimizer, profiling or GPU execution occurred.
