# Frozen mHC input/output capture

Status: **COMPLETED_ONE_FROZEN_FORWARD**. The historical CLRS training result remains **COMPLETED_NON_SUCCESS**. This diagnostic uses one retained training array and the final checkpoint only; it does not retest generalization or change the historical interpretation.

## Ranked mechanistic interpretation

1. **Attention mixers directly attenuate positional distinctions that are present at their inputs.** Across the four attention blocks, centered-energy output/input ratios are 0.153–0.734. Query ratios are 0.1601, 0.1528, 0.4568 and 0.2496. Candidate ratios are 0.7337, 0.5772, 0.6677 and 0.2308. Thus already-homogeneous core output alone is insufficient to explain every observed mixer output. There is direct evidence of further attenuation and directional concentration across these mixers. Whether the attenuated distinctions would help predecessor prediction is **not measured**.

2. **Later GDN/FFN cores already supply highly aligned, predominantly radial variation.** Before mHC, GDN radial fractions are 99.41–99.89%; FFN radial fractions are 99.19–99.99%. Dominant-right-axis centered-energy fractions are about 99.19–99.98%. These observations weaken any claim that the later mixers are the first point at which their inputs become homogeneous. They still attenuate the remaining angular differences.

3. **Angular concentration is not equivalent to loss of all absolute variation.** GDN mixers retain about 98.2–98.8% of centered energy. FFN block6 **increases** it by about 0.767%, while its already tiny angular component decreases further. FFN block7 retains about 97.4–97.9%. Outputs have ≥99.951% of centered energy in their dominant output axis. Large absolute radial variation survives these stages. No task-readability claim follows from its magnitude.

4. **The raw residual path preserves distinctions outside the mixer output.** All captured block outputs equal the actual FP32 sum of the raw residual input and mixer output bit-for-bit. Whole-block centered energy grows in most cases despite mixer compression. Candidate block5 decreases 7069.385386→5119.638945 through a negative residual/increment cross term; query block7 decreases 436.013655→433.197354. These are observed local cancellations, not evidence that a mixer alone erases the complete state.

5. **Historical responsibility and intervention efficacy remain unresolved.** The capture establishes actual forward attenuation and already-aligned late core outputs on this array. It cannot tell when training created them, whether omitted upstream projections caused them, whether compressed distinctions were task-useful, or whether changing a mixer would improve sorting. The normal persistent memory, GDN recurrence and caches remain active; homogeneous live representations do not prove all state channels lack task information.

## Exact identities and replay

- Checkpoint: `experiments/results/gatec_clrs_insertion_sort_6200_v1/training_results_v1/model_update1000_v1.pt`.
- SHA-256: `2c0e3d6ad3711d8f0ee209eb3e3b0f8db8964e6eeaf2be77b7bd56120f29e6c8`.
- Loaded parameter/buffer identity: `7ab9239da5b1a4957cde546bb95e4c2961800b337f8bf677f8202e662e94f1a0`.
- Input: `official/train/row0000`, exactly the locked localization row and first sealed training-pool row. Original input identity: `e147c8817d320036834f4318fe40485a7fa37e9fa243d2ef500b1a08017f1b7c`. Locked-input file SHA-256 and all source/adapter identities are recorded in `manifest_v1.json`; they were checked before execution.
- Canonical blocks/actions: `[0,1,2,3,4,5,6,7]`; 16 original-element tokens then 16 query tokens. Candidate/query index j denotes original input element j.
- Final scores, h0, all eight block outputs and all committed streams match existing final-checkpoint tensors **bit-for-bit**. CE is **2.77286672592163**, exactly the retained value. Approved numerical tolerance remains atol=1e-6, rtol=1e-5; no tolerance adjustment was needed.
- CUDA on NVIDIA GB10, PyTorch 2.10.0+cu130/CUDA13.0; FP32, TF32 disabled, eval mode, all parameters frozen, no-grad trajectory. Parameters and buffers are unchanged after execution.

## What the hooks captured

The existing approved adapter and observer call the original production fixed canonical step. No arithmetic, action, normalization, commit or cache implementation is replaced. Additional post-forward hooks return `None`; they copy the actual input and result to detached CPU tensors after each invocation.

- SWA blocks0/1: immediate returned value from `SlidingWindowAttention.forward`, **after** its internal `o_proj`; mHC precedes raw residual addition (`src/blocks.py:27-110,388-410`).
- GQA blocks2/3: immediate returned attention-core output, including its internal output projection; mHC precedes raw residual addition (`src/gqa.py`, `src/blocks.py:388-410`).
- GDN blocks4/5: the returned `y` from `GDNv3.forward`, after head normalization, gate and output projection; neither recurrent state nor conv buffer is the mixer input (`src/gdn_v3.py:158-197`, `src/blocks.py:412-438`).
- FFN blocks6/7: the immediate returned SwiGLU sublayer tensor **inside** FFNHeavyCore, after its sublayer pre-norm and before residual addition. This is not the output of the whole FFNHeavyCore (`src/blocks.py:113-150`).
- `src/mhc.py:75-83`: actual output is `(x @ P.T) * scale`. No intervening projection or residual addition occurs inside the observation boundary.
- Every active mixer has32 invocations of `[1,1,64]` FP32. The registered prelude mixer has0 invocations. Contextual terminal commit work follows its normal paths and adds no extra selected mixer calls.

## All blockwise energy and residual comparisons

Ec is the sum of squared deviations from the per-feature mean across the 16 corresponding positions. Radial fractions are ratios of sums of exact pairwise squared-distance components, not means of fractions.

|Block/type|Positions|Mixer input Ec|Mixer output Ec|Output/input|Input radial fraction|Output radial fraction|Whole-block Ec before → after|
|---|---|---:|---:|---:|---:|---:|---|
|0/SWA|candidate|109.327815|80.2100635|0.733665658|0.331394171|0.355352467|103.807629 → 208.245482|
|0/SWA|query|0.0226445059|0.00362567291|0.160112697|0.0961633823|0.999864456|6.58952946 → 6.58953556|
|1/SWA|candidate|54.1529203|31.2567032|0.577193308|0.612951357|0.999973165|208.245482 → 354.117653|
|1/SWA|query|0.157084705|0.0239965833|0.152762061|0.0942057764|0.999856341|6.58953556 → 6.62551244|
|2/GQA|candidate|72.6477624|48.5058918|0.667685972|0.407904827|0.556721446|354.117653 → 581.951487|
|2/GQA|query|0.879839825|0.401953031|0.456847962|0.199934739|0.999913158|6.62551244 → 7.27046005|
|3/GQA|candidate|18.6245271|4.29920738|0.230835787|0.0437351568|0.999286972|581.951487 → 646.887995|
|3/GQA|query|1.28402099|0.32044927|0.249567002|0.0195584573|0.999140634|7.27046005 → 8.48169654|
|4/GDN|candidate|4184.77686|4133.72601|0.987800819|0.997773446|0.999999893|646.887995 → 7069.38539|
|4/GDN|query|7.68055448|7.58655471|0.98776133|0.998856183|0.999999973|8.48169654 → 8.89048982|
|5/GDN|candidate|595.708551|585.502396|0.9828672|0.994110941|0.999999577|7069.38539 → 5119.63895|
|5/GDN|query|0.0400496371|0.0393227114|0.981849382|0.997014568|0.999999978|8.89048982 → 9.46584975|
|6/FFN|candidate|42902.8391|43231.6984|1.00766521|0.999899672|0.999999997|5119.63895 → 69635.5117|
|6/FFN|query|494.152744|497.941747|1.00766767|0.999934548|0.999999998|9.46584975 → 436.013655|
|7/FFN|candidate|0.144701094|0.141633846|0.978802874|0.996617176|0.999999846|69635.5117 → 69632.3714|
|7/FFN|query|0.0062622134|0.00609974075|0.974055076|0.991910658|0.999997542|436.013655 → 433.197354|

No aggregate reduction threshold was introduced after seeing outcomes. The tables report actual ratios and absolute energies. Positional variation is not equated with task usefulness.

## Direction, radial/angular geometry and singular spectra

|Block|Positions|Mean pair cosine input → output|Centered input energy along dominant input axis|Centered output energy along dominant output axis|Mean same-position input/output cosine|
|---|---|---|---:|---:|---:|
|0|candidate|0.757675333849 → 0.750000473152|0.735737857|0.999961223|0.748797414|
|0|query|0.999803579565 → 0.999999991589|0.160560272|0.999931713|0.74886177|
|1|candidate|0.902066018247 → 0.999985317382|0.577937448|0.99994425|0.496853631|
|1|query|0.998813373746 → 0.999999719431|0.152950374|0.999887722|0.334027463|
|2|candidate|0.54077812558 → 0.0834153611408|0.675448991|0.999964943|0.489781853|
|2|query|0.990036861369 → 0.99999757017|0.462161081|0.999968095|0.473625333|
|3|candidate|0.990748837154 → 0.999997519907|0.231986489|0.999570524|0.785878847|
|3|query|0.999339863345 → 0.999999755082|0.25079591|0.999510072|0.764661761|
|4|candidate|0.996173565382 → 0.999999756438|0.998484744|0.999999899|0.997425622|
|4|query|0.999999769641 → 0.999999999995|0.998444764|0.999999879|0.999089152|
|5|candidate|0.9995256791 → 0.999999978144|0.995102008|0.999999593|0.998427135|
|5|query|0.999999979601 → 1|0.99407357|0.999999847|0.998514582|
|6|candidate|0.999999874614 → 0.999999999997|0.999790629|0.999999991|0.999638883|
|6|query|0.999999998884 → 1|0.999792545|0.999999992|0.999637506|
|7|candidate|0.999999999904 → 1|0.996634295|0.999999844|0.999666559|
|7|query|0.99999999999 → 1|0.991883524|0.999997551|0.999666527|

Candidate outputs from blocks0 and2 include opposite signs along essentially the same axis: minimum pairwise cosines are approximately −0.999983 and −0.999978. Their lower mean cosine is therefore **not** evidence of a rich set of independent directions. Their centered energy is nevertheless ≥99.996% aligned with the leading output axis. The full pairwise records retain these sign differences.

For every stage, radial/angular squared distances satisfy `||hi−hj||²=(ri−rj)²+ri*rj*||ui−uj||²` under explicit floating-point roundoff checks. Absolute centered singular spectra, common energies, row norms, pairwise means/minima and same-position rotation cosines are recorded in `blockwise_metrics_v1.json`. `pairwise_geometry_v1.csv` contains all7680 block/group/stage/pair records. These are descriptive calculations, not inferential samples. `captured_tensors_v1.pt` retains actual inputs/outputs and surrounding residuals for independent reproduction.

The prior audit's source-derived effective operators are reused, not re-executed as modules. CPU algebra `X @ A.T` agrees with actual CUDA outputs under the original tolerances (maximum absolute difference5.3474e-5 at large coordinates, with relative tolerance satisfied). The whole-block outputs remain bitwise equal to the prior evidence. Tiny differences between `block_output−block_input` and actual mixer output are expected subtraction/addition rounding; the newly captured output is used directly instead of treating such differences as a new operation.

## Connection to the previous pointer-score audit

The prior audit showed that terminal normalization removes predominantly radial variation, with learned gains adding only modest attenuation. It also showed a tiny final query-dependent interaction (norm0.03593% of the common-preference matrix), locally CE-aligned but unable to change the all-candidate13 argmax. Those results are reused from the existing report; no additional pointer-model trajectory or score-space intervention was run here. The present evidence fills the missing mHC boundary, without proving that restoring a particular distinction would improve predecessor accuracy.

## Accounting and preservation

{
  "model_constructions": 1,
  "checkpoint_loads": 1,
  "rollouts": 1,
  "attempted_tokens": 32,
  "completed_tokens": 32,
  "failed_tokens": 0,
  "selected_bank_attempted": 256,
  "selected_bank_completed": 256,
  "commits": 32,
  "contextual_commit_entries": 224,
  "backwards": 0,
  "optimizer_objects": 0,
  "optimizer_updates": 0,
  "peak_CUDA_allocated": 30306304,
  "peak_CUDA_reserved": 54525952
}

There was one construction/load and exactly one32-token CUDA trajectory. All256 mixer captures were finite. The original observer verifies selected order, frozen in-flight contexts/anchors/memory reads, unchanged parent state, disjoint returned state, cache-version advances and one terminal commit per token. Per-token action/commit records are in `token_traces_v1.jsonl.gz`; state/metadata summaries are in `committed_state_summaries_v1.json`.

All2481 pre-existing protected files, source/adapter seals, checkpoint identities, local HEAD and original tracked diff passed the post-forward integrity check. Historical bytecode exceptions are preserved. The final integrity and remote publication receipts are separate artifacts; no previous result or Git history is rewritten.

## Limits and stopping boundary

This is one array and one checkpoint, not independent replication. No training history of these activations, other-array prevalence, state-information oracle, causal intervention efficacy or historical origin is established. Intermediates inside cores before their output projections were not captured. No inverse/pseudoinverse, backward, optimizer, new input, warm-up, retry or additional model call was used.

**The authorized diagnostic is complete. No architecture or training change is recommended from this capture alone, and no follow-up execution is launched.**
