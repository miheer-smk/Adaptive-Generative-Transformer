"""Render descriptive report from retained capture and CPU algebra only."""
import json,csv
from pathlib import Path
OUT=Path(__file__).resolve().parent
def main():
    result=json.loads((OUT/'capture_results_v1.json').read_text());ledger=result['ledger']
    comparison=json.loads((OUT/'forward_comparisons_v1.json').read_text())
    records=json.loads((OUT/'blockwise_metrics_v1.json').read_text())
    rows=list(csv.DictReader((OUT/'blockwise_summary_v1.csv').open()))
    table=['|Block/type|Positions|Mixer input Ec|Mixer output Ec|Output/input|Input radial fraction|Output radial fraction|Whole-block Ec before → after|',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        f=lambda n:float(r[n])
        table.append(f"|{r['block']}/{r['species']}|{r['positions']}|{f('input_centered_energy'):.9g}|{f('output_centered_energy'):.9g}|{f('energy_ratio'):.9g}|{f('input_radial_fraction'):.9g}|{f('output_radial_fraction'):.9g}|{f('block_input_centered_energy'):.9g} → {f('block_output_centered_energy'):.9g}|")
    angle=['|Block|Positions|Mean pair cosine input → output|Centered input energy along dominant input axis|Centered output energy along dominant output axis|Mean same-position input/output cosine|',
        '|---|---|---|---:|---:|---:|']
    for r in records:
        s=r['stage_statistics'];v=r['same_position_input_output_cosines']
        angle.append(f"|{r['block']}|{r['positions']}|{s['mixer_input']['pairwise_cosine_mean']:.12g} → {s['mixer_output']['pairwise_cosine_mean']:.12g}|{r['input_alignment_to_dominant_right']['centered_energy_fraction']:.9g}|{r['output_alignment_to_dominant_left']['centered_energy_fraction']:.9g}|{sum(v)/len(v):.9g}|")
    text=f'''# Frozen mHC input/output capture

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
- Final scores, h0, all eight block outputs and all committed streams match existing final-checkpoint tensors **bit-for-bit**. CE is **{comparison['CE']:.15g}**, exactly the retained value. Approved numerical tolerance remains atol=1e-6, rtol=1e-5; no tolerance adjustment was needed.
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

{chr(10).join(table)}

No aggregate reduction threshold was introduced after seeing outcomes. The tables report actual ratios and absolute energies. Positional variation is not equated with task usefulness.

## Direction, radial/angular geometry and singular spectra

{chr(10).join(angle)}

Candidate outputs from blocks0 and2 include opposite signs along essentially the same axis: minimum pairwise cosines are approximately −0.999983 and −0.999978. Their lower mean cosine is therefore **not** evidence of a rich set of independent directions. Their centered energy is nevertheless ≥99.996% aligned with the leading output axis. The full pairwise records retain these sign differences.

For every stage, radial/angular squared distances satisfy `||hi−hj||²=(ri−rj)²+ri*rj*||ui−uj||²` under explicit floating-point roundoff checks. Absolute centered singular spectra, common energies, row norms, pairwise means/minima and same-position rotation cosines are recorded in `blockwise_metrics_v1.json`. `pairwise_geometry_v1.csv` contains all7680 block/group/stage/pair records. These are descriptive calculations, not inferential samples. `captured_tensors_v1.pt` retains actual inputs/outputs and surrounding residuals for independent reproduction.

The prior audit's source-derived effective operators are reused, not re-executed as modules. CPU algebra `X @ A.T` agrees with actual CUDA outputs under the original tolerances (maximum absolute difference5.3474e-5 at large coordinates, with relative tolerance satisfied). The whole-block outputs remain bitwise equal to the prior evidence. Tiny differences between `block_output−block_input` and actual mixer output are expected subtraction/addition rounding; the newly captured output is used directly instead of treating such differences as a new operation.

## Connection to the previous pointer-score audit

The prior audit showed that terminal normalization removes predominantly radial variation, with learned gains adding only modest attenuation. It also showed a tiny final query-dependent interaction (norm0.03593% of the common-preference matrix), locally CE-aligned but unable to change the all-candidate13 argmax. Those results are reused from the existing report; no additional pointer-model trajectory or score-space intervention was run here. The present evidence fills the missing mHC boundary, without proving that restoring a particular distinction would improve predecessor accuracy.

## Accounting and preservation

{json.dumps(dict(model_constructions=result['model_constructions_completed'],checkpoint_loads=result['checkpoint_loads'],rollouts=result['rollouts_completed'],attempted_tokens=ledger['attempted_tokens'],completed_tokens=ledger['completed_tokens'],failed_tokens=ledger['failed_tokens'],selected_bank_attempted=ledger['selected_bank_attempted'],selected_bank_completed=ledger['selected_bank_completed'],commits=ledger['commits'],contextual_commit_entries=ledger['contextual_commit_entries'],backwards=0,optimizer_objects=0,optimizer_updates=0,peak_CUDA_allocated=ledger['peak_CUDA_allocated'],peak_CUDA_reserved=ledger['peak_CUDA_reserved']),indent=2)}

There was one construction/load and exactly one32-token CUDA trajectory. All256 mixer captures were finite. The original observer verifies selected order, frozen in-flight contexts/anchors/memory reads, unchanged parent state, disjoint returned state, cache-version advances and one terminal commit per token. Per-token action/commit records are in `token_traces_v1.jsonl.gz`; state/metadata summaries are in `committed_state_summaries_v1.json`.

All2481 pre-existing protected files, source/adapter seals, checkpoint identities, local HEAD and original tracked diff passed the post-forward integrity check. Historical bytecode exceptions are preserved. The final integrity and remote publication receipts are separate artifacts; no previous result or Git history is rewritten.

## Limits and stopping boundary

This is one array and one checkpoint, not independent replication. No training history of these activations, other-array prevalence, state-information oracle, causal intervention efficacy or historical origin is established. Intermediates inside cores before their output projections were not captured. No inverse/pseudoinverse, backward, optimizer, new input, warm-up, retry or additional model call was used.

**The authorized diagnostic is complete. No architecture or training change is recommended from this capture alone, and no follow-up execution is launched.**
'''
    with (OUT/'report_v1.md').open('x') as f:f.write(text)
    print('Report written from retained measurements only.')

if __name__=='__main__':main()
