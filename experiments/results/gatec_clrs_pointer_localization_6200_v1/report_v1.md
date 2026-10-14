# CLRS pointer-failure localization, seed6200

Status: **COMPLETED_NO_UPDATE_DIAGNOSTIC**. Historical training remains **COMPLETED_NON_SUCCESS**.

One retained input: `official/train/row0000`, size16. Fresh initialization exactly matches the recorded identity; update1000 checkpoint loads strictly and matches its recorded final parameter identity. Both forwards match retained original scores and CE at atol1e-6/rtol1e-5.

|Measured stage: mean pairwise L2|Initial|Final|
|---|---:|---:|
|Committed query states|1.70421808|0.00323669788|
|Committed candidates|2.79656437|0.0117884484|
|Projected queries|0.902911542|0.00177651336|
|Projected keys|1.49492488|0.00893302596|

CE initial/final: 2.78796863556/2.77286672592. Both predict candidate13 for every query and get0/16 correct; candidate13 is this array’s maximum and never a predecessor target.

Mean logit span initial/final: 0.428040292/0.0044340305. Mean maximum probability: 0.0706362324/0.0626421133; uniform is0.0625. Mean target rank: 9.5/9.4375. Full raw/centered scores, probabilities, targets, ranks, margins and adjoints are retained.

The final model retains distinct encoded query inputs and prelude states. Selected live-block norms grow sharply, while relative query variation contracts. Query-token mean norm reaches about1893 before terminal normalization; after the ordinary commit the mean norm is7.704 and mean pairwise query distance is0.003237. Candidate states show the same substantial contraction.

This localizes the major measured loss of distinctions upstream of the task projections, along the selected-block/terminal-normalization path. It is a measured association with a dominant shared live-state component, not proof that a particular operator or commit causes the historical optimization failure. The projections do not map distinct inputs to exact equality and preserve/increase relative variation in this case.

All retained candidate/query/projected-query/projected-key task-loss adjoints are connected, finite and nonzero. Directions vary across elements/queries. The final projected-query mean adjoint norm is much smaller than initially; score-level CE gradients remain substantial and match the analytic CE derivative. This demonstrates attenuated connected credit rather than a severed graph. It does not establish useful historical gradient alignment.

Query-state adjoints include later recurrent computation. Projected-key adjoints combine all query losses. Do not interpret these as independent local intervention estimates.

## Scope and limits

One array and two checkpoints do not locate the historical onset, prove the gradient spikes caused collapse, establish operator causality or generalize this mechanism to other inputs. Initial scores are already weak and incorrect; training reduced distinction substantially here without changing the repeated argmax. No training, optimizer construction, update, warm-up, retry or third rollout occurred.

## Exact ledger
```json
{
  "attempted_tokens": 64,
  "completed_tokens": 64,
  "failed_tokens": 0,
  "selected_bank_attempted": 512,
  "selected_bank_completed": 512,
  "commits": 64,
  "backward_attempted": 2,
  "backward_completed": 2,
  "optimizer_objects": 0,
  "optimizer_updates": 0,
  "phase_counts": {
    "initial": {
      "attempted": 32,
      "completed": 32,
      "failed": 0,
      "selected_attempted": 256,
      "selected_completed": 256,
      "commits": 32
    },
    "final": {
      "attempted": 32,
      "completed": 32,
      "failed": 0,
      "selected_attempted": 256,
      "selected_completed": 256,
      "commits": 32
    }
  },
  "contextual_commit_entries": 448,
  "elapsed_seconds": 4.4101756159216166,
  "peak_CUDA_allocated": 163053056,
  "peak_CUDA_reserved": 171966464
}
```

## Integrity
```json
{
  "protected_files": 2416,
  "changed_or_missing": [],
  "HEAD_unchanged": true,
  "tracked_diff_unchanged": true,
  "historical_exception_preserved": true,
  "success": true
}
```

All parameter/buffer identities are unchanged across each backward. Full tensor evidence is in `initial_tensor_evidence_v1.pt` and `final_tensor_evidence_v1.pt`; descriptive measurements are in `analysis_v1.json`, per-query records in the model result files, and metadata/state summaries in model-specific committed-state files. Publication receipts are added separately.
