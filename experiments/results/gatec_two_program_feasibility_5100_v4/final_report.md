# V4 locked two-program feasibility: stopped at current screening

P0 archival and all96 P0→P0 current cases were accepted. The candidate P1 was rejected when a P1-generated prefix followed by a P0 boundary produced an incorrect current answer. No future continuation or joint bootstrap ran. No automatic retry is authorized or performed.

## Tests and engineering

Final focused suite:32 passed,0 failed,0 skipped, including genuine CPU-mask/CUDA-logits P0 execution. Substrate suite:30 passed,5 explicitly deselected (training=True and backward fixtures, to honor the no-backward restriction). Initial development fixture run:30 passed,2 failed; both failures were an incorrect unit expectation of the production broadcast mask shape, preserved in development_test_failure.json and the original logs. The corrected fixture was validated before sealing; no study had started.

No production code was edited. The observer-only functional correction is the local `.to(device=row['logits'].device)` on the mask operand; the original CPU mask is returned unchanged. New research extraction and barriers are detailed in research_amendment.diff, runner_v4.py, protocol_amendment.md and the annotated620-entry matrix.

All1979 completed study calls passed observed legality, selected-action/block correspondence, P=8 accounting, in-flight/parent/sibling purity, cache-version and terminal-only once-commit assertions. There were no mechanical call failures. P1 executed [0,1,2,3,4,5,5,7], with the repeat resolving to block5. All calls had D=P=8. These are mechanical statements, not task equivalence.

## Archival comparison and monitor

The complete eight-token canonical-control gate compared64 bank transitions per path and all8 token records per path, including required raw hidden/feature/terminal/logit arrays, retained persistent/context/commit tensor summaries, null topology and exact cache/window/version/token metadata. Missing fields and unmatched records fail closed. Original tolerance:atol1e-6,rtol1e-5; discrete exact. Final CE=0.0014546061865985394.

Historical full-tensor agreement concerns only retained full arrays. Historical memory/GDN/cache summaries remain summary evidence; they do not establish elementwise historical equality. Full returned-state reference/adapter comparisons are separately established within v4. Within-run storage ownership checks are not cross-run address comparisons.

Seed3401 isolated monitor:REPRODUCED,128/128 retained records. Reconstructed with the original full-model constructor order and load_model(3401), within preserved Python/CPU/CUDA RNG states; its independently owned frozen controller was never registered as a model router or action provider. No extra AGT rollouts were used.

## Current screening and first rejection

|History → boundary|Observed|Accepted|Unverified|
|---|---:|---:|---:|
|P0 → P0|96|96|0|
|P0 → P1|4|4|92|
|P1 → P0|4|3|92|
|P1 → P1|3|3|93|

Total current cases observed:107 of384;106 passed and1 failed. Remaining277 cases are unverified. These partial, ordered results are not full-panel estimates for P1.

First mandatory failure:acquisition/L16/row03, length16, seed5100. Bits:0000101000110010; bits SHA-256:8ee3bd216c58a9e17d3403ec3f5c182a5823371fc064608119cf8d770b230e68. The first15 tokens used P1. At human token16/internal15, a complete P0 program predicted0 for truth1, CE=6.418639659881592. Its full logits were[[3.758411407470703, -2.6585960388183594]]. P0 supplied-state reference agreed with the adapter's transitions, complete returned state and logits on this same P1-generated incoming state. Thus the failure is current-task/state-switch acceptance, not executor mismatch.

On the identical sequence under P0 history → P0 boundary, prediction=1 and CE=0.0022292304784059525, logits=[[-3.588261127471924, 2.5167391300201416]]. The canonical target was retained; the divergent state was never repaired or reset. This CE difference is descriptive; exact correctness caused the declared rejection. No intermediate-prefix-readout correctness requirement was invented.

## Execution boundary and budget

Archival:16 completed tokens. P0 history construction:1696. P1 history construction:60 (four length16 prefixes). Boundary cases:107. Supplied-state P0 reference checks:100. Total:1979 completed tokens,15832 selected bank executions,1979 commits; elapsed study time:57.049s. GDN checks:block4=1979,block5=2046; pure plan checks=1979. Deterministic candidate constructions=0. Future calls=0; future AGT rollouts=0; joint bootstrap evaluations=0. The40,848-token cap was not extended. Unit fixtures and historical work remain separate; v1's one completed reference token/eight calls/one commit is not pooled.

The failed current token completed mechanically, followed by its supplied-state reference check; rejection occurred at current acceptance before the remaining P1→P1 boundary for row03. No later prefix, surviving-case future, replacement sample or additional study attempt ran.

## Conclusions

Engineering:VERIFIED on executed calls. P0 archival equivalence:ACCEPTED under full retained coverage and within-v4 checks. P0 current correctness:ACCEPTED on all96 locked prefixes. P1 required current/switch contract:FAILED at the stated case; reject this fixed candidate under the protocol. Future suffix compatibility:NOT EXECUTED/UNVERIFIED. Original final joint acceptance:NOT EVALUABLE. The bounded research hypothesis is not established; a required current-stage condition failed. This result neither rejects every possible alternative program nor supports selector training or universal committed-state equivalence.

## Identities and preservation

Backbone SHA-256:53b56e7e90f7eb7a6570107b6dfe3a297caac48d637fe27e5f00fbe94c6f09c3. Retained input-manifest SHA-256:cbc80bd2e8b12627d03245db1926403af068479e090f28c7b7e154286162ef80. Encoded-input SHA-256:1c52bd823aa65bd6c4a574cfd5043e79f0dad4660b4517839a9bf8a1dcf780ac. Revision:75f138e6b909c5ca9884e9a7755de63489d3d6f2; tracked-diff SHA-256:45b3d093bd7fd7976019bf4572dc574784ed271b3294b86194f80c05826a8eac. Runtime:CUDA NVIDIA GB10, FP32 B1 D64,16slots,[0,0] budget/zero presence, zero compatibility, HALT/convergence disabled, frozen backbone and independently initialized unused registered controller. Torch/CUDA versions are recorded in study_started.json; historical capability/NumPy warnings are retained without environment changes.

All617 protected identities, all checkpoints, retained inputs, original tracked diff, v1–v3 artifacts and sealed v4 implementation remain unchanged. The historical bytecode exception is unchanged; no new exception. `git diff --check` clean. No training, backward pass, optimizer, selector, parameter update, production edit or Git commit occurred.

Complete evidence:implementation_seal.json;requirements_matrix_annotated.json;archival_comparisons.jsonl;monitor_diagnostic.json;token_traces.jsonl.gz;execution_ledger.json;current_cases.json;current_comparison_table.csv;paired_comparisons.json;current_screen_analysis.json;failure.json;integrity_after.json;final_verification.json. Earlier generated report.md is preserved; this final_report.md adds the concise numerical interpretation without rewriting it.
