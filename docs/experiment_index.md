# Experiment and evidence index

Audited **2026-10-25**. This index preserves recorded classifications; editorial summaries are explicitly separate. The latest completed milestone is the feature-complement pilot. No executed successor or newly authorized next experiment was found.

The remote default branch is an older history snapshot. The evidence links below are commit-pinned. A publication commit identifies accessible evidence, not necessarily the original execution HEAD: several early runs used an uncommitted, fingerprinted source diff. See [repository state](maintenance/repository_state_20261025.json), [machine-readable inventory](evidence_inventory.json), [reproduction boundaries](reproducibility.md), and [known discrepancies](limitations.md#evidence-conflicts-and-unavailable-material).

## Milestone map

|Milestone|Scientific question|Recorded status or report verdict|Evidence|Scope limit|
|---|---|---|---|---|
|[Gate-A persistent GDN state](#gate-a)|Does committed GDN recurrence influence parity?|Report: causal state involvement; transfer not established|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/GATE_A_REPORT.md)|One available successful backbone; representation probes are not causal proof; no unique-state explanation.|
|[Stage-1 canonical parity reference](#stage1)|Can the fixed bank learn parity?|Reported working fixed-route reference|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/GATE_A_REPORT.md)|Not a language-model result, routing discovery, or multi-backbone replication.|
|[Early Stage-3 validation](#stage3-historical)|Can controller-only task loss discover useful routing?|Historical mixed validation; discovery not established|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/STAGE3_VALIDATION_RECORD.md)|Earlier eight-decision protocols are not retroactively eight-execution experiments.|
|[Execution accounting correction](#accounting)|Does P count actual selected block executions?|VERIFIED accounting/equivalence; regression not green|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_execution_accounting_v1/gatec_canonical_replay.md)|One canonical fixture; historical diagnostic assumptions and Stage4 failures remain recorded.|
|[Learned-policy closed-loop replay](#replay)|Does the imitated controller recover the canonical route?|A: canonical route recovered and parity preserved|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_learned_policy_replay_seed3305_v1/replay.md)|One train control; not discovery or diversity.|
|[Locked validation and OOD replay](#replay-panels)|Does canonical replay hold on locked panels?|Report: canonical recovery on locked panels|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_locked_policy_validation_ood_seed3305_v1/ood_test.md)|Panel overlap/exposure caveats remain; no new route discovered.|
|[Imitation seed replication](#imitation)|Can fresh controllers reproduce the teacher route?|strict_replication_success = true|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_imitation_seed_replication_4401_4402_v1/report.md)|Two controller seeds share one backbone; supervision prescribes the route.|
|[Warm task-loss maintenance](#maintenance)|Does task loss preserve an already competent imitated policy?|strong_bounded_maintenance = true|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_paired_task_maintenance_4401_v1/report.md)|Preservation of an imitated route does not demonstrate useful diverse routing or discovery.|
|[Fresh task-loss routing](#fresh-routing)|Can a fresh controller discover reference-level parity computation?|strong_reference_level_discovery = false|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_fresh_task_discovery_4700_v1/report.md)|One initialization, small panels, historical nine-bytecode integrity exception; original failure preserved.|
|[Action Values](#action-values)|How do retained branch values relate to ST credit?|Completed bounded diagnostic (report)|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_st_action_value_4700_v1/report.md)|Diagnostic branches are not a deployable oracle or a replacement estimator.|
|[Temporal Credit](#temporal-credit)|Which temporal paths contribute to the existing backward?|Completed bounded diagnostic (report)|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_temporal_vjp_4700_update1040_v1/report.md)|Components were not used as replacement training gradients.|
|[Matched-prefix offline analysis](#matched-prefix)|Does generated prefix state hurt parity with root probabilities held fixed?|UNRESOLVED|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_matched_prefix_analysis_4700_v1/report.md)|Saved-branch conditional Monte Carlo; snapshots not independent replicates; no new model calls.|
|[Independent-noise gradient averaging](#gradient-average)|Does averaging independent ST gradients repair discovery?|intervention_helped = false; useful_policy_repair = false|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_paired_gradient_average_4900_v1/paired_report.md)|One controller initialization; four locked tapes; duplicate control rollouts are not independent observations.|
|[Complete-program source audit](#program-audit)|Can existing execution/state contracts support a finite program study?|Recommendation: separate specification, not training|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_complete_program_source_audit_v1/report.md)|Source feasibility is not empirical success; the full standalone blueprint is unavailable here.|
|[Two-program feasibility v1](#program-v1)|Can the fixed alternative satisfy current/future committed-state contracts?|STOPPED BEFORE P1|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_two_program_feasibility_5100_v1/report.md)|No P1 scientific outcome.|
|[Two-program feasibility v2](#program-v2)|Can the fixed alternative satisfy current/future committed-state contracts?|STOPPED_DURING_INSPECTION|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_two_program_feasibility_5100_v2/report.md)|No repair/model execution; not a scientific negative.|
|[Two-program feasibility v3](#program-v3)|Can the fixed alternative satisfy current/future committed-state contracts?|STOPPED_AT_SPECIFICATION_BARRIER|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_two_program_feasibility_5100_v3/report.md)|No model execution or implementation amendment.|
|[Two-program feasibility v4](#program-v4)|Can the fixed alternative satisfy current/future committed-state contracts?|STOPPED / CURRENT_CANDIDATE_REJECTED|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_two_program_feasibility_5100_v4/final_report.md)|Ordered partial cases only; zero future calls/joint bootstrap; P1-to-P1 row03 unobserved.|
|[v4 offline forensics](#program-forensics)|What do retained prefixes/boundaries establish about the rejected case?|Completed offline forensic analysis (report)|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_two_program_v4_forensics_5100_v1/report.md)|Summary differences are not tensor distances or causal attribution; missing cells remain unobserved.|
|[Receiver-anchored prerequisites](#receiver-prerequisites)|Can gradients traverse real committed-state handoffs with detached receiver parameters?|20 passed; 3 documented intentional skips|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_receiver_anchored_prerequisites_6000_v1/prerequisite_report_final_v1.md)|Earlier failures and one failed call remain; fixtures are not task-learning success.|
|[Receiver-anchored compatibility](#receiver)|Does cross-program continuation training add compatibility beyond matched native training?|COMPLETED_NON_SUCCESS|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_receiver_anchored_compatibility_6000_v1/concise_report_v1.md)|One initialization; parity saturation; reused panels; zero-width interval is not universal equivalence.|
|[Streaming keyed-register qualification](#keyed-register)|Can fixed programs acquire a nontrivial mutable-register benchmark?|engineering/integrity/resource stop|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/db0e62ca0fad464fd840ac2754eb759bf53bb44b/experiments/results/gatec_keyed_register_qualification_6100_v1/closure_report_v1.md)|Incomplete and unqualified; online training metrics are not held-out evidence; no nine-hour retry assumed.|
|[CLRS insertion-sort engineering](#clrs-engineering)|Can a serialized AGT pointer adapter faithfully execute the official target task?|ENGINEERING_PASSED|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/0998852e515bde46be6049e373a357217c8c115a/experiments/results/gatec_clrs_insertion_sort_6200_v1/engineering_report_v1.md)|512-token fixtures are not benchmark competence; not a reproduction of the official CLRS GNN.|
|[Original CLRS fixed-program training](#clrs-training)|Does the fresh model learn predecessor pointers at n16 and transfer to n64?|COMPLETED_NON_SUCCESS|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/386ea1be31abe05d1ccf31f59511c1fd1d470aed/experiments/results/gatec_clrs_insertion_sort_6200_v1/training_results_v1/final_report_v1.md)|One seed, near-uniform CE; no learned routing or full CLRS-suite claim.|
|[Pointer Localization](#pointer-localization)|Where do query/candidate distinctions and adjoints survive?|COMPLETED_NO_UPDATE_DIAGNOSTIC|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/f40f65d6c4008f407669085cd5655ccc52f3424c/experiments/results/gatec_clrs_pointer_localization_6200_v1/report_v1.md)|One fitting array, two checkpoints; not a history of gradient alignment.|
|[Residual Audit](#residual-audit)|How do residual energy, radial coding and target-conditioned scores relate?|COMPLETED_OFFLINE_AUDIT|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/d3cf866419c6fa93507e70008d9d8ea58fd51960/experiments/results/gatec_clrs_residual_score_audit_6200_v1/report_v1.md)|One array; omitted pre-mHC core activations were not reconstructed; numerical failures retained.|
|[Mhc Capture](#mhc-capture)|Do actual mixers attenuate distinctions already present at their inputs?|COMPLETED_MHC_INPUT_OUTPUT_CAPTURE|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/d9d5eb0061b860b796f64fbc559420caff863509/experiments/results/gatec_clrs_mhc_capture_6200_v1/report_v1.md)|One array/checkpoint; attenuation is not proof of task relevance or historical causation.|
|[Raw Readout](#raw-readout)|Do saved raw radii improve the existing readout?|COMPLETED_OFFLINE_COUNTERFACTUAL_READOUT_ANALYSIS|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1ed1d773264b3f5f53bd1ba169e819709d33cbb2/experiments/results/gatec_clrs_counterfactual_readout_6200_v1/report_v1.md)|Offline same-weight readouts do not simulate changed recurrent state evolution.|
|[Frozen fitting-representation cache](#frozen-cache)|Can committed states/targets be retained with verified alignment?|COMPLETED_VERIFIED_TRAINING_CACHE|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/377c222b7a274c7d41c39df67c4336c57bd57bee/experiments/results/gatec_clrs_frozen_readout_cache_6300_v1/report_v1.md)|Cache preparation only; fitting arrays were seen by original core training.|
|[Original frozen bilinear readout](#bilinear-original)|Can one bilinear readout expose held-out decodability?|INCONCLUSIVE_NONCONVERGED_OPTIMIZATION|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/94a7b19079dad3fd21afc801f62d65ed9101b613/experiments/results/gatec_clrs_frozen_readout_cache_6300_v1/decodability_results_v1/report_v1.md)|Nonconverged optimization; reused panels; not representation impossibility.|
|[Separately sealed certified refit](#bilinear-refit)|Does a certified solve of the same objective recover competence?|CERTIFIED_FIXED_OBJECTIVE_NON_SUCCESS|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/97f3b78a17d4060d2b0fef8b52c606596de6ca5e/experiments/results/gatec_clrs_bilinear_refit_6300_v1/report_v1.md)|One regularized bilinear objective, historically exposed held-out panels; not all-readout impossibility.|
|[Feature-complement causal pilot](#feature-complement)|Does the exact P+Q intervention improve query-conditioned task acquisition?|PARTIAL_EFFECT_TASK_NON_SUCCESS|[Report](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/ea183970feea1f49677f186e26065ce4e333f709/experiments/results/gatec_clrs_feature_complement_pilot_6400_v1/report_v1.md)|One paired seed; reused panels; one predicted candidate/array; increased variation is not task success.|

## Full provenance and reproduction records

An entry point is a source pointer, **not a claim that running it in place is safe or portable**. Most historical runners deliberately refuse an existing output directory and some assume DGX-specific paths. No experiment runner or model test was executed to prepare this index. Availability refers to the audited Git tree; LFS payload reachability is qualified separately.

<a id="gate-a"></a>
### Gate-A persistent GDN state

**Question:** Does committed GDN recurrence influence parity?

**Recorded status/verdict:** `Report: causal state involvement; transfer not established`. **Evidence type:** experiment.

**Observed:** Zero/shuffle interventions affect parity; donor-state patching did not establish parity transfer.

**Configuration/data:** Frozen D64 seed3305 parity; canonical route

**Limit:** One available successful backbone; representation probes are not causal proof; no unique-state explanation.

**Report:** [GATE_A_REPORT.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/GATE_A_REPORT.md).
**Protocol/configuration:** [docs/TASK_A_SPEC.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/docs/TASK_A_SPEC.md).
**Numerical records:** [experiments/gateA_a4_state_ablation.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/gateA_a4_state_ablation.json), [experiments/gateA_a6_state_patching.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/gateA_a6_state_patching.json).
**Source entry:** [experiments/gateA_a4_state_ablation.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/gateA_a4_state_ablation.py).
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

|Checkpoint|SHA-256|Availability|
|---|---|---|
|[experiments/runs/stage1_parity_seed3305_best.pt](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/runs/stage1_parity_seed3305_best.pt)|`53b56e7e90f7eb7a6570107b6dfe3a297caac48d637fe27e5f00fbe94c6f09c3`|Local + ordinary Git object; content identity verified|

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="stage1"></a>
### Stage-1 canonical parity reference

**Question:** Can the fixed bank learn parity?

**Recorded status/verdict:** `Reported working fixed-route reference`. **Evidence type:** experiment.

**Observed:** The seed3305 backbone supplies the frozen positive control used by later routing work.

**Configuration/data:** D64; 16 memory slots; final-token parity CE; fixed sweep

**Limit:** Not a language-model result, routing discovery, or multi-backbone replication.

**Report:** [GATE_A_REPORT.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/GATE_A_REPORT.md).
**Protocol/configuration:** [docs/TASK_A_SPEC.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/docs/TASK_A_SPEC.md).
**Numerical records:** [experiments/runs/stage1_taskA_tiny.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/runs/stage1_taskA_tiny.json).
**Source entry:** [experiments/train_stage1_taskA_len8.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/train_stage1_taskA_len8.py).
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

|Checkpoint|SHA-256|Availability|
|---|---|---|
|[experiments/runs/stage1_parity_seed3305_best.pt](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/runs/stage1_parity_seed3305_best.pt)|`53b56e7e90f7eb7a6570107b6dfe3a297caac48d637fe27e5f00fbe94c6f09c3`|Local + ordinary Git object; content identity verified|

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="stage3-historical"></a>
### Early Stage-3 validation

**Question:** Can controller-only task loss discover useful routing?

**Recorded status/verdict:** `Historical mixed validation; discovery not established`. **Evidence type:** experiment.

**Observed:** Strong supervised imitation preserved parity; fresh task-loss controllers did not recover reference performance.

**Configuration/data:** Frozen Stage1 backbone; several controller initializations

**Limit:** Earlier eight-decision protocols are not retroactively eight-execution experiments.

**Report:** [STAGE3_VALIDATION_RECORD.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/STAGE3_VALIDATION_RECORD.md).
**Protocol/configuration:** [GATE_B_STAGE3_SPEC.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/GATE_B_STAGE3_SPEC.md).
**Numerical records:** [experiments/runs/stage3_validation_manifest_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/runs/stage3_validation_manifest_v1.json).
**Source entry:** [experiments/run_stage3_taskA_warmstart.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/run_stage3_taskA_warmstart.py).
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="accounting"></a>
### Execution accounting correction

**Question:** Does P count actual selected block executions?

**Recorded status/verdict:** `VERIFIED accounting/equivalence; regression not green`. **Evidence type:** experiment.

**Observed:** Canonical paths agree; an alternating ADVANCE/RESTART trace has P=8 and D=15.

**Configuration/data:** Corrected Stage3 fixed P=8; no updates

**Limit:** One canonical fixture; historical diagnostic assumptions and Stage4 failures remain recorded.

**Report:** [experiments/results/gatec_execution_accounting_v1/gatec_canonical_replay.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_execution_accounting_v1/gatec_canonical_replay.md).
**Protocol/configuration:** [GATE_B_STAGE3_SPEC.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/GATE_B_STAGE3_SPEC.md).
**Numerical records:** report tables and linked historical artifacts; no additional aggregate JSON is asserted here.
**Source entry:** [experiments/audit_gatec_canonical_replay.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/audit_gatec_canonical_replay.py).
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="replay"></a>
### Learned-policy closed-loop replay

**Question:** Does the imitated controller recover the canonical route?

**Recorded status/verdict:** `A: canonical route recovered and parity preserved`. **Evidence type:** experiment.

**Observed:** First no-update replay recovered the canonical control on its retained fixture.

**Configuration/data:** Frozen seed3305 backbone; imitated controller; P=8

**Limit:** One train control; not discovery or diversity.

**Report:** [experiments/results/gatec_learned_policy_replay_seed3305_v1/replay.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_learned_policy_replay_seed3305_v1/replay.md).
**Protocol/configuration:** no separate locked protocol located for this record; use the report and linked parent specifications.
**Numerical records:** report tables and linked historical artifacts; no additional aggregate JSON is asserted here.
**Source entry:** [experiments/replay_gatec_learned_policy.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/replay_gatec_learned_policy.py).
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

|Checkpoint|SHA-256|Availability|
|---|---|---|
|[experiments/runs/stage3_imitation_replication_3305_v1.pt](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/runs/stage3_imitation_replication_3305_v1.pt)|`259e2b86786c6cd037e3ba0db71d91fd05cdb76753bdb462be414bd1b9400df3`|Local + ordinary Git object; content identity verified|

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="replay-panels"></a>
### Locked validation and OOD replay

**Question:** Does canonical replay hold on locked panels?

**Recorded status/verdict:** `Report: canonical recovery on locked panels`. **Evidence type:** experiment.

**Observed:** Reference-equivalent canonical routing is supported on retained validation/OOD inputs.

**Configuration/data:** D64; deterministic; zero budget features

**Limit:** Panel overlap/exposure caveats remain; no new route discovered.

**Report:** [experiments/results/gatec_locked_policy_validation_ood_seed3305_v1/ood_test.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_locked_policy_validation_ood_seed3305_v1/ood_test.md).
**Protocol/configuration:** [experiments/results/gatec_locked_policy_validation_ood_seed3305_v1/manifest.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_locked_policy_validation_ood_seed3305_v1/manifest.json).
**Numerical records:** report tables and linked historical artifacts; no additional aggregate JSON is asserted here.
**Source entry:** [experiments/evaluate_gatec_locked_policy.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/evaluate_gatec_locked_policy.py).
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="imitation"></a>
### Imitation seed replication

**Question:** Can fresh controllers reproduce the teacher route?

**Recorded status/verdict:** `strict_replication_success = true`. **Evidence type:** experiment.

**Observed:** Both fresh seeds4401/4402 recovered canonical routes and frozen-reference parity.

**Configuration/data:** 5000 supervised updates per fresh controller; final checkpoints

**Limit:** Two controller seeds share one backbone; supervision prescribes the route.

**Report:** [experiments/results/gatec_imitation_seed_replication_4401_4402_v1/report.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_imitation_seed_replication_4401_4402_v1/report.md).
**Protocol/configuration:** [experiments/results/gatec_imitation_seed_replication_4401_4402_v1/manifest.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_imitation_seed_replication_4401_4402_v1/manifest.json).
**Numerical records:** [experiments/results/gatec_imitation_seed_replication_4401_4402_v1/results.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_imitation_seed_replication_4401_4402_v1/results.json).
**Source entry:** [experiments/replicate_gatec_locked_imitation.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/replicate_gatec_locked_imitation.py).
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

|Checkpoint|SHA-256|Availability|
|---|---|---|
|[experiments/runs/stage3_imitation_replication_4401_locked_v1.pt](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/runs/stage3_imitation_replication_4401_locked_v1.pt)|`4e14f56d3762aac1442a8dcc6fd86eb950e0013986409ff31b48d514fd972fcd`|Local + ordinary Git object; content identity verified|
|[experiments/runs/stage3_imitation_replication_4402_locked_v1.pt](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/runs/stage3_imitation_replication_4402_locked_v1.pt)|`771dd6cb845e73eac7fcb5b6641ed200af76fb22e7f35921cd7792139a190766`|Local + ordinary Git object; content identity verified|

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="maintenance"></a>
### Warm task-loss maintenance

**Question:** Does task loss preserve an already competent imitated policy?

**Recorded status/verdict:** `strong_bounded_maintenance = true`. **Evidence type:** experiment.

**Observed:** Competence/reference equivalence was maintained under the bounded continuation.

**Configuration/data:** 104 task-CE updates from seed4401 imitation; matched zero-update arm

**Limit:** Preservation of an imitated route does not demonstrate useful diverse routing or discovery.

**Report:** [experiments/results/gatec_paired_task_maintenance_4401_v1/report.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_paired_task_maintenance_4401_v1/report.md).
**Protocol/configuration:** [experiments/results/gatec_paired_task_maintenance_4401_v1/manifest.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_paired_task_maintenance_4401_v1/manifest.json).
**Numerical records:** [experiments/results/gatec_paired_task_maintenance_4401_v1/results.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_paired_task_maintenance_4401_v1/results.json).
**Source entry:** [experiments/run_gatec_paired_task_maintenance.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/run_gatec_paired_task_maintenance.py).
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

|Checkpoint|SHA-256|Availability|
|---|---|---|
|[experiments/runs/stage3_task_maintenance_4401_104_locked_v1.pt](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/runs/stage3_task_maintenance_4401_104_locked_v1.pt)|`c2dcee934edbc60e5e3faf0167ad202f9e25bf76ef47029a723d3fdcc4e39504`|Local + ordinary Git object; content identity verified|

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="fresh-routing"></a>
### Fresh task-loss routing

**Question:** Can a fresh controller discover reference-level parity computation?

**Recorded status/verdict:** `strong_reference_level_discovery = false`. **Evidence type:** experiment.

**Observed:** Parameters/routes changed; reference-level discovery was not established.

**Configuration/data:** 1040 predetermined controller-only updates; four stochastic tapes

**Limit:** One initialization, small panels, historical nine-bytecode integrity exception; original failure preserved.

**Report:** [experiments/results/gatec_fresh_task_discovery_4700_v1/report.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_fresh_task_discovery_4700_v1/report.md).
**Protocol/configuration:** [experiments/results/gatec_fresh_task_discovery_4700_v1/manifest.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_fresh_task_discovery_4700_v1/manifest.json).
**Numerical records:** [experiments/results/gatec_fresh_task_discovery_4700_v1/results.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_fresh_task_discovery_4700_v1/results.json).
**Source entry:** [experiments/run_gatec_fresh_task_discovery.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/run_gatec_fresh_task_discovery.py).
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

|Checkpoint|SHA-256|Availability|
|---|---|---|
|[experiments/runs/stage3_fresh_task_4700_update1040_locked_v1.pt](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/runs/stage3_fresh_task_4700_update1040_locked_v1.pt)|`37537106f47e3a003e4b5d2ff98ded93a8f98e3a2b60ee7814e502d28ddaa6d4`|Local + ordinary Git object; content identity verified|

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="action-values"></a>
### Action Values

**Question:** How do retained branch values relate to ST credit?

**Recorded status/verdict:** `Completed bounded diagnostic (report)`. **Evidence type:** diagnostic.

**Observed:** Bounded counterfactual bank and full-ST gradients were retained for matched probes.

**Configuration/data:** Saved seed4700 snapshot1040; matched final-token states

**Limit:** Diagnostic branches are not a deployable oracle or a replacement estimator.

**Report:** [experiments/results/gatec_st_action_value_4700_v1/report.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_st_action_value_4700_v1/report.md).
**Protocol/configuration:** [experiments/results/gatec_st_action_value_4700_v1/manifest.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_st_action_value_4700_v1/manifest.json).
**Numerical records:** [experiments/results/gatec_st_action_value_4700_v1/results.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_st_action_value_4700_v1/results.json).
**Source entry:** [experiments/diagnose_gatec_st_action_values_4700.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/diagnose_gatec_st_action_values_4700.py).
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="temporal-credit"></a>
### Temporal Credit

**Question:** Which temporal paths contribute to the existing backward?

**Recorded status/verdict:** `Completed bounded diagnostic (report)`. **Evidence type:** diagnostic.

**Observed:** A bounded no-update temporal-gradient decomposition characterized the existing estimator.

**Configuration/data:** Saved seed4700 snapshot1040; matched final-token states

**Limit:** Components were not used as replacement training gradients.

**Report:** [experiments/results/gatec_temporal_vjp_4700_update1040_v1/report.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_temporal_vjp_4700_update1040_v1/report.md).
**Protocol/configuration:** [experiments/results/gatec_temporal_vjp_4700_update1040_v1/manifest.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_temporal_vjp_4700_update1040_v1/manifest.json).
**Numerical records:** [experiments/results/gatec_temporal_vjp_4700_update1040_v1/results.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_temporal_vjp_4700_update1040_v1/results.json).
**Source entry:** [experiments/diagnose_gatec_temporal_vjp_4700.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/diagnose_gatec_temporal_vjp_4700.py).
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="matched-prefix"></a>
### Matched-prefix offline analysis

**Question:** Does generated prefix state hurt parity with root probabilities held fixed?

**Recorded status/verdict:** `UNRESOLVED`. **Evidence type:** offline analysis.

**Observed:** Six snapshot1040 pairs give +1.65821 pp with approximate conditional interval including zero.

**Configuration/data:** 12 B/C pairs across snapshots0/1040; primary six final pairs

**Limit:** Saved-branch conditional Monte Carlo; snapshots not independent replicates; no new model calls.

**Report:** [experiments/results/gatec_matched_prefix_analysis_4700_v1/report.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_matched_prefix_analysis_4700_v1/report.md).
**Protocol/configuration:** [experiments/results/gatec_matched_prefix_analysis_4700_v1/manifest.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_matched_prefix_analysis_4700_v1/manifest.json).
**Numerical records:** [experiments/results/gatec_matched_prefix_analysis_4700_v1/results.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_matched_prefix_analysis_4700_v1/results.json).
**Source entry:** [experiments/analyze_gatec_matched_prefix_4700.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/analyze_gatec_matched_prefix_4700.py).
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="gradient-average"></a>
### Independent-noise gradient averaging

**Question:** Does averaging independent ST gradients repair discovery?

**Recorded status/verdict:** `intervention_helped = false; useful_policy_repair = false`. **Evidence type:** experiment.

**Observed:** Neither independent-noise treatment nor duplicate-noise control acquired a useful policy.

**Configuration/data:** 520 updates/arm; two rollouts/update; fresh original controller

**Limit:** One controller initialization; four locked tapes; duplicate control rollouts are not independent observations.

**Report:** [experiments/results/gatec_paired_gradient_average_4900_v1/paired_report.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_paired_gradient_average_4900_v1/paired_report.md).
**Protocol/configuration:** [experiments/results/gatec_paired_gradient_average_4900_v1/manifest.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_paired_gradient_average_4900_v1/manifest.json).
**Numerical records:** [experiments/results/gatec_paired_gradient_average_4900_v1/results.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_paired_gradient_average_4900_v1/results.json).
**Source entry:** [experiments/run_gatec_paired_gradient_average_4900.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/run_gatec_paired_gradient_average_4900.py).
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

|Checkpoint|SHA-256|Availability|
|---|---|---|
|[experiments/runs/stage3_gradient_average_4900_C_update0520_locked_v1.pt](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/runs/stage3_gradient_average_4900_C_update0520_locked_v1.pt)|`0b414c7bfa9fb5e12dd7fb81dcab21325dadf56cf131e051a76ed79f32e9e95d`|Local + ordinary Git object; content identity verified|
|[experiments/runs/stage3_gradient_average_4900_T_update0520_locked_v1.pt](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/runs/stage3_gradient_average_4900_T_update0520_locked_v1.pt)|`f2da58059806802eaf8e16ff958c95558b088c404a2e9cd7c79f866a40ab21ac`|Local + ordinary Git object; content identity verified|

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="program-audit"></a>
### Complete-program source audit

**Question:** Can existing execution/state contracts support a finite program study?

**Recorded status/verdict:** `Recommendation: separate specification, not training`. **Evidence type:** source audit.

**Observed:** Reusable executor, state cloning and continuation interfaces were identified.

**Configuration/data:** Read-only source inventory; no model calls

**Limit:** Source feasibility is not empirical success; the full standalone blueprint is unavailable here.

**Report:** [experiments/results/gatec_complete_program_source_audit_v1/report.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_complete_program_source_audit_v1/report.md).
**Protocol/configuration:** [experiments/results/gatec_complete_program_source_audit_v1/manifest.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_complete_program_source_audit_v1/manifest.json).
**Numerical records:** report tables and linked historical artifacts; no additional aggregate JSON is asserted here.
**Source entry:** inspect the report artifact list; no single portable reproduction entry is asserted.
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="program-v1"></a>
### Two-program feasibility v1

**Question:** Can the fixed alternative satisfy current/future committed-state contracts?

**Recorded status/verdict:** `STOPPED BEFORE P1`. **Evidence type:** feasibility / stopped attempt.

**Observed:** Observer CPU-mask/CUDA-logits mismatch stopped the archival gate.

**Configuration/data:** Frozen D64 parity; v4 P1 executes [0,1,2,3,4,5,5,7]

**Limit:** No P1 scientific outcome.

**Report:** [experiments/results/gatec_two_program_feasibility_5100_v1/report.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_two_program_feasibility_5100_v1/report.md).
**Protocol/configuration:** [experiments/results/gatec_two_program_feasibility_5100_v1/protocol_manifest.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_two_program_feasibility_5100_v1/protocol_manifest.json).
**Numerical records:** [experiments/results/gatec_two_program_feasibility_5100_v1/results.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_two_program_feasibility_5100_v1/results.json).
**Source entry:** inspect the report artifact list; no single portable reproduction entry is asserted.
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="program-v2"></a>
### Two-program feasibility v2

**Question:** Can the fixed alternative satisfy current/future committed-state contracts?

**Recorded status/verdict:** `STOPPED_DURING_INSPECTION`. **Evidence type:** feasibility / stopped attempt.

**Observed:** Inspection exposed sequential-gate and archival-comparison gaps.

**Configuration/data:** Frozen D64 parity; v4 P1 executes [0,1,2,3,4,5,5,7]

**Limit:** No repair/model execution; not a scientific negative.

**Report:** [experiments/results/gatec_two_program_feasibility_5100_v2/report.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_two_program_feasibility_5100_v2/report.md).
**Protocol/configuration:** [experiments/results/gatec_two_program_feasibility_5100_v2/manifest.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_two_program_feasibility_5100_v2/manifest.json).
**Numerical records:** report tables and linked historical artifacts; no additional aggregate JSON is asserted here.
**Source entry:** inspect the report artifact list; no single portable reproduction entry is asserted.
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="program-v3"></a>
### Two-program feasibility v3

**Question:** Can the fixed alternative satisfy current/future committed-state contracts?

**Recorded status/verdict:** `STOPPED_AT_SPECIFICATION_BARRIER`. **Evidence type:** feasibility / stopped attempt.

**Observed:** Requirements matrix identified unresolved gate scope and monitor targets.

**Configuration/data:** Frozen D64 parity; v4 P1 executes [0,1,2,3,4,5,5,7]

**Limit:** No model execution or implementation amendment.

**Report:** [experiments/results/gatec_two_program_feasibility_5100_v3/report.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_two_program_feasibility_5100_v3/report.md).
**Protocol/configuration:** [experiments/results/gatec_two_program_feasibility_5100_v3/manifest.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_two_program_feasibility_5100_v3/manifest.json).
**Numerical records:** report tables and linked historical artifacts; no additional aggregate JSON is asserted here.
**Source entry:** inspect the report artifact list; no single portable reproduction entry is asserted.
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="program-v4"></a>
### Two-program feasibility v4

**Question:** Can the fixed alternative satisfy current/future committed-state contracts?

**Recorded status/verdict:** `STOPPED / CURRENT_CANDIDATE_REJECTED`. **Evidence type:** feasibility / stopped attempt.

**Observed:** P0 archival/current gates passed; P1-history/P0-boundary row03 failed current correctness.

**Configuration/data:** Frozen D64 parity; v4 P1 executes [0,1,2,3,4,5,5,7]

**Limit:** Ordered partial cases only; zero future calls/joint bootstrap; P1-to-P1 row03 unobserved.

**Report:** [experiments/results/gatec_two_program_feasibility_5100_v4/final_report.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_two_program_feasibility_5100_v4/final_report.md).
**Protocol/configuration:** [experiments/results/gatec_two_program_feasibility_5100_v4/protocol_amendment.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_two_program_feasibility_5100_v4/protocol_amendment.md).
**Numerical records:** [experiments/results/gatec_two_program_feasibility_5100_v4/results.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_two_program_feasibility_5100_v4/results.json).
**Source entry:** [experiments/results/gatec_two_program_feasibility_5100_v4/runner_v4.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_two_program_feasibility_5100_v4/runner_v4.py).
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="program-forensics"></a>
### v4 offline forensics

**Question:** What do retained prefixes/boundaries establish about the rejected case?

**Recorded status/verdict:** `Completed offline forensic analysis (report)`. **Evidence type:** offline analysis.

**Observed:** Four executed P1-prefix cases were descriptively aligned; ledger and component coverage reconciled.

**Configuration/data:** Zero model calls; acquisition/L16/row03 plus three preceding controls

**Limit:** Summary differences are not tensor distances or causal attribution; missing cells remain unobserved.

**Report:** [experiments/results/gatec_two_program_v4_forensics_5100_v1/report.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_two_program_v4_forensics_5100_v1/report.md).
**Protocol/configuration:** no separate locked protocol located for this record; use the report and linked parent specifications.
**Numerical records:** [experiments/results/gatec_two_program_v4_forensics_5100_v1/analysis_summary.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_two_program_v4_forensics_5100_v1/analysis_summary.json).
**Source entry:** inspect the report artifact list; no single portable reproduction entry is asserted.
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="receiver-prerequisites"></a>
### Receiver-anchored prerequisites

**Question:** Can gradients traverse real committed-state handoffs with detached receiver parameters?

**Recorded status/verdict:** `20 passed; 3 documented intentional skips`. **Evidence type:** engineering.

**Observed:** CPU/CUDA fixtures established forward/state and writer-directed gradient paths under the retained scope.

**Configuration/data:** Bounded disposable fixtures; zero optimizer updates

**Limit:** Earlier failures and one failed call remain; fixtures are not task-learning success.

**Report:** [experiments/results/gatec_receiver_anchored_prerequisites_6000_v1/prerequisite_report_final_v1.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_receiver_anchored_prerequisites_6000_v1/prerequisite_report_final_v1.md).
**Protocol/configuration:** [experiments/results/gatec_receiver_anchored_prerequisites_6000_v1/protocol_clarifications_v1.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_receiver_anchored_prerequisites_6000_v1/protocol_clarifications_v1.md).
**Numerical records:** [experiments/results/gatec_receiver_anchored_prerequisites_6000_v1/final_results_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_receiver_anchored_prerequisites_6000_v1/final_results_v1.json).
**Source entry:** inspect the report artifact list; no single portable reproduction entry is asserted.
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="receiver"></a>
### Receiver-anchored compatibility

**Question:** Does cross-program continuation training add compatibility beyond matched native training?

**Recorded status/verdict:** `COMPLETED_NON_SUCCESS`. **Evidence type:** experiment.

**Observed:** Both arms saturated robust/switching panels; delta0 with interval[0,0]; all four operation-ablation drops0.

**Configuration/data:** 512 common updates then512/arm; new namespaced P1 repeats block3

**Limit:** One initialization; parity saturation; reused panels; zero-width interval is not universal equivalence.

**Report:** [experiments/results/gatec_receiver_anchored_compatibility_6000_v1/concise_report_v1.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_receiver_anchored_compatibility_6000_v1/concise_report_v1.md).
**Protocol/configuration:** [experiments/results/gatec_receiver_anchored_compatibility_6000_v1/protocol_original_copy.txt](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_receiver_anchored_compatibility_6000_v1/protocol_original_copy.txt).
**Numerical records:** [experiments/results/gatec_receiver_anchored_compatibility_6000_v1/final_results.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_receiver_anchored_compatibility_6000_v1/final_results.json), [experiments/results/gatec_receiver_anchored_compatibility_6000_v1/final_analysis.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_receiver_anchored_compatibility_6000_v1/final_analysis.json).
**Source entry:** [experiments/results/gatec_receiver_anchored_compatibility_6000_v1/study_runner_v1.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_receiver_anchored_compatibility_6000_v1/study_runner_v1.py).
**Evidence revision:** [`1dd215e24407`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1dd215e24407b8e497abd7e4c92de5702fc25821). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `archive/agt-v3-complete-dgx-snapshot-20261013T183904Z`.

|Checkpoint|SHA-256|Availability|
|---|---|---|
|[experiments/results/gatec_receiver_anchored_compatibility_6000_v1/checkpoints/T_update0512.pt](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_receiver_anchored_compatibility_6000_v1/checkpoints/T_update0512.pt)|`0d4a7e75496148c8da5b96d733990736271dcc4da8f602e801b71b50b966d9c7`|Local + matching Git LFS pointer; payload not re-downloaded in this audit|
|[experiments/results/gatec_receiver_anchored_compatibility_6000_v1/checkpoints/C_update0512.pt](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1dd215e24407b8e497abd7e4c92de5702fc25821/experiments/results/gatec_receiver_anchored_compatibility_6000_v1/checkpoints/C_update0512.pt)|`4fd7cf9e3636b5ec62e21e82884e473b86c48172d6c65bb69cbe43ed3bd17438`|Local + matching Git LFS pointer; payload not re-downloaded in this audit|

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="keyed-register"></a>
### Streaming keyed-register qualification

**Question:** Can fixed programs acquire a nontrivial mutable-register benchmark?

**Recorded status/verdict:** `engineering/integrity/resource stop`. **Evidence type:** experiment.

**Observed:** 78/128 updates completed before closure; final development/ablations did not execute.

**Configuration/data:** Single initialization; 100000 attempted-token /7200-second cap

**Limit:** Incomplete and unqualified; online training metrics are not held-out evidence; no nine-hour retry assumed.

**Report:** [experiments/results/gatec_keyed_register_qualification_6100_v1/closure_report_v1.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/db0e62ca0fad464fd840ac2754eb759bf53bb44b/experiments/results/gatec_keyed_register_qualification_6100_v1/closure_report_v1.md).
**Protocol/configuration:** [experiments/results/gatec_keyed_register_qualification_6100_v1/protocol_approved_verbatim.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/db0e62ca0fad464fd840ac2754eb759bf53bb44b/experiments/results/gatec_keyed_register_qualification_6100_v1/protocol_approved_verbatim.md).
**Numerical records:** [experiments/results/gatec_keyed_register_qualification_6100_v1/final_results.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/db0e62ca0fad464fd840ac2754eb759bf53bb44b/experiments/results/gatec_keyed_register_qualification_6100_v1/final_results.json).
**Source entry:** [experiments/results/gatec_keyed_register_qualification_6100_v1/pilot_runner_v1.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/db0e62ca0fad464fd840ac2754eb759bf53bb44b/experiments/results/gatec_keyed_register_qualification_6100_v1/pilot_runner_v1.py).
**Evidence revision:** [`db0e62ca0fad`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/db0e62ca0fad464fd840ac2754eb759bf53bb44b). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `research/keyed-register-qualification-6100-20261014T131545Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="clrs-engineering"></a>
### CLRS insertion-sort engineering

**Question:** Can a serialized AGT pointer adapter faithfully execute the official target task?

**Recorded status/verdict:** `ENGINEERING_PASSED`. **Evidence type:** engineering.

**Observed:** Official pinned sampler examples, alignment, gradients, canonical execution and CPU/CUDA fixtures were verified.

**Configuration/data:** Fresh seed6200 D64; four-feature encoder; dynamic pointer head

**Limit:** 512-token fixtures are not benchmark competence; not a reproduction of the official CLRS GNN.

**Report:** [experiments/results/gatec_clrs_insertion_sort_6200_v1/engineering_report_v1.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/0998852e515bde46be6049e373a357217c8c115a/experiments/results/gatec_clrs_insertion_sort_6200_v1/engineering_report_v1.md).
**Protocol/configuration:** [experiments/results/gatec_clrs_insertion_sort_6200_v1/scope_lock.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/0998852e515bde46be6049e373a357217c8c115a/experiments/results/gatec_clrs_insertion_sort_6200_v1/scope_lock.json).
**Numerical records:** [experiments/results/gatec_clrs_insertion_sort_6200_v1/final_results_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/0998852e515bde46be6049e373a357217c8c115a/experiments/results/gatec_clrs_insertion_sort_6200_v1/final_results_v1.json), [experiments/results/gatec_clrs_insertion_sort_6200_v1/official_data_manifest_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/0998852e515bde46be6049e373a357217c8c115a/experiments/results/gatec_clrs_insertion_sort_6200_v1/official_data_manifest_v1.json).
**Source entry:** [experiments/results/gatec_clrs_insertion_sort_6200_v1/clrs_adapter_v1.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/0998852e515bde46be6049e373a357217c8c115a/experiments/results/gatec_clrs_insertion_sort_6200_v1/clrs_adapter_v1.py).
**Evidence revision:** [`0998852e515b`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/0998852e515bde46be6049e373a357217c8c115a). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `research/clrs-insertion-sort-engineering-6200-20261014T145602Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="clrs-training"></a>
### Original CLRS fixed-program training

**Question:** Does the fresh model learn predecessor pointers at n16 and transfer to n64?

**Recorded status/verdict:** `COMPLETED_NON_SUCCESS`. **Evidence type:** experiment.

**Observed:** 1000 updates/evaluations completed; development3.125% and transfer1.171875% pointer accuracy; exact arrays0.

**Configuration/data:** Official size16 train1000; train subset128; dev32; n64 transfer32

**Limit:** One seed, near-uniform CE; no learned routing or full CLRS-suite claim.

**Report:** [experiments/results/gatec_clrs_insertion_sort_6200_v1/training_results_v1/final_report_v1.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/386ea1be31abe05d1ccf31f59511c1fd1d470aed/experiments/results/gatec_clrs_insertion_sort_6200_v1/training_results_v1/final_report_v1.md).
**Protocol/configuration:** [experiments/results/gatec_clrs_insertion_sort_6200_v1/training_results_v1/scope_lock_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/386ea1be31abe05d1ccf31f59511c1fd1d470aed/experiments/results/gatec_clrs_insertion_sort_6200_v1/training_results_v1/scope_lock_v1.json).
**Numerical records:** [experiments/results/gatec_clrs_insertion_sort_6200_v1/training_results_v1/final_results_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/386ea1be31abe05d1ccf31f59511c1fd1d470aed/experiments/results/gatec_clrs_insertion_sort_6200_v1/training_results_v1/final_results_v1.json), [experiments/results/gatec_clrs_insertion_sort_6200_v1/training_results_v1/analysis_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/386ea1be31abe05d1ccf31f59511c1fd1d470aed/experiments/results/gatec_clrs_insertion_sort_6200_v1/training_results_v1/analysis_v1.json).
**Source entry:** [experiments/results/gatec_clrs_insertion_sort_6200_v1/training_results_v1/train_once_v1.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/386ea1be31abe05d1ccf31f59511c1fd1d470aed/experiments/results/gatec_clrs_insertion_sort_6200_v1/training_results_v1/train_once_v1.py).
**Evidence revision:** [`386ea1be31ab`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/386ea1be31abe05d1ccf31f59511c1fd1d470aed). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `research/clrs-insertion-sort-training-6200-20261014T170026Z`.

|Checkpoint|SHA-256|Availability|
|---|---|---|
|[experiments/results/gatec_clrs_insertion_sort_6200_v1/training_results_v1/model_update1000_v1.pt](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/386ea1be31abe05d1ccf31f59511c1fd1d470aed/experiments/results/gatec_clrs_insertion_sort_6200_v1/training_results_v1/model_update1000_v1.pt)|`2c0e3d6ad3711d8f0ee209eb3e3b0f8db8964e6eeaf2be77b7bd56120f29e6c8`|Local + ordinary Git object; content identity verified|

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="pointer-localization"></a>
### Pointer Localization

**Question:** Where do query/candidate distinctions and adjoints survive?

**Recorded status/verdict:** `COMPLETED_NO_UPDATE_DIAGNOSTIC`. **Evidence type:** diagnostic / offline analysis.

**Observed:** Two retained-checkpoint passes localize representation/score behavior and connected gradients.

**Configuration/data:** Frozen update1000; official/train/row0000

**Limit:** One fitting array, two checkpoints; not a history of gradient alignment.

**Report:** [experiments/results/gatec_clrs_pointer_localization_6200_v1/report_v1.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/f40f65d6c4008f407669085cd5655ccc52f3424c/experiments/results/gatec_clrs_pointer_localization_6200_v1/report_v1.md).
**Protocol/configuration:** [experiments/results/gatec_clrs_pointer_localization_6200_v1/manifest_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/f40f65d6c4008f407669085cd5655ccc52f3424c/experiments/results/gatec_clrs_pointer_localization_6200_v1/manifest_v1.json).
**Numerical records:** [experiments/results/gatec_clrs_pointer_localization_6200_v1/final_results_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/f40f65d6c4008f407669085cd5655ccc52f3424c/experiments/results/gatec_clrs_pointer_localization_6200_v1/final_results_v1.json).
**Source entry:** [experiments/results/gatec_clrs_pointer_localization_6200_v1/analyze_diagnostic_v1.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/f40f65d6c4008f407669085cd5655ccc52f3424c/experiments/results/gatec_clrs_pointer_localization_6200_v1/analyze_diagnostic_v1.py).
**Evidence revision:** [`f40f65d6c400`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/f40f65d6c4008f407669085cd5655ccc52f3424c). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `research/clrs-pointer-localization-6200-20261014T181120Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="residual-audit"></a>
### Residual Audit

**Question:** How do residual energy, radial coding and target-conditioned scores relate?

**Recorded status/verdict:** `COMPLETED_OFFLINE_AUDIT`. **Evidence type:** diagnostic / offline analysis.

**Observed:** Common-component growth and weak/misaligned score interactions were quantified offline.

**Configuration/data:** Frozen update1000; official/train/row0000

**Limit:** One array; omitted pre-mHC core activations were not reconstructed; numerical failures retained.

**Report:** [experiments/results/gatec_clrs_residual_score_audit_6200_v1/report_v1.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/d3cf866419c6fa93507e70008d9d8ea58fd51960/experiments/results/gatec_clrs_residual_score_audit_6200_v1/report_v1.md).
**Protocol/configuration:** [experiments/results/gatec_clrs_residual_score_audit_6200_v1/manifest_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/d3cf866419c6fa93507e70008d9d8ea58fd51960/experiments/results/gatec_clrs_residual_score_audit_6200_v1/manifest_v1.json).
**Numerical records:** [experiments/results/gatec_clrs_residual_score_audit_6200_v1/final_results_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/d3cf866419c6fa93507e70008d9d8ea58fd51960/experiments/results/gatec_clrs_residual_score_audit_6200_v1/final_results_v1.json).
**Source entry:** [experiments/results/gatec_clrs_residual_score_audit_6200_v1/audit_numeric_v3.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/d3cf866419c6fa93507e70008d9d8ea58fd51960/experiments/results/gatec_clrs_residual_score_audit_6200_v1/audit_numeric_v3.py).
**Evidence revision:** [`d3cf866419c6`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/d3cf866419c6fa93507e70008d9d8ea58fd51960). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `research/clrs-residual-score-audit-6200-20261016T061547Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="mhc-capture"></a>
### Mhc Capture

**Question:** Do actual mixers attenuate distinctions already present at their inputs?

**Recorded status/verdict:** `COMPLETED_MHC_INPUT_OUTPUT_CAPTURE`. **Evidence type:** diagnostic / offline analysis.

**Observed:** A single frozen forward captures actual active-mixer inputs/outputs and surrounding residuals.

**Configuration/data:** Frozen update1000; official/train/row0000

**Limit:** One array/checkpoint; attenuation is not proof of task relevance or historical causation.

**Report:** [experiments/results/gatec_clrs_mhc_capture_6200_v1/report_v1.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/d9d5eb0061b860b796f64fbc559420caff863509/experiments/results/gatec_clrs_mhc_capture_6200_v1/report_v1.md).
**Protocol/configuration:** [experiments/results/gatec_clrs_mhc_capture_6200_v1/manifest_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/d9d5eb0061b860b796f64fbc559420caff863509/experiments/results/gatec_clrs_mhc_capture_6200_v1/manifest_v1.json).
**Numerical records:** [experiments/results/gatec_clrs_mhc_capture_6200_v1/final_results_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/d9d5eb0061b860b796f64fbc559420caff863509/experiments/results/gatec_clrs_mhc_capture_6200_v1/final_results_v1.json).
**Source entry:** [experiments/results/gatec_clrs_mhc_capture_6200_v1/analyze_capture_v1.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/d9d5eb0061b860b796f64fbc559420caff863509/experiments/results/gatec_clrs_mhc_capture_6200_v1/analyze_capture_v1.py).
**Evidence revision:** [`d9d5eb0061b8`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/d9d5eb0061b860b796f64fbc559420caff863509). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `research/clrs-mhc-input-output-capture-6200-20261016T065806Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="raw-readout"></a>
### Raw Readout

**Question:** Do saved raw radii improve the existing readout?

**Recorded status/verdict:** `COMPLETED_OFFLINE_COUNTERFACTUAL_READOUT_ANALYSIS`. **Evidence type:** diagnostic / offline analysis.

**Observed:** Raw-state rankings remained query-independent; restoring magnitude did not recover sorting.

**Configuration/data:** Frozen update1000; official/train/row0000

**Limit:** Offline same-weight readouts do not simulate changed recurrent state evolution.

**Report:** [experiments/results/gatec_clrs_counterfactual_readout_6200_v1/report_v1.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1ed1d773264b3f5f53bd1ba169e819709d33cbb2/experiments/results/gatec_clrs_counterfactual_readout_6200_v1/report_v1.md).
**Protocol/configuration:** [experiments/results/gatec_clrs_counterfactual_readout_6200_v1/manifest_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1ed1d773264b3f5f53bd1ba169e819709d33cbb2/experiments/results/gatec_clrs_counterfactual_readout_6200_v1/manifest_v1.json).
**Numerical records:** [experiments/results/gatec_clrs_counterfactual_readout_6200_v1/final_results_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1ed1d773264b3f5f53bd1ba169e819709d33cbb2/experiments/results/gatec_clrs_counterfactual_readout_6200_v1/final_results_v1.json).
**Source entry:** [experiments/results/gatec_clrs_counterfactual_readout_6200_v1/analyze_readouts_v1.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/1ed1d773264b3f5f53bd1ba169e819709d33cbb2/experiments/results/gatec_clrs_counterfactual_readout_6200_v1/analyze_readouts_v1.py).
**Evidence revision:** [`1ed1d773264b`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/1ed1d773264b3f5f53bd1ba169e819709d33cbb2). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `research/clrs-counterfactual-pointer-readout-6200-20261016T072029Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="frozen-cache"></a>
### Frozen fitting-representation cache

**Question:** Can committed states/targets be retained with verified alignment?

**Recorded status/verdict:** `COMPLETED_VERIFIED_TRAINING_CACHE`. **Evidence type:** extraction.

**Observed:** 128 n16 fitting arrays verified; row0000 reused and127 newly extracted.

**Configuration/data:** Frozen core; committed candidates/queries; zero backwards

**Limit:** Cache preparation only; fitting arrays were seen by original core training.

**Report:** [experiments/results/gatec_clrs_frozen_readout_cache_6300_v1/report_v1.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/377c222b7a274c7d41c39df67c4336c57bd57bee/experiments/results/gatec_clrs_frozen_readout_cache_6300_v1/report_v1.md).
**Protocol/configuration:** [experiments/results/gatec_clrs_frozen_readout_cache_6300_v1/manifest_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/377c222b7a274c7d41c39df67c4336c57bd57bee/experiments/results/gatec_clrs_frozen_readout_cache_6300_v1/manifest_v1.json).
**Numerical records:** [experiments/results/gatec_clrs_frozen_readout_cache_6300_v1/cache_manifest_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/377c222b7a274c7d41c39df67c4336c57bd57bee/experiments/results/gatec_clrs_frozen_readout_cache_6300_v1/cache_manifest_v1.json).
**Source entry:** inspect the report artifact list; no single portable reproduction entry is asserted.
**Evidence revision:** [`377c222b7a27`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/377c222b7a274c7d41c39df67c4336c57bd57bee). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `research/clrs-frozen-readout-cache-6300-20261016T083448Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="bilinear-original"></a>
### Original frozen bilinear readout

**Question:** Can one bilinear readout expose held-out decodability?

**Recorded status/verdict:** `INCONCLUSIVE_NONCONVERGED_OPTIMIZATION`. **Evidence type:** readout fitting.

**Observed:** 256 accepted iterations missed the certificate; held-out competence was not recovered.

**Configuration/data:** 128 cached fitting arrays; FP64 L-BFGS; FP32 primary readout

**Limit:** Nonconverged optimization; reused panels; not representation impossibility.

**Report:** [experiments/results/gatec_clrs_frozen_readout_cache_6300_v1/decodability_results_v1/report_v1.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/94a7b19079dad3fd21afc801f62d65ed9101b613/experiments/results/gatec_clrs_frozen_readout_cache_6300_v1/decodability_results_v1/report_v1.md).
**Protocol/configuration:** [experiments/results/gatec_clrs_frozen_readout_cache_6300_v1/decodability_results_v1/scope_lock_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/94a7b19079dad3fd21afc801f62d65ed9101b613/experiments/results/gatec_clrs_frozen_readout_cache_6300_v1/decodability_results_v1/scope_lock_v1.json).
**Numerical records:** [experiments/results/gatec_clrs_frozen_readout_cache_6300_v1/decodability_results_v1/final_results_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/94a7b19079dad3fd21afc801f62d65ed9101b613/experiments/results/gatec_clrs_frozen_readout_cache_6300_v1/decodability_results_v1/final_results_v1.json), [experiments/results/gatec_clrs_frozen_readout_cache_6300_v1/decodability_results_v1/metrics_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/94a7b19079dad3fd21afc801f62d65ed9101b613/experiments/results/gatec_clrs_frozen_readout_cache_6300_v1/decodability_results_v1/metrics_v1.json).
**Source entry:** [experiments/results/gatec_clrs_frozen_readout_cache_6300_v1/decodability_results_v1/fit_readout_v1.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/94a7b19079dad3fd21afc801f62d65ed9101b613/experiments/results/gatec_clrs_frozen_readout_cache_6300_v1/decodability_results_v1/fit_readout_v1.py).
**Evidence revision:** [`94a7b19079da`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/94a7b19079dad3fd21afc801f62d65ed9101b613). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `research/clrs-bilinear-decodability-6300-20261016T092110Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="bilinear-refit"></a>
### Separately sealed certified refit

**Question:** Does a certified solve of the same objective recover competence?

**Recorded status/verdict:** `CERTIFIED_FIXED_OBJECTIVE_NON_SUCCESS`. **Evidence type:** exploratory readout fitting.

**Observed:** First certificate at341 accepted updates; competence and own-common-component advantage were not established.

**Configuration/data:** Fresh zero B; same train-only whiteners; frozen core; no new AGT calls

**Limit:** One regularized bilinear objective, historically exposed held-out panels; not all-readout impossibility.

**Report:** [experiments/results/gatec_clrs_bilinear_refit_6300_v1/report_v1.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/97f3b78a17d4060d2b0fef8b52c606596de6ca5e/experiments/results/gatec_clrs_bilinear_refit_6300_v1/report_v1.md).
**Protocol/configuration:** [experiments/results/gatec_clrs_bilinear_refit_6300_v1/protocol_v1.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/97f3b78a17d4060d2b0fef8b52c606596de6ca5e/experiments/results/gatec_clrs_bilinear_refit_6300_v1/protocol_v1.md).
**Numerical records:** [experiments/results/gatec_clrs_bilinear_refit_6300_v1/final_results_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/97f3b78a17d4060d2b0fef8b52c606596de6ca5e/experiments/results/gatec_clrs_bilinear_refit_6300_v1/final_results_v1.json), [experiments/results/gatec_clrs_bilinear_refit_6300_v1/metrics_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/97f3b78a17d4060d2b0fef8b52c606596de6ca5e/experiments/results/gatec_clrs_bilinear_refit_6300_v1/metrics_v1.json).
**Source entry:** [experiments/results/gatec_clrs_bilinear_refit_6300_v1/fit_readout_v1.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/97f3b78a17d4060d2b0fef8b52c606596de6ca5e/experiments/results/gatec_clrs_bilinear_refit_6300_v1/fit_readout_v1.py).
**Evidence revision:** [`97f3b78a17d4`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/97f3b78a17d4060d2b0fef8b52c606596de6ca5e). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `research/clrs-bilinear-refit-6300-20261017T055819Z`.

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.

<a id="feature-complement"></a>
### Feature-complement causal pilot

**Question:** Does the exact P+Q intervention improve query-conditioned task acquisition?

**Recorded status/verdict:** `PARTIAL_EFFECT_TASK_NON_SUCCESS`. **Evidence type:** experiment.

**Observed:** Treatment gains7.8125 pp over control on dev but0 over its own common component; all competence gates fail.

**Configuration/data:** 1000 fresh seed6200 updates/arm; eight active mixers only;82432 tokens

**Limit:** One paired seed; reused panels; one predicted candidate/array; increased variation is not task success.

**Report:** [experiments/results/gatec_clrs_feature_complement_pilot_6400_v1/report_v1.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/ea183970feea1f49677f186e26065ce4e333f709/experiments/results/gatec_clrs_feature_complement_pilot_6400_v1/report_v1.md).
**Protocol/configuration:** [AGT_v3_feature_complement_causal_pilot_locked_protocol_6400_v1.md](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/ea183970feea1f49677f186e26065ce4e333f709/AGT_v3_feature_complement_causal_pilot_locked_protocol_6400_v1.md).
**Numerical records:** [experiments/results/gatec_clrs_feature_complement_pilot_6400_v1/final_results_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/ea183970feea1f49677f186e26065ce4e333f709/experiments/results/gatec_clrs_feature_complement_pilot_6400_v1/final_results_v1.json), [experiments/results/gatec_clrs_feature_complement_pilot_6400_v1/evaluation_summaries_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/ea183970feea1f49677f186e26065ce4e333f709/experiments/results/gatec_clrs_feature_complement_pilot_6400_v1/evaluation_summaries_v1.json), [experiments/results/gatec_clrs_feature_complement_pilot_6400_v1/mechanistic_summary_v1.json](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/ea183970feea1f49677f186e26065ce4e333f709/experiments/results/gatec_clrs_feature_complement_pilot_6400_v1/mechanistic_summary_v1.json).
**Source entry:** [experiments/results/gatec_clrs_feature_complement_pilot_6400_v1/run_paired_v1.py](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/ea183970feea1f49677f186e26065ce4e333f709/experiments/results/gatec_clrs_feature_complement_pilot_6400_v1/run_paired_v1.py).
**Evidence revision:** [`ea183970feea`](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/ea183970feea1f49677f186e26065ce4e333f709). Published evidence snapshot; not necessarily the execution HEAD.
**Known branch:** `research/clrs-feature-complement-6400-20261017T102044Z`.

|Checkpoint|SHA-256|Availability|
|---|---|---|
|[experiments/results/gatec_clrs_feature_complement_pilot_6400_v1/C_update1000_v1.pt](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/ea183970feea1f49677f186e26065ce4e333f709/experiments/results/gatec_clrs_feature_complement_pilot_6400_v1/C_update1000_v1.pt)|`f410dacdfa19ef06fa9f27fe3bf594b1f341e3571fb0da8a1303e9692be2610a`|Local + ordinary Git object; content identity verified|
|[experiments/results/gatec_clrs_feature_complement_pilot_6400_v1/T_update1000_v1.pt](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/ea183970feea1f49677f186e26065ce4e333f709/experiments/results/gatec_clrs_feature_complement_pilot_6400_v1/T_update1000_v1.pt)|`57458b259619eb00ff9bbbceb4a0978b22b0fdf668e7ae1c80166b35da2a7300`|Local + ordinary Git object; content identity verified|

Read reports without dependencies. Model entry points are archived, often hardware/path-specific and guarded against reusing sealed output directories; not rerun by this maintenance task.
