# AGT-v3 feature-complement causal pilot

**PARTIAL_EFFECT_TASK_NON_SUCCESS**. Status: COMPLETED_PAIRED_EXPLORATORY_PILOT. This is one paired exploratory intervention under the locked insertion-sort task/recipe, not learned routing or broad AGT superiority.

## Protocol, preservation and exact intervention

Protocol `AGT_v3_feature_complement_causal_pilot_locked_protocol_6400_v1.md` SHA256 `0456b5625384b9da130a2c219c0a92d3e1a1d6858901a76161a8828677ae2574`. Reviewed source commit `1ed1d773264b3f5f53bd1ba169e819709d33cbb2`; local source HEAD `75f138e6b909c5ca9884e9a7755de63489d3d6f2` with the preserved historical diff. All41 checked local source files match the reviewed remote content. Branch `research/clrs-feature-complement-6400-20261017T102044Z`. Production and historical scientific artifacts are unchanged; source-level diffs and seals document research-only code.

Control uses `(X @ P.T)*g`. Treatment uses exactly `(X @ (P+Q).T)*g`, with `Q=I64-ones(64,64)/64`, at six outer bank mixers0–5 and FFN-core mixers6/7. No prelude-mixer change, coefficient, renormalization, moved gain, parameter-capacity increase or residual/commit/normalization change. Active-site calls and the unexecuted prelude mixer are checked on every token.

Operator-only checks passed6 CPU cases and3 genuinely executed CUDA cases, with no full AGT construction/calls/backwards. Starting equality and independent ownership of the two actual arms are recorded in `starting_equivalence_v1.json` when that gate ran. Exact original initial parameter/buffer identity: `c28e83425e505734e1b6a39f4c6d5ecf414f1b1fb28a77e01ddc31bbfa92280e`. No failed-final-checkpoint initialization.

## Locked training, data and execution

Original sealed1000 size16 training arrays, order seed6201, fresh initialization seed6200. Original native mean pointer CE, AdamW LR1e-3 WD0 betas(.9,.999) eps1e-8, clip global norm1, FP32 TF32off, full BPTT, original eval-mode flags, frozen controller/vocabulary head. C then T on each identical row; CPU/CUDA RNG streams matched without intervention draws. Exactly1000 prescribed updates/arm at completion; no initial diagnostic, warm-up, hints, selector, whitening, extra loss/readout fit, checkpoint selection or retry.

Training pool/target/source/order hashes and original configuration seals are in `preflight_source_configuration_v1.json`. Development/size64 pools are reused and historically examined, not untouched confirmation. Training subset is the original first128 rows; development32 size16 and transfer32 size64 rows retain sealed order and targets. Native final heads are evaluated once per prescribed row. No task error filters a case.

## Task results

|Panel|Arm/condition|Predecessor accuracy|Exact-array accuracy|CE|Invalid chains|
|---|---|---:|---:|---:|---:|
|training_subset|C/native|3.3203125%|0%|2.772870812|100%|
|training_subset|C/query_independent|3.3203125%|0%|2.772870821|100%|
|training_subset|C/uniform|6.0546875%|0%|2.772588722|100%|
|training_subset|C/self_pointer|6.25%|0%|unavailable|100%|
|training_subset|C/original_order|5.17578125%|0%|unavailable|0%|
|training_subset|T/native|11.0351562%|0%|2.755798803|100%|
|training_subset|T/query_independent|11.0351562%|0%|2.755807276|100%|
|training_subset|T/uniform|6.0546875%|0%|2.772588722|100%|
|training_subset|T/self_pointer|6.25%|0%|unavailable|100%|
|training_subset|T/original_order|5.17578125%|0%|unavailable|0%|
|development|C/native|3.125%|0%|2.772864395|100%|
|development|C/query_independent|3.125%|0%|2.772864376|100%|
|development|C/uniform|6.640625%|0%|2.772588722|100%|
|development|C/self_pointer|6.25%|0%|unavailable|100%|
|development|C/original_order|7.6171875%|0%|unavailable|0%|
|development|T/native|10.9375%|0%|2.755964026|100%|
|development|T/query_independent|10.9375%|0%|2.7559704|100%|
|development|T/uniform|6.640625%|0%|2.772588722|100%|
|development|T/self_pointer|6.25%|0%|unavailable|100%|
|development|T/original_order|7.6171875%|0%|unavailable|0%|
|size_transfer|C/native|1.171875%|0%|4.158960282|100%|
|size_transfer|C/query_independent|1.171875%|0%|4.158960291|100%|
|size_transfer|C/uniform|1.51367188%|0%|4.158883083|100%|
|size_transfer|C/self_pointer|1.5625%|0%|unavailable|100%|
|size_transfer|C/original_order|1.3671875%|0%|unavailable|0%|
|size_transfer|T/native|2.34375%|0%|4.159142853|100%|
|size_transfer|T/query_independent|2.34375%|0%|4.159142598|100%|
|size_transfer|T/uniform|1.51367188%|0%|4.158883083|100%|
|size_transfer|T/self_pointer|1.5625%|0%|unavailable|100%|
|size_transfer|T/original_order|1.3671875%|0%|unavailable|0%|

Uniform probability metrics use established deterministic lowest-index ties, with analytic independent-random expectations separately recorded. Competition rank1 on tied scores is not discrimination. Self-pointer and original-order hard controls have no invented CE, score margin or target rank. Full scores, targets, predictions, ranks, ties and target margins are retained per array. Invalid chains are task errors, not executor failures.

## Prospective gates and paired uncertainty

{
  "C": {
    "development": {
      "predecessor_accuracy": 0.03125,
      "predecessor_threshold": 0.95,
      "predecessor_pass": false,
      "exact_array_accuracy": 0.0,
      "exact_threshold": 0.8,
      "exact_pass": false
    },
    "size_transfer": {
      "predecessor_accuracy": 0.01171875,
      "predecessor_threshold": 0.9,
      "predecessor_pass": false,
      "exact_array_accuracy": 0.0,
      "exact_threshold": 0.5,
      "exact_pass": false
    },
    "all_competence_pass": false
  },
  "T": {
    "development": {
      "predecessor_accuracy": 0.109375,
      "predecessor_threshold": 0.95,
      "predecessor_pass": false,
      "exact_array_accuracy": 0.0,
      "exact_threshold": 0.8,
      "exact_pass": false
    },
    "size_transfer": {
      "predecessor_accuracy": 0.0234375,
      "predecessor_threshold": 0.9,
      "predecessor_pass": false,
      "exact_array_accuracy": 0.0,
      "exact_threshold": 0.5,
      "exact_pass": false
    },
    "all_competence_pass": false
  }
}

|Size16 development contrast|Point estimate|Adjusted one-sided lower bound|Resolved positive|
|---|---:|---:|---|
|T_minus_C|7.8125 pp|6.44042969 pp|True|
|T_minus_own_query_independent|0 pp|0 pp|False|

Exactly20000 paired complete-array bootstrap replicates, seed6400, shared bank, linear empirical quantile.025 for each of two contrasts. The bank was sealed before training. All queries within an array stay together. Intervals condition on one paired initialization/recipe and previously examined panels; they do not establish seed reproducibility or untouched population confirmation. Strongest positive classification additionally requires treatment to pass all competence gates while control fails overall competence.

## Secondary mechanistic measurements

Each prescribed evaluation captures every active mixer input/output, whole-block input/output, pre-terminal-normalization and committed vectors, native query/key projections and scores. Source labels distinguish core outputs from later FFN transforms. No mixer inversion or extra forward is used. Per-array candidate/query roles remain separate. Recorded centered energy and common components, actual corresponding-pair difference transport, norms, angular/radial decomposition and terminal normalization summaries are in each arm/panel mechanistic JSONL; raw tensors are retained. Every captured whole-block residual identity is checked exactly.

The native score matrix is decomposed into repeated query-independent candidate preference plus query-dependent interaction; target/incorrect margins, component norms, rank changes and positive/negative interaction counts are retained. Geometry is descriptive only: no new binary collapse threshold, independent-pair inference, component attribution or task-success claim is introduced. A change in variation cannot substitute for competence or the paired contrasts.

## Resource and integrity accounting

{
  "attempted_tokens": 82432,
  "completed_tokens": 82432,
  "failed_tokens": 0,
  "selected_bank_attempted": 659456,
  "selected_bank_completed": 659456,
  "commits": 82432,
  "contextual_commit_entries": 577024,
  "backward_attempted": 2000,
  "backward_completed": 2000,
  "backward_failed": 0,
  "optimizer_attempted": 2000,
  "optimizer_updates": 2000,
  "optimizer_failed": 0
}

Execution boundary elapsed 1491.849522s under the21600s paired ceiling. Peak allocated/reserved CUDA bytes: 192163328/201326592. Per-arm phase and attempted/completed/failed counters, actual selected calls, contextual commit entries, backwards and optimizer steps are retained. Complete token-trace reconciliation is in `complete_trace_verification_v1.json`. No additional model trajectories, diagnostics, profiling or warm-ups were run.

All2675 pre-existing file identities are preserved except the explicitly authorized Git HEAD switch/log append for the new branch; original Git log bytes remain a verified prefix. Original branch ref, HEAD commit, index and tracked diff are preserved. All14 historical tracked modifications and2093 original untracked paths remain. The supplied protocol adds one new root file; pilot evidence is isolated. No reset, clean, stash, source overwrite or deletion occurred.

Final checkpoints and optimizer/update metadata, when completed, are named C_update1000_v1.pt and T_update1000_v1.pt; their individual identity records contain SHA256. All new file hashes, source amendment diff and configuration seals accompany publication. The original training remains COMPLETED_NON_SUCCESS; original readout remains INCONCLUSIVE_NONCONVERGED_OPTIMIZATION; converged refit remains CERTIFIED_FIXED_OBJECTIVE_NON_SUCCESS.

## Interpretation and stop

Outcome: **PARTIAL_EFFECT_TASK_NON_SUCCESS**. Any positive effect concerns the complete prescribed intervention under this task/recipe; it does not uniquely identify historical collapse causation. No claim of internally learned insertion sort, seed replication, untouched confirmation, learned adaptive routing or general architectural superiority is established. No variant, seed, extra update, coefficient, gain/normalization change, task, fitted readout, selector or follow-up is launched. Stop after verified publication and report; await a new research decision.
