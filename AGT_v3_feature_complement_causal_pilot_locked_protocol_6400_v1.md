AGT-v3 Feature-Complement Causal Pilot — Locked Protocol
Protocol status: Approved for one bounded paired exploratory pilot by GPT-6.1 Sol and adversarially reviewed GO by Astra.
Scope: This document authorizes only the single paired control/treatment pilot specified here. It does not authorize further architecture variants, seeds, task changes, hyperparameter tuning, or follow-up experiments.
Repository: miheer-smk/Adaptive-Generative-Transformer
DGX working tree to inspect: /home/iiitn/Desktop/Miheer Projects/AGT/AGT_v3_POC
Reviewed source reference: 1ed1d773264b3f5f53bd1ba169e819709d33cbb2
Historical training checkpoint (archival reference only; do not initialize from it): 2c0e3d6ad3711d8f0ee209eb3e3b0f8db8964e6eeaf2be77b7bd56120f29e6c8
Prior bilinear-readout refit commit (do not repeat): 97f3b78a17d4060d2b0fef8b52c606596de6ca5e
1. Research question
Does adding the specified fixed feature-complement path at the eight active bank-mixer sites improve acquisition of query-conditioned CLRS insertion-sort predecessor prediction under the original task interface and locked training recipe?
A positive result concerns the effect of this complete intervention under this recipe. It does not uniquely identify the historical cause of representational collapse and does not establish learned adaptive routing or broad AGT-v3 superiority.
2. Preflight and preservation gate
Before implementation or training:
1. Inspect the actual DGX checkout, branch, Git status, source/configuration seals, archived seed-6200 initialization record, original CLRS configuration, data/sample manifests, and existing study artifacts.
2. Record existing modifications. Do not reset, clean, overwrite, delete, or silently normalize the existing working tree. Create a unique, separate research branch for the pilot. Do not force-push.
3. Confirm the original initialization, training settings, data order, evaluation definitions, and training code can be reconstructed faithfully. Verify existing integrity seals. If any required component cannot be reconstructed, stop before training and report the precise blocker; do not substitute settings.
4. Do not change historical classifications or historical reports. Prior classifications remain: original CLRS training COMPLETED_NON_SUCCESS; original bilinear readout study INCONCLUSIVE_NONCONVERGED_OPTIMIZATION; separate converged bilinear refit CERTIFIED_FIXED_OBJECTIVE_NON_SUCCESS.
5. Do not perform a preliminary full-model diagnostic forward, profiling run, warm-up, extra training trajectory, or exploratory run. Source-level and isolated operator-level checks are allowed. Initialize the two actual arms from the archived starting state and verify their starting parameter/state equivalence before their first training update; do not load the failed final checkpoint.
3. Exact intervention — do not alter
The implementation uses row-vector batches and mixes the final 64-dimensional feature axis. The control operation is:
Y_control = (X @ P.T) * g
Define the fixed matrix:
Q = I_64 - ones(64, 64) / 64
The treatment operation is exactly:
Y_treatment = (X @ (P + Q).T) * g
Equivalently in column-vector convention, M_T(z) = D_g (P + Q) z, with u = ones(64)/sqrt(64) and Q = I - u u.T.
Apply the treatment at precisely the eight active sites: six outer block mixers plus the two mixers inside the FFN cores. Do not modify the unexecuted prelude mixer.
Keep the existing trainable P and learned gain g; the fixed Q adds no trainable parameters. Preserve the gain after the combined matrix. Do not move Q outside the gain, change the axis or transpose convention, add a coefficient, renormalize P + Q, or modify the existing mixer constraints or initialization. Do not change residual additions, RMSNorm, block order, memory, caches, terminal commits, serialization, attention, labels, pointer head, losses, or any other computation.
Before starting the training clock, run source-level and isolated operator checks for the correct [*, 64] axis and shapes, device/dtype compatibility, transpose orientation, Q @ u = 0, the complement action, and unchanged post-combination gain placement. These checks must not construct/run a separate full AGT model or issue a full-model forward. If the actual implementation layout differs from the reviewed source, do not guess; stop and report it.
4. Paired arms and locked training recipe
- Control: original computation unchanged.
- Treatment: original computation with only the specified P + Q intervention at the eight active sites.
- Start both from independent copies of the exact original seed-6200 initialization, not the failed final checkpoint. Verify parameter/state equality before the first update; preserve independent optimizer and recurrent states.
- Match stochastic draws wherever applicable. The intervention consumes no randomness.
- Training arrays: the original 1,000 size-16 CLRS insertion-sort training arrays.
- Training-order seed: 6201.
- Updates: exactly 1,000 per arm; evaluate final update only.
- Optimizer: original AdamW, learning rate 1e-3, weight decay 0.
- Gradient clipping: global norm 1.
- Precision: FP32.
- Backpropagation: full BPTT.
- Preserve the original loss, frozen/trainable parameter ownership, mode flags, initialization order, optimizer settings, serialization, targets and data handling.
- Fixed canonical program: [0, 1, 2, 3, 4, 5, 6, 7].
- Do not add auxiliary losses, readout fitting, whitening, checkpoint selection, early stopping, parameter searches, or changes to the task/interface.
5. Fixed evaluation panels and outputs
Use the original sealed panels without selecting samples from observed representations, failure severity or treatment outcomes:
- Training-subset evaluation: 128 size-16 arrays.
- Development: 32 size-16 arrays.
- Size-64 transfer: 32 arrays.
For each arm at update 1,000, evaluate the native trained pointer head once. Report predecessor accuracy, exact-array accuracy, cross-entropy and invalid predecessor-chain frequency. Also report the existing query-independent score-component comparison and the uniform, self-pointer and original-order controls using their established metric conventions; do not invent smoothed CE/ranks/margins for controls where those are unavailable.
During these same prescribed evaluation forwards, retain the planned mechanistic measurements at active mixers and the terminal boundary:
- centered energy across positions, separately for candidates and queries, at mixer inputs and outputs;
- transport of observed input differences, not only worst-case singular values;
- norms and angular dispersion before and after terminal normalization;
- query-dependent score interaction and correct-target margins.
These are secondary explanatory measurements. Increased activation variation or changed geometry alone is not task success. No additional diagnostic forwards or profiling are permitted.
These panels have been examined historically. Development/transfer panels are held out from the original readout fitting but are not untouched confirmation. The pilot is exploratory and does not estimate training-seed variability.
6. Prospective success criteria and statistics
Competence gates
Both criteria on each listed panel must be satisfied:
Panel	Predecessor accuracy	Exact-array accuracy
Size-16 development	>= 95%	>= 80%
Size-64 transfer	>= 90%	>= 50%


Paired contrasts
On size-16 development, require both of these contrasts to have resolved positive lower bounds:
1. Treatment minus matched control predecessor accuracy.
2. Treatment minus its own query-independent component predecessor accuracy.
Use exactly 20,000 paired bootstrap replicates over complete arrays, prospective analysis seed 6400, and empirical quantile 0.025 as the one-sided lower bound for each of the two contrasts (Bonferroni adjustment for the two comparisons). All queries for one array must remain together in a resample. Do not alter the contrast definition, resampling unit, seed, number of replicates, quantile, or criterion after seeing results.
The strongest positive classification requires treatment to satisfy all competence gates and both resolved positive contrasts, while the control remains unsuccessful on the competence gates. Passing these exploratory panels is not independent population-level confirmation.
7. Resource ceiling and accounting
Hard ceiling: six hours total on the same DGX for both arms, their prescribed evaluations and required captures. This is a hard stop, not a runtime guarantee.
Predeclared workload estimate:
Work	Model-token executions across both arms
Training: 2 x 1,000 x 32	64,000
Training-subset evaluation	8,192
Development evaluation	2,048
Size-64 transfer evaluation	8,192
Total	82,432


Corresponding expected accounting: 659,456 selected bank executions, 82,432 terminal commits, and 2,000 optimizer updates across the pair. Record attempted and completed calls separately. Capture overhead, synchronization and evaluation must fit inside the six-hour ceiling. No uncounted profiling, warm-ups, diagnostics, retries or extra trajectories. If resource limits prevent both prescribed final evaluations, report an incomplete paired experiment; do not automatically restart.
A numerical or optimization failure of the correctly implemented treatment under the locked recipe is a treatment outcome, not automatic permission to tune it. If an engineering failure prevents valid execution, preserve evidence and classify the pilot as incomplete rather than as scientific non-success.
8. Locked interpretation and stop rules
- Treatment meets competence and both positive contrasts; control remains unsuccessful: retain the variant as a credible research candidate. The result supports an effect of this intervention under this recipe, not a uniquely identified historical mechanism.
- Both arms become competent without a resolved treatment benefit: do not credit the intervention; the historical failure was not reproduced sufficiently to support the proposed bottleneck.
- Treatment restores directional variation but remains incompetent: variation restoration was insufficient; pause this core/interface approach.
- Treatment remains collapsed and unsuccessful: this intervention was insufficient; do not strengthen or alter the mixer automatically.
- Treatment improves but misses competence: report any supported partial effect, while preserving task-level non-success.
- Engineering/resource failure prevents completion: report incomplete; make no scientific failure claim and do not retry automatically.
After the pilot, stop. Do not run another seed, extend the training limit, tune gain, change the complement, alter normalization or interface, fit another readout, train a selector, switch tasks, or start any automatic follow-up. A new research decision is required before any further experiment.
9. Integrity and publication checklist
The final report must include:
- exact source commit, unique branch name and code diff;
- original configuration seals and proof of starting-state equivalence;
- training-data/sample-manifest identities and seeds;
- per-arm update counts, attempted/completed call counts, timing, and resource-accounting totals;
- final checkpoint hashes and retained evaluation outputs;
- every required metric, paired-bootstrap bounds and mechanistic capture;
- all exceptions, interruptions, numerical instabilities or ceiling events;
- explicit final classification, limitations, and a statement that learned adaptive routing has not been established by this pilot.
Verify artifacts and hashes before publishing the new branch. Do not change prior reports or classifications, rewrite history, force-push, or delete data. Use the repository's established GitHub API publication process if the DGX checkout has no Git remote configured. After publication, stop and report the immutable commit/branch and outcome. The following is the authoritative protocol for the approved AGT-v3 feature-complement pilot. Read the entire protocol before taking any action.

First, save it as a new, uniquely named Markdown file in the repository without overwriting, deleting, or modifying any existing work. Verify the saved content against the supplied protocol.

Important safeguards:
- Preserve all 14 existing tracked modifications and 2,093 untracked paths.
- Do not reset, clean, stash, or overwrite the current working tree.
- Do not start training immediately.
- First perform the protocol's prescribed preflight and verify the original configuration, initialization provenance, and implementation requirements.
- If any requirement cannot be verified, stop and report the precise blocker.
- If preflight passes, follow the protocol exactly, including the fixed operator, paired control/treatment design, six-hour hard ceiling, evaluation criteria, and final stop rule.
- Do not run additional experiments or modify the approved design.

After reading the protocol, summarize the required preflight checks and any blockers before proceeding.
