# Frozen-core bilinear-readout decodability study

## Result and boundary

**INCONCLUSIVE_NONCONVERGED_OPTIMIZATION**. All scheduled fitting, extraction and evaluation stages completed within their numerical/resource limits. The fit reached256 accepted iterations without the declared convergence certificate; it was not restarted or extended. The terminal readout improves over the original head on development, but is worse than its own query-independent component and far below competence thresholds. Thus it does not demonstrate competent query-conditioned bilinear decoding. The historical AGT training result remains **COMPLETED_NON_SUCCESS**.

This is one frozen-core readout study, not core training, algorithm-discovery evidence, a learned-routing result or an architectural-superiority comparison. The128 fitting arrays were seen during original AGT training. Development/transfer arrays are held out from **readout fitting** but were examined previously; they are not untouched confirmation data.

## Fit, whitening and sealing

The original training cache SHA-256 is `83b8a4c1d19ba479901d5714c246e89385aadfc76eb971c463fe9ac115352f41`. All per-array hashes, original-index associations, targets, shapes and dtypes were verified before fitting. Its128 trajectories were not repeated.

Separate64×64 uncentered query/candidate moments used all2048 fitting states per role. Symmetric inverse-square-root whiteners retain all64 dimensions. Eigenvalue floors are query5.93574094308398e-07 and candidate5.93573262924164e-07, exactly1e-8 times each largest eigenvalue. Respectively62/59 eigenvalues are floored, not removed. No mean subtraction, per-vector normalization, evaluation-data moment or extra feature was introduced. Matrices/eigensystems are retained in `preconditioners_v1.pt`.

One unrestricted4096-parameter B starts at0. The objective is `meanCE+1e-6/2*||B||²`; the full-batch gradient is `sum_r U_r.T@(P_r-Y_r)@V_r/(8*2048)+1e-6*B`. This exact analytic gradient is supplied to one continuous CPU FP64 PyTorch L-BFGS optimizer with history20 and strong-Wolfe search. The default initial step scale is1; algorithm defaults c1=1e-4/c2=.9 are recorded. Default alternative stopping tolerances are disabled. Caching identical deterministic B evaluations does not change the objective and is disclosed:526 closure calls correspond to271 actual loss/gradient evaluations, including the initial point; all calls/steps are logged.

Stop reason: **accepted_iteration_limit**. Accepted steps256; actual evaluations271; fit duration0.859995093s. Final F=2.7430078767525, CE=2.74150915518742, regularizer=0.00149872156507759, gradient norm=3.30890026861629e-06. Certificate `||gradF||²/(2e-6)` is **5.47441049382448e-06**, exceeding1e-6. Neither an accepted iterate nor a line-search trial met that threshold. Nonconvergence is reported exactly, without rounding into a pass. No alternative iterate was evaluated or selected.

`terminal_readout_v1.pt` and the objective/optimizer records were hashed in `fit_seal_v1.json` **before** opening held-out pools for extraction. That seal remains unchanged after evaluation. FP64 fitting CE recomputed during the prescribed terminal evaluation agrees with sealed CE to floating-point precision. The readout is separate from the archived model; no model checkpoint is replaced.

## Frozen extraction and original-head equivalence

The frozen checkpoint SHA-256 remains `2c0e3d6ad3711d8f0ee209eb3e3b0f8db8964e6eeaf2be77b7bd56120f29e6c8`. Parameter/buffer identity before/after is `7ab9239da5b1a4957cde546bb95e4c2961800b337f8bf677f8202e662e94f1a0`. Existing source/adapter behavior was reused unchanged, with namespaced observer import paths and bounded counters.

All32 development n16 and32 transfer n64 arrays ran once in original order, using fresh initial states, FP32 canonical `[0,1,2,3,4,5,6,7]`, normal RMSNorm/memory/cache/GDN semantics and terminal commits. Encoder/core/normalization/original pointer parameters are frozen. Read-only hooks verify cached candidates and queries equal the original head's inputs. No fitting-array re-extraction, warm-up or retry occurred.

All64 ordinary original-head score matrices and predictions match the archive **bit-for-bit**, before any new-head task metrics were calculated. Both representation roles, exact targets and original input identities are retained in `heldout_representations_v1.pt` (SHA-256 `6132eef9c5638b4575098144ac9d9b8dff3e7fd51a8ad7e8e5807ce4949fd793`). Complete traces verify5120 tokens,40960 actual selected bank calls,5120 commits and35840 contextual commit entries. Each token has D=P=8, the correct selected order, unchanged parent/in-flight state and valid cache versions. Invalid predicted pointer chains below are **task-output errors**, not executor/mechanical failures.

## Primary operational metrics

FP32 CUDA arithmetic explicitly computes `states@whitener`, then `transformed_queries@B`, then candidate dot products divided by8. FP64 sensitivity uses precisely the same sealed matrix and preconditioners, cast to64, with no fitting or selection. CE is calculated stably from retained scores.

|Split|Condition|Pointer accuracy|Exact arrays|Mean CE|Invalid chains|Mean target rank|Mean target margin|Distinct candidates/array|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|fitting|original|3.3203125%|0%|2.772870812|100%|9.372558594|-0.002784876648|1|
|fitting|fitted_FP32|11.376953%|0%|2.741515472|100%|7.502929688|-0.3298002545|3.8515625|
|fitting|uniform|6.0546875%|0%|2.772588722|100%|1|0|1|
|fitting|self_pointer|6.25%|0%|unavailable|100%|unavailable|unavailable|16|
|fitting|original_order|5.1757812%|0%|unavailable|0%|unavailable|unavailable|15|
|fitting|common_FP32|11.181641%|0%|2.754072181|100%|7.607421875|-0.2582410523|1|
|development|original|3.125%|0%|2.772864395|100%|9.375|-0.002720198943|1|
|development|fitted_FP32|9.9609375%|0%|2.757183478|100%|7.80859375|-0.3465033805|3.65625|
|development|uniform|6.640625%|0%|2.772588722|100%|1|0|1|
|development|self_pointer|6.25%|0%|unavailable|100%|unavailable|unavailable|16|
|development|original_order|7.6171875%|0%|unavailable|0%|unavailable|unavailable|15|
|development|common_FP32|11.523438%|0%|2.752649539|100%|7.60546875|-0.254662277|1|
|size_transfer|original|1.171875%|0%|4.158960282|100%|33.44433594|-0.00284307744|1|
|size_transfer|fitted_FP32|1.4648438%|0%|4.363874909|100%|33.27929688|-1.005639912|4.0625|
|size_transfer|uniform|1.5136719%|0%|4.158883083|100%|1|0|1|
|size_transfer|self_pointer|1.5625%|0%|unavailable|100%|unavailable|unavailable|64|
|size_transfer|original_order|1.3671875%|0%|unavailable|0%|unavailable|unavailable|63|
|size_transfer|common_FP32|1.2207031%|0%|4.337194537|100%|33.29052734|-0.9296384654|1|

Competition ranks are1+number of strictly greater scores; stable candidate-index ranks and all ties are also retained. Uniform scores have ties everywhere, so their competition rank1 does **not** indicate discrimination. The table reports deterministic lowest-index argmax for uniform probabilities; analytic independent-uniform expected accuracy is1/n and exact-array probability is n^(-n), separately recorded without random draws. Hard self-pointer/original-order controls have genuine hard-decision metrics; CE, ranks and score margins are unavailable, with no smoothing invented. Full per-query predictions, ranks, ties, target margins and per-array distinct-candidate counts are retained in `per_array_predictions_v1.json`.

## Primary contrasts and competence decisions

|Development paired contrast|Point difference|Adjusted one-sided lower bound|Positive lower bound|
|---|---:|---:|---|
|fitted FP32 − original|6.8359375 pp|4.6875 pp|True|
|fitted FP32 − common_FP32|-1.5625 pp|-3.7109375 pp|False|

These use20,000 paired complete-development-array bootstrap replicates, seed6300, sharing one bank across both contrasts. The two-contrast family uses one-sided lower empirical quantile.025 with linear interpolation. Pointers within an array are not resampled independently. `bootstrap_bank_v1.pt` and `paired_contrasts_v1.json` preserve the bank/hash and results. Intervals describe input-sampling uncertainty conditional on this frozen core/readout and the historically exposed panel, not seed reproducibility.

Size16 requirements (95% predecessor,80% exact): **both fail**. Size64 requirements (90% predecessor,50% exact): **both fail**, reported separately. The original-head contrast has a positive adjusted lower bound; the common-component contrast does not. Strong positive evidence therefore fails in addition to the convergence failure. The modest original-head gain cannot be promoted to useful query-conditioned recovery.

## Target-conditioned decomposition

For the actual transformed/projected query and candidate vectors, verify `Srow=1*b.T+I`, with `b=Kc@mean(Q)/8` and `I=Qc@Kc.T/8`. The common component is an analytical repeated candidate preference, not a second fitted head or a causal deployed policy. Algebraic identities are checked in FP64 from each precision's actual vectors; actual FP32 score arithmetic agrees under the original atol1e-6/rtol1e-5 comparison standard.

On development, interaction improves3805/7680 target-versus-incorrect margins and worsens3875. It flips567 comparisons toward the target and671 away, and worsens total target ranks by104 positions. Fitted CE2.75718348 is worse than common-only2.75264954. Although predictions vary across3.65625 candidates/array rather than1, that variability does not establish useful task credit.

Fitting interaction counts are16706 improved/14014 worsened (30720 dependent comparisons); transfer counts65133/63891 (129024 comparisons). These pair counts are descriptive, not independent statistical observations. `interaction_summary_v1.json`, per-array records and score tensors retain complete component evidence. No target label is used to construct the common component itself.

## Numerical sensitivity

|Split|FP32 pointer accuracy|FP64 pointer accuracy|FP32 CE|FP64 CE|Different predictions|
|---|---:|---:|---:|---:|---:|
|fitting|11.376953%|11.376953%|2.74151547213|2.74150915519|4/2048|
|development|9.9609375%|9.9609375%|2.75718347794|2.75721040982|1/512|
|size_transfer|1.4648438%|1.4160156%|4.36387490897|4.36383529532|10/2048|

FP64 does not recover competence. Maximum FP32/64 score differences are recorded, as are4 fitting,1 development and10 transfer prediction disagreements. No precision, iterate, regularizer, eigenvalue floor or threshold was changed after these results. The primary result remains FP32. No fitted-readout data is pooled across lengths.

## Resource and integrity accounting

Remaining-stage new work:5120 attempted/completed model-token calls,0 failures;40960 attempted/completed selected calls;5120 terminal commits;0 AGT backward passes;0 AGT optimizer objects/updates. There is exactly1 readout optimizer and256 accepted readout updates, with271 analytic full-batch loss/gradient evaluations and0 readout autograd backwards. No extra head, seed, sample, warm-up or retry occurred. Extraction took39.794999459s; peak allocated/reserved CUDA memory during extraction was21,242,880/35,651,584 bytes.

Prior cache extraction is charged31.302s and4064 tokens separately. Full staged extraction totals are9184 newly executed tokens,73472 selected calls and9184 commits; reused historical row0000 adds0 new calls. At report closure, measured elapsed time since the remaining-stage clock marker is851.494s. A conservative600s opening-inspection reserve is additionally charged (not claimed as measured runtime), giving a cumulative budget charge of1482.796s, below2700s. A live guard includes subsequent publication; final terminal verification reports its inclusive charged time. No ceiling is extended.

All2576 pre-existing protected files, original cache/checkpoint/source, local HEAD and tracked diff pass final integrity verification. Historical bytecode exceptions are preserved without additions. `resource_ledger_v1.json`, `complete_record_verification_v1.json`, hashes, input manifests, readout/preconditioners, scores and versioned research-only code accompany this report. Publication adds only this fresh subdirectory on a unique branch and verifies contents independently.

## Interpretation and stop

Under the approved interpretation, poor results without the declared convergence certificate are **inconclusive**. They do not establish universal representation impossibility, a historical cause of the original failure, learned routing, internal sorting-algorithm acquisition or superiority over CLRS learners. A small gain over a weak original head, without competent exact arrays or an advantage over query-independent allocation, is insufficient.

**The authorized one-fit study is complete. No optimizer restart, extra iterations, new seed, core training, alternate readout, selector, task change or follow-up experiment is launched.**
