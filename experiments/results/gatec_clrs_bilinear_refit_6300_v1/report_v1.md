# Exploratory bounded bilinear-readout refit

**CERTIFIED_FIXED_OBJECTIVE_NON_SUCCESS**. The fresh zero-initialized fit reached the first accepted-iterate certificate within every limit, but the approximately optimized fixed objective did not demonstrate competent query-conditioned decoding. The original frozen-readout study remains **INCONCLUSIVE_NONCONVERGED_OPTIMIZATION** and the historical AGT result remains **COMPLETED_NON_SUCCESS**.

## Provenance and scientific boundary

This separately sealed exploratory attempt follows the prior nonconvergence. Protocol SHA256 `a9e10113d8405d9a34a04dcd9fe6960b7b9c169314eb0a4a0f3e075a74f99704`; input/source protocol seal `087aff27ef4e5d768be81dba45ffcac1fa031e39bd8354d79582652487a1cd18`. Training cache `83b8a4c1d19ba479901d5714c246e89385aadfc76eb971c463fe9ac115352f41`; frozen checkpoint `2c0e3d6ad3711d8f0ee209eb3e3b0f8db8964e6eeaf2be77b7bd56120f29e6c8`. The original terminal matrix was identity-verified but not loaded as initialization. Exact training-only preconditioners are reused from the prior file (SHA256 `4117bddef4d4e1b8016a0571c2f7069dbe5ddd0e4dbc9e3015bb75fea98c1608`). Held-out representation SHA256 `6132eef9c5638b4575098144ac9d9b8dff3e7fd51a8ad7e8e5807ce4949fd793`.

All128 fitting arrays were previously seen during original AGT training. Development32 size16 and transfer32 size64 arrays are held out from readout fitting, but examined historically; neither is untouched confirmation. No data or state extraction was repeated.

## Fit and certificate

One4096-parameter FP64 B starts at exactly0, with one fresh optimizer and continuous L-BFGS history20, strong-Wolfe, lr1, c1=1e-4/c2=.9, other stopping tolerances disabled. Whiteners, full-batch meanCE averaging, division8 and regularization1e-6 are unchanged. The exact analytic full-gradient formula is the prior implementation, with no autograd backward. The timer includes optimizer construction and the initial evaluation. Stop reason: **converged**.

Accepted updates **341** /512; wrapper calls341; actual objective/gradient evaluations attempted/completed/failed **360/360/0** /1024; cache hits341; closure calls700. Initial evaluation plus341 accepted points plus18 rejected trial points accounts for360 actual evaluations. No duplicate full recomputations or extra terminal full evaluation occurred. Cache keys contain the exact FP64 B bytes; values/gradients are reused only for the identical matrix and fixed objective.

Fitting wall time **1.55306719616s** /10. Terminal F=2.7430075098149667, CE=2.7415247279357109, regularizer=0.0014827818792558108; gradient norm=1.1756482252332946e-06. Certificate **6.9107437474709814e-07 <=1e-6**. The initial point and all340 preceding accepted points fail the certificate; every341 accepted update passes strong-Wolfe checks. Terminal objective/gradient uses its matching retained full evaluation. Trials alone never authorize selection.

Terminal matrix SHA256 `7480544bc42cfad5f671b04cc61afa566dd04deb2175cb4e218274a04c7f844d`. Full terminal optimizer state, evaluation/accepted-update/cache trace and source identities are retained. `fit_seal_v1.json` SHA256 `e83f851e33ebed7e76c06a5563786054dfe5703c543d0a101730b556be64dd4b` was verified before held-out metrics. Only the first certificate-passing iterate is evaluated. No restart, old history, regularizer sweep or matrix selection occurred.

## Prescribed operational metrics

Primary FP32 CUDA tensor arithmetic and fixed FP64 sensitivity use the prior evaluation arithmetic unchanged. Original archived pointer scores, predictions, targets and input identities match the retained cache exactly for all64 held-out arrays before new-head results. No model is constructed.

|Split|Condition|Pointer accuracy|Exact arrays|CE|Invalid chains|Mean rank|Mean target margin|Distinct candidates/array|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|fitting|original|3.3203125%|0%|2.772870812|100%|9.372558594|-0.002784876648|1|
|fitting|fitted_FP32|11.328125%|0%|2.741530976|100%|7.506347656|-0.3298445255|3.8515625|
|fitting|uniform|6.0546875%|0%|2.772588722|100%|1|0|1|
|fitting|self_pointer|6.25%|0%|unavailable|100%|unavailable|unavailable|16|
|fitting|original_order|5.1757812%|0%|unavailable|0%|unavailable|unavailable|15|
|fitting|common_FP32|11.181641%|0%|2.754075945|100%|7.607910156|-0.2581984063|1|
|development|original|3.125%|0%|2.772864395|100%|9.375|-0.002720198943|1|
|development|fitted_FP32|9.9609375%|0%|2.757161242|100%|7.810546875|-0.3464818808|3.625|
|development|uniform|6.640625%|0%|2.772588722|100%|1|0|1|
|development|self_pointer|6.25%|0%|unavailable|100%|unavailable|unavailable|16|
|development|original_order|7.6171875%|0%|unavailable|0%|unavailable|unavailable|15|
|development|common_FP32|11.523438%|0%|2.752652787|100%|7.60546875|-0.2545887286|1|
|size_transfer|original|1.171875%|0%|4.158960282|100%|33.44433594|-0.00284307744|1|
|size_transfer|fitted_FP32|1.5136719%|0%|4.361190934|100%|33.27148438|-0.9974885777|4.09375|
|size_transfer|uniform|1.5136719%|0%|4.158883083|100%|1|0|1|
|size_transfer|self_pointer|1.5625%|0%|unavailable|100%|unavailable|unavailable|64|
|size_transfer|original_order|1.3671875%|0%|unavailable|0%|unavailable|unavailable|63|
|size_transfer|common_FP32|1.2695312%|0%|4.334617168|100%|33.28662109|-0.9219524385|1|

Uniform score rows use deterministic lowest-index argmax, with analytic independent uniform probabilities reported separately (expected pointer accuracy1/n, exact-array n^(-n)). Competition rank1 for all-tied uniform rows is not discrimination; stable index ranks/ties are retained. Self-pointer/original-order controls are hard-only, with CE/ranks/margins unavailable and no invented smoothing. Invalid chains are task errors, not executor failures. Complete per-array/query records are retained.

## Paired contrasts and competence

|Development FP32 contrast|Point difference|Adjusted lower bound|Positive lower bound|
|---|---:|---:|---|
|Fitted minus original|6.8359375 pp|4.6875 pp|True|
|Fitted minus common_FP32|-1.5625 pp|-3.7109375 pp|False|

The exact previously retained20000 paired whole-development-array bootstrap bank is reused (seed6300); both contrasts use empirical one-sided quantile.025, linear interpolation. Pointers are not independent resampling units. Uncertainty is input-sampling uncertainty conditional on one frozen core and this exploratory fitted matrix. No selection or fitting uses evaluation targets.

Size16 predecessor>=95% and exact>=80%: **fail**. Size64 predecessor>=90% and exact>=50%: **fail**, separately reported. The original-head contrast resolves positively, but fitted-minus-own-common does not; therefore no required query-conditioned advantage or strong competent recovery is demonstrated.

## Score decomposition and numerical sensitivity

The actual projected vectors verify Srow=1*b.T+I. Both matrix components, target-versus-incorrect margins and all target ranks are retained. Components are not separately fitted heads. Pair counts are descriptive dependent comparisons, not independent observations.

fitting: 30720 target/incorrect comparisons; interaction improves16719 and worsens14001. Common-positive17187, actual-positive17395; flips toward/against target2403/2195; total target-rank improvement208. Mean common/interaction Frobenius norms3.100707388/2.406289002.

development: 7680 target/incorrect comparisons; interaction improves3808 and worsens3872. Common-positive4298, actual-positive4193; flips toward/against target566/671; total target-rank improvement-105. Mean common/interaction Frobenius norms2.947374857/2.295089698.

size_transfer: 129024 target/incorrect comparisons; interaction improves65111 and worsens63913. Common-positive62901, actual-positive62932; flips toward/against target3709/3678; total target-rank improvement31. Mean common/interaction Frobenius norms37.02839142/15.83948335.

|Split|FP32 accuracy|FP64 accuracy|FP32 CE|FP64 CE|Prediction disagreements|
|---|---:|---:|---:|---:|---:|
|fitting|11.328125%|11.328125%|2.74153097613|2.74152472794|6/2048|
|development|9.9609375%|9.9609375%|2.75716124157|2.75718819677|3/512|
|size_transfer|1.5136719%|1.4648438%|4.36119093415|4.36115217039|7/2048|

FP64 does not restore competence. Its fitting CE matches the sealed objective CE within floating-point precision; this is prescribed metric evaluation, not another full loss/gradient evaluation. Neither the readout nor precision is selected using these results.

## Resources, preservation and boundary

Zero AGT constructions, model tokens, selected bank calls, terminal commits, core backward passes, AGT optimizers/updates, extraction calls, tests, warm-ups or retries. Exactly one new readout fit. Computational work outside the1.553-second fit comprises identity checks, prescribed terminal readout matrix arithmetic/analysis and publication; it adds no optimizer iterations or objective/gradient evaluations. All2631 pre-existing files (including the prior study and local Git administration), HEAD, tracked diff and historical bytecode exception are unchanged. Numerical warnings from the unchanged Torch environment are not silently suppressed; actual arithmetic completed and all required checks passed.

The approximately optimized solution of this fixed regularized bilinear objective does not demonstrate required competence. This is not proof that every bilinear readout or representation fails, does not establish historical causation or internal insertion-sort learning, and says nothing positive about learned routing. Existing nonconverged and negative historical classifications remain unchanged.

Publication uses a unique branch with the previously verified study commit as parent; all raw content hashes and remote tree entries are verified separately. Remote commit identity is reported in terminal output and the publication receipt. No local Git staging/ref change, history rewrite or deletion is used. **Stop: no third fit, extended limit, alternate readout, AGT training, extra seed/input or automatic follow-up.**
