# Exploratory frozen-core bilinear refit: one attempt

This separately authorized exploratory follow-up is prompted by the prior 256-iteration nonconvergence. The original frozen-readout study remains INCONCLUSIVE_NONCONVERGED_OPTIMIZATION; the original AGT training remains COMPLETED_NON_SUCCESS. Development and size-transfer panels were examined previously and are held out only from readout fitting, not untouched confirmation.

## Inputs and fixed mathematical problem

Reuse the original 128-array training cache in original order, SHA256 83b8a4c1d19ba479901d5714c246e89385aadfc76eb971c463fe9ac115352f41. Frozen checkpoint SHA256 2c0e3d6ad3711d8f0ee209eb3e3b0f8db8964e6eeaf2be77b7bd56120f29e6c8. Reuse exactly the prior training-only query/candidate symmetric uncentered whiteners, preserving all dimensions and eigenvalue floor 1e-8 times largest eigenvalue. No whitening recalculation, mean subtraction, state normalization, new feature or data extraction. Original readout, source, objective, input manifests, checkpoint, held-out tensor cache and original-head equivalence records are hash-verified. Exact paths/hashes are in input_verification_v1.json and protocol_seal_v1.json.

One unrestricted FP64 64x64 B starts at zero, with fresh optimizer state. Scores are (U@Wq)@B@(V@Wk).T/8. Full-batch CE averages all 2048 training queries. F=meanCE+(1e-6/2)*sum(B^2). The same deterministic CPU FP64 tensor operations and exact analytical full-objective gradient as the prior implementation are used. No AGT or readout autograd backward is required.

## Optimizer and simultaneous hard limits

One continuous PyTorch L-BFGS history20, strong-Wolfe, lr1, c1=1e-4, c2=.9, tolerance_grad=0, tolerance_change=0, max_iter1 calls retaining the full optimizer history. The unchanged pinned PyTorch source defines line-search behavior. Count actual moved, strong-Wolfe-accepted updates, not wrapper calls. Check the initial and every accepted iterate against ||gradF||_F^2/(2e-6)<=1e-6. A trial alone cannot qualify.

Stop at the first certificate-passing initial/accepted iterate, or first of 512 accepted updates, 1024 actual full objective/gradient evaluations, 10 monotonic seconds. The timer begins immediately before optimizer construction and includes its construction and the initial evaluation. All line-search trials, duplicate recomputations and any terminal verification are charged if actually evaluated. Exact-byte cache hits for the identical B and objective are separately logged and do not count as fresh evaluations. No separate terminal reevaluation is needed when the matching evaluated gradient is retained. Maintain an accepted-point snapshot; on a mid-search limit, restore only the last accepted iterate, recording rejected/truncated work. No restart, resume, extension, alternative readout or hyperparameter change.

## Certificate barrier and evaluation

Save and hash terminal B, fixed whiteners, full optimizer state, fit configuration, full attempted/completed/failed evaluation trace, accepted-update trace, cache hits, objective decomposition, gradient/certificate and source/input identities before held-out fitted metrics. If not certified, stop scientific work and publish a nonconvergence report; do not open held-out tensor contents or calculate new fitted metrics.

If certified, use the one first-passing terminal B with the existing held-out representation cache only. Verify original-head equivalence records and cache hashes before metrics. Use exactly the prior evaluation implementation's arithmetic: FP32 CUDA matrix algebra primary, fixed FP64 sensitivity, no AGT model imports/construction/calls/backward/optimizer. Report fitting/development/transfer separately for original, fitted, uniform, hard self-pointer, hard original-order predecessor, and the fitted head's repeated query-independent candidate preference. Hard controls have no invented CE. Report predecessor/exact accuracy, CE where defined, invalid chains, ranks, margins, distinct predicted candidates, and score decomposition Srow=1*b.T+I with target-conditioned margin comparisons. Do not treat candidate pairs as independent samples.

Reuse the existing 20000 complete-development-array paired bootstrap bank seed6300; linear empirical quantile .025 for the two one-sided primary lower bounds (fitted minus original; fitted minus own common component). No resampling of individual pointers or pooling lengths. Size16 competence requires predecessor>=.95 and exact>=.80; size64 >=.90 and >=.50. Strong evidence requires certified fit, FP32 size16 competence and both positive adjusted lower bounds. FP64-only competence is numerically fragile, not robust operational recovery.

## Interpretation, preservation and closure

Certified poor performance or no resolved query-conditioned advantage means the approximately optimized solution of this fixed regularized objective does not demonstrate required competence; it does not prove all bilinear readouts or representations fail. Without a certificate, optimization qualification remains unresolved and no evaluation/extension is allowed. No inference about historical causes, internally learned insertion sort, creation of information during training, or learned routing is licensed.

All pre-existing files, historical exceptions, HEAD and tracked diff are fingerprinted and rechecked. New files only in this directory. No samples, checkpoint changes, state extraction, AGT execution, tests, core updates or model construction. Publish on one unique GitHub branch through Git Database API preserving history; verify every raw content hash and branch commit. No force push/local staging/deletion. Stop after report and verified publication; no third fit or automatic follow-up.
