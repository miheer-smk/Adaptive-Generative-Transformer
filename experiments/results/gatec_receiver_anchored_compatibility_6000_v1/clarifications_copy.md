# Prospective clarifications — supplied conversation authority

This new document records the user's final clarifications. It does not rewrite
an original protocol or any v4 record. The original protocol authority is the
complete text in the conversation; no local original-protocol file identity is
claimed. Its versioned implementation binding is recorded separately.

## Profiling

Use z0 = 0u only, identical inputs across treatment, control and frozen reference.
Main confirmation uses both suffixes. Each profiling execution starts from the
original initial state and executes the complete base-plus-z0 trajectory.
Exactly one warm-up and five timed repetitions: 32 cases × 5 paths × 6 executions
× 28 average tokens = 26,880 model-token executions. No prefix reuse.

## Distinguishing-operation ablation

U[m,p,i] and A[m,p,i] indicate current-boundary and all-token correctness for
both eight-token suffixes on paired unablated/ablated native trajectories.
a[m,p] = E[U-A], with equal weights for four length/parity strata.
Reuse the confirmation-case bank of 20,000 stratified paired resamples, seed
6010. The family is exactly four arm/program contrasts. Individual one-sided
lower quantile = 0.0125, linear-interpolated empirical quantiles. Pass iff the
unrounded point estimate is >= 0.05 AND the adjusted lower bound is > 0.
No requirement for the lower bound to exceed 0.05; no length-specific gates.

## Latency

Average five timed measurements within case; exclude warm-up. R[p,L] is the
ratio of treatment/control arithmetic case-mean latency with equal parity
weighting, not a mean of individual ratios. Use a separate profiling-case
resampling bank initialized with seed 6010, 20,000 paired parity-stratified
resamples, pairing both programs and arms. The four program/length comparisons
form their own confidence family. Individual one-sided upper quantile = 0.9875,
linear interpolation. All four upper bounds must be <= 1.10. Nonfinite values,
nonpositive denominators or missing prescribed measurements are unassessable.
No trimming, replacement or discarding measurements. The ablation and latency
families remain separate.

## This authorization boundary

Research-only implementation and disposable prerequisite fixtures only.
At most 512 attempted model-token calls and 4,096 actual selected-bank calls.
Record attempted/completed/failed calls and warm-ups separately. No optimizer
steps, parameter updates, training, study checkpoints, study ablation, study
profiling, model selection or held-out confirmation. Backward is restricted to
disposable fixtures. Stop after the prerequisite evidence report.
