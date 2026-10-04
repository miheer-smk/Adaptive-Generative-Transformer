# Stage-3 validation record — 2026-10-04

## Hypothesis and scope

Distinguish controller route expressivity from discovery by task loss,
and identify whether the locked executor has a malformed gradient path.
The bank, controller, masking, ST-Gumbel estimator, memory/context purity,
and single terminal commit are unchanged by this continuation. The prior
audit corrected illegal-action contamination of the relaxed normalization
in `b54afee`; this continuation uses that corrected primitive. No HALT
learning, auxiliary objective, regularizer, or temperature schedule is used.

Repository starting point: `5a88ecb` on
`research/stage3-routing-gate-c-audit`. This is the actual current branch,
not the older Stage-3 branch named in the supplied roadmap. Existing
Stage-4 changes, generated files, backups, and historical experiments
remain untouched. The prior audit is in `GATE_C_AUDIT.md`.

## Phase B: independently verified imitation

Backbone: `experiments/runs/stage1_parity_seed3305_best.pt`, excluding all
old controller weights. The fresh current controller alone is optimized.
Pilot: D64, bank size eight, 395 -> 256 -> 11 controller, zero budget.
Dataset: 8,320 explicitly aligned TOKEN -> BLOCK rows, from the training
length range 4–16. Optimizer: AdamW, lr=1e-3, weight_decay=0, minibatch
1,024. Supervised classification uses the full source-legal action mask,
with HALT only as a distractor class and no HALT target. Fixed-depth task
training and successor mixing exclude HALT. Real evaluation uses exactly
eight decisions and training=False.

| Seed | Supervised updates | Offline accuracy | Decisions 0–7 | Held-out validation exact routes | Held-out OOD exact routes |
|---|---:|---:|---|---|---|
| 3305 | 5,000 | 1.0 | each 1.0 | 130/130 tokens | 392/392 tokens |
| 3306 | 5,000 | 1.0 | each 1.0 | 130/130 tokens | 392/392 tokens |
| 3307 | 5,000 | 1.0 | each 1.0 | 130/130 tokens | 392/392 tokens |

Confusion matrices have 1,040 observations on each of the eight target
diagonal entries, and zero off-diagonal counts. Incorrect actions,
illegal predictions, and RESTART errors are all zero. Actual initial
executor probes also recover eight canonical actions for each of eight
tokens. Offline accuracy measures the supervised training rows; held-out
executor evaluation samples one sequence per length. This is replicated
canonical-route recoverability, not task-loss route discovery.

Authoritative new metadata files:
`experiments/runs/stage3_imitation_replication_{3305,3306,3307}_v1.json`.
Each records checkpoint SHA256, backbone SHA256, seed, dimensions,
budget, depth, optimization settings, error breakdown, and held-out
results. Checkpoints remain local, untracked artifacts. The diagnostic
rejects existing output paths and preserves failed recovery evidence.

## Phase C: gradient and observation diagnostics

Artifact: `experiments/runs/stage3_gradient_diagnostics_3305_v1.summary.json`.
It links the SHA256 of the full locally preserved retained-tensor trace.
The warm-start checkpoint is the independently reproduced seed3401 audit
checkpoint from the preceding work, not a stale historical checkpoint.

Setup: fixed validation sequence length eight, eight independent Gumbel
seeds 4305–4312, fixed tau=2, fresh controller initialization seed3305
versus strong supervised warm start. No optimizer updates. Frozen
parameters retain autograd through input activations. A separate structural
probe enables backbone gradients without running any optimizer; this is
not joint backbone training and does not modify weights.

The warm controller's complete observation tuple matches the fixed-route
teacher exactly at all 64 token/decision rows: h, memory read, slot mean,
pointer, and budget each have maximum absolute difference zero. A focused
two-token test independently verifies this contract with a canonical
scripted controller. The fresh controller follows different routes and
therefore diverges from teacher states; its tuple differences are expected
trajectory divergence, not evidence of input-construction failure.

All 512 routing-logit gradients per arm are finite and nonzero. Every
hard successor equals its selected candidate exactly. Illegal forward
mass, illegal-logit gradient, and HALT-logit gradient are zero. There are
ten successors, excluding HALT. Direct gradients of unselected candidate
states are zero, as expected for the hard forward/ST weight formulation.
Selected successor, live, and anchor gradient magnitudes are recorded
per decision. Controller norm_h, norm_memory_read, first layer, and
output layer all receive finite signals across the sequence. Relevant
bank computations, memory, Prelude, commit, memory write, and output
head are reached in the structural probe. Absent or zero gradients in
unused branches are explicitly retained rather than treated as failures.

| Initialization | Controller gradient L2 mean | Across-draw population SD | Mean pairwise gradient cosine | Minimum teacher-action probability |
|---|---:|---:|---:|---:|
| Fresh | 2.27867 | 5.27596 | 0.43863 | 0.088069 |
| Strong imitation | 0.000014619 | 0.000014237 | 0.32528 | 0.999990 |

These eight draws on one sequence indicate large magnitude variability
from fresh initialization and much smaller gradients for a nearly
saturated warm policy. They do not uniquely identify variance as the
cause of optimization failure, nor measure unbiased gradient accuracy;
ST-Gumbel is the locked surrogate estimator.

## Frozen-backbone route interventions

Thirty-two fixed validation sequences at lengths 4, 8, 12, 16; same
examples across each scripted policy. Every token has eight legal
decisions. These policies are diagnostic controls, not trained routing.

| Policy | Accuracy | CE |
|---|---:|---:|
| Canonical one sweep | 1.00000 | 0.001821 |
| Skip first SWA, finish with REPEAT | 0.68750 | 2.029771 |
| Skip first GDN, finish with REPEAT | 0.53125 | 2.977295 |
| Repeat first GDN, omit final FFN | 0.84375 | 0.883589 |
| Mid-sweep RESTART, partial second sweep | 0.37500 | 1.198829 |

The backbone is sensitive to route deviations, especially GDN omission
and restart/truncation on this small probe. Each intervention changes the
whole within-token route; it does not isolate a single block's unique
causal contribution. Gate-A's locked conclusion is unchanged.

## Phases D–F: controlled replication setup

Implementation checkpoint: `f23a50a`. Seeds 3305, 3306, 3307. Frozen
Stage-1 backbone; fresh versus independently supervised controller
initialization; 100 Task-A updates, batch size one, AdamW lr=1e-4,
weight_decay=0, tau=2, zero budget, eight routing decisions.

Both arms use identical training token sequences and paired Gumbel seeds.
Evaluation examples are shared before/after and across arms/seeds. Train
evaluation uses the last sixteen optimization examples; validation and
an independently sampled in-distribution test each use four sequences
per length 4–16; OOD test uses four per length 17–32. In-distribution
test is a separate draw from Task-A's validation generator, with no
gradient use. Binary-sequence overlaps can occur in small finite spaces.

Independent reload/no-update controls repeat the warm initialization's
evaluation after optimized arms. Reports retain input bits and targets,
before/after CE and accuracy, route distributions and confusion matrices,
first-deviation positions, actual selected block invocation counts,
REPEAT/RESTART rates, distinct blocks visited, per-update parameter
gradient magnitudes, numerical stability, frozen-weight equality, and
result checkpoint fingerprints. Training candidate block invocations
are reported separately from selected-path computation. No efficiency
claim follows merely from fixed depth or route counts.

Primary record: `experiments/runs/stage3_replication_3305_3307_100_v1.json`.
This is the same estimator and task-loss settings as the earlier short
pilot; the longer duration and independent warm-start seeds are explicit.
Completed results are evaluated against the independently reloaded
no-update controls below.

## Engineering status and interpretation boundary

Focused Gate-C tests: 42 passed. Regression excluding two pre-existing
uncommitted Stage-4 test files: 131 passed, 1 skipped. Full worktree
regression: 134 passed, 1 skipped, 2 failed. Both failures remain HALT
execution-count disagreements in unchanged uncommitted Stage-4 code.
The full worktree is not green. A separate clean git archive of committed
checkpoint `bd87568` passes the full suite: 130 passed, 1 skipped. The
worktree has one additional pre-existing Stage-3 test and five Stage-4
tests; three of those Stage-4 tests pass and two fail. Stage 3 is not
declared scientifically complete.
git diff --check passes. CUDA experiments work despite the existing
NumPy and GB10 capability warnings; no environment changes were made.

Infrastructure, route imitation, task-loss discovery, useful route
diversity/generalization, and adaptive-depth learning remain separate
claims. No Stage-4, D768, CLRS, or language experiment is authorized by
these diagnostic results.

## Completed three-seed task-loss comparison

All six primary arms completed 100 updates with finite losses and finite,
nonzero gradients in all six controller parameter tensors. Frozen backbone
weights were bitwise unchanged. Means below use sample standard deviation
across seeds, not uncertainty over the population of Task-A sequences.

| Initialization | Seed | Train accuracy | Validation accuracy | Independent ID accuracy | OOD accuracy |
|---|---:|---:|---:|---:|---:|
| Fresh | 3305 | 0.68750 | 0.36538 | 0.44231 | 0.50000 |
| Fresh | 3306 | 0.31250 | 0.55769 | 0.55769 | 0.53125 |
| Fresh | 3307 | 0.50000 | 0.38462 | 0.48077 | 0.51563 |
| Strong imitation | all three | 1.00000 | 1.00000 | 1.00000 | 1.00000 |

The shared evaluation draws are not class-balanced: validation contains
33/52 positive targets, independent ID 29/52, and OOD 31/64. Their
constant-majority baselines are 0.63462, 0.55769, and 0.51563 respectively.
These controls are in the saved summary; the small evaluation sets support
no statistical claim of performance above random or majority prediction.
The seeds replicate controller optimization on one common frozen backbone,
not independent backbone training or Gate-A replication.

Fresh-controller validation accuracy is 0.43590 ± 0.10591 and CE
1.43238 ± 0.40786. Independent ID accuracy is 0.49359 ± 0.05875;
OOD accuracy is 0.51563 ± 0.01563, CE 1.65302 ± 0.73408.
None of the 3,004 sampled training-token routes across the three fresh
runs was canonical; deterministic endpoint canonical-route fractions
are also zero. Validation routing averages RESTART 0.33558 ± 0.10152,
REPEAT 0.12724 ± 0.05525, 1.78333 ± 0.65105 distinct blocks per token,
and 5.31538 ± 0.81219 actual selected block calls per token. Fewer block
calls accompany failed task performance and support no efficiency claim.

Strong imitation has validation CE 0.00184819, ID CE 0.00180381, OOD
CE 0.00175095, with all accuracies 1.0. Learned-route outputs match the
fixed reference exactly on these examples. Every token uses the canonical
eight blocks; REPEAT and RESTART rates are zero. All three independently
reloaded no-update controls give the same results before and after;
continued task optimization receives no credit for this retained accuracy.
All 3,004 strong-initialization sampled training routes were canonical.
Controller gradients are nonzero but very small for the confident policy.

Lossless full records, including saved datasets and per-update traces,
are stored in `stage3_replication_3305_3307_100_v1.json.gz` and its
`.summary.json` in `experiments/runs/`. The archive decompresses exactly
to the preserved original JSON; SHA256 fingerprints are in the summary.
Result controller checkpoints remain local, with fingerprints in the full
record. These frozen-backbone experiments isolate routing optimization;
they do not test joint backbone learning or another algorithmic task.

## Shorter-supervision control and stochastic evaluation

A second imitation protocol uses 100 supervised updates with unchanged
optimizer/settings and independent seeds 3305–3307. All three controllers
already recover every supervised and held-out deterministic canonical
route. The `partial` artifact name denotes shorter training, not an
incompletely recovered deterministic policy. Their metadata records
100 updates; these are separate experiments from the 5,000-update arm.

Continuing each shorter warm start for 100 Task-A updates retains perfect
deterministic task performance and canonical routes, again identical to
no-update controls. Stochastic training recovers 977/1,025, 975/1,019,
and 909/960 canonical token routes, versus all tokens for strong imitation.
Thus a shorter prior supplies route deviations without changing the
architecture or adding an objective. The corresponding lossless archive
is `stage3_replication_short_imitation_3305_3307_100_v1.json.gz`.

The paired stochastic evaluator uses the actual training=True executor,
fixed tau=2, the same seeds, and six unique sequences of lengths
4, 8, 12, 16, 20, 32 with four noise draws each. It measures 24 trajectories
per seed/arm and counts exhaustive candidate calls separately. Before
continuation, all three shorter warm starts score 23/24. After continuation,
they score 23/24, 23/24, 22/24. No trajectory was corrected; one worsened
at seed3307. Mean accuracy changes from 0.95833 to 0.94444 ± 0.02406;
mean CE changes from 0.26173 to 0.35236 ± 0.15700. Strong no-update
controllers score 24/24, CE 0.00203105. These are small matched probes,
not population estimates or a definitive claim that continuation harms
performance. Record: `stage3_stochastic_continuation_3305_3307_v1.json`.

For an eight-token teacher sequence, the nominal ideal categorical
probability of sampling its complete canonical route is 3.73e-51 at
fresh initialization, 0.999769 for strong seed3401 imitation, and 0.583721
for shorter seed3305 imitation. This product is conditional on teacher
states and uses the ideal Gumbel categorical law; the implementation's
finite clamped random numbers make it a nominal probability. Other useful
routes are not excluded. In the six-sequence paired continuation probe,
that nominal teacher-route probability decreased after Task-A training
for every case at all three shorter-initialization seeds. This motivates
credit diagnostics rather than crediting deterministic retention as learning.

## Hard-metadata credit limitation

The production ST mixture differentiates successor hidden states while
pointer, restart status, and future masks follow the selected hard action.
A focused eight-block additive/identity toy constructs REPEAT and RESTART
with identical local hidden-state successors but different future legal
destinations. Under controlled draws, terminal losses are 1 and 10,000,
yet both branch-logit surrogate gradients are exactly zero. Resampling
only that branch under the ideal categorical law gives a nonzero expected
loss gradient of approximately [-1965.92, +1965.92]. This is a precise
counterexample to complete credit assignment through metadata, not a
claim that the locked ST estimator was implemented incorrectly or that
this toy proves the sole cause of real Task-A failure. The diagnostic
restores the random-number function and never changes experimental noise.
Record: `stage3_metadata_credit_probe_v2.json`.

A separate real-bank probe holds h fixed while contrasting REPEAT versus
RESTART pointer/legality metadata at six eligible teacher states per token.
It samples one validation length-eight sequence and one OOD length-24
sequence, for 192 observations. The next deterministic action changes in
111/192 states for the fresh controller and 192/192 for each strong,
shorter-before, and shorter-after imitation controller. Frozen weights and
in-flight context versions are preserved. For the shorter prior, median
first-layer pointer effects exceed median repeated-live-state effects for
SWA and GQA. These measurements show that metadata matters to future
choices, without estimating its causal contribution to final parity loss.
Record: `stage3_real_metadata_dependency_3305_v1.json`.

The hard metadata derivative is outside the locked hidden-successor
mixture. Softening the pointer or replacing the estimator would be a new
research design decision, so no such production change was made.

## Implementation and provenance checkpoints

- `f23a50a`: retained-gradient measurements and deterministic seeded replication.
- `ea0134a`: independently identified shorter-supervision comparison arms.
- `2a632a6`: nominal canonical-exploration probability and original pilot batch guard.
- `33bf96e`: paired stochastic evaluation with real training-time execution.
- `7d18df3`: hard-metadata counterexample and real-bank dependence controls.
- `3e2e589`: update/data-matched averages of the existing ST gradient samples.

No pre-existing Stage-4 source or tests, backup files, generated bytecode,
or old checkpoints enter these commits. Current authoritative backbone
remains the Stage-1 seed3305 checkpoint. Older route-imitation artifacts
are superseded only by independently reproduced, fingerprinted new files;
they remain preserved and are not used as current evidence.

## Exact next scientific milestone

The matched sampling-variance pilot is complete and supplies no evidence
of a route-discovery improvement. Assess its limitation alongside the
metadata-credit counterexample. Canonical recoverability is established;
useful task-loss route discovery is not. Deterministic retention after
supervised initialization is insufficient because no-update controls
already achieve it, and the paired stochastic probe shows no improvement.

The next source-aligned research decision concerns credit through hard
pointer/mask dynamics. Any proposal that relaxes this metadata or replaces
the estimator changes the locked formulation and requires an explicit
specification decision before implementation. Until that decision, valid
work consists of targeted diagnostics and explicitly identified controlled
optimization studies using the existing primitive. Neither these failures
nor imitation success authorizes Stage 4, D768, CLRS, or enwik8.

## Matched sampling-variance pilot

Implementation `3e2e589`; fresh frozen-backbone controllers, seeds
3305–3307, 20 updates each, one versus eight independent ST samples per
update. The eight-draw arm averages gradients of the same existing task
CE and ST primitive before one AdamW update; a focused real-model test
checks that averaging equals the independently measured gradients.
Backbone, tau=2, lr=1e-4, zero budget, eight decisions, and input sequences
are unchanged. Arms share the first draw of each update, and this pilot
uses a newly documented per-update noise rule, distinct from the earlier
100-update experiment. It is data/update matched and not compute matched.

| Seed | Draws/update | Validation accuracy | Independent ID accuracy | OOD accuracy |
|---|---:|---:|---:|---:|
| 3305 | 1 | 0.50000 | 0.46154 | 0.56250 |
| 3305 | 8 | 0.38462 | 0.42308 | 0.48438 |
| 3306 | 1 | 0.63462 | 0.55769 | 0.48438 |
| 3306 | 8 | 0.63462 | 0.57692 | 0.48438 |
| 3307 | 1 | 0.38462 | 0.46154 | 0.51563 |
| 3307 | 8 | 0.44231 | 0.50000 | 0.51563 |

One-draw OOD accuracy is 0.52083 ± 0.03932, CE 1.84282 ± 1.26822;
eight-draw OOD accuracy is 0.49479 ± 0.01804, CE 1.99761 ± 1.23026.
All six runs are finite, preserve frozen weights, and have zero canonical
sampled training routes and zero canonical endpoint routes. Candidate
calls are 15,120/120,960, 15,912/127,296, 12,240/97,920 by seed and arm:
exactly eight times as many for eight samples. At seed3305's first update,
individual gradient norms range from 0.000144 to 103.395, with averaged
norm 11.984 and RMS sample dispersion 34.807. This demonstrates variability
but not an unbiased task-objective gradient.

No useful routing improvement is established by this 20-update pilot.
It does not eliminate variance as a contributor or rule out a longer
averaging protocol. Averaging retains the hard-metadata credit limitation;
no new estimator, route penalty, or architectural change is introduced.
The lossless record is `stage3_variance_ablation_3305_3307_20_v1.json.gz`
and its readable `.summary.json`. This pilot preserves seeded data/noise
and traces; unlike the main 100-update trials, it does not save endpoint
controller weights, so an independent endpoint reproduction requires
rerunning the recorded 20 updates.
