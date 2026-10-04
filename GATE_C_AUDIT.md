# Gate-C research audit — 2026-10-04

## Reconstructed state

Initial HEAD was `9b72638` on `research/stage4-adaptive-halting-gate-d`,
not the supplied `8373bfc` / Stage-3 branch. The Stage-3 branch points to
`2cd9ae7`. Later commits include inference optimization (`5898bd1`),
functional-parity experiment code (`6c0e966`), and closeout reports
(`4644a15`, `2cd9ae7`). Commit titles and old reports are not new evidence.
No applicable AGENTS.md was found in the searched Desktop tree.

Work continues on `research/stage3-routing-gate-c-audit`, branched from
the actual HEAD while retaining the existing worktree. Unrelated caches,
backups, checkpoints, shell artifacts, and uncommitted Stage-4 work were
preserved. The original v2/masked imitation checkpoints were not reused.

## Implementation audit and correction

- Locked bank: SWA, SWA, GQA, GQA, GDN, GDN, FFN, FFN.
- Controller: normalized h and memory read, 256-wide memory slot mean,
  9-wide pointer, 2-wide budget; D64 input 395 -> 256 -> 11.
- Masks implement initial, monotone advance, repeat, and post-restart
  legality. Restart preserves h and resets the pointer; it consumes a
  routing decision but executes no block.
- Successor order is eight ADVANCE states, REPEAT, RESTART. HALT has no
  successor and is excluded in both fixed-depth training and inference.
- Hard pointer updates follow the sampled forward action. Training
  constructs all candidates (including masked candidates); inference
  executes only the selected block, or no block for RESTART.
- In-flight blocks reconstruct ephemeral views from committed context.
  Caller persistent state is cloned. One terminal commit updates stream,
  memory, and every species-specific context from the final state.
  Fixed-route step and legacy adaptive step remain separate and unchanged.

The original relaxed softmax included illegal actions in its denominator
before zeroing them. This violates the specified legal categorical
relaxation, despite hard actions remaining legal. `b54afee` corrects this
using masked log_softmax before Gumbel relaxation; it also avoids clipping
small legal probabilities. A fixed-noise test compares actual gradients
against the legal categorical reference, including extreme logits.
Earlier task-loss experiments used the defective relaxation and should
not be treated as definitive evidence about the specified estimator.

## Engineering validation

Before edits: focused Gate-C tests 29 passed; full suite 121 passed,
1 skipped, 2 failed. After edits: focused tests 31 passed; full suite
123 passed, 1 skipped, the same 2 failed. Excluding the two uncommitted
Stage-4 test files: 120 passed, 1 skipped. git diff --check passed.

The full suite is **not green**. The failures are the uncommitted Stage-4
tests expecting HALT not to increment execution_count, while the new
uncommitted execute_adaptive includes HALT in len(action_history).
That path also references st_gumbel_action without importing it; its
training branch does not implement a relaxed successor mixture. It is
outside current Gate-C authorization and was not developed or repaired.

NumPy initialization and GB10 capability warnings persist; CUDA runs
completed successfully. No environment reinstall was performed.

## Canonical route recovery

The imitation script already had explicit TOKEN -> BLOCK alignment and
an 8-decision lock. Its remaining mismatch was zero budget during
training versus token-position/sequence-length budget during evaluation.
Both now use zeros. Held-out evaluation now uses the supplied sequence,
checks every token, and verifies exactly eight decisions per token;
formerly it generated another sequence and checked only its final token.

Independent imitation runs: seeds 3401, 3402, 3403; 5,000 AdamW steps,
lr=1e-3, no weight decay, 1,024 route rows/update, frozen Stage-1 seed3305
backbone, fresh current controller, 8,320 aligned training route rows.

| Seed | Overall offline | Decisions 0–7 individually | Initial 8-token executor probe | Held-out val tokens | Held-out OOD tokens |
|---|---:|---|---|---|---|
| 3401 | 1.0 | all 1.0 | all exact | 130/130 | 392/392 |
| 3402 | 1.0 | all 1.0 | all exact | 130/130 | 392/392 |
| 3403 | 1.0 | all 1.0 | all exact | 130/130 | 392/392 |

Every recovered route is ADVANCE(0), ..., ADVANCE(7). Held-out samples
are one sequence per length, not an exhaustive binary-sequence test.
Offline accuracy is training-set accuracy. These results establish
supervised canonical-route recoverability across three initializations;
they do not establish task-loss discovery of useful routing or novelty
beyond the fixed route.

## Controlled task-loss diagnostic

`experiments/run_stage3_controlled_audit.py` pairs fresh and imitation
initializations over seeds 3401–3403. Both arms receive identical training
sequences and Gumbel seeds. All use fixed 8-decision depth, batch size 1,
zero budget, tau=2.0, AdamW lr=1e-4 without weight decay, 20 task-loss-only
updates. No entropy, route penalty, new temperature schedule, HALT loss,
or architectural change is added.

Evaluation cases are generated once and reused before/after and across
all arms: 52 validation sequences (520 tokens), 64 OOD sequences (1,568
tokens), four sequences per length. Fixed and learned paths receive the
same examples. The imitation arm in all three pairs uses the same newly
trained seed3401 checkpoint; task-training seeds are distinct from the
three independent imitation runs above.

Full before/after CE, task accuracy, per-decision accuracy, action counts,
exact-route counts, fixed-route comparison, and per-update gradient
coverage are recorded in `experiments/runs/stage3_controlled_audit.json`.

| Task-training seed | Fresh val accuracy / CE | Fresh OOD accuracy / CE | Imitation val accuracy / CE | Imitation OOD accuracy / CE |
|---|---|---|---|---|
| 3401 | 0.5000 / 2.463275 | 0.578125 / 2.448853 | 1.0 / 0.001848186 | 1.0 / 0.001750953 |
| 3402 | 0.596154 / 2.120354 | 0.515625 / 2.443760 | 1.0 / 0.001848186 | 1.0 / 0.001750953 |
| 3403 | 0.634615 / 1.904514 | 0.484375 / 2.831674 | 1.0 / 0.001848186 | 1.0 / 0.001750953 |

All fresh arms recovered zero canonical token routes. All imitation arms
recovered 520/520 validation and 1,568/1,568 OOD token routes, before and
after training; their maximum absolute logit difference from the paired
fixed-route baseline was zero. They visited all eight blocks once per
token, with zero REPEAT/RESTART. Across fresh arms/splits, REPEAT rates
were 2.69–51.59%, RESTART rates 9.72–46.48%, and selected block calls
averaged 4.28–7.22 per token despite eight routing decisions. These count
selected-path block calls, not exhaustive training candidate overhead.
All six controller parameter tensors had finite, nonzero gradients on
every update in all six arms. Gradient existence alone did not yield
successful routing from fresh initialization.

Fresh imitation reproduction logs are preserved as
`experiments/runs/stage3_imitation_audit_seed{3401,3402,3403}.log`.
New local checkpoints are named `stage3_controller_route_imitation_audit_*`;
they remain untracked. The paired experiment used the initial newly
reproduced seed3401 audit checkpoint, not the historical v2 checkpoint.

## Interpretation and next milestone

Canonical-route representation/recovery is established for these pilot
samples. Fresh task-loss optimization and a strong supervised warm start
must remain separate claims. Preserving a supervised route after a short
task-loss run does not demonstrate that task loss discovered that route.
The pilot supports an initialization/exploration difficulty, but does not
yet isolate Gumbel variance, off-route backbone sensitivity, or long-run
credit assignment as the unique mechanism.

Next: retain matched data and the corrected estimator; characterize
stochastic route error rates and first-deviation decisions at initialization,
then compare longer task-loss runs from random and independently trained
imitation initializations. Include no-update controls and paired seeds;
extend evaluation sample counts before claiming seed-stable learning.
This is a controlled experiment decision, not authorization for a new
regularizer, architecture, learned HALT, D768, CLRS, or enwik8.

Gate-C infrastructure is validated with the estimator correction.
Gate-C task-loss learned-routing scientific success remains **not established**.
Gate-A's locked causal conclusion is unchanged.
