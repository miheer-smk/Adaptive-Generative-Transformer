# AGT-v3 Gate C Report

## Scope

Gate C implements fixed-depth differentiable routing using the unified
controller action space and straight-through Gumbel routing.

The validated fixed-depth routing space is:

- ADVANCE(0..N-1)
- REPEAT
- RESTART

HALT remains excluded from successor-state mixing because it has no
successor state. Adaptive halting belongs to Stage 4.

## Implementation checkpoints

- `59ae46f` — differentiable routing primitives, successor mixing, legality masks
- `28ecd3f` — fixed-depth differentiable executor loop
- `7cbb0e3` — deterministic forward-routing semantics
- `8373bfc` — AGTv3 differentiable token path

## Validation

Gate-C tests: 26 passed.

Full regression: 115 passed, 1 skipped.

`git diff --check`: clean at the final clean checkpoints.

The real AGTv3 differentiable path was verified to propagate finite,
nonzero gradients to the controller.

## Task-A controlled results

The validated Stage-1 fixed-route backbone achieved:

- Validation accuracy: 1.0000
- Test accuracy: 1.0000

A controller-only ST-Gumbel Task-A pilot was then run with:

- frozen Stage-1 seed3305 backbone
- 8 routing decisions
- controller-only optimization
- tau = 2.0
- learning rate = 1e-4

The task-loss-only controller did not learn a successful route policy.
After 100 steps:

- validation accuracy: 0.4904
- test accuracy: 0.5039

The deterministic route was dominated by RESTART behavior.

## Route-imitation diagnostic

A teacher-forced route-imitation diagnostic was run against the canonical
fixed route:

`ADVANCE(0), ADVANCE(1), ..., ADVANCE(7)`

After optimization:

- offline route accuracy: 0.932933

However, closed-loop exact route recovery failed. A single routing error
can move the execution state away from the teacher-forced trajectory.

Therefore the 93.29% offline number is not evidence of successful
closed-loop route recovery.

## Gate C conclusion

### Implementation status: PASS

The following are implemented and validated:

- unified controller inputs
- source-aligned legality masks
- ST-Gumbel categorical routing
- differentiable successor-state mixing
- fixed-depth differentiable executor
- deterministic argmax inference
- AGTv3 integration
- controller gradient propagation

### Scientific learned-routing status: NOT ESTABLISHED

The experiments do not justify claiming that a randomly initialized
controller learns an effective task-optimal routing policy on Task A.

This limitation is recorded explicitly rather than being hidden by
additional hyperparameter tuning.

## Transition

Gate-C scientific validation must be completed before Stage 4. The
implementation milestone alone does not authorize adaptive execution
depth or learned HALT. See `GATE_C_AUDIT.md` for the subsequent audit.

No Gate-C result is claimed as evidence for adaptive halting.
