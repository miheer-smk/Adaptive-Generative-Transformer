# AGT-v3 Gate C Closeout

> Historical infrastructure checkpoint. Current replicated experiments and
> unresolved scientific limitations are recorded in [STAGE3_VALIDATION_RECORD.md](STAGE3_VALIDATION_RECORD.md).
> Stage 3 is not scientifically closed.

## Status

Gate C implementation is complete.

## Locked checkpoints

- 59ae46f — differentiable routing primitives and source-aligned masks
- 28ecd3f — differentiable executor loop
- 7cbb0e3 — deterministic routing semantics validation
- 8373bfc — differentiable routing connected to AGTv3

## Validation

- Gate-C tests: 26 passed at the final AGTv3 integration checkpoint
- Full regression: 115 passed, 1 skipped
- Deterministic forward semantics: PASS
- Real AGTv3 controller gradient flow: PASS

## Scientific outcome

The frozen Stage-1 seed3305 backbone preserves 100% validation and test
accuracy under the canonical fixed route.

The controller-only task-loss experiment with an 8-decision routing budget
did not recover useful routing. After 100 optimization steps:

- validation accuracy: approximately 49.04%
- test accuracy: approximately 50.39%

The controller exhibited substantial RESTART usage and did not preserve the
validated fixed-route computation.

A route-imitation diagnostic demonstrated that the controller can partially
learn the canonical routing policy, reaching approximately 93.29% offline
route accuracy, but the deterministic executor did not fully recover the
canonical route.

Therefore:

**Gate C implementation: PASS**

**Positive E2 / learned-routing scientific result: NOT ESTABLISHED**

No claim of successful learned topology should be made from Gate C alone.

## Boundary

Gate C remains fixed-depth, D64, differentiable routing only.

No adaptive halting, D768 scaling, CLRS, enwik8, MoE, hypernetwork
modulation, or new GDN mechanics are included in this gate.

## Next milestone

Gate-C scientific validation remains open. Complete controlled route
recovery, task-loss ablations, and seed robustness before advancing to
Stage 4. The provisional Gate-D specification is not authorization to
skip this milestone. See `GATE_C_AUDIT.md` for the subsequent audit.
