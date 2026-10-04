# AGT-v3 Gate D / Stage 4 — Provisional Specification

## Status

PROVISIONAL — internal working specification.

The exact numbered Stage-4 source specification was not located during Gate B.
This document therefore records an explicit engineering interpretation rather
than claiming historical source fidelity.

## Stage-4 objective

Extend the validated unified-controller routing mechanism from fixed-depth
execution to adaptive computation depth.

The controller must learn:

**whether to continue computation or terminate the current token computation.**

## Existing action space

For N blocks:

- ADVANCE(0 ... N-1)
- REPEAT
- RESTART
- HALT

No second halting controller is introduced.

## Stage-4 semantic change

Gate C:

    fixed routing budget
    +
    HALT excluded from differentiable successor mixing

Gate D:

    variable routing depth
    +
    HALT becomes an executable termination action

HALT has no successor state.

## Preserved invariants

- Unified controller
- Monotone ADVANCE routing within a sweep
- REPEAT semantics
- RESTART remains zero-compute
- RESTART cannot immediately follow RESTART
- Persistent memory/context remain terminal-commit semantics
- No architectural upper bound is claimed
- No new GDN mechanics
- No MoE
- No hypernetwork modulation
- No unrestricted backward routing

## Core unresolved design decision

HALT has no successor state, so its learning signal cannot come from the
successor-state ST-Gumbel mixture used for ADVANCE/REPEAT/RESTART.

Gate D must therefore explicitly define a stopping-value / halting-credit
mechanism before serious training.

## D0 implementation target

First implement and validate HALT-aware deterministic execution and
measurement only.

Required measurements:

- halt position
- realized computation depth
- route trajectory
- task loss
- accuracy
- zero-HALT / always-HALT controls
- fixed-depth comparison

No halting-training loss is introduced until the stopping-value definition is
explicitly locked.

## Initial scientific question

Can adaptive termination reduce realized computation while preserving task
accuracy relative to the validated fixed-depth reference?
