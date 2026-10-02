# AGT-v3 Stage 1 — Task A Specification

## Status
LOCKED FOR STAGE 1 BACKBONE TRAINING

The roadmap explicitly identifies "parity with varying length" as a Level-1
algorithmic task family and requires a difficulty parameter that varies per
example. The roadmap does not specify the exact Task-A encoding or numeric
split boundaries, so the following is the explicit implementation
formalization used for this experiment.

## 1. Task

Binary sequence parity.

Given

    x = (b_1, b_2, ..., b_L),  b_i in {0,1}

predict

    y = (sum_i b_i) mod 2

## 2. Difficulty

    D(x) = L

## 3. Input representation

Each binary token is mapped to a frozen deterministic feature vector in R^d.

Feature seed:
    3002

The feature vectors are not trainable and are not part of AGT-v3's parameter count.

## 4. Target / supervision

The target is one binary class:

    y in {0,1}

Loss is applied only after the final input token.

Loss:
    CrossEntropyLoss(final_logits, y)

No intermediate-token supervision is used.

## 5. Sequence-length splits

TRAIN:
    L uniformly sampled from {4, ..., 16}

VALIDATION:
    L uniformly sampled from {4, ..., 16}

TEST:
    L uniformly sampled from {17, ..., 32}

Reporting bins:

    TEST-EASY-OOD:   L in {17, ..., 24}
    TEST-HARD-OOD:   L in {25, ..., 32}

## 6. Execution contract

Controller:
    DISABLED

Route:
    SWA -> SWA -> GQA -> GQA -> GDN -> GDN -> FFN -> FFN

Execution depth:
    P = 8

Each token receives exactly one complete bank sweep.

Persistent cross-token state is preserved through successive AGTv3.step(...) calls.

## 7. Forbidden Stage-1 mechanisms

No:
    REPEAT
    RESTART
    HALT decisions
    ST-Gumbel
    routing loss
    entropy regularization
    deliberation cost
    adaptive-depth loss

## 8. Training objective

Plain CE only.

The controller is excluded from optimization.

The trainable AGT backbone remains otherwise unchanged.

## 9. Required measurements

1. Gradient coverage
2. Nonzero parameter updates
3. Training CE decrease
4. Tiny-subset overfit
5. Validation accuracy / CE
6. Test accuracy / CE
7. Test accuracy / CE by OOD difficulty bin

## 10. Reproducibility

Task seed:
    3001

Feature seed:
    3002

No architecture changes are permitted during this Task-A experiment.

Any future change to task encoding, difficulty definition, or split ranges
requires a new task specification/version.
