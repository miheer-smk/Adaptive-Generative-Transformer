# AGT-v3 — Gate B
# Stage-3 / Stage-4 Source Extraction + Repository Gap Analysis

**Status:** SPECIFICATION EXTRACTION ONLY

No Stage-3 implementation, routing implementation, ST-Gumbel implementation,
loss implementation, controller training, D768 modification, or GDN modification
is authorized by this document.

---

# 0. GATE-B PROVENANCE

## Gate-A frozen point

- Gate-A branch: `research/gdn-state-mechanism`
- Gate-A commit:
  `09bed25c79b223f49893fb4f9c2038534498a5e6`
- Gate-A tag:
  `AGT-v3-GateA-complete`
- Immutable AGT-v3 integration base:
  `c5657630781ba463ce9238a1285d35e96a71ce75`

Gate A is closed.

The Gate-A artifacts and successful checkpoint are preserved.

## Repository state used for this analysis

This specification compares the source material against repository state at:

`09bed25c79b223f49893fb4f9c2038534498a5e6`

No claim in this document authorizes modification of that commit.

---

# 1. ORIGINAL AGT-v3 SOURCE MATERIAL LOCATED

The search located multiple AGT-v3 design documents in the project Library.

## S1 — CURRENT MACRO-ARCHITECTURE CORRECTION

**Document:** `Pasted text(20260929-195559).txt`

**Role:** latest explicit architecture correction; current macro-architecture authority.

The document explicitly states that the obsolete AGT-v3 figure must be replaced by the
current locked architecture:

> heterogeneous computational blocks
> + state-conditioned routing
> + adaptive within-token recurrence
> + persistent cross-token latent state
> + restartable monotone sweeps
> + unbounded-in-principle computation depth

It explicitly distinguishes:

- within-token recurrence:
  `h_{t,0} -> ... -> h_{t,P_t}`
- cross-token persistence:
  `H_{t-1} -> H_t -> H_{t+1}`

It explicitly removes:

- local PonderNet halt
- separate global halt controller
- soft-segment readout as a separate controller/mechanism
- unrestricted dynamic topology
- direct backward/cyclic routing

It explicitly requires one unified state-conditioned execution policy and the action set:

`ADVANCE(j')`, `REPEAT`, `RESTART`, `HALT`.

It explicitly states that no architectural upper bound exists on `P_t`, while a serving
budget may still be an external condition.

It explicitly says training may use categorical relaxation / optionally Gumbel-Softmax and
inference uses hard categorical execution.

It explicitly forbids top-k simultaneous core execution.

## S2 — DETAILED SEPTEMBER ARCHITECTURE / RESEARCH MEMO

**Document:** `Pasted markdown(20260913-113624).md`

**Role:** earlier detailed AGT-v3 architecture / controller / training memo.

This source contains the most detailed mathematical treatment of:

- unified routing
- Gumbel routing
- halting marginalization
- routing loss
- deliberation cost
- route entropy
- staged training
- compute-matched baselines
- success/failure criteria

However, it is chronologically older than S1 and contains details that S1 later retracts or
changes. Those conflicts are explicitly recorded below rather than silently reconciled.

## S3 — DYNAMICAL / EXTENSIBILITY DECISION MEMO

**Document:** `Pasted markdown(20260913-120909).md`

**Role:** later decision memo addressing extensibility, convergence, state consistency,
restartable sweeps, depth extrapolation, and the distinction between architectural
extensibility and empirical improvement.

It explicitly establishes:

- no step-indexed parameters
- no step-indexed positional/depth embedding
- no P-dependent normalization
- no hard P_max mask in the halt distribution
- bounded persistent state
- bounded KV growth with terminal-only writes
- recall / h0 reinjection
- primary convergence-based stopping
- budget-randomized training
- restartable monotone sweeps
- difficulty-dependent effective depth as the key empirical question

## S4 — LITERATURE / ARCHITECTURE REVISION

**Document:** `AGT_v3_Literature_DeepDive_and_Architecture_Revision.md`

**Role:** later literature synthesis.

This document explicitly treats the AGT macro-architecture as fixed in spirit while
keeping microarchitecture provisional/open. It recommends, as later research/training
directions:

- recurrence stabilization
- tail-aware rollout curriculum
- model-induced-state/self-forcing training
- intermediate-depth supervision
- overthinking diagnostics
- compute-matched evaluation

These are not silently promoted to locked Stage-3 parameters.

---

# 2. SOURCE VERSION RECONCILIATION RULE

The source set is not a single perfectly versioned Stage-3 document.

Therefore this Gate-B specification uses the following authority rule:

1. S1 (`20260929-195559`) controls the latest macro-architecture.
2. S3 (`20260913-120909`) controls later corrections to the earlier detailed memo where it
   explicitly retracts or changes a design choice.
3. S2 (`20260913-113624`) supplies earlier detailed formulas and experimental prescriptions
   only where they have not been superseded.
4. S4 is treated as literature-driven revision / research guidance, not as permission to
   invent missing locked implementation values.

Older details that conflict with newer source text are marked as historical/revised, not
merged into the current specification.

---

# 3. STAGE-3 SOURCE STATUS

The located AGT-v3 sources do **not** contain a standalone section literally titled
"Stage 3 Specification".

Accordingly, "Stage 3" is reconstructed from the current-v3 architecture definition
plus the detailed v3 controller/training material.

Each item below is classified exactly as:

- `[SPECIFIED]`
- `[PARTIALLY SPECIFIED]`
- `[UNSPECIFIED]`

For an item not actually defined by the source, the exact phrase is used:

`UNSPECIFIED — not defined by the original AGT-v3 source.`

---

# 4. STAGE 3 — ITEMIZED SPECIFICATION

## 4.1 Stage-3 objective

**[SPECIFIED]**

Establish AGT-v3 with:

- heterogeneous computational blocks
- one state-conditioned execution policy
- adaptive within-token computation
- persistent cross-token latent state
- restartable monotone sweeps
- computation depth with no architectural upper bound

The research question is whether useful adaptive heterogeneous computation can be learned
and whether effective computation depth becomes input/difficulty dependent.

---

## 4.2 Controller input

**[PARTIALLY SPECIFIED]**

Latest source gives the policy in the form:

`π_ψ(a_{t,p} | h_{t,p}, M_t, j_{t,p}, b_p)`

with:

`b_p = p / B`

described as optional bounded budget-relative conditioning.

Older S2 material additionally describes pooled multi-stream state and block-position
embedding as controller inputs.

The later S3 memo explicitly retracts the old step-in-sweep feature `φ(p)` and replaces it
with budget-relative conditioning.

Therefore:

- live hidden state: specified
- persistent memory: specified
- current block position: specified
- optional budget-relative conditioning: specified
- exact feature construction / pooling / concatenation implementation: not fully locked

---

## 4.3 Controller architecture

**[PARTIALLY SPECIFIED]**

The source requires one unified state-conditioned execution policy.

Older detailed S2 material describes an MLP-like controller producing action logits.

Exact current hidden width, number of intermediate layers, activation, normalization,
parameterization, and final dimensionality are not locked by the latest architecture
correction.

`UNSPECIFIED — not defined by the original AGT-v3 source.` for any missing exact value.

---

## 4.4 Controller output

**[SPECIFIED]**

The unified policy outputs one next action:

`ADVANCE(j')`
`REPEAT`
`RESTART`
`HALT`

No second controller is part of the current architecture.

---

## 4.5 Action space

**[SPECIFIED]**

`a_{t,p} ∈ { ADVANCE(j'), REPEAT, RESTART, HALT }`

---

## 4.6 ADVANCE semantics

**[SPECIFIED]**

`j' > j`

The next executed block is strictly forward in block index within the current sweep.

---

## 4.7 REPEAT semantics

**[SPECIFIED]**

`j' = j`

The current block is executed again.

---

## 4.8 RESTART semantics

**[SPECIFIED]**

`j <- 0`

while:

`h is preserved`

The restart is a state-preserving sweep boundary, not a direct backward edge.

---

## 4.9 HALT semantics

**[SPECIFIED]**

HALT terminates internal within-token computation and proceeds to terminal commit /
emission.

---

## 4.10 Execution-mask constraints

**[SPECIFIED]**

The current architecture requires:

- monotone forward advancement inside a sweep
- no unrestricted backward jumps
- no direct backward/cyclic topology
- restart as the legal sweep boundary
- one selected transition per conceptual execution step
- no top-k simultaneous core execution

---

## 4.11 ST-Gumbel formulation

**[PARTIALLY SPECIFIED]**

The source says categorical relaxation may be used during training and specifically names
Gumbel-Softmax as optional training relaxation.

Older S2 material says the ADVANCE branch is trained with straight-through Gumbel-Softmax.

Exact current straight-through estimator equations are not defined in S1.

---

## 4.12 Logits

**[PARTIALLY SPECIFIED]**

A policy logit vector exists.

Older S2 formulation gives:

`ℓ_p = π_ψ([Pool(h_p); emb(j_p); m_p; φ(p)])`

but this uses the later-retracted `φ(p)` feature and therefore is not current.

The current source does not lock an exact final logit dimensionality in the latest correction.

---

## 4.13 Soft / straight-through behavior

**[PARTIALLY SPECIFIED]**

Training:

- categorical relaxation may be used
- Gumbel-Softmax may be used
- older S2 explicitly describes straight-through Gumbel for forward routing

Inference:

- hard categorical execution is specified

Exact soft/straight-through implementation for every action type is not completely
specified.

---

## 4.14 Temperature

**[UNSPECIFIED]**

`UNSPECIFIED — not defined by the original AGT-v3 source.`

No final scalar temperature is locked by the latest correction.

---

## 4.15 Temperature initialization

**[UNSPECIFIED]**

`UNSPECIFIED — not defined by the original AGT-v3 source.`

---

## 4.16 Temperature annealing schedule

**[UNSPECIFIED]**

`UNSPECIFIED — not defined by the original AGT-v3 source.`

---

## 4.17 Routing loss

**[PARTIALLY SPECIFIED]**

Older S2 material gives:

`L_route = κ · E[Σ_p 1[j_{p+1} != j_p]] - η · H(π̄)`

and interprets:

- `κ` as deliberation/switching cost
- `η` as batch-marginal route entropy / load-balancing term

The later/current source does not restate these exact coefficients or declare them current
locked constants.

Therefore the existence/form of these historical terms is source-derived, but exact
current coefficients are not locked.

---

## 4.18 Entropy regularization

**[PARTIALLY SPECIFIED]**

Older S2 source specifies a negative batch-marginal action entropy term:

`-η H(π̄)`

Latest current correction does not lock `η`.

Exact coefficient:

`UNSPECIFIED — not defined by the original AGT-v3 source.`

---

## 4.19 Deliberation / computation cost

**[PARTIALLY SPECIFIED]**

Older S2 source explicitly specifies a switching/de deliberation cost proportional to
changes of execution position.

Current source preserves the unified policy and adaptive computation but does not lock a
final numerical coefficient.

---

## 4.20 Adaptive-depth loss

**[PARTIALLY SPECIFIED]**

The later source strongly specifies adaptive depth as an empirical objective, but the
exact final adaptive-depth objective is not fully locked.

The dynamical memo instead emphasizes:

- convergence-based stopping
- depth-randomized training
- multi-depth supervision
- difficulty-dependent effective depth

A single canonical "adaptive-depth loss" formula is not fixed by the latest correction.

---

## 4.21 Auxiliary losses

**[PARTIALLY SPECIFIED]**

Older sources mention:

- halt-distribution KL
- route-switching / deliberation cost
- route entropy
- optional state norm regularization
- deep supervision in the later training revision

These do not all survive as current mandatory locked terms.

Exact final auxiliary-loss inventory:

`UNSPECIFIED — not defined by the original AGT-v3 source.`

---

## 4.22 Loss coefficients

**[UNSPECIFIED]**

No final locked values are specified for:

- halt coefficient
- entropy coefficient
- deliberation coefficient
- deep-supervision coefficient
- adaptive-depth coefficient
- any optional state-norm coefficient

---

## 4.23 Gradient flow

**[PARTIALLY SPECIFIED]**

Source requires a differentiable training path through relaxed categorical routing and
emphasizes state-consistency.

The source also says:

- state used in gradient computation must be generated under current parameters
- stale caches are invalid
- detaching recurrent state changes the gradient
- route replay in off-policy settings requires rerouting or correction

Exact Stage-3 BPTT truncation length / memory strategy is not locked.

---

## 4.24 What is trainable

**[PARTIALLY SPECIFIED]**

Stage-3 conceptually trains the execution policy and the heterogeneous computational
backbone jointly through a staged procedure.

However the exact freeze/unfreeze schedule in the latest source is not preserved as a
single locked Stage-3 contract.

Older S2 training phases are historical guidance, not current frozen values.

---

## 4.25 What remains frozen

**[PARTIALLY SPECIFIED]**

The following are clearly architectural exclusions/current invariants:

- no GDN redesign
- no outer reasoning loop
- no separate PonderNet/global halt controllers
- no unrestricted backward routing
- no step-indexed block parameters
- no step-indexed iteration embeddings
- no fixed architectural recurrence ceiling

Exact parameter-freeze staging for training is not fully specified.

---

## 4.26 Training-time behavior

**[PARTIALLY SPECIFIED]**

Training may use:

- categorical relaxation
- budget-randomized rollout depth
- multi-depth / deep supervision
- staged routing/controller optimization
- deep-start initialization
- model-induced-state training in later revision

The precise final training schedule is not locked by the latest architecture correction.

---

## 4.27 Inference-time behavior

**[SPECIFIED]**

Inference uses:

- hard categorical execution
- one selected transition at a time
- monotone sweeps
- RESTART as the state-preserving sweep boundary
- convergence as a stopping signal
- learned HALT as part of the unified policy
- terminal state update and next-token emission

The architecture has no fixed architectural upper bound on within-token depth.

---

## 4.28 D64 experimental setup

**[UNSPECIFIED]**

`UNSPECIFIED — not defined by the original AGT-v3 source.`

The source contains architectural scale examples such as D=768, but it does not provide
a canonical D64 Stage-3 experiment specification.

The Gate-A D64 parity experiment is a Stage-1 validation artifact and must not be silently
promoted into Stage 3.

---

## 4.29 Dataset / task

**[PARTIALLY SPECIFIED]**

The older research plan names Level-1 algorithmic families including:

- parity
- modular arithmetic
- sorting
- cellular automata
- Sudoku / constraint tasks

and Level-2 controlled reasoning / OOD tasks.

However no single canonical Stage-3 dataset/task is fixed.

Current source-set status:

`[PARTIALLY SPECIFIED]`

---

## 4.30 Route constraints

**[SPECIFIED]**

Current architecture requires:

- forward monotonicity within sweep
- REPEAT allowed
- RESTART allowed
- HALT allowed
- no unrestricted backward routing
- no arbitrary direct cyclic edge

---

## 4.31 Number of blocks

**[PARTIALLY SPECIFIED]**

Older S2 minimum implementation gives `N=8`.

The current correction preserves a heterogeneous block bank but explicitly warns that the
exact species/ratio should not be treated as permanently locked.

Therefore:

- existence of a heterogeneous bank: specified
- `N=8` as an older implementation target: specified historically
- current permanently locked exact count/species: not fully specified

---

## 4.32 Evaluation metrics

**[PARTIALLY SPECIFIED]**

The source strongly recommends measuring:

- task performance
- inference compute / FLOPs
- executed steps/depth
- route entropy
- per-block utilization
- input-route dependence / mutual information
- depth vs known difficulty
- convergence residual
- overthinking behavior
- seed variance
- memory diagnostics
- latency / memory / KV cost under matched compute

The exact canonical Stage-3 metric table is not frozen.

---

## 4.33 Baseline comparisons

**[PARTIALLY SPECIFIED]**

Older source proposes:

- static dense transformer
- active-parameter-matched static transformer
- homogeneous looped model
- convergence-halted homogeneous model
- heterogeneous fixed-route
- adaptive topology
- random-route control
- length-only route control
- persistent-state ablations
- state-conditioned routing comparison

These are strong source-derived experimental requirements in the research plan, but the
latest macro correction does not define a single mandatory Stage-3 baseline table.

---

## 4.34 Success criteria

**[PARTIALLY SPECIFIED]**

The older detailed research plan gives concrete criteria including:

- learned routing must be non-trivial
- route should depend on input
- compute should correlate with known difficulty
- learned route should beat matched random/length controls
- later depth extrapolation should remain useful
- gains must be evaluated at matched compute

Specific numerical thresholds such as:

`ρ(P_eff, difficulty) > 0.3`

appear in the older research plan.

Because the latest architecture correction does not restate this as a locked Stage-3 gate,
the threshold is historical/source-derived, not an independently authorized new constant.

---

## 4.35 Failure / falsification criteria

**[PARTIALLY SPECIFIED]**

The older source explicitly identifies failure modes:

- route collapse
- shallow-halt collapse
- routing indistinguishable from random/length controls
- heterogeneity adding no value over homogeneous recurrence
- persistent-state routing adding no value
- instability under deeper execution
- no difficulty/depth relationship

Exact Stage-3 statistical decision rules are not fully frozen in the latest correction.

---

## 4.36 Transition criteria to Stage 4

**[UNSPECIFIED]**

`UNSPECIFIED — not defined by the original AGT-v3 source.`

The source contains a long-term progression including v3 and a later v3.5 frozen-model
retrofit, but does not provide an authoritative Stage-4 transition contract by that name.

---

# 5. STAGE 4 — SOURCE EXTRACTION

## 5.1 Critical source finding

The located AGT-v3 source set does **not** contain a document/section that defines a canonical
"Stage 4" specification.

Therefore missing Stage-4 details must not be invented from:

- generic adaptive-depth literature
- PonderNet defaults
- MoE routing
- ANIRA
- Ouro
- FPRM
- personal design intuition

## 5.2 Stage-4 objective

**[UNSPECIFIED]**

`UNSPECIFIED — not defined by the original AGT-v3 source.`

## 5.3 Stage-4 controller behavior

**[UNSPECIFIED]**

`UNSPECIFIED — not defined by the original AGT-v3 source.`

## 5.4 Stage-4 adaptive-depth mechanism

**[UNSPECIFIED]**

The source discusses adaptive within-token depth, convergence-based stopping, and later
test-time depth extension, but does not label an authoritative Stage-4 implementation.

## 5.5 Stage-4 losses

**[UNSPECIFIED]**

`UNSPECIFIED — not defined by the original AGT-v3 source.`

## 5.6 Stage-4 computation cost

**[PARTIALLY SPECIFIED]**

The overall research program explicitly treats compute/depth as an evaluation axis and
discusses deliberation cost. No separate authoritative Stage-4 cost specification exists.

## 5.7 Stage-4 difficulty definition

**[PARTIALLY SPECIFIED]**

Earlier source proposes graded algorithmic tasks with explicit difficulty knobs such as:

- sequence length
- expression complexity
- graph size / reachable subgraph
- Sudoku difficulty
- reasoning horizon

No unique Stage-4 difficulty definition is locked.

## 5.8 Stage-4 training procedure

**[UNSPECIFIED]**

`UNSPECIFIED — not defined by the original AGT-v3 source.`

## 5.9 Stage-4 inference procedure

**[PARTIALLY SPECIFIED]**

The architecture-level inference semantics are specified:

- hard categorical execution
- adaptive trajectory
- convergence stopping
- terminal update

But no Stage-4-specific procedure is defined.

## 5.10 Stage-4 experiment design

**[UNSPECIFIED]**

`UNSPECIFIED — not defined by the original AGT-v3 source.`

## 5.11 Stage-4 baselines

**[PARTIALLY SPECIFIED]**

Earlier source provides a broad baseline ladder, but no canonical Stage-4-only baseline table.

## 5.12 Stage-4 success criteria

**[UNSPECIFIED]**

`UNSPECIFIED — not defined by the original AGT-v3 source.`

---

# 6. IMPORTANT HISTORICAL REVISIONS — DO NOT SILENTLY MERGE

## 6.1 Old `φ(p)` controller feature

Older S2:

`φ(p)` = step-in-sweep feature.

Later S3 explicitly retracts this.

Current status:

**RETRACTED**

Use of step-indexed iteration/depth features in the core architecture is not authorized.

---

## 6.2 Old hard P_max mask

Older S2:

`P <= P_max`

with ADVANCE/REPEAT masked at the bound.

Later S3 explicitly retracts the hard architectural halt mask.

Current status:

**RETRACTED as an architectural requirement**

A deployment/training budget may still exist externally.

---

## 6.3 PonderNet as primary controller

Older S2 uses PonderNet-style halting marginalization.

Later current architecture explicitly removes the separate PonderNet controller and instead
uses one unified policy containing HALT, with convergence as a stopping signal.

Current status:

**Separate PonderNet controller: RETRACTED**

A PonderNet-style probabilistic treatment in older training mathematics remains historical,
not current locked architecture.

---

## 6.4 RESTART

Older detailed Section R omitted RESTART from its reduced action set.

Later current architecture explicitly adds:

`RESTART`

with:

`j <- 0, h preserved`

Current status:

**CURRENT / SPECIFIED**

---

## 6.5 Exact block composition

Older S2 gives an example N=8 composition.

Later S1 explicitly says the exact species and ratios must not be implied as permanently
locked.

Current status:

**Heterogeneity locked conceptually; exact composition provisional.**

---

## 6.6 Temperature / coefficients

Older documents contain formulas with symbols such as:

`β_h`, `κ`, `η`, `β_z`

but do not establish one final current numerical parameter set.

Current status:

**No numerical values should be invented.**

---

# 7. REPOSITORY GAP ANALYSIS AT GATE-A-COMPLETE

Repository reference point:

`09bed25c79b223f49893fb4f9c2038534498a5e6`

---

## A. ALREADY IMPLEMENTED

### A1. Persistent cross-token state

**Repository:**

- `src/agt.py`
- `src/terminal_commit.py`
- `src/caches.py`

The persistent state object contains:

- summary state
- persistent memory
- typed execution context

This matches the architecture-level persistent-state concept.

### A2. GDN recurrent context

**Repository:**

- `src/gdn_v3.py`
- `src/blocks.py`
- `src/terminal_commit.py`
- `src/caches.py`

The GDN recurrent state and terminal commit semantics are already implemented and were
mechanistically validated in Gate A.

No Stage-3 work should modify this mechanism.

### A3. Heterogeneous block bank

**Repository:**

- `src/blocks.py`
- `src/config.py`

The Gate-A repository contains an 8-block heterogeneous bank.

Important mismatch with current source semantics is documented below: the exact ratio is
currently concrete in code while S1 says exact species/ratios are not permanently locked.

### A4. Unified action vocabulary

**Repository:**

- `src/controller.py`

The controller declares:

`ADVANCE`, `REPEAT`, `RESTART`, `HALT`

which is consistent with the latest action vocabulary.

### A5. Adaptive executor machinery

**Repository:**

- `src/executor.py`
- `src/agt.py`

The executor understands the action family and enforces execution constraints.

### A6. Restart / monotone execution machinery

**Repository:**

- `src/executor.py`

The existing executor contains restart handling and execution-mask logic compatible with the
current architecture at the interface level.

### A7. Convergence mechanism

**Repository:**

- `src/convergence.py`
- `src/executor.py`

A convergence module exists and is integrated with executor logic.

### A8. Terminal commit

**Repository:**

- `src/terminal_commit.py`

The repository has a terminal commit path and atomic persistent-state update.

---

# 8. B. MISSING FROM THE CURRENT REPOSITORY

## B1. Stage-3 differentiable routing training

The current repository contains controller/executor interfaces, but no validated Stage-3
training implementation matching the source-derived categorical-relaxation training
contract.

## B2. Authoritative current ST-Gumbel implementation

No validated current ST-Gumbel formulation is established.

The repository must not invent a temperature or estimator.

## B3. Final current routing objective

The repository does not yet contain a source-authoritative implementation of:

- switching/deliberation objective
- route entropy term
- halt-distribution objective
- any final adaptive-depth objective

## B4. Difficulty-correlated compute diagnostics

The repository lacks a completed Stage-3 diagnostic system measuring, in one coherent driver:

- route dependence on input
- depth vs known difficulty
- route entropy
- block utilization
- convergence residual over depth
- seed variance
- matched-compute comparisons

## B5. Source-authoritative Stage-3 experimental harness

`docs/TASK_A_SPEC.md` is Stage 1 fixed-route only.

It cannot be reused as if it were the Stage-3 experimental contract.

## B6. Compute-matched baseline ladder

The repository does not yet implement the source-planned baseline ladder for:

- static
- homogeneous loop
- convergence-halted homogeneous
- fixed-route heterogeneous
- adaptive topology
- random-route control
- length-only control
- persistent-state controls

as one matched evaluation framework.

## B7. Stage-3 route replay / strict recompute contract

The research memo calls for strict current-parameter recomputation of persistent state and
route state. The present repository does not expose this as a completed Stage-3 training
contract.

---

# 9. C. CONFLICTING / REQUIRES CHANGE

## C1. Controller dimensions

**Repository:**

`src/controller.py`

Current implementation is a provisional controller.

The Gate-A D64 parameter audit observed a compact controller of the form:

`D -> D -> 11`

with D64 giving:

`64 -> 64 -> 11`

The current source set does not establish this as the final controller architecture.

Older project roadmap discussion had referenced a different controller dimensionality.

Current status:

**CONFLICTING / REQUIRES EXPLICIT DECISION**

Do not silently reconcile these dimensions.

The final controller feature width, hidden width, and output indexing scheme must be
explicitly fixed before implementation.

---

## C2. Exact action-logit dimensionality

**Repository:**

`src/controller.py`

The action vocabulary is present.

**Source:**

Current source specifies action semantics but does not lock a final serialized logit
layout for all possible `ADVANCE(j')` actions.

Current status:

**PARTIALLY IMPLEMENTED / REQUIRES SPECIFICATION**

---

## C3. Current exact block composition

**Repository:**

`[SWA, SWA, GQA, GQA, GDN, GDN, FFN, FFN]`

**Source:**

S1 says heterogeneity is the macro requirement and warns against presenting exact species
and ratios as permanently locked.

Current status:

**CONFLICTING / REQUIRES EXPLICIT DECISION**

Gate-A fixed-route composition must remain frozen for Gate-A reproducibility, but its status
as the future Stage-3 bank must be explicitly decided.

---

## C4. Convergence role

**Repository:**

`src/convergence.py`

Convergence infrastructure exists.

**Current source:**

Convergence is a stopping signal and diagnostic; there is no second halt controller.

Current status:

**REQUIRES TRAINING/INFERENCE INTEGRATION VALIDATION**

The existence of the convergence module is not the same thing as demonstrating that Stage-3
training/inference satisfies the current convergence semantics.

---

## C5. Gumbel and temperature

**Repository:**

No validated current ST-Gumbel training protocol exists.

**Source:**

Gumbel-Softmax is optional at the architecture level; exact temperature/schedule is not
locked.

Current status:

**MISSING + HUMAN DECISION REQUIRED**

---

## C6. Loss coefficients

**Repository:**

No validated final Stage-3 objective exists.

**Source:**

Symbols exist in historical detailed memos, but no final current numerical values are locked.

Current status:

**MISSING + HUMAN DECISION REQUIRED**

---

## C7. D64 Stage-3 task

**Repository:**

D64 parity exists only as Stage-1 fixed-route validation.

**Source:**

No canonical D64 Stage-3 task is locked.

Current status:

**MISSING + HUMAN DECISION REQUIRED**

---

## C8. D768

The older detailed memo contains D768 implementation examples.

The current Gate-B instruction explicitly forbids touching D768 now.

Current status:

**OUT OF BOUNDARY**

No D768 implementation is authorized by Gate B.

---

# 10. REQUIRED IMPLEMENTATION CHANGES AFTER GATE-B APPROVAL

These are requirements to resolve, NOT implementation instructions.

## 10.1 Resolve controller specification

Explicitly decide:

- controller input feature construction
- controller hidden dimension
- action-logit serialization
- whether `b_p` is enabled
- exact position encoding representation, if any, consistent with the current source

## 10.2 Resolve differentiable routing specification

Explicitly decide, only where source is silent:

- exact ST-Gumbel estimator
- temperature
- temperature schedule
- treatment of RESTART
- treatment of REPEAT
- treatment of HALT during relaxation

## 10.3 Resolve objective

Explicitly decide:

- task-loss formulation
- halt regularization
- switching/deliberation coefficient
- route entropy coefficient
- adaptive-depth term
- any deep-supervision term
- any auxiliary state regularization

## 10.4 Resolve D64 Stage-3 benchmark

Select and freeze:

- task
- difficulty axis
- train/validation/test split
- compute budgets
- evaluation points
- seed count
- baseline arms

No existing Stage-1 parity result should be silently reused as Stage-3 evidence.

## 10.5 Build instrumentation before training

The research source treats diagnostics as a first-class result.

Required instrumentation should measure at minimum:

- route distribution
- route entropy
- per-block utilization
- executed depth
- repeated-block count
- restart count
- convergence residual
- depth vs difficulty
- input-route dependence
- seed variance
- compute / latency / memory

---

# 11. HUMAN DECISIONS REQUIRED

The following are deliberately NOT decided by Gate B:

1. Final controller hidden dimension.
2. Final controller input feature vector.
3. Whether `b_p = p/B` is enabled.
4. Exact ST-Gumbel estimator.
5. Gumbel temperature.
6. Initial temperature.
7. Temperature annealing schedule.
8. Deliberation/switching coefficient.
9. Route entropy coefficient.
10. Adaptive-depth coefficient.
11. Deep-supervision coefficient.
12. Any state-norm auxiliary coefficient.
13. Canonical D64 Stage-3 task.
14. Canonical D64 difficulty variable.
15. Final exact heterogenous block composition for Stage 3.
16. Final Stage-3 success thresholds where the sources do not lock them.
17. Any Stage-4-specific specification.

No value should be inserted merely because it is "reasonable".

---

# 12. PROPOSED GATE-C BOUNDARY

**Gate C should begin only after explicit approval of the unresolved Gate-B decisions.**

Gate C implementation boundary:

### IN SCOPE

- D64 only
- current frozen Gate-A backbone as the starting substrate
- one unified controller
- source-approved action semantics
- differentiable categorical routing
- exact chosen ST-Gumbel protocol
- exact chosen objective
- exact chosen D64 task
- instrumentation-first implementation
- fixed-route and null controls
- deterministic/equivalence tests
- no GDN redesign

### OUT OF SCOPE

- D768 scaling
- real language-model training
- real-world datasets
- new GDN mathematics
- MoE
- outer reasoning loop
- second controller
- unrestricted topology
- arbitrary backward routing
- architectural recurrence ceiling
- unapproved loss terms
- unapproved controller features
- unapproved temperature schedule

### Gate-C stop condition

Do not promote to the next gate merely because training produces a positive number.

Gate C must first establish that:

1. the routing implementation matches the frozen source-derived specification,
2. null / fixed-route / random-route controls behave correctly,
3. gradients flow through the intended routing path,
4. the controller does not trivially collapse,
5. the measured result is reproducible under the registered protocol.

Only then should adaptive-depth performance claims be made.

---

# 13. FINAL GATE-B STATUS

## Located source documents

1. `Pasted text(20260929-195559).txt` — current macro-architecture correction
2. `Pasted markdown(20260913-120909).md` — extensibility / convergence decision memo
3. `Pasted markdown(20260913-113624).md` — detailed older controller / training memo
4. `AGT_v3_Literature_DeepDive_and_Architecture_Revision.md` — later literature synthesis

## Stage 3

A source-derived architecture is identifiable.

However, several critical numerical / implementation details are not source-locked,
especially:

- temperature
- temperature schedule
- final controller dimensionality
- exact feature construction
- final loss coefficients
- canonical D64 Stage-3 benchmark
- final transition gate

## Stage 4

No authoritative Stage-4 specification was located.

Therefore Stage-4 implementation details must remain unspecified pending an actual
source document or explicit human design decision.

## Repository status

The repository already contains substantial infrastructure:

- persistent state
- heterogeneous bank
- unified action vocabulary
- executor
- restart / monotone execution machinery
- convergence module
- terminal commit

The repository does not yet contain a source-authoritative validated Stage-3 differentiable
routing training system.

## Scientific boundary

Gate B ends here.

No Stage-3 implementation is authorized by this document until the unresolved
specification decisions are explicitly approved.

# SOURCE-LOCKED RECONCILIATION

This section is the authoritative Gate-B reconciliation of the original AGT-v3
source materials located in the Library.

It does not authorize implementation.

---

## B.1 Source hierarchy

### S1 — latest macro-architecture correction

Source:

`Pasted text(20260929-195559).txt`

This is the latest architecture correction and therefore has precedence over
older topology/controller formulations where the two conflict.

S1 explicitly specifies:

- one unified state-conditioned execution policy;
- persistent cross-token state;
- adaptive within-token recurrence;
- restartable monotone sweeps;
- unbounded-in-principle computation depth;
- action space
  `{ADVANCE(j'), REPEAT, RESTART, HALT}`;
- `ADVANCE: j' > j`;
- `REPEAT: j' = j`;
- `RESTART: j <- 0` while preserving live hidden state;
- `HALT: terminate internal computation`;
- categorical relaxation during training;
- hard categorical execution at inference;
- no unrestricted backward/cyclic topology;
- no top-k simultaneous core execution.

Source locations:
S1 lines 42–52, 145–170, 236–246, 373–383, 483–501.

Classification: **architecture-level current source**.

### S2 — detailed AGT-v3 mathematical/research memo

Source:

`Pasted markdown(20260913-113624).md`

This document supplies the detailed mathematical formulation, loss design,
training curriculum, experimental hypotheses, kill criteria, and minimum
implementable architecture.

Important qualification:

S2 is a detailed/research specification and recommendation document, not a
later architecture-correction freeze. Where S2 conflicts with S1, the
conflict is recorded below rather than silently reconciled.

Relevant sections:

- F.2/F.3 — controller simplification / outer-loop discussion
- G.1–G.9 — mathematical AGT-v3 formulation
- P.3–P.6 — curriculum, state-consistency and compute matching
- Q.1–Q.3 — hypotheses, gating experiment, kill criteria
- R — minimum implementable AGT-v3
- S — long-term gated architecture path

---

# B.2 Stage-3 specification extraction

## 1. Stage-3 objective

**[SPECIFIED]**

The principal objective is to determine whether heterogeneous conditional
computation with persistent state produces a genuinely useful learned
execution topology.

Operationally, Q.2 asks:

> Does the router learn a non-trivial, input-dependent,
> difficulty-correlated policy and beat the two null controls at matched
> compute?

Source: S2 §Q.2.

---

## 2. Controller input

**[SPECIFIED — WITH A SOURCE CONFLICT]**

S2 G.4 specifies:

`ell_p = pi_psi([Pool(h_p); emb(j_p); m_p; phi(p)])`

where:

- `Pool(h_p)` is the pooled live within-token state;
- `emb(j_p)` is the current block index embedding;
- `m_p` is the memory read;
- `phi(p)` is a position-in-sweep feature.

S1 instead presents the abstract policy as:

`pi_psi(a_{t,p} | h_{t,p}, M_t, j_{t,p}, b_p)`

with `b_p = p/B` described as an optional bounded budget-relative condition.

**Unresolved:** the exact final controller feature representation is not
uniquely fixed across the two sources.

Do not choose between `phi(p)`, `b_p`, raw `h_p`, pooled `h_p`, or any
concatenation beyond what the source explicitly states.

---

## 3. Controller architecture

**[UNSPECIFIED]**

The source specifies the controller's functional inputs/outputs but does not
provide one unambiguously locked MLP width/depth specification.

S2 gives the source-level controller as `pi_psi(...)`, not a unique layer
stack.

Any repository `D -> D -> ...` implementation must therefore be treated as
repository state, not automatically as the canonical Stage-3 architecture.

---

## 4. Controller output

**[PARTIALLY SPECIFIED]**

S1 explicitly defines the unified policy output as:

`{ADVANCE(j'), REPEAT, RESTART, HALT}`.

S2 G.4 defines logits over `N+2` actions corresponding to:

`ADVANCE(j')`, `REPEAT`, `HALT`.

**Conflict:** S1 adds explicit `RESTART`; S2's detailed mathematical action
space omits it.

Final action-logit dimensionality is therefore **not to be silently selected**.

---

## 5. Action space

**[PARTIALLY SPECIFIED]**

Current source hierarchy:

S1:
`ADVANCE(j'), REPEAT, RESTART, HALT`.

S2:
`ADVANCE(j'>j), REPEAT, HALT`.

The current repository's 11-output controller for an 8-block action encoding
must not be declared the source truth until the RESTART discrepancy is
resolved.

---

## 6. ADVANCE semantics

**[SPECIFIED]**

`ADVANCE(j')` requires forward movement:

`j' > j`.

No unrestricted backward jump is allowed.

---

## 7. REPEAT semantics

**[SPECIFIED]**

S1:
the current block is repeated.

S2:
`j_{p+1}=j_p` and the same block is executed again.

---

## 8. RESTART semantics

**[SPECIFIED — CURRENT SOURCE ONLY]**

S1 explicitly defines:

`j <- 0`

while preserving the live hidden state.

This is a current architecture correction.

S2's G.4 does not include RESTART in its mathematical action space.

Therefore RESTART exists in the latest architecture source but is not fully
defined inside the older G/P/Q mathematical training specification.

---

## 9. HALT semantics

**[SPECIFIED]**

HALT terminates within-token computation.

S2 additionally gives a PonderNet-style marginalized halt formulation.

---

## 10. Execution-mask constraints

**[PARTIALLY SPECIFIED]**

S2 explicitly masks actions to preserve monotonicity:

`A(j_p) = {ADVANCE(j'): j'>j_p} U {REPEAT} U {HALT}`

with REPEAT disallowed at `j=0`.

A training/deployment `P_max` is described as a safeguard, not an architectural
depth constant.

S1 additionally requires restartable monotone sweeps and no unrestricted
backward/cyclic routing.

Because RESTART is absent from S2's mathematical mask, its precise mask
interaction is **UNSPECIFIED**.

---

## 11. ST-Gumbel formulation

**[PARTIALLY SPECIFIED]**

S1 specifies categorical relaxation during training and hard categorical
execution at inference.

S2 specifies straight-through Gumbel-Softmax for the ADVANCE branch.

The exact combined four-action relaxation under S1's RESTART formulation is
not specified.

---

## 12. Logits

**[PARTIALLY SPECIFIED]**

S2 gives:

`ell_p = pi_psi([...]) in R^(N+2)`.

That dimension corresponds to the older three-category action formulation.

Under S1's explicit RESTART action, the final logits dimensionality is not
independently specified by the latest source.

---

## 13. Soft / straight-through behavior

**[SPECIFIED — PARTIALLY]**

S2:

- HALT is marginalized through `rho_p`;
- ADVANCE uses straight-through Gumbel-Softmax.

S1:

- categorical relaxation during training;
- hard categorical execution at inference.

The exact soft relaxation of RESTART/REPEAT in the newer four-action system
is **UNSPECIFIED**.

---

## 14. Temperature

**[UNSPECIFIED]**

No exact ST-Gumbel temperature value is locked in the located sources.

---

## 15. Temperature initialization

**[UNSPECIFIED]**

No exact initialization temperature is specified.

---

## 16. Temperature annealing schedule

**[UNSPECIFIED]**

No numeric annealing schedule is specified.

---

## 17. Routing loss

**[SPECIFIED — COEFFICIENTS UNSPECIFIED]**

S2 gives:

`L_route = kappa * E[sum_p 1[j_{p+1} != j_p]]
           - eta * H(pi_bar)`

where:

- `kappa` is a switching/deliberation penalty;
- `eta` weights batch-marginal action entropy.

This is source-defined in form.

The final numerical values of `kappa` and `eta` are **UNSPECIFIED**.

---

## 18. Entropy regularization

**[SPECIFIED — COEFFICIENT UNSPECIFIED]**

S2 explicitly includes batch-marginal route entropy:

`- eta * H(pi_bar)`.

The source does not lock a numerical `eta`.

---

## 19. Deliberation/computation cost

**[SPECIFIED — COEFFICIENT UNSPECIFIED]**

S2 D.4 specifies a switching penalty:

`kappa * 1[j_{k+1} != j_k]`

and ties this to option-collapse prevention.

A PonderNet-style halt prior is also specified.

The numeric `kappa` is **UNSPECIFIED**.

---

## 20. Adaptive-depth loss

**[PARTIALLY SPECIFIED]**

S2 does not define one separately named "adaptive-depth loss."

Instead, depth is handled through:

- PonderNet-style halting marginalization;
- routing/switching cost;
- entropy/load-balancing regularization;
- depth-dependent execution and evaluation.

Therefore a separate additional adaptive-depth loss is **UNSPECIFIED** unless
introduced by a later human decision.

---

## 21. Auxiliary losses

**[PARTIALLY SPECIFIED]**

S2 G.8 includes:

`beta_z * ||Z_t||^2/d`

as an optional mild state-norm regularizer.

S2 also specifies halt KL and route entropy.

The source itself says the state-norm term may be dropped if convex memory
update proves sufficient.

Thus:

- halt KL: specified;
- route switching cost: specified;
- route entropy: specified;
- state-norm regularizer: optional/source-proposed;
- any additional loss: unspecified.

---

## 22. Loss coefficients

**[UNSPECIFIED]**

The symbols are specified (`beta_h`, `kappa`, `eta`, `beta_z`) but the final
numeric coefficients are not locked.

Do not invent values.

---

## 23. Gradient flow

**[PARTIALLY SPECIFIED]**

Source-level constraints:

- straight-through Gumbel is used for ADVANCE;
- HALT admits PonderNet-style marginalization;
- current-parameter states must be recomputed rather than stale;
- truncation boundaries must be explicit;
- route replay must account for policy staleness if used.

The exact implementation of gradient flow for the four-action newer RESTART
policy is **UNSPECIFIED**.

---

## 24. What is trainable

**[SPECIFIED — BY CURRICULUM PHASE]**

S2 P.3:

Phase 1:
- blocks train;
- fixed path;
- fixed depth.

Phase 2:
- router is unfrozen;
- entropy regularization and deliberation cost active;
- halting remains fixed.

Phase 3:
- halting is unfrozen;
- halt-KL warmup.

Phase 4:
- persistent-state gates unfrozen;
- entropy regularization annealed.

Exact frozen parameter subsets outside this curriculum are not fully
enumerated.

---

## 25. What remains frozen

**[PARTIALLY SPECIFIED]**

The staged curriculum implies:

- controller is frozen initially;
- router is frozen during block-only warm-up;
- halting stays fixed until Phase 3;
- persistent-state gates remain frozen until Phase 4.

Exact treatment of every backbone parameter and all memory parameters is
not separately enumerated.

---

## 26. Training-time behavior

**[SPECIFIED — HIGH LEVEL]**

Training is staged rather than fully joint from step zero.

Instrument at phase boundaries:

- route entropy;
- `I(input;route)`;
- mean halt depth;
- `||Delta h||` decay;
- per-block utilization.

Variable-depth and model-induced-state training principles are recommended in
the source program, but exact final rollout distribution is not numerically
locked here.

---

## 27. Inference-time behavior

**[SPECIFIED — HIGH LEVEL]**

S1 specifies hard categorical execution at inference.

The execution process follows the unified policy with monotone sweeps,
restart semantics from S1, and HALT termination.

Exact production serving budget and stop policy are not completely specified.

---

## 28. D64 experimental setup

**[UNSPECIFIED]**

The located Stage-3 minimum implementation in S2 specifies `d=768`, not D64.

The current Stage-1 parity experiment is D64, but it is explicitly a fixed-route
Stage-1 experiment and is not a source-defined Stage-3 setup.

A D64 Stage-3 translation is therefore a human decision, not source fact.

---

## 29. Dataset / task

**[PARTIALLY SPECIFIED]**

S2 identifies Level-1 and Level-2 task families including:

- parity with varying length;
- modular arithmetic;
- sorting;
- cellular automata;
- Sudoku-Extreme;
- compositional OOD;
- multi-hop synthetic knowledge graphs;
- length extrapolation.

The source does not lock one unique Stage-3 benchmark/task.

The central requirement is a controlled difficulty parameter whose variation
can be compared against executed depth.

---

## 30. Route constraints

**[SPECIFIED — WITH RESTART GAP]**

S1 locks monotone sweeps and restartable execution.

S2 locks monotone forward advancement and same-block repeat.

Precise post-RESTART masking rules are unspecified.

---

## 31. Number of blocks

**[SPECIFIED IN MINIMUM IMPLEMENTATION]**

S2 Section R specifies:

`N = 8`.

However, S2 G.1/G.4 formulate the equations generically in `N`.

---

## 32. Evaluation metrics

**[SPECIFIED]**

Required source-level measurements include:

- test performance;
- mean ± seed standard deviation;
- `I(input;route)`;
- correlation of executed depth with known difficulty;
- route entropy;
- per-block utilization;
- mean/p95 executed steps;
- repeats;
- distinct blocks;
- cycles;
- KV bytes per token;
- wall-clock latency;
- training and inference FLOPs;
- peak memory.

---

## 33. Baseline comparisons

**[SPECIFIED]**

The central controls include:

- dense/fixed computation baseline;
- `R_ctrl`: frozen random path of equal length;
- `L_ctrl`: path determined by sequence length;
- homogeneous looped baseline;
- compute-matched static baseline.

S2 explicitly states that capability comparisons should be made at equal
inference FLOPs, with equal training-FLOP comparisons as a secondary axis.

---

## 34. Success criteria

**[SPECIFIED]**

Q.1/Q.2 define the main criteria:

- non-trivial input-dependent routing;
- routing differs from random and length-only controls;
- depth correlates with known difficulty;
- state-conditioned routing improves over the non-routing state-memory arm;
- matched-compute comparisons are required.

H6 specifically predicts:

`E2 > E1`

with measurable route shifts conditioned on persistent memory `M`.

---

## 35. Failure / falsification criteria

**[SPECIFIED]**

Source kill criteria include:

- KILL-1: dynamic topology indistinguishable from random-path control;
- KILL-2: optimization failure across the required seeds;
- KILL-3: state-conditioned routing fails to improve over its ablation;
- KILL-4: AGT is matched by an Ouro-style simpler looped model at equal
  active parameters and sequential steps.

Router collapse / near-zero mutual information between input and route is a
critical failure mode.

---

## 36. Transition criteria to Stage 4

**[PARTIALLY SPECIFIED]**

There is no separately titled "Stage 4" in the located source.

The source instead defines a gated architectural path:

- v3 proceeds only if Q.2 passes and KILL-1/KILL-2 are cleared;
- v4a is unlocked only if H6/state-conditioned-routing evidence is positive;
- v4b is unlocked only if H4/heterogeneity evidence is positive;
- later topology relaxation is deferred to v5 if monotone routing saturates.

Thus a single Stage-4 transition criterion is **not explicitly defined**.

---

# B.3 STAGE 4 — SOURCE EXTRACTION

## Explicitly named Stage 4

**[UNSPECIFIED]**

No located source contains a distinct, authoritative section titled
"Stage 4" with a complete objective/controller/loss/dataset/metric/success
specification.

The closest authoritative long-term source is Section S:

### v4a — state

**[SPECIFIED]**

Unlocked only if H6 passes.

Direction:
richer persistent memory, including possible Titans-style test-time-learned
memory or mutable KV state.

Gate:
measurable gains on horizon-scaled tasks.

### v4b — operator

**[SPECIFIED]**

Unlocked only if H4 passes.

Direction:
state-conditioned block behavior such as low-rank parameter modulation
`theta_j + Delta(M_t)`.

The source does not define a complete training recipe for v4b.

### v5 — topology relaxation

**[SPECIFIED]**

Only after v3 monotone routing is demonstrably saturating.

Relax monotonicity and use the outer-loop-as-sweep factorization as the
servable implementation of that later relaxation.

---

## Stage-4 items requested by this Gate-B protocol

### Objective
**[UNSPECIFIED]**

### Controller behavior
**[UNSPECIFIED]**

### Adaptive-depth mechanism
**[UNSPECIFIED]**

### Losses
**[UNSPECIFIED]**

### Costs
**[UNSPECIFIED]**

### Difficulty definition
**[UNSPECIFIED]**

### Training procedure
**[UNSPECIFIED]**

### Inference procedure
**[UNSPECIFIED]**

### Experiment design
**[UNSPECIFIED]**

### Baselines
**[UNSPECIFIED]**

### Success criteria
**[UNSPECIFIED]**

The source gives only the v4a/v4b/v5 gating directions above.

---

# B.4 REPOSITORY GAP ANALYSIS AT GATE-A-COMPLETE

Repository snapshot:

`09bed25c79b223f49893fb4f9c2038534498a5e6`

## A. ALREADY IMPLEMENTED

### Unified controller scaffold
Repository:
`src/controller.py`

Current source-level role:
controller module and action vocabulary scaffold.

Status:
**IMPLEMENTED AS SCAFFOLD, NOT STAGE-3 TRAINING.**

### Executor
Repository:
`src/executor.py`

Status:
**IMPLEMENTED**

The executor enforces the existing action execution machinery and adaptive-step
integration.

### Adaptive AGT step
Repository:
`src/agt.py`

Status:
**IMPLEMENTED**

`adaptive_step()` exists, but this does not mean learned adaptive routing has
been validated.

### Convergence monitor
Repository:
`src/convergence.py`

Status:
**IMPLEMENTED**

This is compatible with the source concept that convergence can act as a
stopping signal.

### Persistent state / memory
Repository:
`src/agt.py`
`src/caches.py`
`src/terminal_commit.py`
`src/memory.py`

Status:
**IMPLEMENTED**

These support the persistent cross-token state contract already validated by
Gate A.

### Fixed-route D64 task
Repository:
`data/task_a.py`
`experiments/train_stage1_taskA.py`

Status:
**IMPLEMENTED FOR STAGE 1 ONLY**

This must not be reclassified as the Stage-3 routing experiment.

---

# B.5 B. MISSING

### Stage-3 learned controller training
Status:
**MISSING**

No source snapshot at Gate-A-complete contains the complete staged Stage-3
controller-training loop.

### ST-Gumbel routing implementation
Status:
**MISSING**

No Stage-3 ST-Gumbel training path is present in the Gate-A snapshot.

### Source-defined routing losses
Status:
**MISSING**

No Stage-3 implementation of the source-defined switching cost / entropy /
halt-KL objective is present in the frozen Gate-A snapshot.

### Stage-3 route instrumentation
Status:
**MISSING**

Required:

- route entropy;
- `I(input;route)`;
- route/difficulty correlation;
- per-block utilization;
- mean/p95 executed depth;
- repeat count;
- distinct-block count;
- compute matching;
- seed variance.

### R_ctrl
Status:
**MISSING**

No frozen-random-path matched-compute control is present.

### L_ctrl
Status:
**MISSING**

No sequence-length-only route control is present.

### Stage-3 compute-matched benchmark framework
Status:
**MISSING**

The repository does not yet provide the full source-required FLOP/latency/
active-parameter matching protocol.

### D64 Stage-3 benchmark definition
Status:
**MISSING / HUMAN DECISION REQUIRED**

The source does not define a D64 Stage-3 experiment.

---

# B.6 C. CONFLICTING / REQUIRES CHANGE

## Controller action space

Repository:
`src/controller.py`

Source conflict:

- current architecture correction: `ADVANCE / REPEAT / RESTART / HALT`;
- detailed mathematical memo: `ADVANCE / REPEAT / HALT`.

**Requires explicit resolution before implementation.**

---

## Controller output dimensionality

Repository:
`src/controller.py`

Source conflict:

- current repository has a fixed output dimensionality;
- S2 G.4 expresses `R^(N+2)`;
- S1 has a fourth action category via RESTART.

**Do not silently select the final dimension.**

---

## Controller input features

Repository:
`src/controller.py`

Source requires state-conditioned routing using current computation state,
block index, and memory-related conditioning.

Exact repository feature construction must be checked against S2 G.4 and
S1's abstract policy.

**Requires review before implementation.**

---

## RESTART masking semantics

Repository:
`src/executor.py`

Source:
S1 explicitly requires restart while preserving hidden state.

S2 does not mathematically define RESTART.

**Requires explicit specification before implementation.**

---

## P_max

Repository:
`src/executor.py`

Current implementation has a finite execution safeguard.

Source S2 specifies:

`P_max = 24`

as a training/deployment safeguard, not as an architectural constant.

**Current repository value must not be silently assumed to equal the source
value.**

---

## D64 versus D768

Repository Gate-A research:
D64.

Source minimum implementation:
D768.

**This is a scale discrepancy, not an architecture discrepancy. A D64
Stage-3 experiment requires an explicit human-approved translation.**

---

## Heterogeneous block composition

Repository Gate-A:
the experimentally validated fixed bank is:

`[SWA, SWA, GQA, GQA, GDN, GDN, FFN, FFN]`.

Source S2 Section R gives an example composition with genuine architectural
heterogeneity that differs from the Gate-A experimental bank.

Therefore the source does NOT authorize silently replacing the Gate-A bank
before the Stage-3 decision is made.

---

# B.7 HUMAN DECISIONS REQUIRED

The following are intentionally unresolved:

1. Does current S1's four-action `RESTART` replace S2's older three-action
   mathematical formulation for Stage 3?

2. What exact controller architecture is authorized?

3. What final action-logit dimension follows from the resolved action space?

4. Should the Stage-3 experiment be D64 as an explicitly authorized pilot, or
   use the D768 minimum implementation from S2?

5. What exact controller features are allowed if S1 only describes budget
   conditioning abstractly and S2 gives `phi(p)`?

6. What exact ST-Gumbel temperature and annealing schedule should be used?

7. What exact values should `kappa`, `eta`, `beta_h`, and optional `beta_z`
   take, unless a later source is found?

8. Should the S2 PonderNet-style halt marginalization survive unchanged under
   the newer RESTART-inclusive policy?

9. What exact Stage-3 benchmark/task should be selected from the Level-1/Level-2
   family?

10. What precise source-compatible compute-matching protocol is required for
    the first pilot?

No answer is chosen by this document.

---

# B.8 PROPOSED GATE-C BOUNDARY

This is a **boundary, not an implementation authorization**.

Gate C may begin only after the human decisions in B.7 are resolved.

Gate C should be limited to:

- the approved action-space formulation;
- the approved controller input/architecture;
- the approved categorical-relaxation formulation;
- the approved D64 or D768 pilot scale;
- the approved routing losses;
- the approved training schedule;
- mandatory route diagnostics;
- R_ctrl and L_ctrl;
- no GDN modification;
- no memory redesign;
- no D768 change unless explicitly selected as the Stage-3 scale;
- no Stage-4 mechanisms;
- no outer loop;
- no topology relaxation beyond the approved source formulation.

Gate C ends at the first reproducible routing-mechanism result and its
matched-compute controls.

---

# B.9 GATE-B CONCLUSION

Gate B is complete as a specification exercise.

The located original source supports a strong, testable Stage-3 program, but
does NOT support silently filling several implementation gaps.

Most importantly:

**The latest architecture source and the older detailed mathematical source
are not perfectly identical.**

The main unresolved architectural conflict is:

`S1: ADVANCE / REPEAT / RESTART / HALT`

versus

`S2: ADVANCE / REPEAT / HALT`.

Additionally, temperature values, annealing, several loss coefficients, exact
controller architecture, and a D64 Stage-3 setup are not source-locked.

Therefore no implementation should begin until those items are explicitly
resolved.

Gate B ends here.
