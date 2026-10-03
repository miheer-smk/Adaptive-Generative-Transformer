# AGT-v3 — Gate B
# Stage-3 / Stage-4 Source Extraction and Repository Gap Analysis

**Status: SPECIFICATION EXTRACTION ONLY**

No Stage-3 implementation, controller training, ST-Gumbel code, routing-loss code,
temperature schedule, D768 modification, GDN modification, or adaptive-routing
experiment is authorized by this document.

---

# 0. SOURCE AUTHORITY AND CHRONOLOGY

The located AGT-v3 materials do not contain one document explicitly titled
"AGT-v3 Stage 3 Specification" or "AGT-v3 Stage 4 Specification".

Therefore this document does **not** fabricate numbered stages.

The source set is reconciled chronologically and semantically.

## S1 — Current macro-architecture correction

**Source:**
`Pasted text(20260929-195559).txt`

This is the latest located AGT-v3 architecture correction and is treated as the
current macro-architecture authority where it conflicts with older design material.

Source-defined current architecture:

- heterogeneous computational blocks
- state-conditioned routing
- adaptive within-token recurrence
- persistent cross-token latent state
- restartable monotone sweeps
- unbounded-in-principle within-token computation depth
- one unified state-conditioned execution policy
- actions:
  `ADVANCE(j')`, `REPEAT`, `RESTART`, `HALT`
- `ADVANCE` requires `j' > j`
- `REPEAT` keeps the same block
- `RESTART` sets `j <- 0` while preserving live hidden state
- `HALT` terminates internal computation
- convergence is a stopping signal
- categorical relaxation may be used during training
- hard categorical execution at inference
- no top-k simultaneous core execution
- terminal persistent-state update
- persistent state:
  `H_t = (s_t, M_{t+1}, C_t)`
- current source explicitly rejects the old separate PonderNet/global-halt topology.

## S2 — AGT-v3 architectural research/adversarial memo

**Source:**
`Pasted markdown(20260913-113624).md`

This is an adversarial research memo containing architectural analysis,
hypotheses, experimental gates, and a "minimum implementable AGT-v3".

Important distinction:

> This memo contains recommendations/proposals in addition to architectural
> semantics. They are source-derived research proposals, not automatically
> locked architecture.

Relevant sections:

- P — Experimental Plan
- Q — Success / Failure Criteria
- R — Minimum Implementable AGT-v3
- S — Long-Term Architectural Path
- T — Final Decision

## S3 — Older internal-block blueprint

**Source:**
`Pasted text(20260911-094508).txt`

This older blueprint explicitly described:

- local PonderNet recurrence
- soft segment readout
- separate topology controller
- separate global HALT
- straight-through Gumbel-Softmax
- sequence-level routing
- strict forward topology
- `j_{k+1} > j_k`

However, the later S1 architecture correction explicitly says those older mechanisms
are obsolete and replaces them with one unified policy and restartable monotone
sweeps.

Therefore S3 is retained as historical source material but is **not current
macro-architecture authority** where it conflicts with S1.

## S4 — AGT-v3 literature / architecture revision memo

**Source:**
`AGT_v3_Literature_DeepDive_and_Architecture_Revision.md`

This source describes the current conceptual direction and explicitly labels
several items as provisional/open rather than locked.

---

# 1. NUMBERED STAGE STATUS

## Stage 3

**Classification: [PARTIALLY SPECIFIED]**

The located sources define the Stage-3-relevant macro architecture and the proposed
minimum experiment, but they do not provide a complete stage-specific training
contract containing every controller dimension, feature, temperature, annealing
schedule, loss coefficient, gradient estimator detail, and D64 benchmark protocol.

The exact source-derived content is therefore extracted below without filling gaps.

## Stage 4

**Classification: [UNSPECIFIED AS A NUMBERED STAGE]**

No located AGT-v3 source defines a numbered "Stage 4" whose scope is explicitly
"adaptive-depth" with a complete training/inference specification.

The source instead defines later architectural stages:

- **v3.5** — state-conditioned router retrofitted onto a frozen 7–8B model
- **v4a** — richer state/memory, conditional on H6
- **v4b** — state-conditioned block behaviour, conditional on H4
- **v5** — relaxation of monotonicity if the v3 routing regime saturates
- **v6** — scaling

These are extracted separately in Section 10.

---

# 2. STAGE-3 SOURCE-DERIVED SPECIFICATION

## 2.1 Stage-3 objective

**[PARTIALLY SPECIFIED]**

The current architecture defines the objective direction as learning adaptive
computation through:

`heterogeneous block bank + one state-conditioned execution policy +
persistent cross-token state + adaptive within-token computation`

The research memo states the central empirical question as whether adaptive
heterogeneous computation produces a useful, depth-scalable, difficulty-dependent
trajectory under matched compute.

The minimum v3 hypothesis is:

> a model learns to select a variable-length path through heterogeneous
> computational modules conditioned on state that persists across tokens.

However, no unique mathematical Stage-3 objective function is specified.

**UNSPECIFIED:** exact scalar training objective for Stage 3.

---

## 2.2 Controller input

**[SPECIFIED]**

Current macro-architecture source gives:

\[
\pi_\psi
\left(
a_{t,p}\mid h_{t,p},M_t,j_{t,p},b_p
\right)
\]

where:

- `h_{t,p}` = live within-token hidden state
- `M_t` = persistent latent memory
- `j_{t,p}` = current block index
- `b_p = p/B` = optional bounded budget-relative conditioning

Important: the source calls `b_p` optional.

Therefore the source requires the policy to be state-conditioned and exposes
persistent memory to routing, but does not force every optional feature into the
implementation.

---

## 2.3 Controller architecture

**[PARTIALLY SPECIFIED]**

The source requires **one shared / unified policy**.

The research memo's minimum implementable architecture uses one unified action policy.

No canonical hidden width, number of layers, activation function, normalization,
or feature-concatenation layout is defined by the current source.

**UNSPECIFIED:**

- controller hidden dimension
- number of controller layers
- controller activation
- exact controller feature vector
- exact embedding for block index
- exact memory-read projection used by the controller
- whether budget-relative conditioning is enabled in the first implementation

---

## 2.4 Controller output

**[SPECIFIED]**

The current source defines:

\[
a_{t,p}\in
\{
\operatorname{ADVANCE}(j'),
\operatorname{REPEAT},
\operatorname{RESTART},
\operatorname{HALT}
\}.
\]

The current architecture explicitly requires one unified execution policy.

---

## 2.5 Action space

**[SPECIFIED]**

Exactly:

- `ADVANCE(j')`
- `REPEAT`
- `RESTART`
- `HALT`

No second PonderNet controller and no separate global halt controller are part of
the current macro-architecture.

---

## 2.6 ADVANCE semantics

**[SPECIFIED]**

\[
j' > j
\]

`ADVANCE` selects a later block in the current monotone sweep.

---

## 2.7 REPEAT semantics

**[SPECIFIED]**

\[
j'=j
\]

The current block is executed again.

---

## 2.8 RESTART semantics

**[SPECIFIED]**

\[
j\leftarrow0,\qquad h\text{ preserved}.
\]

Restart changes the reachable block index while preserving the live latent
computational state.

The source explicitly forbids a direct unrestricted backward edge.

---

## 2.9 HALT semantics

**[SPECIFIED]**

HALT terminates internal computation and transfers control to terminal state
commit / output processing.

The current source explicitly does not introduce a second halt controller.

---

## 2.10 Execution-mask constraints

**[PARTIALLY SPECIFIED]**

Source-defined constraints:

- forward advancement must be monotone: `j' > j`
- repeat stays at the same index
- restart resets index to zero
- one next action/block is selected per transition
- no top-k simultaneous core execution
- no unrestricted backward/cyclic topology
- no architectural upper bound on `P_t`

**UNSPECIFIED:**

- exact tensor mask implementation
- exact legality-mask representation
- whether HALT is masked at p=0 or other edge cases
- exact maximum serving/training budget policy
- exact anti-consecutive-RESTART rule

---

# 3. ST-GUMBEL / CATEGORICAL ROUTING

## 3.1 ST-Gumbel formulation

**[PARTIALLY SPECIFIED]**

The older internal-block blueprint explicitly specified:

> STRAIGHT-THROUGH GUMBEL-SOFTMAX

The current later architecture correction relaxes this to:

> categorical relaxation during training

and optionally:

> Gumbel-Softmax

with hard categorical execution at inference.

Thus the conceptual estimator family is source-supported, but the exact current
ST formulation is not fully re-specified in the latest correction.

---

## 3.2 Logits

**[PARTIALLY SPECIFIED]**

A controller produces action logits / a categorical policy over the allowed
actions.

**UNSPECIFIED:**

- exact logits tensor layout
- exact masking order
- whether masked logits are `-inf` or another sentinel
- exact normalization point
- exact parameterization of `ADVANCE(j')` logits

---

## 3.3 Soft / straight-through behavior

**[PARTIALLY SPECIFIED]**

Training may use a categorical relaxation; the older blueprint specifically
named straight-through Gumbel-Softmax.

Inference uses hard categorical execution.

The source does not fully specify the backward estimator in the current correction.

---

## 3.4 Temperature

**[UNSPECIFIED]**

No canonical Stage-3 temperature value is specified by the located sources.

---

## 3.5 Temperature initialization

**[UNSPECIFIED]**

No canonical initial temperature is specified.

---

## 3.6 Temperature annealing schedule

**[UNSPECIFIED]**

No canonical annealing schedule is specified.

---

# 4. ROUTING LOSSES / REGULARIZATION

## 4.1 Routing loss

**[UNSPECIFIED]**

No unique Stage-3 routing loss equation is defined.

The research memo discusses routing evaluation and optimization risks but does not
establish a final canonical routing-loss formula.

---

## 4.2 Entropy regularization

**[PARTIALLY SPECIFIED]**

The minimum implementable v3 in Section R of S2 proposes:

- batch-marginal entropy regularizer with coefficient `η`

However:

- `η` is not numerically specified
- this proposal is not present as a locked formula in the latest S1 correction

Therefore entropy regularization is a source-derived proposal, not a resolved
locked hyperparameter.

---

## 4.3 Deliberation / computation cost

**[PARTIALLY SPECIFIED]**

Section R of S2 proposes:

- deliberation cost `κ`

The exact definition and coefficient are not fully specified.

The current S1 architecture establishes adaptive computation but does not provide
the scalar cost equation.

---

## 4.4 Adaptive-depth loss

**[UNSPECIFIED]**

No final adaptive-depth loss term or mathematical formula is specified by the
located AGT-v3 source set.

---

## 4.5 Auxiliary losses

**[PARTIALLY SPECIFIED]**

S2 discusses experiments such as:

- sparse intermediate-depth supervision
- representation / trajectory consistency
- overthinking protection
- recurrence-stability diagnostics

but explicitly describes some of these as experiments, provisional ideas, or
open choices rather than mandatory permanent losses.

Therefore there is no final locked auxiliary-loss package.

---

## 4.6 Loss coefficients

**[UNSPECIFIED]**

No canonical numerical values are provided for:

- routing coefficient
- entropy coefficient
- deliberation coefficient
- adaptive-depth coefficient
- auxiliary-loss coefficients

---

# 5. GRADIENT FLOW / PARAMETER STATUS

## 5.1 Gradient flow

**[PARTIALLY SPECIFIED]**

Source requirements:

- categorical relaxation may provide training gradients
- hard categorical execution is used at inference
- recurrent state must remain parameter-consistent
- cached states generated by previous optimizer-step parameters must not be used
  as differentiable state inputs

The research memo's state-consistency contract additionally says:

- no silent detachment across the token axis
- explicit BPTT truncation must be reported if used
- route replay must account for policy staleness in off-policy settings
- halt-state values must not be stale
- KV state generated by old parameters must not be reused after parameter updates

No final Stage-3 BPTT truncation length is specified.

---

## 5.2 What is trainable

**[PARTIALLY SPECIFIED]**

The architecture requires trainable policy parameters `ψ`.

The minimum v3 memo also assumes trainable heterogeneous block parameters during
the staged training programme.

No final universal freeze/unfreeze schedule is defined by the current source.

---

## 5.3 What remains frozen

**[PARTIALLY SPECIFIED]**

Gate A / Stage-1 explicitly froze routing/controller optimization.

That does NOT establish the Stage-3 freeze set.

The current sources do not provide one final authoritative Stage-3 list of
frozen versus trainable modules.

---

## 5.4 Training-time behavior

**[PARTIALLY SPECIFIED]**

The current source defines:

- adaptive policy during optimization
- categorical relaxation may be used
- hard execution is not the conceptual training-only route
- one selected action transition per conceptual execution step
- persistent state is carried across tokens
- terminal commit persists final token state

S2 additionally proposes staged controller unfreezing and deep-start initialization.

Those proposals are not fully locked in S1.

---

## 5.5 Inference-time behavior

**[SPECIFIED]**

Inference is:

- hard categorical execution
- one selected action/block per transition
- monotone advancement
- repeat when selected
- restart to block zero while preserving latent hidden state
- halt when selected / stopping criterion satisfied
- terminal persistent-state update
- next-token generation from updated persistent state

---

# 6. D64 EXPERIMENTAL SPECIFICATION

## 6.1 D64 setup

**[UNSPECIFIED]**

The located current macro-architecture does not define a canonical D64 Stage-3
configuration.

S2 contains small-scale experimental guidance but does not define one complete
D64 Stage-3 specification.

---

## 6.2 Dataset / task

**[PARTIALLY SPECIFIED]**

S2 recommends Level-1 / Level-2 task families and emphasizes tasks with known
difficulty variation.

Examples discussed include:

- parity
- modular arithmetic chains
- sorting
- cellular automata
- Sudoku-like constraint propagation
- compositional/OOD synthetic reasoning
- multi-hop retrieval

However no single Stage-3 mandatory dataset/task is locked.

The Gate-A parity task is a Stage-1 fixed-route task and must **not** be silently
reinterpreted as the final Stage-3 benchmark.

---

## 6.3 Route constraints

**[SPECIFIED]**

Current source:

- monotone forward advancement
- repeat allowed
- restart allowed
- halt allowed
- no unrestricted backward/cyclic topology
- no top-k simultaneous execution

---

## 6.4 Number of blocks

**[PARTIALLY SPECIFIED]**

Current source uses symbolic:

\[
B_1,\ldots,B_N.
\]

S2 Section R proposes an 8-block heterogeneous bank for a minimum v3 implementation.

However S1 explicitly says exact block species and ratios are not permanently
locked.

Therefore `N=8` is a source-derived proposed implementation value, not a fully
resolved macro-architecture invariant.

---

## 6.5 Evaluation metrics

**[PARTIALLY SPECIFIED]**

S2 Q.2 requires, for the routing non-triviality experiment:

- test performance
- mean and seed variation
- estimated `I(input ; route)`
- correlation between executed depth and known difficulty
- per-block utilization
- route entropy

P.6 additionally requires compute matching including:

- active parameters/token
- training FLOPs
- inference FLOPs/token
- peak memory
- wall-clock latency
- mean/p95 executed blocks
- mean repeats
- mean distinct blocks
- mean cycles
- KV bytes/token

No single consolidated Stage-3 metric table is formally locked.

---

## 6.6 Baseline comparisons

**[PARTIALLY SPECIFIED]**

The memo's corrected arm ladder includes:

- A: static dense transformer
- A': static transformer at equal active parameters
- A'': MoE at equal active parameters
- B: adaptive duration only
- B': fixed-depth loop at matched mean depth
- C: topology only
- D: duration + topology
- D_hom: homogeneous control
- E1: persistent state without state-conditioned routing
- E2: persistent state + state-conditioned routing
- R_ctrl: frozen random path of equal length
- L_ctrl: path determined by sequence length only

S2 explicitly says D vs R_ctrl vs L_ctrl is the central routing non-triviality
gate.

---

## 6.7 Success criteria

**[PARTIALLY SPECIFIED]**

The strongest source-defined gate is Q.2:

> the router should learn a non-trivial, input-dependent, difficulty-correlated
> policy and beat null controls at matched compute.

S2 H2 / H6 additionally require evidence for dynamic topology and state-conditioned
topology.

No exact numeric Stage-3 benchmark threshold is universally locked beyond the
source's hypothesis-specific formulations.

---

## 6.8 Failure / falsification criteria

**[PARTIALLY SPECIFIED]**

Source-defined kill criteria include:

- **KILL-1:** dynamic topology approximately matches random/length-only controls at
  matched compute
- **KILL-2:** majority of seeds fail to obtain non-trivial routing despite
  optimization safeguards
- **KILL-3:** state-conditioned routing provides no measurable benefit over the
  persistent-state baseline
- **KILL-4:** AGT approximately matches a simpler looped model at matched
  active parameters and sequential steps

The source also identifies router collapse as the primary optimization failure
mode.

---

## 6.9 Transition criteria toward later stages

**[PARTIALLY SPECIFIED]**

S2 Section S states:

- v3 proceeds only if Q.2 passes and KILL-1/KILL-2 are cleared
- v3.5 is a cheap scale signal
- v4a richer memory occurs only if H6 passes
- v4b state-conditioned block behaviour occurs only if H4 passes
- v5 relaxes monotonicity only if the v3 monotone regime is demonstrably
  saturating
- v6 scaling follows consistent iso-FLOP evidence at multiple model sizes

No source-defined "Stage 3 -> Stage 4" transition is named in those terms.

---

# 7. STAGE 4 — EXACT SOURCE STATUS

## 7.1 Numbered Stage 4

**[UNSPECIFIED]**

No located AGT-v3 source defines a complete numbered Stage 4 specification.

Therefore the following items are NOT filled by inference:

- exact Stage-4 objective
- exact Stage-4 controller behavior
- exact Stage-4 adaptive-depth mechanism
- exact Stage-4 loss package
- exact Stage-4 computation cost
- exact difficulty definition
- exact training procedure
- exact inference procedure
- exact experiment design
- exact baselines
- exact success criteria

---

# 8. LATER SOURCE-DEFINED ARCHITECTURAL PATH

The source does define later versions, which must not be mislabeled as a numbered
Stage 4.

## v3.5 — Frozen large-model routing signal

**[SPECIFIED]**

Retrofit a state-conditioned router onto a frozen 7–8B open-weight model.

Gate:

> routing remains non-trivial at scale.

---

## v4a — richer persistent state

**[SPECIFIED]**

Only if H6 passes.

Possible state directions named in the source:

- Titans-style test-time-learned memory
- mutable KV state

Gate:

> measurable gains on horizon-scaled tasks.

Exact implementation remains open.

---

## v4b — state-conditioned block behaviour

**[SPECIFIED]**

Only if H4 passes.

Named mechanism:

\[
\theta_j+\Delta(M_t)
\]

using low-rank parameter modulation.

Source labels this as:

- grade-2 novelty
- large optimization risk

Exact implementation not locked.

---

## v5 — topology relaxation

**[SPECIFIED]**

Only if the monotone routing regime is demonstrably saturating.

Source proposes relaxing monotonicity and using the outer-loop/sweep factorization
as the schedulable implementation.

This is explicitly later than v3.

---

## v6 — scaling

**[SPECIFIED]**

Only after v3–v5 produce consistent iso-FLOP advantage at multiple model sizes.

The source identifies iso-FLOP scaling curves as the deliverable.

---

# 9. REPOSITORY GAP ANALYSIS AT GATE-A-COMPLETE

Reference repository state:

`09bed25c79b223f49893fb4f9c2038534498a5e6`

## A. ALREADY IMPLEMENTED

### A1 — Unified action vocabulary

Repository:

`src/controller.py`
`Controller.ACTIONS`

Current repository contains:

- ADVANCE
- REPEAT
- RESTART
- HALT

Matches S1 action semantics.

### A2 — Execution controller object

Repository:

`src/controller.py`
`Controller`

A shared controller module exists.

However its exact input/output representation is still provisional.

### A3 — Adaptive execution machinery

Repository:

`src/executor.py`
`AdaptiveExecutor`

Repository can execute action sequences with:

- ADVANCE
- REPEAT
- RESTART
- HALT

and applies execution/mask constraints.

### A4 — Restart semantics

Repository:

`src/executor.py`

Restart preserves the live hidden state while resetting execution position.

This matches S1.

### A5 — Adaptive model entry point

Repository:

`src/agt.py`
`AGTv3.adaptive_step()`

The adaptive execution path is integrated into the model.

### A6 — Persistent cross-token state

Repository:

`src/agt.py`
`src/terminal_commit.py`
`src/caches.py`

Persistent state includes:

\[
H_t=(s_t,M_{t+1},C_t).
\]

This matches S1 at the architectural level.

### A7 — Terminal commit

Repository:

`src/terminal_commit.py`

The repository enforces terminal persistent-state update and exactly-once commit.

Matches S1.

### A8 — No top-k simultaneous core execution

Repository:

`src/executor.py`

Execution is represented as one selected action transition rather than simultaneous
top-k block execution.

Matches S1.

### A9 — Monotone forward advancement

Repository:

`src/executor.py`
execution legality / mask handling

Matches the current macro-architecture requirement.

### A10 — Convergence module exists

Repository:

`src/convergence.py`

A deterministic convergence gate exists.

This is architecturally aligned with the existence of convergence-based stopping,
but the exact stopping observable conflicts with the current source, described below.

---

# 10. B. MISSING FROM THE GATE-A-COMPLETE REPOSITORY

## B1 — Stage-3 categorical relaxation

Repository has controller logits, but no confirmed canonical ST-Gumbel training
implementation at the Stage-3 level.

**Missing:** exact source-authorized categorical relaxation training path.

Source basis:
S1 §17 / S3 §7.

---

## B2 — Stage-3 training objective

No complete Stage-3 routing objective exists.

**Missing:**

- source-defined routing loss
- entropy term
- deliberation term
- adaptive-depth term
- coefficients

Source status:
mostly unspecified in Gate B.

---

## B3 — Stage-3 controller training loop

Repository only contains the Stage-1 training loop with controller disabled:

`experiments/train_stage1_taskA.py`

There is no authoritative Stage-3 controller training procedure.

---

## B4 — D64 Stage-3 benchmark

The repository's validated D64 parity experiment is fixed-route Stage 1.

There is no source-authorized Stage-3 routing benchmark.

---

## B5 — Required routing diagnostics

The source requires / strongly emphasizes:

- route entropy
- per-block utilization
- `I(input ; route)`
- executed depth versus difficulty
- route collapse monitoring
- compute-matched comparisons

These are not yet implemented as a complete Stage-3 evaluation suite.

---

## B6 — Random-path / length-only routing controls

S2 Q.2 specifically identifies:

- `R_ctrl`
- `L_ctrl`

as decisive routing controls.

They do not exist as a completed Gate-B / Stage-3 evaluation package.

---

## B7 — Multi-seed routing study

Source Q.2 calls for multiple seeds.

No Stage-3 routing study exists.

---

## B8 — Difficulty-correlated adaptive computation test

The current Gate-A parity setup proves fixed-route competence and mechanism relevance,
but does not test whether routing depth tracks known problem difficulty.

---

# 11. C. CONFLICTING / REQUIRES CHANGE OR EXPLICIT DECISION

## C1 — Old PonderNet vs current unified controller

Older source S3:

- separate local PonderNet
- separate global HALT
- soft segment readout

Current source S1:

- one unified policy
- REPEAT / RESTART / HALT are policy actions
- no separate PonderNet controller
- no separate global halt controller

**Resolution status:** S1 supersedes S3.

No code change should be made yet because Gate B is specification-only.

---

## C2 — Old sequence-level routing vs current token-indexed architecture

S3 states:

> Current prototype routing is sequence-level.

S1 defines token-indexed recurrent execution:

\[
h_{t,p}
\]

and persistent state across token transitions.

S2 P.4 / R then recommend token-level routing.

**Resolution status:** the later sources supersede the older sequence-level prototype.

Repository's adaptive per-token path is therefore closer to current architecture than
the old S3 blueprint.

---

## C3 — Controller dimensionality discrepancy

Repository:

`src/controller.py`

Current implementation is a provisional architecture whose controller output is
dimensioned for the action space at the repository's model dimension.

A prior AGT design artifact referenced a substantially different feature/controller
shape (historically described as something like `1803 -> 256 -> 11`), whereas the
current repository uses a `D -> D -> 11`-style controller.

The located current S1 source does **not** define the 1803-dimensional feature
vector or a final controller hidden dimension.

**Resolution status:** unresolved.

This must NOT be silently reconciled.

---

## C4 — Controller input / persistent-memory conditioning

S1 explicitly defines the policy as conditioned on:

\[
(h_{t,p},M_t,j_{t,p},b_p)
\]

and S2 Section R emphasizes:

> router conditioned on `read(M_t)`.

The repository has a controller implementation, but there is not yet a completed
Gate-B-verified demonstration that the controller receives the exact source-defined
memory-conditioned feature set.

**Resolution status:** requires explicit implementation review before Stage 3.

---

## C5 — Convergence observable

S1 specifies:

\[
\delta_{\mathrm{out},t,p}
=
\|\ell_{t,p+1}-\ell_{t,p}\|
\]

and:

\[
\delta_{\mathrm{out}}\rightarrow0
\Rightarrow
\operatorname{HALT}.
\]

Repository:

`src/convergence.py`

currently implements deterministic convergence from state representations / normalized
state change rather than the source's explicitly written output-logit difference.

**Resolution status:** conflict requiring explicit decision.

Do not silently substitute one for the other.

---

## C6 — Architectural depth ceiling

S1 explicitly requires:

\[
P_t\in\mathbb N
\]

and:

> NO ARCHITECTURAL UPPER BOUND ON `P_t`.

S2 Section R proposes:

> `P_max = 24` as a training safeguard.

Repository executor currently contains an execution-depth safety mechanism.

**Resolution status:**

The distinction between:

- architectural maximum
- training safeguard
- serving budget

must be resolved explicitly.

A finite safeguard must not be described as an architectural bound.

---

## C7 — Action set mismatch in older minimum-v3 memo

S2 Section R writes the minimum policy as:

`{ADVANCE(j'>j), REPEAT, HALT}`

while S1 later defines:

`{ADVANCE(j'), REPEAT, RESTART, HALT}`.

**Resolution status:** S1 is later and therefore authoritative.

RESTART is part of current architecture.

---

## C8 — PonderNet terminology

S3 presents PonderNet local recurrence as a distinct mechanism.

S1 explicitly removes the old PonderNet architecture and instead treats repeat and
halting as unified policy actions.

**Resolution status:** use current unified terminology for implementation.

---

# 12. ALL CURRENTLY UNSPECIFIED ITEMS

The following must remain unspecified unless a source is subsequently located:

1. exact Stage-3 scalar objective
2. exact controller hidden dimension
3. exact controller feature vector
4. exact controller activation / normalization
5. exact memory-read feature projection for routing
6. exact use of budget-relative conditioning
7. exact logits parameterization and masking tensor
8. exact ST-Gumbel formula under current architecture
9. temperature
10. temperature initialization
11. temperature annealing schedule
12. routing-loss formula
13. entropy coefficient
14. deliberation-cost coefficient
15. adaptive-depth-loss formula
16. auxiliary-loss package
17. exact loss-weight coefficients
18. exact BPTT truncation length
19. exact Stage-3 trainable/frozen module set
20. exact D64 Stage-3 task
21. exact D64 training dataset size
22. exact D64 sequence/task protocol
23. exact fixed block count as a permanent lock
24. exact Stage-3 baseline table
25. exact benchmark threshold
26. exact route-information estimator
27. exact route-entropy definition
28. exact depth/difficulty correlation statistic
29. exact Stage-4 specification
30. exact Stage-4 adaptive-depth loss
31. exact Stage-4 controller behaviour
32. exact Stage-4 training schedule
33. exact Stage-4 inference budget
34. exact Stage-4 baseline suite
35. exact Stage-4 success threshold

These are intentionally **not filled in**.

---

# 13. REQUIRED IMPLEMENTATION CHANGES AFTER GATE-B APPROVAL

These are consequences of the source/code comparison, not implementation instructions
for the current Gate-B phase.

## Required before Stage-3 implementation

1. Resolve the controller feature/dimension discrepancy.
2. Define a source-consistent routing feature interface including persistent-memory
   conditioning if S1/S2 is retained as authoritative.
3. Define the exact categorical-relaxation estimator.
4. Define the exact routing objective and coefficient set.
5. Resolve convergence observable: logits versus state change.
6. Formalize training-safeguard versus architectural depth semantics.
7. Implement route diagnostics.
8. Implement `R_ctrl` and `L_ctrl`.
9. Define a difficulty-controlled D64 task family.
10. Define a reproducible multi-seed protocol.
11. Freeze the source-derived Stage-3 contract before modifying model code.

---

# 14. HUMAN DECISIONS REQUIRED

These decisions are intentionally left open:

### Decision H1 — Stage-3 source authority

Should the latest S1 macro-architecture correction be treated as the final
authority over the older S3 PonderNet blueprint?

The present document records that chronology but does not make a new scientific
architecture choice.

### Decision H2 — Controller dimensionality

Which source, if any, supplies the missing historical `1803 -> 256 -> 11`
specification?

Until that source is recovered, the final dimensions remain unresolved.

### Decision H3 — Categorical estimator

Does Stage 3 require the older explicit ST-Gumbel formulation from S3, or only the
broader current S1 requirement that training may use a categorical relaxation?

### Decision H4 — Convergence

Should Stage 3 implement the explicitly written output-logit convergence signal,
or is the repository's state-change convergence intended to supersede it?

### Decision H5 — Depth safeguard

What finite execution safeguard, if any, is permitted while preserving the source's
"No architectural upper bound" claim?

### Decision H6 — Exact Stage-3 task

Which task/difficulty family is intended as the mandatory D64 routing gate?

### Decision H7 — Exact Stage-4 interpretation

What document, if any, defines the numbered Stage 4?

Until identified, no Stage-4 implementation should be started.

---

# 15. GATE-C IMPLEMENTATION BOUNDARY

Gate C may begin only after explicit approval.

The boundary is:

**Gate C may implement only the source-approved D64 differentiable routing
mechanism required to test state-conditioned topology.**

Gate C must NOT silently introduce:

- new GDN mechanics
- new memory hierarchy
- D768 scaling
- real-world dataset training
- outer reasoning loops
- arbitrary backward routing
- second halting controller
- MoE routing
- hypernetwork parameter modulation
- unspecified losses
- unspecified controller dimensions
- unspecified temperature schedules

The first Stage-3 implementation target should be the smallest experiment capable
of testing:

\[
E2:
\text{persistent state + state-conditioned routing}
\]

against the source-mandated controls, especially:

\[
R_{\mathrm{ctrl}}
\]

and

\[
L_{\mathrm{ctrl}}.
\]

The source's own Q.2 gate is the decisive reason for this boundary.

---

# 16. GATE-B CONCLUSION

Gate B establishes:

1. The latest located source defines a coherent current macro-architecture.
2. The older PonderNet + separate-global-halt design is superseded by the later
   unified action policy.
3. Persistent cross-token state and state-conditioned routing are central source-defined
   mechanisms.
4. Categorical relaxation is source-supported, but the exact ST-Gumbel implementation
   is not fully specified by the latest source.
5. Temperature, annealing, routing loss, entropy coefficient, deliberation coefficient,
   adaptive-depth coefficient, and several controller dimensions are not fully specified.
6. A complete numbered Stage 4 specification was not located.
7. The repository contains substantial execution infrastructure but lacks the full
   source-authorized Stage-3 training/evaluation contract.
8. The controller dimensionality discrepancy remains unresolved and must not be guessed.
9. The convergence observable contains a real source/repository mismatch and must not
   be silently reconciled.
10. No Stage-3 implementation is justified until the unresolved human decisions above
    are explicitly settled.

**Gate B ends here.**

No training.
No ST-Gumbel implementation.
No routing-loss implementation.
No D768.
No GDN modification.
No Stage-3 experiment.

---

# 17. SOURCE INDEX

### S1 — Current macro architecture correction
`Pasted text(20260929-195559).txt`

Primary current semantics:

- unified state-conditioned policy
- persistent `H_t`
- ADVANCE / REPEAT / RESTART / HALT
- monotone advancement
- restart with hidden-state preservation
- unbounded-in-principle within-token depth
- convergence stopping
- categorical relaxation / hard inference
- no top-k execution

### S2 — Architectural research / adversarial memo
`Pasted markdown(20260913-113624).md`

Primary research-plan content:

- P / experimental plan
- Q / hypotheses and kill criteria
- R / minimum implementable v3
- S / later architectural path
- T / final research framing

### S3 — Older internal-block blueprint
`Pasted text(20260911-094508).txt`

Historical design content:

- local PonderNet recurrence
- soft segment readout
- separate global HALT
- sequence-level routing
- ST-Gumbel
- monotone routing

S3 is retained for provenance and change tracking, not current authority.

### S4 — Literature / architecture revision
`AGT_v3_Literature_DeepDive_and_Architecture_Revision.md`

Primary role:

- architectural stabilization
- recurrence / routing risk analysis
- provisional/open distinction
- current research direction.

---

# FINAL GATE-B STATUS

**Gate A:** COMPLETE
**Gate A commit:** `09bed25c79b223f49893fb4f9c2038534498a5e6`
**Gate A tag:** `AGT-v3-GateA-complete`

**Gate B:** SPECIFICATION EXTRACTION COMPLETE
**Stage 3:** PARTIALLY SPECIFIED
**Stage 4:** NOT FOUND AS A NUMBERED SPECIFICATION
**Implementation:** NOT STARTED
**Training:** NOT STARTED
**D768 changes:** NONE
**GDN changes:** NONE

**STOP. Await explicit approval before Gate-C implementation.**
