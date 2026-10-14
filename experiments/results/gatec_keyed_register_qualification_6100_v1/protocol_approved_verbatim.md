**Finalized prospective protocol: AGT-v3 streaming keyed-register qualification pilot, v1.** This specifies one bounded fixed-program pilot. It does not authorize execution, routing training or confirmation.

The receiver-anchored study remains `COMPLETED_NON_SUCCESS`; its records and interpretation remain unchanged.

**1. Task, tokenization and encoding**

Use eight binary registers, with legal keys `0…7`.

| Command | Symbolic semantics |
|---|---|
| `SET(k,b)` | `R[k] ← b`, where `b∈{0,1}` |
| `XOR(k,b)` | `R[k] ← R[k] XOR b` |
| `COPY(k,j)` | Read the old scalar `R[j]`, then assign it to `R[k]` |
| `QUERY(k)` | Return `R[k]`; leave symbolic registers unchanged |

`COPY(k,k)` is legal and is an identity operation. Registers contain scalar bits, not aliased references. The generator below uses distinct copy source/destination keys.

Each command is **one input token**. Length 64/96 includes eight initialization commands and eight query commands. Answers are output logits, not additional input tokens.

Every episode begins with eight `SET` commands assigning every register exactly once. The symbolic interpreter may initialize registers to zero internally, but no query occurs before all eight assignments.

Use a fixed, nontrainable 64-dimensional FP32 encoding:

| Coordinates | Contents |
|---|---|
| `0:4` | Opcode one-hot: SET, XOR, COPY, QUERY |
| `4:12` | Destination/query-key one-hot |
| `12:20` | Source-key one-hot for COPY; otherwise zero |
| `20:22` | Operand-bit one-hot for SET/XOR; otherwise zero |
| `22:64` | Zero |

No normalization, random embedding, program-ID feature, answer feature or extra trainable encoder is added. The existing Prelude accepts model-width vectors. [Prelude interface](</home/iiitn/Desktop/Miheer Projects/AGT/AGT_v3_POC/src/prelude.py:42>)

Only query positions receive CE loss. Class indices `0/1` represent the queried bit through the existing two-class head.

Inference processes one command at a time. Predictions are read after its normal terminal commit. Labels never enter embeddings, state initialization, routing or subsequent inputs. Earlier predicted answers are not fed back.

**2. Exact episode generation and counterfactual families**

All generation uses independent `random.Random` streams:

| Purpose | Seed |
|---|---:|
| Training families | 6100 |
| Development families | 6101 |
| Training episode order | 6102 |
| Development uncertainty | 6103 |
| Reserved later length-128 confirmation | 6104 |
| Reserved later length-192 confirmation | 6105 |

Encoding consumes no randomness. Model construction retains the validated seed-3305 initialization procedure.

For reproducible draws, define `randbelow(n)` by rejection sampling from `getrandbits(n.bit_length())`; for `n=1`, return zero without a draw. Permutations use descending Fisher–Yates with that helper. Record the Python version.

For each length, generate families in ascending accepted-family order; process length 64 before 96 without resetting the split’s RNG.

For a base episode:

1. Draw a permutation of the eight keys and eight independent initialization bits in key-ID order.
2. Emit the eight initialization SETs in permutation order.
3. Let `a` and `d` be the permutation’s first two keys. These are generator roles, never additional model features.
4. At internal indices 8 and 9, emit `XOR(a,b8)` and `XOR(d,b9)`, drawing both operand bits.
5. Define eight query positions:
   \[
   q_r=8+(r+1)(L-8)/8-1,\quad r=0,\ldots,7.
   \]
6. Shuffle the seven keys other than `a` for the first seven queries; query `a` last.
7. Fill remaining positions, in ascending order, with uniformly chosen SET/XOR/COPY opcodes:
   - SET: uniform destination among the six other keys; uniform bit.
   - XOR: uniform destination among all eight keys; uniform bit.
   - COPY: uniform destination among the six other keys; uniform distinct source among those six.

Thus subsequent SET/COPY commands cannot erase or propagate the two protected differences. Both protected keys can still receive XOR updates.

Each family contains four episodes:

- **V0:** base episode.
- **V1:** V0 with both XOR operands at indices 8/9 flipped.
- **V2:** V0 with every SET value complemented; XOR operands unchanged.
- **V3:** V2 with both XOR operands at indices 8/9 flipped.

The designated counterfactual pairs are V0/V1 and V2/V3.

Within either pair:

- Opcode/key sequences and all commands after index 9 are identical.
- Final-query last-16-command windows are identical.
- Latest SET values are identical.
- Global register XOR and raw operand-bit XOR are identical at query boundaries.
- The final queried answer differs.

V2 complements every query answer relative to V0; V3 complements V1. Consequently, **each query position has exactly balanced labels within every family**.

These are analytically specified properties. Future executable interpreter tests must verify them; none ran here.

| Split | Families per length | Episodes per length | Total episodes |
|---|---:|---:|---:|
| Training | 16 | 64 | 128 |
| Development | 32 | 128 | 256 |

Reject a candidate family if it duplicates another family’s structural command template, complete input, initialization block or complete incoming query history. Apply exclusions across both splits and available comparable historical inputs. Within-family shared initialization/history is intentional.

Common short lexical fragments are allowed; “history disjointness” means complete supervised histories, not every possible one-token prefix.

Cap generation at 100,000 candidate families per split. If quotas cannot be filled, stop without relaxing exclusions. Record rejection reasons and hashes.

Length-128/192 confirmation inputs **must not be generated or evaluated in this pilot**. Their reserved streams will later use the same construction and exclude all documented pilot exposures. Unknown historical exposure remains disclosed.

**3. Programs and existing semantics**

Source inspection confirms the intended monotone advances and repeats:

| Program | Actions | Executed blocks | Selected species counts |
|---|---|---|---|
| Balanced B | `0,1,2,3,4,5,6,7` | Same | 2 SWA, 2 GQA, 2 GDN, 2 FFN |
| Attention-biased A | `0,1,2,3,8,5,6,7` | `0,1,2,3,3,5,6,7` | 2 SWA, 3 GQA, 1 GDN, 2 FFN |
| Recurrence-biased R | `0,1,2,4,8,5,6,7` | `0,1,2,4,4,5,6,7` | 2 SWA, 1 GQA, 3 GDN, 2 FFN |

Action 8 repeats the current pointer. Therefore all three scripts have **D=8, P=8**, with no RESTART or HALT. This is source-confirmed legality, not a newly executed validation. [Masks](</home/iiitn/Desktop/Miheer Projects/AGT/AGT_v3_POC/src/masks.py:68>), [dispatch/accounting](</home/iiitn/Desktop/Miheer Projects/AGT/AGT_v3_POC/src/executor.py:699>)

Names describe selected operations, not demonstrated specialization.

Repeated operations read the committed context with changed live hidden input. They do not create intermediate persistent commits. GDN repetitions do not chain their discarded ephemeral states into successive persistent updates. [GDN contract](</home/iiitn/Desktop/Miheer Projects/AGT/AGT_v3_POC/src/gdn_v3.py:14>)

Preserve normal terminal commits for all contextual entries, including unselected blocks. [Commit construction](</home/iiitn/Desktop/Miheer Projects/AGT/AGT_v3_POC/src/terminal_commit.py:199>)

**4. Model, objective and update schedule**

Initialize one research model using the validated non-controller Stage-1 loader:

`experiments/runs/stage1_parity_seed3305_best.pt`

Verified file SHA-256:

`53b56e7e90f7eb7a6570107b6dfe3a297caac48d637fe27e5f00fbe94c6f09c3`

Configuration:

- D64, 16 memory slots, two output classes.
- Shared existing non-controller parameters trainable.
- Controller frozen and unused.
- No new capacity or program-specific parameters.
- Batch size one; FP32; model in evaluation mode with autograd enabled during training.
- Deterministic scripted, selected-only routing; zero budget and compatibility inputs.
- No ST-Gumbel, exhaustive candidates, auxiliary objective or compatibility loss.

The existing selected-only executor path uses `training=False`; this flag must **not** be confused with disabling autograd. [Validated research loader/path](</home/iiitn/Desktop/Miheer Projects/AGT/AGT_v3_POC/experiments/results/gatec_receiver_anchored_compatibility_6000_v1/study_adapter_v1.py:174>)

Shuffle the canonical 128-episode manifest once with seed 6102. Each episode appears exactly once.

At update \(i\):

\[
L_i=\frac13\sum_{p\in\{B,A,R\}}
\frac18\sum_{r=0}^{7}\mathrm{CE}(\ell_{p,q_r},y_{q_r}).
\]

Run B, A, R in that order, each from an independent normal initial state. Hold parameters constant across all three rollouts.

- Zero gradients once.
- Backpropagate each route loss divided by three.
- Preserve full BPTT through every command and commit.
- Clip the accumulated global gradient norm to 1 once.
- Perform one AdamW step.

AdamW: LR `1e-4`, weight decay `0`, betas `(0.9,0.999)`, epsilon `1e-8`. No scheduler.

Exactly **128 optimizer steps and 384 training backwards**. Finite losses/gradients and actual parameter displacement are required. Unused parameters may legitimately have absent gradients.

Save initial identity and the final update-128 checkpoint. There is no intermediate checkpoint selection, early stopping for task performance or development-driven extension.

**5. Fixed diagnostics and exact accounting**

Choose the first two development families at each length, including all four variants: **16 episodes**, eight per length. Seal this selection before training.

Evaluate these episodes under all three programs at update zero. Final development supplies their matched update-128 results without extra unablated calls.

If the development gates pass, run preliminary ablations on these same 16 episodes.

At every token, execute the fifth selected operation normally, then substitute its incoming live hidden tensor for its returned live hidden output:

- B: suppress GDN-4 output.
- A: suppress the second GQA-3 output.
- R: suppress the second GDN-4 output.

All other selected operations, normal execution counts and terminal commits remain active. Earlier occurrences remain active. Contextual commit work still occurs, and its payloads may change because terminal hidden state changes.

**Ablations are descriptive preliminary diagnostics.** Four independent families cannot support the intended inferential operation-relevance claim. No ablation confidence interval or significance claim is permitted.

The pre-study model-fixture menu is fixed:

| Fixture, per device | Tokens |
|---|---:|
| Retained eight-token canonical control: fixed reference plus B adapter | 16 |
| One initialized token under A and R | 2 |
| Fixed 12-command task fixture under all three programs | 36 |
| Duplicate final-token branch from each retained fixture incoming state | 3 |
| **Per device** | **57** |

Run this menu once on CPU and once on CUDA: **114 tokens**.

The 12-command fixture is eight ascending-key SETs with values `0,1,0,1,0,1,0,1`, followed by `XOR(0,1)`, `COPY(1,0)`, `QUERY(0)`, `QUERY(1)`.

Check route correspondence, complete-state isolation, commits and autograd using disposable models. Allow one state VJP and one native-loss backward per program/device: **12 fixture backward/VJP evaluations**, zero fixture optimizer steps.

For canonical fixtures compare complete fresh reference/adapter states and metadata, using numerical `atol=1e-6`, `rtol=1e-5` and exact discrete comparisons. Check archived CUDA control CE within `1e-6` of `0.0014546061865985394`. This does not claim reproduction of unrelated historical controller monitors.

| Work | Formula | Tokens |
|---|---|---:|
| Training | `128 × 80 × 3` | 30,720 |
| Final development | `256 × 80 × 3` | 61,440 |
| Initial diagnostic | `16 × 80 × 3` | 3,840 |
| Preliminary ablations | `16 × 80 × 3` | 3,840 |
| CPU/CUDA fixtures | `57 × 2` | 114 |
| **Maximum scheduled total** | | **99,954** |

Full completion entails **799,632 bank executions**, **99,954 terminal commits**, and **396 backward/VJP evaluations**.

There are no profiling repetitions or model warm-ups. Any additional model execution is unscheduled. The remaining 46 tokens do not authorize probes, retries or extensions.

Hard limits: **100,000 attempted model-token calls, 800,000 bank calls, and one 7,200-second accelerator allocation**. CPU fixture tokens count toward the token ceiling. The allocation clock includes preparation performed during allocation, logging and analysis; it cannot be paused.

**6. Metrics, uncertainty and qualification decisions**

For episode \(i\), program \(p\):

\[
S_{p,i}=1
\quad\text{iff all eight query answers are correct.}
\]

Complete-episode accuracy is primary. Also report query accuracy, CE, margins, query position, protected/ordinary keys and both lengths.

Use families as uncertainty clusters. Bootstrap 20,000 times with seed 6103, resampling 32 development families independently within each length. Keep four variants, all programs and all query outcomes together. Use equal length weighting and fixed linear-interpolated empirical quantiles.

Report conditional input-sampling intervals, not training-seed reproducibility. Degenerate intervals do not establish universal equivalence.

Apply these unrounded gates:

| Gate | Requirement |
|---|---|
| Symbolic correctness | Independent interpreter agreement; all construction, encoding and separation assertions pass |
| Shortcut rejection | No declared shortcut reaches 80% complete-episode accuracy at either length |
| Solvability | At least one program reaches query accuracy ≥95% and complete-episode accuracy ≥80% at both lengths |
| Headroom | Strongest observed fixed-program complete-episode accuracy is below 95% at each length |
| Complementarity | Conditions below hold at both lengths |

Shortcut controls are constant predictions; latest SET value; last operand bit; global register XOR and its complement; and a last-16-command interpreter reset to zero at its window boundary. No shortcut fitting or extra model execution is required.

For complementarity define:

\[
O_L=\operatorname{mean}_i\max_p S_{p,i},\qquad
G_L=O_L-\max_p\operatorname{mean}_i S_{p,i}.
\]

Require:

- \(G_L\ge0.10\) at each length.
- Its Bonferroni-adjusted one-sided bootstrap lower bound is positive, using quantile `0.025` for the two-length family.
- At least two programs each have unique-success frequency ≥5% at each length.

Recompute maxima within each bootstrap replicate. \(O_L\) is a clairvoyant whole-episode screening bound—not achievable selector performance.

For preliminary ablations report paired counts and mean signed accuracy decrease. Flag apparent relevance only if A decreases by ≥5 points and at least one of B/R does likewise. This is a descriptive flag; **inferential relevance remains unestablished even if it passes**.

These gates concern this shared model and fixed budget. They do not establish superiority over independently optimized fixed baselines.

**7. Ordered stopping and interpretation**

Execution order is fixed:

1. Preservation/preflight and input sealing.
2. Symbolic/encoding/separation checks and shortcut screen.
3. CPU/CUDA model fixtures.
4. Initial 16-episode diagnostic.
5. Exactly 128 native updates.
6. Complete final development evaluation for all three programs.
7. Qualification calculations.
8. Preliminary ablations only if all prior gates pass.
9. Final accounting, integrity and report.

Task errors do not truncate final development. Failed qualification gates stop before ablations.

Report one applicable boundary: generation blocked; engineering/integrity/resource stop; shortcut failure; solvability failure; saturation failure; complementarity failure; preliminary relevance unsupported; or **provisionally qualified, inferential operation relevance unconfirmed**.

An initially competent diagnostic subset limits attribution to learning. Final competence establishes observed post-training capability; it does not by itself prove optimizer-caused generalization.

No outcome establishes learned selector advantage. No failed attempt triggers automatic replacement data, another seed or extended training.

**8. Evidence and preservation**

Prospective fresh directory:

`experiments/results/gatec_keyed_register_qualification_6100_v1/`

Stop if it already exists. Expected outputs include:

- Protocol and implementation seals; versioned research modules/tests.
- Input, encoding, rejection, exposure and order manifests with hashes.
- Fixed diagnostic selection.
- Fixture comparisons and gradient records.
- Initial, training, development and ablation records.
- Final checkpoint with optimizer state and update count.
- Token/block/commit ledger, qualification analysis, failures and final report.
- Initial/final integrity inventories.

Trace joins must use explicit phase, program, family, variant, episode and token IDs. Record attempted/completed/failed calls separately, actual blocks, commits, cache versions and resource consumption.

Preserve all existing files and tracked diff. Disable bytecode/cache writes to protected locations. Historical integrity exceptions remain disclosed. Retain complete fixture tensors; label ordinary trace hashes/summaries accurately rather than implying retained full-state arrays.

Actual runtime, longer-sequence memory requirements and implementation correctness remain execution-time prerequisites. Insufficient storage, unavailable CUDA, nonfinite optimization or broken isolation must stop the attempt without changing the method.

**No files were written, datasets generated, models imported, tests executed or experiments run. HEAD and the original tracked-diff hash remained unchanged.**
