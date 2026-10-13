# Receiver-anchored continuation: bounded prerequisite evidence

## Result and boundary

The final focused checks establish engineering feasibility of the prescribed
writer-directed gradient path on disposable CPU and NVIDIA GB10 CUDA fixtures.
They establish neither native acquisition nor switching improvement. **Training
remains unauthorized.** No common acquisition, treatment/control training,
model selection, study profiling, distinguishing-operation study ablation,
held-out confirmation, optimizer step or study checkpoint occurred.

The original protocol authority is the user's complete conversation text.
No local original-protocol filename or SHA-256 identity is claimed. Its final
clarifications are separately recorded and hashed in
`protocol_clarifications_v1.md` and the implementation seals. These documents do
not amend or reinterpret v4. Historical P1 and all stopped attempts are unchanged.

## Implementation and tests

All new files are confined to this fresh evidence directory. The current adapter
is `receiver_anchored_v2.py`; current statistical acceptance code is
`statistical_rules_v3.py`. Main focused tests are `test_receiver_anchored_v3.py`;
supplemental tests are `test_supplemental_boundaries_v1.py`. Versioned predecessors
and every failed preflight/fixture record remain present. No production source
was edited. The adapter calls the actual production complete-token path;
execution, cache/GDN handling, memory and commit mathematics are not duplicated.

Final selections: **15 passed, 3 skipped** in `pytest_results_v3.xml`, plus
**5 passed** in `supplemental_results.xml`: **20 passed, 3 skipped**.
The skips are the two CPU instances of CUDA-only resource probes and the CUDA
instance of the CPU synthetic optimizer-ownership fixture. CUDA control,
program, clone, directed-gradient, resource and cache tests genuinely executed.
No full repository regression was run: uninstrumented historical tests would
fall outside this bounded model-token-accounted authorization.

The scope exercised:

- Immutable namespaced P0/P1 scripts and fail-closed legality/completion checks.
- Production requested/selected/executed correspondence, no exhaustive candidates,
  D8/P8 and exactly one seven-entry terminal commit.
- Full parent/returned-state topology, values, exact metadata and nonaliasing.
- Cache cardinalities, versions, absolute positions, GDN states/buffers and
  in-flight committed-context, anchor and memory-read purity.
- Connected committed-state clones and detached-parameter functional receivers.
- Original native objective gradients, full writer-directed continuation gradients,
  complete handoff-state VJP reconstruction and explicit detached-state negative control.
- Independent model/buffer/optimizer-moment ownership using synthetic optimizer
  moments. Their fixture step metadata is 512; **executed optimizer steps are zero**.
- Objective weights, equal episode weighting and clipping of disposable gradient
  values only; gradients were never assigned to parameters for an update.
- Locked acceptance arithmetic using synthetic robust-case/latency arrays only.

## Forward evidence

CPU and CUDA each executed all eight tokens of the retained archival control
through fresh same-checkpoint canonical and scripted P0 paths. At every token,
all returned-state fields and metadata and all observed macro tensors met the
established atol=1e-6/rtol=1e-5 comparison; discrete metadata was exact.
Per-token maximum output-logit difference was zero on both devices.

Final CE on both devices: **0.0014546061865985394**.

This is full fresh reference/adapter comparison and separate agreement with the
retained control CE. It is not a claim to have replayed every historical archival
tensor or monitor. Complete comparison schemas are in `test_measurements.json`.

P1 executed `[0,1,2,3,3,5,6,7]`, with actions `[0,1,2,3,8,5,6,7]`.
The omitted selected GDN4 operation still produced its ordinary terminal entry.

## Gradient evidence

| Disposable fixture | CPU | CUDA |
|---|---:|---:|
| Native parameter-gradient L2 | 60.53039551 | 60.53028107 |
| Continuation writer-parameter-gradient L2 | 21.85624504 | 21.85624695 |

All measured connected gradients were finite. Direct continuation gradients
with respect to writer parameters matched the VJP through the complete cloned
handoff state under the established numerical comparison. Receiver parameter
values had `requires_grad=False`; receiver computation remained differentiable
with respect to its incoming state. Native/writer parameters remained connected.
The explicit detached-state negative-control loss had `requires_grad=False`
without detaching the loss or enclosing the positive receiver in `no_grad`.

Across the two handoffs, 26 of 32 incoming tensor fields were connected and
nonzero on both devices; six were disconnected. These were both Prelude cache
K/V pairs and the P0-writer GDN4 recurrence/buffer consumed by a P1 receiver
that never selects GDN4. They were still handed off and committed normally.
The source Prelude is a projection constructor rather than a reader of its
separately maintained Prelude attention cache. No unused field was forced into
the task graph. This is not universal complete-state behavioral equivalence.
Per-field and per-parameter gradient records are retained.

## Statistical rule boundaries

Synthetic ablation contrasts had point estimates 0.0625 and adjusted lower
bounds 0.0390625: all passed. This verifies that a positive lower bound need not
exceed 0.05. Smaller point estimates and constant no-effect fixtures failed.

Synthetic latency contrasts had point estimates approximately 1.05 but adjusted
upper bounds 1.225: all failed. Constant-ratio 1.05 fixtures passed; 1.11 fixtures
failed. Missing/nonfinite measurements and nonpositive denominators were
unassessable. The ratio-of-arithmetic-means fixture produced 2.5, not the mean
of individual ratios, 2.0.

The two four-comparison confidence families are separate. Both use 20,000
paired stratified resamples seeded 6010 and linear-interpolated quantiles.
Profiling has a separate bank. Equal stratum weighting was checked against
unequal synthetic stratum sizes. These are mathematical tests, not scientific
confirmation or inference about learned models.

## Resource ledger

| Quantity | Final observed count |
|---|---:|
| Attempted model tokens | 387 / 512 |
| Completed model tokens | 386 |
| Failed model-token calls | 1 |
| Attempted/completed selected-bank calls | 3,088 / 3,088, ceiling 4,096 |
| Attempted/completed terminal commits | 386 / 386 |
| Backward/VJP evaluations, including scalar arithmetic fixture | 13 |
| Optimizer updates | 0 |
| AGT warm-up tokens | 0 |

Actual selected-block totals: A0=386, A1=386, A2=386, block3=572,
block4=200, block5=386, block6=386, block7=386.
Observed terminal payloads: 2,702 across 386 seven-entry plans. Source-implied
contextual work includes 772 GDN entries, 1,544 bank attention entries and 386
Prelude attention entries. Per-core commit kernel timing/FLOPs were not measured.

Each CUDA representative arm fixture executed one complete effective batch
(length16 plus length24), including native rollouts and additional receivers:
112 tokens and one full-objective gradient evaluation, with no optimizer step.

| Arm fixture | Forward/backward seconds | Tokens/second | Peak allocated bytes |
|---|---:|---:|---:|
| T | 2.59387016 | 43.1787 | 364,393,472 |
| C | 2.54887157 | 43.9410 | 364,128,768 |

The overall 534,288-token ceiling requires 12.3678 tokens/second over 12 hours.
Applying the slower fixture rate to every allocated token gives a crude
3.4372-hour projection. **This is not a full-study runtime bound:** it excludes
optimizer-step cost, complete confirmation-state retention/I/O and offline
analysis, and it is based on two fixed disposable batches. It includes fixture
observations and ledger writes. Deployment latency was not measured. Peak
memory and throughput support plausibility, not guaranteed resource fit.

The environment emitted the preserved PyTorch warning about GB10 capability
12.1 versus listed support through12.0; the actual CUDA tests completed. NumPy
is unavailable; no package installation was performed.

## Retained implementation failures

1. The first launcher required unavailable NumPy and stopped before source
   sealing, checkpoint deserialization or model construction. Its failure,
   sources and report remain untouched. Versioned statistical fixtures use the
   existing CPU PyTorch dependency instead.
2. The first CPU control fixture interpreted archival `[1,8]` bits as a flat
   vector and added an extra dimension. Production rejected the shape before
   any bank or commit call. The corrected fixture extracts the retained B1 row
   without changing contents, encoding, labels, thresholds or source semantics.
   This one failed call remains counted. Its failed XML, source seal, report
   and trace are retained separately.
3. Offline XML aggregation encountered ElementTree truth-value handling. Its
   failure note is retained; explicit None checking corrected report extraction
   without model work or changes to test results.

These are disclosed implementation/fixture corrections within one bounded
prerequisite allowance, not scientific repairs, replacement samples or study
attempts. Earlier reports are not rewritten as successful.

## Integrity and conclusion

All **704 pre-existing files** were rehashed unchanged, including all 617
historically protected identities, v1–v4, forensic records and checkpoints.
Original tracked-diff SHA-256 remains
`45b3d093bd7fd7976019bf4572dc574784ed271b3294b86194f80c05826a8eac`.
Earlier source seals remain unchanged. `git diff --check` is clean. No new file
outside this attempt directory was introduced. The historical bytecode exception
and its original failure record remain intact; no new integrity exception arose.

Engineering verdict: required forward and directed-gradient mechanics verified
on these disposable fixtures. Resource verdict: bounded study is plausible,
not guaranteed within its ceiling. Native-acquisition, compatibility-benefit,
noncollapse and cost-acceptance verdicts: **NOT TESTED**.

Stop boundary: prerequisite report only. A separately approved training action
is necessary before any acquisition begins. No automatic continuation is authorized.
