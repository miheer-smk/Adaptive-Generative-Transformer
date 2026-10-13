# Receiver-anchored compatibility study: final concise report

**COMPLETED_NON_SUCCESS.** The complete locked study ran without engineering, integrity or nonfinite-optimization failures. The treatment did not establish a compatibility benefit over the matched native-continuation control. All four distinguishing-operation ablations failed the required task-relevance criterion. This closes the proposed formulation on parity under the approved stopping rule; no follow-up was launched.

## Execution and decisions

| Phase | Executed result |
|---|---|
| Prerequisite verification | Existing 20 passes and 3 documented intentional skips verified; no repeat suite |
| P0 archival calibration | Full control comparison passed; final CE 0.0014546061865985394 |
| Common acquisition | 512 updates; development native gate passed for both programs and lengths |
| Paired acquisition | 512 treatment and 512 control updates, independent complete optimizer/model forks |
| Native-only selection | Both 256 and 512 eligible; shared selected update 512 minimizes worst native mean CE |
| Four-cell confirmation | Complete for both models, all cases, suffixes and continuation schedules |
| Frozen reference, ablation and profiling | Complete prescribed work |
| Final acceptance | Failed primary practical-benefit and noncollapse criteria |

## Held-out outcomes

Both arms achieved 100% current and robust-case accuracy in every native and switched program/length stratum. Both also achieved 100% alternating-continuation robust-case accuracy in every cell and length. Each arm has 512 cases with four-cell pattern 1111; the other fifteen patterns have zero cases. A robust case requires a correct current boundary and every token of both eight-token suffixes.

| Native program/length | Treatment mean CE (nats) | Control mean CE (nats) | Current/robust accuracy, both |
|---|---:|---:|---|
| P0/L16 | 0.000139989635528 | 0.000139992287605 | 100% / 100% |
| P0/L24 | 0.000139639558699 | 0.000139646592694 | 100% / 100% |
| P1/L16 | 0.000144286468877 | 0.000144292136141 | 100% / 100% |
| P1/L24 | 0.000144307141101 | 0.000144315875504 | 100% / 100% |

Native competence, native comparability and treatment behavioral-compatibility criteria passed, including horizon CE requirements. Adjusted native paired accuracy intervals were approximately ±1.697 percentage points with zero observed discordances, within the prescribed ±2 points.

**Primary estimand Δ = 0.0; paired 95% bootstrap interval [0.0, 0.0].** Both length-specific estimates and intervals were also zero. The required lower bound of five percentage points was not met. The matched control already switched successfully on every evaluated case. These degenerate finite-panel intervals do not establish universal equivalence.

Frozen archival P0 achieved 100% current/robust accuracy at both lengths. Its mean CE was 0.0017763416611909204 at L16 and 0.001783607566213665 at L24. No reference errors were removed from the panel.

## Ablation and latency

For T/P0, T/P1, C/P0 and C/P1, each paired robust-accuracy decrease was **0 percentage points**, with adjusted one-sided lower bound 0. All four fail both required conditions (point ≥5 points and lower bound >0). Suppressing each program’s distinguishing fifth returned hidden output at every native token left observed robust accuracy at 100%. This does not prove global computational equivalence or identify a causal state component.

| Program/length | T/C arithmetic mean latency ratio | Adjusted one-sided upper bound | Acceptance |
|---|---:|---:|---|
| P0/L16 | 1.014106724 | 1.044277340 | ≤1.10: met |
| P0/L24 | 1.001952763 | 1.008055787 | ≤1.10: met |
| P1/L16 | 1.005100338 | 1.011870081 | ≤1.10: met |
| P1/L24 | 0.999418010 | 1.004522188 | ≤1.10: met |

Profiling used the locked z0 suffix only, 32 prospectively fixed cases, five paths, one warm-up and five timed full trajectories per case/path. Both suffixes remained in confirmation. Repetitions were averaged before the separate paired bootstrap. Ablation and latency used separate four-comparison confidence families and fixed linear-interpolated quantiles. No measurements were trimmed or replaced. FLOPs are not supported; measured end-to-end latency is the cost criterion.

## Resource and mechanical accounting

Study calls: 533,776 attempted, 533,776 completed, 0 failed; 4,270,208 selected bank executions; 533,776 terminal commits.

Training: 1,536 optimizer updates (512 common, 512 T, 512 C), 3,072 backward evaluations. All normal completed tokens have D=8/P=8, legal requested/selected/executed correspondence and one commit. P1 contains one REPEAT and no RESTART. The full offline trace verification checks all 533,776 rows. Clean latency traces retain runtime-checked block identities without forensic decision arrays.

Including the prior prerequisite ledger: 534,163 attempted, 534,162 completed, 4,273,296 selected executions; its one failed fixture remains separately recorded, not pooled into outcomes.

New study accelerator window: 7573.896793 seconds (2.103860 hours). Charged prior fixture time: 22.660482 seconds. Peak allocated/reserved memory: 311,910,912/316,669,952 bytes. Offline statistical analysis: 0.548930 seconds. All locked ceilings were respected.

Contextual commit work includes 1,067,552 SWA bank, 1,067,552 GQA bank, 1,067,552 GDN bank and 533,776 Prelude SWA payload calls, including unselected contextual blocks. Prelude, commit, selected-bank and total timings are retained in profiling records. No exhaustive routing candidate construction, Gumbel noise, selector or controller update occurred.

Full confirmation incoming/boundary tensors and metadata are retained in 1,024 state files. Prefix readouts remain descriptive rather than independent observations.

## Provenance and preservation

Git revision: 75f138e6b909c5ca9884e9a7755de63489d3d6f2. Protocol SHA-256: `d87c63ec804e0551ad52a6acf974f3b484767cc50abd640e8440e9d6a10410ec`. Addendum SHA-256: `53a9c1b9e6f383e464335e93c625a2428ab21510a347ec9f8a3bfc5ba8373ced`.

The validated adapter was copied into the new directory; its only adaptation changes two fixture-only 4,096-block assertions to the authorized ledger ceiling. Exact diff and original/new hashes are sealed. No gradient, commit, mask, routing or production implementation was changed. The 395→256→11 controller remained frozen and unused. Research models have 1,934,324 parameters, of which 1,829,993 existing non-controller parameters were trained; no new capacity was added.

Inputs, encodings, orders and profiling selection were sealed before model outcomes. The exclusion inventory contained 1,429 distinct retained historical inputs from 47 readable JSON artifacts. Complete inputs and prescribed incoming histories are disjoint across the declared splits. Unknown Stage-1 exposure and unavailable historical inputs remain limitations. Held out means outside documented acquisition/selection exposure.

The study uses per-token task parity supervision and full committed-state handoff credit. It does not rehabilitate historical final-only CE/ST-Gumbel discovery, establish program synthesis, universal compatibility, seed reproducibility, architectural novelty or advantage over the best fixed program. Uncertainty is conditional on these acquired models and retained input sampling.

Selected checkpoint identities:

- T: `checkpoints/T_update0512.pt`, SHA-256 `0d4a7e75496148c8da5b96d733990736271dcc4da8f602e801b71b50b966d9c7`.
- C: `checkpoints/C_update0512.pt`, SHA-256 `4fd7cf9e3636b5ec62e21e82884e473b86c48172d6c65bb69cbe43ed3bd17438`.

All 776 protected pre-existing file identities and the original tracked diff remained unchanged at runner completion; final verification is recorded separately in integrity_final.json. The historical bytecode exception remains unchanged. Archival checkpoints and production source were preserved. Only newly authorized research checkpoints were created. No Git commit, retry, replacement model, extra samples or follow-up experiment occurred.

Detailed machine-readable results: final_results.json and final_analysis.json. Complete trace/ledger evidence: token_traces.jsonl.gz, complete_trace_verification.json and execution_ledger.json. The exact execution boundary is completion of the prescribed study and offline final verification; all scientific phases ran, none were skipped or left unverified.
