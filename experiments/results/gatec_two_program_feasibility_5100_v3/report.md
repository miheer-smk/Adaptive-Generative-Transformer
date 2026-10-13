# v3: stopped at the specification barrier

No implementation amendments,tests or model execution were performed. v1 remains STOPPED BEFORE P1. v2 remains an inspection-only stop.

## Completed inspection

The original602 protected identities were verified; the expanded baseline protects607 files including all additional immutable v1/v2 artifacts. The original tracked diff and archived checkpoint hashes match. The original sealed source,test,input manifest,encoded tensors,generation attempts and exclusion inventory remain hash-identical.

The requirements matrix has620 entries. The schema inventory traverses all64 step and8 token records in each archived path A/B/C,not merely the first record or an intersection of available fields. It enumerates17 step fields,13 token fields and every nested retained persistent/context/commit summary field,including null variants,shapes,versions,window counts and consumption metadata. Detailed extraction mappings,tolerances,missing-field behavior,dependencies,budgets and proposed negative tests are recorded per requirement.

## Observer diagnosis: confirmed,not repaired

The sealed observer at229 applies a CPU boolean successor-mask operand of shape10 to CUDA FP32 logits of shape1x10. Production `src/executor.py:647-652` moves the returned full mask onto the logits device only after that observer returns. A local mask-only `to(device=logits.device)` would retain shape,dtype,values,legal indices and action order,with no in-place mutation or RNG. This amendment was not applied because the specification barrier did not pass.

## Unresolved requirement1: current acceptance scope

The locked manifest records gate names,pointwise thresholds and a joint current/future confidence statistic. It does not define independent current-gate scope (per-case,panel,study) or a separate current-stage statistical decision. The sealed `bootstrap_prefix_worst` at546-562 requires two current cells and18 groups of future values per prefix. Its confidence bound cannot be evaluated before those future values exist. It is unresolved whether current acceptance is exact correctness/pointwise checks only or includes a distinct pre-future confidence criterion. Creating a current-only statistic or choosing a scope would revise an unspecified scientific requirement.

## Unresolved requirement2: P0/P1 prerequisite ordering

The original portfolio includes P0 current evaluation under both P0-generated and P1-generated histories. P1→P0 requires P1 prefix execution. The authoritative artifacts do not define which P0 current subset must pass before that P1 prefix may run,or whether a panel-wide/all-history P0 gate is intended. An all-history P0 prerequisite would itself need P1-generated state. The exact dependency ordering must be specified before implementation; samples,history conditions and denominators cannot be silently changed.

## Unresolved requirement3: retained controller-monitor output identity

Archived `paths.A/C.steps[*].controller_logits` is produced by the monitored registered model controller,not by the forced scripted action provider (`experiments/audit_gatec_canonical_replay.py:115-120`). The historical harness constructs that fresh controller with seed3401 (`:376`,and retained fixture `seed`). The locked feasibility loader uses seed3305 (`v1/complete_program_feasibility.py:load_frozen`). Scripted provider logits are ±100 and are not the archived monitor values. Comparing these unlike functions is not valid. The required comparison target/role for this retained diagnostic field must be resolved; it cannot be silently omitted or relabeled. This is unrelated to selecting canonical actions and does not establish an executor problem.

## Historical versus within-run evidence

Historical live/terminal hidden arrays and output logits are retained elementwise. Historical persistent memory,GDN states/buffers and KV caches are retained as shape/l2/max_abs summaries plus exact version/window/token metadata,not full raw tensors. These summaries can be compared as summaries; they cannot establish elementwise historical-state equality. New reference-versus-adapter full-state comparison is a separate evidence layer. Ownership,storage addresses and plan/state object identity are within-run assertions,not cross-run numerical equality targets.

## Precise execution boundary

V3 stopped before creating amended observer/adapter/runner/test files. No CPU/CUDA tests ran or skipped; no CUDA validation claim is made. No checkpoint was loaded into a model. Reference,adapter,P1,continuation,bank-call,commit and bootstrap counts are all zero. No optimizer,gradient,training,selector,parameter tuning,production edit or Git commit occurred. V1 partial work (one reference token,eight bank calls,one commit) and v2 zero-work records remain separate historical ledgers.

Engineering: diagnosis confirmed,implementation validation blocked. P0 archival equivalence: unexecuted. P1 current correctness: unexecuted. Future compatibility: unexecuted. Research hypothesis: untested. No scientific acceptance or rejection of P1 is justified.

Final protected-file and tracked-diff verification is recorded in integrity_after.json; no new integrity exception was created.
