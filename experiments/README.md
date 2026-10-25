# Research implementations and evidence

Use the [central experiment index](../docs/experiment_index.md) for research questions, exact statuses, immutable report links, source revisions, data and checkpoint identities. The [research overview](../docs/research_overview.md) explains how the questions developed, including unsuccessful and stopped attempts.

- Top-level Python files are research runners, measurements and diagnostics. Some assume a particular environment or DGX path and may execute a model immediately.
- `runs/` contains historical checkpoints and control records.
- `results/` contains isolated study directories: protocols, sealed implementations, exported data, results, tensors, traces, integrity and publication records. These paths remain unchanged.

Do not infer completion from a directory or an approved protocol alone. Read final execution boundaries and machine-readable outcomes. In particular, the closed receiver study, interrupted keyed-register pilot, original CLRS training, nonconverged probe, separate certified refit and feature-complement pilot retain separate classifications.

No runner here is a documentation check. For offline presentation regeneration use [scripts/documentation](../scripts/documentation/README.md). Reproduction requires its own authorization, environment, inputs, budgets and fresh output path; see [reproducibility](../docs/reproducibility.md) and [artifact policy](../docs/artifact_policy.md).
