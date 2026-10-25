# Contributing to AGT-v3

Start with the [research status](README.md#current-research-status), [source map](docs/architecture.md) and [experiment index](docs/experiment_index.md). A proposed architecture, an implemented feature and an experimentally supported claim are different things. Describe which one your change addresses.

## Documentation and analysis

Prefer commit-pinned links to historical evidence over copying or moving it. Preserve original statuses, denominators, split definitions and uncertainty conventions. Figures must have numerical inputs, source identities, generation commands and limitations in the [figure provenance](docs/figure_sources/README.md). Never generate missing measurements or label a proposed study as completed.

For presentation-only changes, run the [documentation checks](scripts/documentation/README.md). Do not run model-executing tests as a routine documentation check. No current CI badge or universal dependency lock is asserted.

## Code and experiments

Keep production changes separate from research variants. Before changing execution semantics, inspect the [development guide](docs/development.md): decisions, selected executions, contextual work and commits have distinct accounting. A passing forward fixture does not establish useful gradients, task competence or measured efficiency.

New experiments require a separately approved protocol and allocation. Declare hypotheses, data exclusions, seeds, selection rules, metrics, budgets, stopping rules and intended classifications before execution. Use a fresh output namespace; never resume a closed attempt silently. Record source and working-tree identities, environments, attempted/completed/failed calls, backward work and optimizer updates. Preserve failed calls and unsuccessful outcomes.

Do not edit sealed inputs, reports, checkpoints, protocols, manifests or implementation snapshots. Corrections belong in a separately dated, linked addendum with an explanation. Do not reset, clean, rewrite history, force-push or delete evidence to make a branch look tidy. Follow the [artifact policy](docs/artifact_policy.md) for publication and large files.

## Review checklist

- State the concrete behavior or claim affected and its evidence.
- Identify production code versus isolated research code.
- Explain validation actually executed, known failures and unverified assumptions.
- Include provenance for data, plots and claims; disclose reused evaluation panels.
- Verify remote paths and payloads before declaring an upload complete.
- Avoid unrelated changes, credentials and large duplicate copies.

No project-wide license or preferred citation metadata was located. This contribution guide does not grant a new license or invent publication metadata; maintainers must resolve those separately.
