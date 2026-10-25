# Development map and validation boundaries

The [architecture guide](architecture.md) maps implementation symbols to pinned source. The current DGX working tree contains historical modifications; this documentation adds navigation without normalizing or overwriting them. The [maintenance record](maintenance/repository_state_20261025.json) identifies the starting state.

## Shared implementation

[`src/`](../src/README.md) contains the model, heterogeneous blocks, persistent memory, typed contextual state, action masks, controller, executor and differentiable route step. `src/agt.py` provides the canonical sequential path. `src/executor.py` owns routed execution; `src/routing.py` constructs differentiable successor candidates. `src/config.py` contains blueprint-oriented settings that must not be substituted for an execution-specific configuration.

Archived `.bak` files and legacy modules remain for provenance. Their presence does not make them the active path. In particular, the bank uses `gdn_v3.py`; do not infer bank behavior from the older GDN module. Research adapters, instrumentation and scoped implementations live under `experiments/` and individual result directories. Consult each implementation seal before reproducing a historical run.

## Contracts that changes must preserve

- Count decisions separately from actual selected executions. RESTART consumes a decision without executing a selected block.
- Separate selected work from exhaustive ST candidates and terminal contextual work. A selected-block count is not total compute.
- Keep the live stream distinct from the complete persistent state. Clone every owned tensor and nested cache; tensor cloning may preserve autograd while avoiding storage aliasing.
- Preserve one terminal commit per completed token, typed context versions/positions, parent and sibling isolation, and ordinary updates for unselected contextual entries.
- Distinguish frozen parameters from detached incoming states. Receiver parameter anchoring requires gradients through state; disabling autograd on the branch would change the method.
- Keep task adapters and pilot settings explicit. A frozen controller or scripted route does not demonstrate learned routing.

## Tests are executable research work

The [test directory](../tests/README.md) includes model construction, forwards, backwards and training fixtures. Historical reports record test outcomes with their own configurations. An accounting-correction regression record includes failures tied to legacy assumptions; no new blanket pass claim is made here. Later prerequisite suites establish bounded properties for specific adapters and environments, not every code path.

For a code change, select checks appropriate to the changed contract and separately authorize any research execution. For a documentation change use:

```bash
python3 -B scripts/documentation/validate_docs.py
```

This guide was prepared without importing AGT, deserializing a checkpoint or running the model/test suite. The only rendered calculations read retained numerical evidence. See [reproducibility](reproducibility.md) for the isolated plotting environment.
