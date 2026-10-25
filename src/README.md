# Shared AGT implementation

This directory contains shared model behavior, not a single universal experimental configuration. Start with the [source-grounded architecture](../docs/architecture.md) and [implementation/blueprint reconciliation](../docs/implementation_vs_blueprint.md).

|Area|Files|
|---|---|
|Canonical token path and complete state|`agt.py`|
|Bank blocks, feature mixers and normalization|`blocks.py`, `mhc.py`, `norms.py`|
|Attention and recurrent operators|`swa.py`, `gqa.py`, `gdn_v3.py`|
|Persistent memory and contextual state|`memory.py`, `caches.py`, `terminal_commit.py`|
|Routing features, legality and execution|`controller.py`, `masks.py`, `executor.py`, `routing.py`|
|Blueprint-oriented constants|`config.py`|

Inspect the actual imports for active modules. Legacy files and `.bak` copies remain for provenance. Research variants are identified by their own seals under [experiments](../experiments/README.md); the feature-complement pilot does not silently replace the shared mixer. Pilot D64, memory-slot and execution-budget values must come from that pilot's configuration.

See [development contracts](../docs/development.md) before changing routing, state ownership, gradients or terminal commit. Reading this guide runs no model.
