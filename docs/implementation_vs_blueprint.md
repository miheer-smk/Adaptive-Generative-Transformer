# Implementation versus blueprint

This is a current reconciliation, not an edit to the historical blueprint or notes. `src/config.py` and `docs/IMPLEMENTATION_DECISIONS.md` reference **AGT-v3 blueprint pre-freeze v0.1, 30 Sep2026**. The full standalone blueprint was not found in the audited checkout/remote evidence listing. Diagram fidelity beyond those retained references is therefore **unverified**. The implementation and execution-specific protocols are authoritative for this guide.

|Topic|Retained conceptual/declarative record|Actual source or experiment|Implication|
|---|---|---|---|
|Width|`src/config.py` declares768|`AGTv3` constructor defaults256; retained pilots explicitly instantiate64|Do not describe D64 results as D768 reproduction|
|Parameter count|Provisional86,381,128 in central config|CLRS adapter records1,942,836 total /1,838,377 trainable|A declared constant is not a measured instantiated-model size|
|Residual/mHC structure|Central `N_HC=2`|`src/mhc.py` mixes the final feature axis with one matrix and coordinatewise gain|Do not draw a demonstrated two-stream residual architecture|
|Backpropagation|Central `K_BP=4`|CLRS/receiver protocols require full BPTT and real state handoff graphs|Central constants do not override run-specific training code|
|Bank|Two of each species|Actual locked order SWA,SWA,GQA,GQA,GDN,GDN,FFN,FFN|Order and operator counts must be explicit|
|GDN implementation|Older `src/gdn.py` remains|Bank imports `GDNv3` from `gdn_v3.py`|Legacy code is preserved, not an alternative silently substituted|
|Block count argument|`num_blocks` exists in constructor/controller|Bank validates eight entries|Changing that argument alone does not resize the bank|
|Memory read|Conceptual descriptions may suggest live-state-conditioned reads|Validated token paths read from previous committed s once and reuse R|Per-decision memory re-reading was not established|
|Context name|Prelude takes a tensor named `C_prev`|Persistent typed cache is `state.ctx`; compatibility tensor is separate and often zero|Do not collapse these two interfaces in diagrams|
|Depth|Executor default max256|Fixed-depth pilots explicitly use eight actual executions|P=8 is a pilot setting, not a universal adaptive budget|
|Restart accounting|Earlier reports used eight decisions|Corrected Stage 3 increments P only for ADVANCE/REPEAT|Historical experiments keep their original protocols|
|HALT|Represented in controller/full mask; separate adaptive paths exist|Excluded from fixed-depth Stage 3; off in listed fixed-program studies|HALT/adaptive-depth learning is not demonstrated|
|Normalization/readout|Old DEVIATIONS.md says duplicate final norms|Current OutputHead is only a bias-free projection; terminal commit supplies RMSNorm|Old note is stale; do not change the sealed note to hide it|
|Prelude block|Registered dedicated SWA block/context owner|Canonical macro Prelude constructs h0; the dedicated Prelude mixer's forward is unexecuted in these pilots|The feature-complement study changes eight active mixers, not this ninth registered mixer|
|Commit of skipped entries|A selected route might suggest selected-only state updates|All context-bearing bank entries plus Prelude receive terminal payloads|Selected bank counts understate complete work|
|Program portfolio|Source audit proposed finite complete-program contracts|Research-only scripted adapters exist; no deployed learned whole-program selector|Do not label a hand-designed script as program discovery|
|Task output|Conceptual language-model head|Parity uses class logits; CLRS adds a four-feature encoder and dynamic query/key pointer readout|CLRS is an adapter study, not a full language-model or CLRS GNN result|

Source locations are collected in [architecture](architecture.md). The [complete-program source audit](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/ea183970feea1f49677f186e26065ce4e333f709/experiments/results/gatec_complete_program_source_audit_v1/report.md) provides the earlier detailed inventory, including its own source hashes and scope. Newer research adapters do not retroactively turn its proposed interfaces into production functionality.

The guide does not resolve missing owner decisions by changing code. It records the full-blueprint availability gap, stale notes, known legacy tests and per-run differences in [limitations](limitations.md).
