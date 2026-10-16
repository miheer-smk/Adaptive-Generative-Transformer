# Frozen training-representation cache

Status: **COMPLETED_VERIFIED_TRAINING_CACHE**. This prepares training data only. No bilinear readout was fitted, no fitted-readout metric was calculated, and no development/size-transfer representation was extracted. Historical CLRS training remains **COMPLETED_NON_SUCCESS**.

## Cache and exact scope

`training_representation_cache_v1.pt` contains exactly128 size16 training arrays in the original sampler order `official/train/row0000` through `official/train/row0127`:

- `candidate_committed_s`: CPU FP32 `[128,16,64]`, state.s committed immediately after original input element j.
- `query_committed_s`: CPU FP32 `[128,16,64]`, state.s committed immediately after query i, which asks about original element i.
- `predecessor_targets`: CPU int64 `[128,16]`, original official predecessor indices in candidate coordinates.
- Explicit array IDs, original row indices, input hashes, separate candidate/query original-element indices, checkpoint identity and program descriptor.

There is no AGT autograd graph, optimizer state, copied full checkpoint, learned readout or extra parameter in this artifact. Its size is **1,082,197 bytes**. Its SHA-256 is **`83b8a4c1d19ba479901d5714c246e89385aadfc76eb971c463fe9ac115352f41`**. Numeric representation/target payload is1,064,960 bytes; serialization and identity metadata account for the remaining bytes.

Row0000 is reused bit-for-bit from the verified final-checkpoint localization capture. It contributes **zero new model calls**. Exactly127 new isolated arrays were executed: rows0001–0127. The historical shuffled optimization order (seed6201) is hash-verified and preserved but is not used as the cache order. No new sample was generated or substituted.

## Source, input and checkpoint provenance

- Final checkpoint: `experiments/results/gatec_clrs_insertion_sort_6200_v1/training_results_v1/model_update1000_v1.pt`.
- SHA-256: `2c0e3d6ad3711d8f0ee209eb3e3b0f8db8964e6eeaf2be77b7bd56120f29e6c8`.
- Loaded model parameter/buffer identity before and after: `7ab9239da5b1a4957cde546bb95e4c2961800b337f8bf677f8202e662e94f1a0`.
- Training-pool SHA-256: `a3e7a17717c44ef7202bf2072f23981a089e5b09c39694740834dccc227b6851`.
- Official CLRS source revision: `6d5728b9efadf8648413c8c473929a733caacced`.
- Adapter SHA-256: `53b9e7106a233e34d6a072ab1ff80bb69e82c3d09e507466df36447eeeb67f00`.
- The selected row0000 capture tensor SHA-256 is `607ca991f8f021c839948b2a85cca8709470e5651b8b09aedbacf32ce970ab42`.

Preflight checked engineering/training/localization implementation seals, archived training records, official source hashes and original sampler row identities. For every selected row, original float64 key/position bytes reproduce the recorded input hash; FP32 keys/positions, targets and32×4 features match the retained training-only official NPZ export. Targets and query/candidate roles preserve the approved original-index convention. No symbolic generator or model was used to reconstruct labels.

`input_manifest_v1.json` seals all128 IDs, source/reuse status, canonical row hashes, original-input hashes, target hashes, feature-byte hashes, labels and explicit indices. `manifest_v1.json` binds exact source/input/checkpoint files. `extraction_seal_v1.json` was written before model execution. The original tracked source changes and historical bytecode exception are preserved.

## Execution and state handling

The existing sealed `InsertionSortAdapter` was reused unchanged. It creates the prescribed fresh initial persistent state at the start of each array and processes16 original-element tokens then16 query tokens. The fixed canonical production path executes `[0,1,2,3,4,5,6,7]` with bank order `[SWA,SWA,GQA,GQA,GDN,GDN,FFN,FFN]`. No controller or exhaustive candidates are called.

One model was constructed and loaded strictly from the final checkpoint; all parameters, including the core, encoder, registered controller, vocabulary head and original query/key matrices, were frozen before trajectories. Eval mode, FP32, TF32 disabled, PyTorch2.10.0+cu130/CUDA13.0 on NVIDIA GB10 match the original runtime configuration. No seed6300 model is trained or initialized as a replacement;6300 names this cache operation.

The research observer is a versioned copy of the previously sealed observer. `observer_diff_v1.patch` records only descriptive text, the900-second deadline, token/bank caps and a backward method that refuses execution. The original token dispatch, contextual purity, cache-version, parent-state isolation and terminal-commit checks are unchanged. Candidate/query state capture and read-only head hooks return no replacement values. Every saved representation is detached and copied to CPU; no transient runtime state crosses array boundaries.

For every new token, actual selected blocks, D=8/P=8, one terminal commit, frozen in-flight contexts/anchors/memory reads, unchanged parent and disjoint returned state were checked. Every contextual cache version advances once. The complete retained trace has4064 rows over127 arrays, in exact input order, with tokens0–31 in each array and fresh empty cache metadata at token0. Normal memory writes, GDN state/buffer evolution, attention caches, terminal RMSNorm and all seven contextual commit entries remain active.

## Numerical equivalence

Every cache entry has16 candidate states and16 query states with expected dtype, bounds, role/index association and finite values. Row0000's serialized cache states match the reused localization tensors bit-for-bit. Candidate storage remained unchanged while later input/query tokens ran. Read-only original head hooks confirm that the queried state.s values and the key-table rows are precisely the cached values.

All128 archived original-head score matrices and all2048 predictions are available and used as required comparisons. New ordinary-forward scores have maximum absolute error **0** against the retained records; predictions agree exactly. CPU FP64 replay of the existing checkpoint query/key matrices from the cache has maximum absolute score error **1.84917706814e-07**, within unchanged atol1e-6/rtol1e-5; all predictions remain exact. CPU serialized-cache verification independently checks these comparisons without importing AGT or executing a model.

This is reproduction of the original frozen head, **not** a fitted-readout evaluation. No decodability, trainability or generalization claim is made. The historical negative task result is not reclassified.

## Resource ledger

|Operation|Attempted|Completed|Failed|
|---|---:|---:|---:|
|New arrays|127|127|0|
|New model tokens|4064|4064|0|
|Selected bank executions|32512|32512|0|
|Terminal commits|4064|4064|0|

Reused arrays:1. New model constructions:1; checkpoint loads into that model:1. CPU-only verification additionally reads checkpoint parameter tensors; it does not construct another model or optimizer. Contextual terminal-commit entries: **28,448**. Backwards: **0**. Optimizer objects/updates: **0**. Warm-ups: **0**. Retries: **0**. Held-out arrays: **0**.

Extraction wall time: **31.301698611 seconds**, within the900-second limit. The clock began before CUDA/model initialization and included construction/load, row0000 algebraic replay, all127 new trajectories and per-array checks. A real-time signal alarm and per-token/array resource checks were armed; no deadline extension occurred. Offline serialized-cache/integrity verification and publication occur after extraction and introduce no model calls.

Peak CUDA allocated: **20,866,560 bytes**; peak reserved: **35,651,584 bytes**. Storage preflight estimated ≤20MiB total and found204,768,153,600 free bytes. No archive, full-project copy or duplicate checkpoint was created. No dependency was installed; preflight used the already-existing CPU environment's NumPy for training-export identity checks.

## Integrity, outputs and stopping boundary

The preflight inventory fingerprints **2,543 pre-existing files**. `integrity_final_v1.json` records the full end check, including original HEAD/tracked diff and the historical integrity exception. Inputs, source, checkpoint, prior experiments and sealed implementations remain unchanged.

Core evidence: `training_representation_cache_v1.pt`, `cache_manifest_v1.json`, `input_manifest_v1.json`, `per_array_extraction_v1.jsonl`, `token_traces_v1.jsonl.gz`, `execution_ledger_v1.json`, `numerical_equivalence_v1.json`, `cache_verification_v1.json`, versioned code, source seals and integrity inventories. Publication uses a unique branch and separate remote-verification receipt, without local staging or rewriting history.

**The training-only cache extraction is complete. No bilinear fitting, fitted-readout metrics, development/size64 extraction, further model call or subsequent experiment is authorized or started.**
