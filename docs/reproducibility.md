# Reproducibility: four distinct workflows

## 1. Read and verify evidence

Start with the [experiment index](experiment_index.md) and [machine-readable inventory](evidence_inventory.json). Each record binds reports, protocols, numerical files, source entry points, checkpoint SHA-256 identities and availability to a publication revision. These revisions are not automatically execution commits: the original DGX checkout contained14 tracked changes and much untracked evidence. Execution manifests and source hashes are the finer-grained identity.

The remote default branch points to 75f138e6, an older source-history snapshot. The latest completed evidence snapshot is [ea183970feea](https://github.com/miheer-smk/Adaptive-Generative-Transformer/commit/ea183970feea1f49677f186e26065ce4e333f709). Use the index's pinned links or the appropriate research/archive branch. Do not merge branches indiscriminately to obtain artifacts.

Historical reports can contain DGX-local paths and absolute source links. Those bytes remain immutable; use the new index and [source map](architecture.md) for GitHub navigation. Some older reproduction paths point to a particular environment or output directory. A script's presence is not proof of a portable, independently validated workflow.

### Artifact availability

The [availability audit](maintenance/evidence_availability.json) checked131 indexed artifact identities against12 publication trees. Ordinary Git object identities were compared with local bytes. Matching LFS pointer OID/size was checked for the indexed LFS records; their large payloads were **not re-downloaded** in this documentation task. Prior publication verification records remain the evidence for those uploads. A matching pointer alone is not a fresh full-payload verification.

The Stage 1 parity seed 3305 checkpoint, imitation controls, original CLRS checkpoint and both feature-pilot final checkpoints are indexed with hashes. Do not confuse seed 3306/3307 **controller** artifacts with unavailable additional Stage 1 backbone checkpoints. The Gate-A report explicitly declines multi-backbone replication.

There is no artifact migration in this task. Existing LFS rules are [commit-pinned here](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/ea183970feea1f49677f186e26065ce4e333f709/.gitattributes). Do not rely on `.gitignore` or the source-only default branch to inventory all evidence. Large LFS-backed examples include some receiver confirmation/state/checkpoint files and historical raw traces. Use the recorded hash/size, materialize only required objects, and distinguish an LFS pointer from its payload.

## 2. Regenerate documentation figures

Only nine retained JSON/CSV inputs are required; [result_source_hashes.json](figure_sources/result_source_hashes.json) lists their exact paths/revisions. The plotting scripts never import AGT, Torch, runners or fitting code, and never deserialize checkpoints. No experimental bootstrap, probe or model is rerun: figures use saved aggregate values and saved confidence bounds.

```bash
# Run from a checkout containing this documentation and its listed inputs.
python3 -m venv /tmp/agt-docs-env
/tmp/agt-docs-env/bin/python -m pip install -r docs/figure_sources/requirements.txt
/tmp/agt-docs-env/bin/python scripts/documentation/render_figures.py
/tmp/agt-docs-env/bin/python scripts/documentation/render_figures.py --check-data
/tmp/agt-docs-env/bin/python scripts/documentation/render_diagrams.py
python3 -B scripts/documentation/build_index.py
python3 -B scripts/documentation/validate_docs.py
```

Equivalent commands were executed in an isolated temporary documentation environment for this guide. The plotting library is pinned to Matplotlib3.10.7; resolved versions are in [environment](figure_sources/environment.json). PNG/SVG metadata and fonts can vary across platforms; the numeric-data check is the invariant. Missing/mismatched retained inputs cause an explicit failure; model execution is never a fallback. Reading reports needs none of these packages.

An incomplete source-only checkout cannot regenerate figures by itself. Obtain the exact listed inputs at their pinned revisions, respecting existing LFS storage where applicable. No whole-project archive or large checkpoint copy is required to regenerate the presentation.

## 3. Lightweight checks versus model tests

`validate_docs.py` is a standard-library documentation checker. It checks the new Markdown, local file/image references, curated evidence revisions, diagram graph references, prohibited model imports and preservation hashes. `render_figures.py --check-data` compares plotted data with the retained source values. Neither constructs a model or executes a test runner.

The repository's [tests](../tests/README.md) include real forwards, backwards and training-related fixtures. The full suite is **not** a lightweight documentation check. No project test suite was run for this guide, and no current green CI status is claimed. Historical test results have execution-specific scopes and known failure records.

## 4. Reproduce a scientific experiment

This requires a separately reviewed plan and compute allocation. Read the locked protocol and implementation seal first, then identify data, checkpoint, environment, paths, time/token budgets and stop rules. Create a new isolated run namespace without rewriting historical files. Many runners intentionally refuse an existing `*_started*.json` marker/output directory, so invoking them in place should fail rather than resume a closed study.

|Recorded setting|Evidence and caveat|
|---|---|
|D64, memory16 slots of width256, FP32, batch1|Common pilot settings, not central architecture defaults|
|Python3.12, PyTorch2.10.0+cu130, CUDA13.0, NVIDIA GB10|Recorded CLRS/receiver environment; not a newly verified portability matrix|
|TF32 disabled; original adapter eval-mode flags during gradient recording|Preserved CLRS training settings; `.eval()` does not disable autograd|
|Official CLRS revision `6d5728b9efadf8648413c8c473929a733caacced`|Task-specific sampler export; not claimed byte-equivalent to released TFRecords|
|CLRS train1000 n16, dev32 n16, transfer32 n64|Sealed original examples; n input plus n query tokens; no hints/permutation enforcement|
|Original CLRS1000 updates, LR1e-3, clip1|Full BPTT; final checkpoint only; see execution-specific scope lock for every optimizer detail|
|Receiver study512 common then 512 per arm|Different task/supervision/parameter policy from controller-only Stage 3 or CLRS|

The export's [CPU requirements](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/ea183970feea1f49677f186e26065ce4e333f709/experiments/results/gatec_clrs_insertion_sort_6200_v1/cpu_requirements_locked_v1.txt) describe a historical task-export environment, not a root installation recipe for all AGT work. No root requirements/pyproject lock was found. We do not fabricate a universal `pip install` command or claim experiments can be reproduced from code alone. Some environments emit known NumPy/CUDA-capability warnings; actual passed fixtures and execution records, not warning suppression, are the evidence.

The [prerequisite tests](experiment_index.md#receiver-prerequisites), [CLRS engineering record](experiment_index.md#clrs-engineering) and [feature-pilot starting equivalence](experiment_index.md#feature-complement) document what was actually checked. Their success is not permission to rerun or extend a closed experiment.
