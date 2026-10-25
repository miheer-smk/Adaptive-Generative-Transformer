# Artifact preservation and publication

Scientific evidence includes unsuccessful runs, inspection stops, failed calls, implementation seals, data exclusions and integrity exceptions. Preserve their exact classifications and paths. The [experiment index](experiment_index.md) provides access without relocating sealed directories.

## Repository content

Keep source, versioned research implementations, protocols, environment records, manifests, reports, numerical summaries, tests and compact figures discoverable in Git. Preserve existing checkpoints and large evidence where they already live. Do not introduce an artifact migration as a documentation cleanup.

New large payloads may need separately approved LFS or external storage. Reference them with immutable object identity, size, retrieval location, source revision and access requirements. A pointer or hash is not a payload. Do not claim accessibility unless verified; do not silently replace a missing artifact with a newly generated one. Git cannot preserve all filesystem metadata, caches or empty directories.

## Existing record

The source-history default branch does not contain all research. Use [commit-pinned evidence links](experiment_index.md), not a claim that every file is on the default branch. Existing [LFS rules](https://github.com/miheer-smk/Adaptive-Generative-Transformer/blob/ea183970feea1f49677f186e26065ce4e333f709/.gitattributes) cover some large artifacts. This documentation task checked indexed pointer OIDs/sizes against local payloads but did not re-download those payloads. Prior publication records remain the upload-verification evidence.

Do not move or rename a sealed report, checkpoint, input, script with path-sensitive reproduction, or protected identity. Do not erase historical bytecode exceptions. A later correction must retain the original and explain its scope in a new linked record. Do not merge unrelated experimental branches indiscriminately.

## Completed-run publication

Keep an active run isolated. At its prescribed boundary, validate reports, logs, data identities, checkpoints, counters and checksums before publication. Use a unique branch or an explicitly intended research branch; never force-push or rewrite old commits. Verify the remote commit, expected paths, ordinary blob content and any LFS payloads before declaring success. GitHub is the established remote evidence location; publication is not permission for another experiment.

Disk pressure does not justify deleting the only usable copy, truncating traces or rewriting history. Avoid unnecessary duplicates and broad staging. A storage or upload blocker must be reported. Changes to retention, migration, automatic publishing or deletion policy require separate review.

## Presentation artifacts

Figures belong under [docs/figures](figures/README.md), with editable sources, read-only numerical generation scripts, source hashes and scope limitations under [figure_sources](figure_sources/README.md). They are derived presentation, not replacement experimental evidence. Documentation preservation records are dated and separate from scientific integrity manifests.
