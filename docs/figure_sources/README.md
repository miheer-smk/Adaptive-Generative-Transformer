# Editable figures and provenance

- [figure_provenance.json](figure_provenance.json): six result plots, captions, exact sources/revisions, units, transformations, limitations and output hashes.
- [result_source_hashes.json](result_source_hashes.json): nine required JSON/CSV inputs. Missing or changed evidence causes generation to fail.
- [figure_data.json](figure_data.json): derived plotting values; fractions/percentages and aggregation definitions remain explicit in provenance.
- [diagrams.json](diagrams.json): editable nodes, edges, layout, captions and scopes for four diagrams.
- [Mermaid gallery](mermaid_diagrams.md): GitHub-native sources generated from the same graph as the static SVG/PNG fallbacks.
- [diagram_provenance.json](diagram_provenance.json): generation commands and source maps.
- [milestones.json](milestones.json): curated inventory inputs for the central experiment index.
- [requirements.txt](requirements.txt) and [environment.json](environment.json): isolated presentation tooling, separate from any research environment.

Diagram sources: [architecture](architecture.mmd), [token execution](token_execution.mmd), [evidence map](evidence_map.mmd), [CLRS serialization](clrs_pipeline.mmd). Captions and scope statements are in the gallery and the consuming documents. Sources are implementation-grounded schematics, not AI-generated illustrations.

Generation scripts live under [scripts/documentation](../../scripts/documentation/README.md). They do not import model code, deserialize checkpoints, execute a runner, fit a readout or run a new bootstrap. Existing confidence bounds are plotted as recorded. The [validation record](../maintenance/validation_20261025.json) states exactly which checks were performed and any rendering limitations.
