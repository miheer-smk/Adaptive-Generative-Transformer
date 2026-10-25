# Offline repository presentation tools

Run from the repository root. These tools operate on documentation, source text and retained JSON/CSV only. They never import AGT or Torch, deserialize a checkpoint, execute a model runner or fit anything.

```bash
python3 -B scripts/documentation/build_index.py
python3 -B scripts/documentation/validate_docs.py

python3 -m venv /tmp/agt-docs-env
/tmp/agt-docs-env/bin/python -m pip install -r docs/figure_sources/requirements.txt
/tmp/agt-docs-env/bin/python scripts/documentation/render_figures.py
/tmp/agt-docs-env/bin/python scripts/documentation/render_figures.py --check-data
/tmp/agt-docs-env/bin/python scripts/documentation/render_diagrams.py
```

`build_index.py` checks curated local evidence and renders the Markdown/machine-readable indexes. `render_figures.py` validates nine source hashes and plots saved measurements; `--check-data` checks derived values without changing assets. `render_diagrams.py` renders four vector/raster fallbacks and editable Mermaid from the shared graph. `validate_docs.py` uses the standard library for link/image, graph, provenance, source-import and preservation checks; it never runs project tests.

The plotting environment is independent of `agt-v3-env`. [Resolved versions](../../docs/figure_sources/environment.json), [provenance](../../docs/figure_sources/figure_provenance.json) and [reproducibility limitations](../../docs/reproducibility.md) are recorded. Inputs unavailable in a source-only checkout must be obtained at their listed revisions; no model execution is a fallback. SVG/PNG byte identity can depend on platform/fonts, while the retained numerical data must match exactly.

The dated preservation check compares against this documentation task's starting inventory. It is a maintenance check, not a replacement for each study's scientific integrity seal. It may legitimately fail after future authorized changes; never rewrite the old inventory to hide a change.
