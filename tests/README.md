# Tests and recorded scope

These tests cover execution accounting, routing, gradients, state isolation, caches and task/research fixtures. Many instantiate AGT, run forwards/backwards or require CUDA. **Do not run the full suite as a documentation check.** No fresh test-suite or green-CI claim accompanies this guide.

Historical pass/fail/skip counts apply to the precise revision and environment in each report. Accounting changes intentionally invalidate some legacy budget assumptions; later focused prerequisite suites do not erase those records. Consult the [experiment index](../docs/experiment_index.md) and [development guide](../docs/development.md) before choosing a suite.

The safe presentation-only checker is separate:

```bash
python3 -B scripts/documentation/validate_docs.py
```

It reads documentation and retained numerical files without importing or executing the model.
