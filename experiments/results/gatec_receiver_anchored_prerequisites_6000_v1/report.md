# Receiver-anchored prerequisites — one bounded attempt

Status: STOPPED / PARTIAL

No training, optimizer step, parameter update, study checkpoint, study profiling, ablation, confirmation or commit occurred.
The authoritative protocol is conversation text. The separately hashed clarification and binding documents do not claim a local original-protocol identity.

## Executed tests

Pytest exit code: 1. Exact tests/results: pytest_results.xml; selection: test_selection.json.
CPU and CUDA results are separately keyed in test_measurements.json. CPU skips for accelerator-only probes do not count as CUDA evidence.

## Forward and gradient evidence

P0 comparisons cover each of eight control tokens, complete typed returned states, exact metadata, macro tensors and commits against fresh same-parameter canonical execution. Archived control CE is checked separately; historical full-tensor reproduction is not claimed.
P1 checks actual blocks [0,1,2,3,3,5,6,7], including the unselected GDN4 terminal entry. Exhaustive candidate construction is forbidden.
Gradient measurements compare original continuation gradients with the complete cloned-state VJP. Native gradients and a genuinely disconnected explicit detached-state negative control are separate. Individual state fields may have connected zero or disconnected gradients.
Receiver parameter values have requires_grad=False; native/writer parameters remain differentiable. Model/parameter/buffer identities are checked after fixtures. Optimizer ownership uses synthetic moments without any optimizer step.

## Accounting

```json
{}
```

## Resource estimate

```json
{
  "required_overall_tokens_per_second": 12.367777777777778,
  "measured_fixture_rates": [],
  "conservative_all_tokens_at_slowest_fixture_rate_hours": null,
  "not_a_guarantee": "fixture time includes observation/ledger overhead but excludes optimizer work, complete study I/O and statistical analysis; no deployment latency measured"
}
```
A representative effective batch for each arm uses 112 model tokens and full graphs at lengths16/24. This is disposable prerequisite work, not an acquisition update. No empirical learning or study-cost verdict follows.

## Integrity and execution boundary

All 704 pre-existing files rehashed: True. Original tracked diff unchanged: True.
Historical bytecode exception retained: experiments/results/gatec_fresh_task_discovery_4700_v1/artifact_integrity_incident.json retained unchanged; no cache restoration or historical rewriting
Only this new evidence directory was permitted for writes. No automatic retry or training continuation is authorized.
