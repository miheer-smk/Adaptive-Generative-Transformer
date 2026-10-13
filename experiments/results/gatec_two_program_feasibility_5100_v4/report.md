# V4 two-program feasibility

{
  "status": "STOPPED",
  "boundary": "remaining current screen; P1 histories and boundary combinations",
  "failure": {
    "error": "CurrentRejection(\"Mandatory current screen failed: {'phase': 'current', 'sequence_id': 'acquisition/L16/row03', 'history_program': 1, 'program': 0}: {'checks': {'current_correct': False, 'executions': True, 'commits': True}, 'accepted': False}\")",
    "traceback": "Traceback (most recent call last):\n  File \"/home/iiitn/Desktop/Miheer Projects/AGT/AGT_v3_POC/experiments/results/gatec_two_program_feasibility_5100_v4/runner_v4.py\", line 310, in execute\n    accepted(sa, case, canonical=canonical)\n  File \"/home/iiitn/Desktop/Miheer Projects/AGT/AGT_v3_POC/experiments/results/gatec_two_program_feasibility_5100_v4/runner_v4.py\", line 199, in accepted\n    raise CurrentRejection(f'Mandatory current screen failed: {case}: {result}')\nCurrentRejection: Mandatory current screen failed: {'phase': 'current', 'sequence_id': 'acquisition/L16/row03', 'history_program': 1, 'program': 0}: {'checks': {'current_correct': False, 'executions': True, 'commits': True}, 'accepted': False}\n",
    "boundary": "remaining current screen; P1 histories and boundary combinations",
    "first_mechanical": null,
    "completed_tokens": 1979,
    "future_calls": 0,
    "elapsed_seconds": 56.30407442105934,
    "classification": "CURRENT_CANDIDATE_REJECTED"
  },
  "P0_archival": true,
  "P0_current": true,
  "entire_current_screen": null,
  "monitor": {
    "status": "REPRODUCED",
    "role": "non-gating monitor diagnostic",
    "evaluated_records": 128,
    "first_mismatch": null,
    "parameter_sha256": "589b08b5a0280c15558e4968473c651d48a658cab58e6f617b9d569e3895cb4b",
    "provenance": "original stage3_measurements.load_model(3401, cuda); full constructor RNG ordering; isolated RNG; retained features"
  },
  "future_calls": 0,
  "completed_tokens": 1979,
  "bank_calls": 15832,
  "commits": 1979,
  "integrity": {
    "protected_count": 617,
    "changed": [],
    "tracked_diff_unchanged": true,
    "historical_exception": "experiments/results/gatec_fresh_task_discovery_4700_v1/artifact_integrity_incident.json retained unchanged; no cache restoration or historical rewriting"
  }
}

Engineering checks passed on completed calls; failed calls remain separately accounted. Historical tensor comparisons and historical summary comparisons are separate from full within-v4 reference/adapter evidence. Monitor status is non-gating and does not authorize a claim of complete diagnostic reproduction. No future or final-joint verdict is made for unexecuted stages. No gradients, training, optimizer, selector, production changes, checkpoint updates or commits occurred. V1 remains STOPPED BEFORE P1; v2 is inspection-only; v3 remains specification-barrier stop.

P1 candidate rejected at mandatory current screening; future compatibility and final joint acceptance remain UNVERIFIED. This is bounded evidence, not a general program-portfolio rejection.
