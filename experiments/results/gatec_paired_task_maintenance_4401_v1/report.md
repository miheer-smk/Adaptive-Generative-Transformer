# Bounded warm task-loss routing with matched ST-Gumbel control

Strong bounded-maintenance criterion: **True**.
Manifest SHA-256 `10ebe3df594899975212b731b1eaaf0bdf66daebc1b95b85f2345569c1976430`. T: exactly 104 final-parity CE updates from seed4401 imitation; U: zero updates.

| Phase | Arm | Accuracy | Mean CE | Canonical token / sequence fraction | RESTART / REPEAT | Candidate constructions |
|---|---|---|---|---|---|---|
| initial_deterministic_val | U | 1.0 | 0.00172144223828 | 1.0 / 1.0 | 0 / 0 | 0 |
| initial_stochastic_val | U | 1.0 | 0.00172144223828 | 1.0 / 1.0 | 0 / 0 | 4160 |
| final_deterministic_val | U | 1.0 | 0.00172144223828 | 1.0 / 1.0 | 0 / 0 | 0 |
| final_deterministic_val | T | 1.0 | 0.00172144223828 | 1.0 / 1.0 | 0 / 0 | 0 |
| final_deterministic_ood_test | U | 1.0 | 0.00176086770625 | 1.0 / 1.0 | 0 / 0 | 0 |
| final_deterministic_ood_test | T | 1.0 | 0.00176086770625 | 1.0 / 1.0 | 0 / 0 | 0 |
| final_stochastic_val | U | 1.0 | 0.00172144223828 | 1.0 / 1.0 | 0 / 0 | 4160 |
| final_stochastic_val | T | 1.0 | 0.00172144223828 | 1.0 / 1.0 | 0 / 0 | 4160 |
| final_stochastic_ood_test | U | 1.0 | 0.00176086770625 | 1.0 / 1.0 | 0 / 0 | 12544 |
| final_stochastic_ood_test | T | 1.0 | 0.00176086398824 | 0.9993622448979592 / 0.984375 | 0 / 1 | 12544 |

Controller parameter change L2: 0.362365156412; changed scalar values: 103048.
Controller gradient L2 mean/min/max: {'mean_l2': 8.951998135696811e-05, 'minimum_l2': 6.263840317081955e-07, 'maximum_l2': 0.0014046646193351573, 'finite_updates': 104, 'nonzero_update_count': 104}.

Raw sequences were recomputed under each arm. Full BPTT across the sequence and committed state/caches was retained; frozen backbone computations remained in T autograd. No extra detach, clipping, scheduler or auxiliary loss.

Noise was indexed by sequence/update, token, decision and ten successor slots. Validation used the same recorded noise before and after updates. U re-evaluation matched exactly. Nine physical speculative bank calls per stochastic decision are distinct from P=8 realized executions; deterministic paths construct no candidates.

Canonical full-route log-probabilities are diagnostics on independent reference states, never an optimization term. Controller probability, margin, per-sequence loss, gradients, parameter deltas, routes and first divergences are retained in JSON/JSONL.

Interpret sampling effects using initial U stochastic versus deterministic validation; update effects using final T versus matched U. Preserved task performance alone does not establish informative route learning. If routes and task losses remain unchanged, finite parameter updates and confidence changes do not demonstrate route discovery or task improvement.

Single fixed-noise panels limit stochastic conclusions. Six validation samples overlap reconstructed imitation training, two are duplicates, and prior panel exposure is recorded. No untouched-generalization claim.

This experiment does not establish fresh-init task-loss trainability, useful diversity, state-conditioned topology, adaptive depth or learned HALT. Production code, frozen backbone and historical checkpoints were not changed. No commit.
