# Locked Gate-C ood_test evaluation

Mechanical execution: VERIFIED.

| Metric | Result |
|---|---|
| sequences | 64 |
| tokens | 1568 |
| reference_accuracy | 1.0 |
| learned_accuracy | 1.0 |
| reference_mean_ce | 0.001750953351802309 |
| learned_mean_ce | 0.001750953351802309 |
| paired_prediction_disagreements | 0 |
| paired_mean_ce_delta | 0.0 |
| canonical_action_route_fraction | 1.0 |
| canonical_executed_block_fraction | 1.0 |
| exact_sequence_canonical_fraction | 1.0 |
| teacher_accuracy | 1.0 |
| mean_D | 8.0 |
| maximum_D | 8 |
| restart_count | 0 |
| repeat_count | 0 |
| P_distribution | {8: 1568} |

## Results by length

| Length | Sequences | Reference accuracy / CE | Learned accuracy / CE | Canonical tokens | Teacher accuracy |
|---|---|---|---|---|---|
| 17 | 4 | 1.0 / 0.00156812983914 | 1.0 / 0.00156812983914 | 1.0 | 1.0 |
| 18 | 4 | 1.0 / 0.00189765135292 | 1.0 / 0.00189765135292 | 1.0 | 1.0 |
| 19 | 4 | 1.0 / 0.00207366805989 | 1.0 / 0.00207366805989 | 1.0 | 1.0 |
| 20 | 4 | 1.0 / 0.00169233835186 | 1.0 / 0.00169233835186 | 1.0 | 1.0 |
| 21 | 4 | 1.0 / 0.00138680072268 | 1.0 / 0.00138680072268 | 1.0 | 1.0 |
| 22 | 4 | 1.0 / 0.00174712610897 | 1.0 / 0.00174712610897 | 1.0 | 1.0 |
| 23 | 4 | 1.0 / 0.00177421214175 | 1.0 / 0.00177421214175 | 1.0 | 1.0 |
| 24 | 4 | 1.0 / 0.00176761014154 | 1.0 / 0.00176761014154 | 1.0 | 1.0 |
| 25 | 4 | 1.0 / 0.0014538023388 | 1.0 / 0.0014538023388 | 1.0 | 1.0 |
| 26 | 4 | 1.0 / 0.00163773365784 | 1.0 / 0.00163773365784 | 1.0 | 1.0 |
| 27 | 4 | 1.0 / 0.00194812464179 | 1.0 / 0.00194812464179 | 1.0 | 1.0 |
| 28 | 4 | 1.0 / 0.00200317363488 | 1.0 / 0.00200317363488 | 1.0 | 1.0 |
| 29 | 4 | 1.0 / 0.00162165035727 | 1.0 / 0.00162165035727 | 1.0 | 1.0 |
| 30 | 4 | 1.0 / 0.00173647666816 | 1.0 / 0.00173647666816 | 1.0 | 1.0 |
| 31 | 4 | 1.0 / 0.00201649582596 | 1.0 / 0.00201649582596 | 1.0 | 1.0 |
| 32 | 4 | 1.0 / 0.00169025978539 | 1.0 / 0.00169025978539 | 1.0 | 1.0 |

First divergences are recorded in the JSON and first-divergence JSONL. State comparisons assert equality only while full autonomous action prefixes agree.

Teacher measurements evaluate the policy on reference states without affecting reference or learned execution. Primary CE/accuracy use only the final token of each sequence.

These sequences were used in prior diagnostics. Exact overlap with imitation and Stage-1 training data is unknown. OOD lengths exceed declared training lengths. This frozen supervised checkpoint result does not establish task-loss route discovery, adaptive depth, useful diversity, HALT benefit, or superiority.
