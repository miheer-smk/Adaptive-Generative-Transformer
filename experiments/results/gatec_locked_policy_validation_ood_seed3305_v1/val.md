# Locked Gate-C val evaluation

Mechanical execution: VERIFIED.

| Metric | Result |
|---|---|
| sequences | 52 |
| tokens | 520 |
| reference_accuracy | 1.0 |
| learned_accuracy | 1.0 |
| reference_mean_ce | 0.0018481857103044884 |
| learned_mean_ce | 0.0018481857103044884 |
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
| P_distribution | {8: 520} |

## Results by length

| Length | Sequences | Reference accuracy / CE | Learned accuracy / CE | Canonical tokens | Teacher accuracy |
|---|---|---|---|---|---|
| 4 | 4 | 1.0 / 0.00191533673205 | 1.0 / 0.00191533673205 | 1.0 | 1.0 |
| 5 | 4 | 1.0 / 0.00194772687973 | 1.0 / 0.00194772687973 | 1.0 | 1.0 |
| 6 | 4 | 1.0 / 0.00175684012356 | 1.0 / 0.00175684012356 | 1.0 | 1.0 |
| 7 | 4 | 1.0 / 0.00190612493316 | 1.0 / 0.00190612493316 | 1.0 | 1.0 |
| 8 | 4 | 1.0 / 0.00168215666781 | 1.0 / 0.00168215666781 | 1.0 | 1.0 |
| 9 | 4 | 1.0 / 0.00175385471084 | 1.0 / 0.00175385471084 | 1.0 | 1.0 |
| 10 | 4 | 1.0 / 0.00186399114318 | 1.0 / 0.00186399114318 | 1.0 | 1.0 |
| 11 | 4 | 1.0 / 0.00183616552386 | 1.0 / 0.00183616552386 | 1.0 | 1.0 |
| 12 | 4 | 1.0 / 0.00184433258255 | 1.0 / 0.00184433258255 | 1.0 | 1.0 |
| 13 | 4 | 1.0 / 0.00192434934434 | 1.0 / 0.00192434934434 | 1.0 | 1.0 |
| 14 | 4 | 1.0 / 0.0019347150228 | 1.0 / 0.0019347150228 | 1.0 | 1.0 |
| 15 | 4 | 1.0 / 0.0018173729477 | 1.0 / 0.0018173729477 | 1.0 | 1.0 |
| 16 | 4 | 1.0 / 0.00184344762238 | 1.0 / 0.00184344762238 | 1.0 | 1.0 |

First divergences are recorded in the JSON and first-divergence JSONL. State comparisons assert equality only while full autonomous action prefixes agree.

Teacher measurements evaluate the policy on reference states without affecting reference or learned execution. Primary CE/accuracy use only the final token of each sequence.

These sequences were used in prior diagnostics. Exact overlap with imitation and Stage-1 training data is unknown. OOD lengths exceed declared training lengths. This frozen supervised checkpoint result does not establish task-loss route discovery, adaptive depth, useful diversity, HALT benefit, or superiority.
