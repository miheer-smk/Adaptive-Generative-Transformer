# Gate-C bounded imitation seed replication

Strict replication success: **True**.
Manifest SHA-256: `b661d0042c56880fc324a7efd758ab744753d7fc63fa6770838cb50408c6fb0c`. Fresh seeds: 4401, 4402; 5,000 updates each; final checkpoints only.

| Policy | Split | Teacher accuracy | Canonical token / sequence recovery | Reference / learned accuracy | Reference / learned CE | D mean/max | RESTART / REPEAT |
|---|---|---|---|---|---|---|---|
| historical_3305 | val | 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.00172144223828 / 0.00172144223828 | 8.0 / 8 | 0 / 0 |
| historical_3305 | ood_test | 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.00176086770625 / 0.00176086770625 | 8.0 / 8 | 0 / 0 |
| fresh_4401 | val | 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.00172144223828 / 0.00172144223828 | 8.0 / 8 | 0 / 0 |
| fresh_4401 | ood_test | 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.00176086770625 / 0.00176086770625 | 8.0 / 8 | 0 / 0 |
| fresh_4402 | val | 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.00172144223828 / 0.00172144223828 | 8.0 / 8 | 0 / 0 |
| fresh_4402 | ood_test | 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.00176086770625 / 0.00176086770625 | 8.0 / 8 | 0 / 0 |

Balanced panel: four samples per length, two per parity, at every length 4–32. Panel fixed before training; no evaluation-based checkpoint selection.

Training and prior-evaluation overlaps are in the manifest with exact panel IDs. Overlapping short inputs are not unseen-sequence evaluation. Stage-1 exposure and unretained historical probe membership remain unknown.

Both fresh controllers were independently initialized by the original full-model initialization procedure. Raw teacher features were reused from the fingerprinted original collection code. Full minibatch indices, RNG states, initialization weights, training losses, final weights and hashes are retained.

Replay uses the unchanged completed-token production executor. Selected actions, actual bank calls, masks, P/D accounting, in-flight committed-state/GDN purity, seven-entry version increments and exactly one terminal commit per completed token were checked. No candidate construction during deterministic inference.

While complete action prefixes agree, the established state/feature comparison standard and exact persistent/output checks apply. Post-route-divergence states are not compared as equivalent decision-index computations.

Absolute parity and reference equivalence are distinct. Reproducing a reference error would preserve imitation equivalence but limit a perfect-parity claim.

This bounded result does not establish task-loss discovery, useful diversity, state-conditioned topology, adaptive depth, learned HALT or superiority. No production change, backbone update, checkpoint overwrite, additional seed, tuning, or commit.
