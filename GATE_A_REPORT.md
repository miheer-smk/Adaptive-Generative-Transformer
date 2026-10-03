# AGT-v3 Gate A — GDN Recurrent-State Mechanism Report

## Provenance

- Gate-A branch: `research/gdn-state-mechanism`
- HEAD: `c5657630781ba463ce9238a1285d35e96a71ce75`
- Immutable base: `c565763` (`AGT-v3: integrate adaptive executor with persistent token path`)
- D64
- Frozen checkpoint: `experiments/runs/stage1_parity_seed3305_best.pt`
- Fixed route: `[SWA, SWA, GQA, GQA, GDN, GDN, FFN, FFN]`
- Controller optimization: disabled
- Task: binary parity
- Final-token CE only

## A0 — State identification

The recurrent GDN state is the delta-rule matrix `S`, with D64 shape:

`[B, 6, 128, 64]`

= 49,152 scalars per example.

The persistent causal target is:

- GDN1: `state.ctx.gdn[4].committed_state`
- GDN2: `state.ctx.gdn[5].committed_state`

The implementation distinguishes ephemeral GDN state from committed persistent state. The committed state is the recurrent state consumed across token boundaries.

## A3 — Linear representation

Independent standardized linear probes trained on exhaustive L4–L8 representations found:

| Length | GDN1 output | GDN2 output | GDN1 state | GDN2 state |
|---:|---:|---:|---:|---:|
| 8 | 0.9922 | 1.0000 | 1.0000 | 1.0000 |
| 16 | 0.9844 | 1.0000 | 1.0000 | 1.0000 |
| 32 | 0.9512 | 1.0000 | 1.0000 | 1.0000 |
| 64 | 0.9395 | 1.0000 | 1.0000 | 1.0000 |
| 128 | 0.9414 | 1.0000 | 1.0000 | 1.0000 |

Untrained controls were approximately chance.

**Conclusion:** persistent GDN state is strongly parity-associated in the trained model. This is representational evidence, not causal proof.

## A4 — State ablation

Only persistent `committed_state` was zeroed between tokens.

| Condition | L8 CE | L8 acc. | L128 CE | L128 acc. |
|---|---:|---:|---:|---:|
| Clean | 0.001761 | 1.0000 | 0.001818 | 1.0000 |
| GDN1 zero | 1.200073 | 0.8125 | 1.274523 | 0.7988 |
| GDN2 zero | 0.801575 | 0.8750 | 0.790391 | 0.8750 |

The target convolution buffer and non-target GDN context were preserved.

**Conclusion:** persistent GDN state is causally involved in parity behavior.

## A5 — State shuffle

Only persistent state was batch-shuffled using a fixed derangement.

| Condition | L8 CE | L8 acc. | L128 CE | L128 acc. |
|---|---:|---:|---:|---:|
| Clean | 0.001761 | 1.0000 | 0.001815 | 1.0000 |
| GDN1 shuffle | 1.369105 | 0.7812 | 1.352675 | 0.7871 |
| GDN2 shuffle | 0.001763 | 1.0000 | 0.151354 | 0.9766 |

**Conclusion:** state ownership matters, particularly for GDN1. The two GDN states are not behaviorally interchangeable under this intervention.

## A6 — State patching

Opposite-parity donor/recipient pairs were matched at an 8-token prefix. Donor committed state was copied into the recipient immediately before token 9.

GDN1, GDN2, and joint patching produced essentially zero donor-probability increase and zero donor-label prediction flips.

Same-parity donor controls also showed no meaningful change.

**Conclusion:** state-level parity transfer was **not established**.

## A7 — Controls

The intervention controls were:

| Condition | L8 CE | L8 acc. | L128 CE | L128 acc. |
|---|---:|---:|---:|---:|
| Clean | 0.001761 | 1.0000 | 0.001815 | 1.0000 |
| GDN1 state zero | 1.200073 | 0.8125 | 1.310733 | 0.7930 |
| GDN2 state zero | 0.801575 | 0.8750 | 0.854465 | 0.8652 |
| GDN1 output zero | 3.176485 | 0.5000 | 3.178188 | 0.4922 |
| GDN2 output zero | 3.162324 | 0.5000 | 3.068963 | 0.5137 |
| Null state intervention | 0.001761 | 1.0000 | 0.001815 | 1.0000 |

The null intervention was bitwise identical to clean at both lengths.

**Conclusion:** the causal state effects are not explained by intervention machinery alone. Output ablation is more destructive than isolated persistent-state ablation, consistent with distributed/redundant computation.

## A8 — Replication

Only one successful checkpoint artifact is currently present:

`experiments/runs/stage1_parity_seed3305_best.pt`

Seed3306 and seed3307 runs were previously observed, but their checkpoint artifacts are not present in the current working tree.

Therefore multi-checkpoint replication is **not claimed**.

## Gate-A justified claim

Under the frozen D64 Stage-1 fixed-route parity protocol, the trained AGT-v3 model exhibits a strong parity association in persistent GDN recurrent state, and intervention on that persistent state causally affects parity behavior.

The evidence is compatible with distributed or redundant computation involving the GDN pathways.

The experiments do **not** establish that persistent GDN state alone is the unique parity representation, nor did state patching demonstrate parity-specific transfer.

Adaptive routing/controller competence, D768 reproduction, and real-dataset validation remain unestablished.

## Artifacts

- `experiments/gateA_a3_state_vs_output.json`
- `experiments/gateA_a4_state_ablation.json`
- `experiments/gateA_a5_state_shuffle.json`
- `experiments/gateA_a6_state_patching.json`
- `experiments/gateA_a7_controls.json`
- `experiments/runs/stage1_parity_seed3305_best.pt`
