# Seed-4700 diagnostic protocol resolution

Artifact inspection only. No model was constructed, no new noise was generated, and no continuation, gradient probe, training or optimizer update ran. The six selections in `gatec_st_action_value_4700_identity_precheck_v1.json` are unchanged. The preserved bytecode exception remains disclosed; historical experiment files and relevant source hashes match their recorded identities.

## Existing noise mapping: no cross-split rule

**No existing canonical mapping converts a training sequence ID into historical evaluation-tape-0 coordinates.**

`run_gatec_fresh_task_discovery.py:124–137` assigns training visits IDs such as `epoch01/train/L16/row0`; the epoch and source ID are recorded in `training_schedule.json`. Lines 243–245 construct training noise from those visits with seed 4702, and evaluation noise from the ordered final panel with child seeds derived from root seed 4704.

`run_gatec_paired_task_maintenance.py:39–63` defines the shared `IndexedNoise` implementation. A private CPU generator creates successive FP32 `[L,15,10]` uniform tensors in the supplied sequence-list order. Lookup is exactly `uniforms[sequence_id][token, decision, action_slot]`. Sequence IDs are dictionary keys, not hashed seeds or cross-split aliases. Injection substitutes only that recorded uniform row into the original sampler. The maintenance manifest and harness use separate training, validation and OOD namespaces; they provide no training-to-evaluation translation.

Historical evaluation **tape 0** means the first tape: manifest key `final_tape_1`, file `noise_final_tape_1.pt`, sampler seed **924088217**, derived from root **4704**. Its schema has 96 evaluation IDs. Neither `train/L16/row0` nor `train/L16/row1` is present. The training tape instead contains 1,040 epoch-qualified visit IDs. An epoch choice for evaluation is not specified.

Verified saved examples, using exact dictionary lookup and reconstruction of an already-recorded action (no new model evaluation):

| Artifact | Line | Recorded noise coordinate |
|---|---:|---|
| Fresh experiment `decisions.jsonl.gz` | 57001 | `[4702, epoch01/train/L16/row0, 0, 0]` |
| Same | 502350 | `[924088217, val/L16/row0, 0, 0]` |
| Same | 502359 | `[924088217, val/L16/row0, 1, 0]` |

The inspected source, schemas, schedules, traces and bounded mapping/coordinate searches contained no additional alias table or ID-derived seed rule.

## Proposed diagnostic-local rule — not implemented

For the missing **training-prefix** noise, propose an append-only extension:

1. Use a new private CPU generator with the historical tape-0 sampler seed **924088217** and the recorded runtime/sampler conventions.
2. Recreate all 96 original final-panel tables in their original order. Require bitwise equality with every historical table before any measurement.
3. Continue that generator to append `train/L16/row0`, then `train/L16/row1`, in their original training-manifest order, each shaped `[16,15,10]`.
4. Save and hash a **new** extension artifact before measurement. Existing evaluation sequences retain their exact historical tables; training prefixes use their newly appended tables. Never edit historical noise or substitute another sequence's coordinates.

This is new diagnostic prefix noise for the training sequences, not recovered historical evaluation provenance. It requires an explicit protocol adoption before the action-value diagnostic proceeds.

For the two independent diagnostic banks, propose one private CPU generator per bank: **4800** for continuation values and **4801** for ST probes. Generate tables before outcomes in nested order: tape index `0..7`, then the unchanged six-sequence selection order. Each table is FP32 `[L,15,10]`, using the existing `torch.rand` distribution. The sealed lookup key is:

`(bank, tape_index, exact_sequence_id, internal_token_index, within_token_decision_index, successor_action_slot)`.

Exclude checkpoint, branch action, prefix condition and outcomes from this key. Thus all eight ADVANCE branches share continuation coordinates, both checkpoints share logical coordinates, and the ST bank remains independent. Execution reads sealed tensors without advancing a flat random stream. Fifteen decision positions cover the legal P=8 bound; they are not a stopping cap. Preserve the original clamp/Gumbel transform, tau=2, masks and successor slots 0–9; HALT remains excluded.

No extension or bank has been generated, sealed, implemented or used in this task.

## Token numbering

The **experiment artifact convention is zero-based**:

- `data/task_a.py:100–110` preserves bit order directly in `x = token_features[bits]`.
- `StochasticPath.run` in `run_gatec_paired_task_maintenance.py:198–235` iterates `range(L)`, reads `x[:, self.token]`, and records that same integer as `token`.
- `ObservedPath.run` in `replay_gatec_learned_policy.py:249–285` follows the same convention for fixed and learned paths.
- `src/agt.py:425` commits with `token_index=work_state.tokens_committed`; `terminal_commit.py:93,239` starts this count at zero and increments it once per token.
- Saved tape-0 token records, lines 43745–43746 of `tokens.jsonl.gz`, show `token=0` with incoming/outgoing commit counts 0/1, followed by `token=1` with counts 1/2.

Normalize probe A to **human-facing token number 2 / internal index 1 / decision index 0**, after exactly one processed and committed token. The final-token probes are human 16/internal 15 for length 16 and human 32/internal 31 for length 32. A human-facing *first token* would instead be internal index 0. The original unqualified English cannot independently establish author intent; this explicit normalization follows the actual diagnostic trace convention.

## Identities and remaining boundary

| Artifact | SHA-256 |
|---|---|
| Fresh experiment manifest | `5c891e9b6939822cac7e937353c4077aae909d40dc405c01c5ab1a72d769bf38` |
| Historical evaluation tape 0 | `8161e35e5bb36183b294fa54bb3e2d260c4420b4f7ee22c4907948882f9d17f7` |
| Historical training noise | `0455a5fe5f550b7477208134afec98cb27b15102b36030b010afb9f6614db238` |
| Locked six-sequence precheck | `c904cabc071b81796feb78ff6e338d4d29eac9258b5923c3a3387575da9c2dd8` |

The companion JSON lists exact consulted paths, source locations and file hashes, including the maintenance manifest, canonical replay records, fresh traces and generator. The sole operational boundary remaining is adoption of a **new** training-prefix extension rule; no existing artifact supplies it. Production semantics, weights, historical noise and the six selections were not changed. No ST action-value experiment or commit occurred.
