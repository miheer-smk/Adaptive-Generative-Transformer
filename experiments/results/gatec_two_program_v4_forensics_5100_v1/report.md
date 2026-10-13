# Offline v4 forensic analysis: recorded switching failure

This standard-library-only analysis used existing v4 evidence. No model or checkpoint was loaded, no forward/backward or test executed, and no future or missing boundary was reconstructed. No bootstrap or new inferential analysis was performed.

The case scope was sealed before derived measurements: primary acquisition/L16/row03; controls row00,row01,row02, identified by the preceding accepted P1→P0 calls in retained execution order. These partial ordered cases are not full-panel P1 performance estimates. V4 still rejects P1 at its current/switch contract.

## Matched four-cell observations

|Case|History → boundary|Recorded truth|Prediction|Recorded CE|V4 acceptance|
|---|---|---:|---:|---:|---|
|row00|P0 → P0|1|1|0.00209814589471|EXECUTED_ACCEPTED|
|row00|P0 → P1|1|1|0.00219247629866|EXECUTED_ACCEPTED|
|row00|P1 → P0|1|1|0.00210016802885|EXECUTED_ACCEPTED|
|row00|P1 → P1|1|1|0.00219330890104|EXECUTED_ACCEPTED|
|row01|P0 → P0|0|0|0.00142341828905|EXECUTED_ACCEPTED|
|row01|P0 → P1|0|0|0.00145413004793|EXECUTED_ACCEPTED|
|row01|P1 → P0|0|0|0.00142472772859|EXECUTED_ACCEPTED|
|row01|P1 → P1|0|0|0.00144079793245|EXECUTED_ACCEPTED|
|row02|P0 → P0|0|0|0.00162695546169|EXECUTED_ACCEPTED|
|row02|P0 → P1|0|0|0.00213228655048|EXECUTED_ACCEPTED|
|row02|P1 → P0|0|0|0.00150543323252|EXECUTED_ACCEPTED|
|row02|P1 → P1|0|0|0.00168170116376|EXECUTED_ACCEPTED|
|row03|P0 → P0|1|1|0.00222923047841|EXECUTED_ACCEPTED|
|row03|P0 → P1|1|1|0.00255841296166|EXECUTED_ACCEPTED|
|row03|P1 → P0|1|0|6.41863965988|EXECUTED_REJECTED|
|row03|P1 → P1|—|—|—|UNOBSERVED|

Row03 P0→P1 is a genuinely retained same-input boundary, not inferred. Row03 P1→P1 is unobserved. Same-input canonical comparison is row03 P0-history/P0-boundary. The supplied-state reference under P1 history is a different comparison: it uses the same divergent incoming state as the failed adapter. It is not a canonical-history oracle.

Recorded labels/predictions/CE match the trace and CSV associations. CE and probabilities were checked arithmetically from recorded logits using the locked numerical tolerance; these checks do not replace published results. No action-position grouping was used to infer tokens. Complete crosschecks are retained in arithmetic_and_identity_crosschecks.json.

## Representation and coverage

- **h_before,h_after,h0,memory_read,memory_mean**: Retained full array. Each selected prefix/boundary bank decision. 
- **pointer_one_hot,budget,script-provider logits**: Retained full array. Adapter/controller observations only; unavailable on reference-hook prefix traces. Script logits are not learned-controller or seed3401 monitor logits. No normalized395-vector for these selected cases.
- **output logits and probabilities**: Retained full array. Completed prefix and boundary outputs. Published outputs, not model recomputation
- **prediction,label,correct**: Exact metadata. Completed prefix and boundary outputs. Prefix readouts are descriptive, not new acceptance gates
- **CE**: Retained summary. Published scalar loss after each completed token. Arithmetic from recorded logits is corroboration only
- **persistent/context s,M,GDN state/buffer,KV/SWA keys/values**: Retained summary. Bank-observer incoming snapshots, and current-case commit_pre/commit. Only shape/l2/max_abs; no elementwise state distance or equality from equal summaries
- **terminal at current boundary**: Retained full array. current_cases.summary.archival_commit.terminal; cross-check last h_after. 
- **prefix terminal**: Retained full array. Last bank h_after; no separately retained prefix commit terminal record. 
- **prefix commit_pre/commit**: Missing/unavailable. Direct commit-time JSON snapshots absent; observer appends archival_commit after writing token trace. Next-token incoming summary may be linked via exact recorded state fingerprint
- **returned-state fields: dtype/shape/cache versions/window counts/tokens_committed/_consumed**: Exact metadata. Every token trace state_metadata. No explicit independent rotary-position values; logical token/window counters only
- **incoming/outgoing state SHA256**: Exact metadata. Every completed token. Fingerprints can bind observations; do not expose raw state arrays
- **full state equality/purity/storage isolation/commit payload comparisons**: Runtime comparison outcome without retained underlying tensors. Archival comparison flags; published gates/reports; adapter assertion path. No new full-state equality test; selected supplied-reference agreement is reconstructed descriptively only for retained live arrays and fingerprints
- **serialized state bundles committed_states_00/01/02.pt**: Missing/unavailable for this analysis. Files exist; deliberately not opened/deserialized; v4 code says they contain incoming/root clones. Not claimed absent on disk; their actual serialized tensor schema is not inspected. Row03 bundle was never created.
- **GDN temporary state,gate/projection internals,full memory slots/raw committed states**: Missing/unavailable in JSON. Not retained as full arrays in selected readable records. Cannot identify a semantic bit, causal component or information loss from summaries
- **row03 P1→P1 and all suffix continuations**: Missing/unavailable. Not executed; no reconstruction or interpolation permitted. 

Prefix JSON lacks direct commit snapshots because the observer writes the token trace before appending archival_commit to its returned summary (sealed research_programs_v4.py:390–396). Boundary current_cases summaries retain commit_pre/terminal/commit. A next incoming prefix/boundary snapshot is used as a returned-state observation only when its explicit case/history/token identity is unique and its incoming hash matches the previous outgoing hash. All such links and their coverage are recorded; this is not inference of raw tensor equality from summary equality.

## Paired prefix timelines and first retained differences

- **acquisition/L16/row00**: first prescribed action/block difference={'token': 0, 'decision': 6, 'P0_action': 6, 'P1_action': 8, 'P0_blocks': [6], 'P1_blocks': [5], 'interpretation': 'Prescribed compound program substitution, not a defect'}; first retained full-array difference: token0/decision6/h_after, max|Δ|=0.14370501, actual-array L2(Δ)=1.11096855. Final same-P0 boundary first retained live-array difference: decision0/h_before, max|Δ|=0.116220057.
- **acquisition/L16/row01**: first prescribed action/block difference={'token': 0, 'decision': 6, 'P0_action': 6, 'P1_action': 8, 'P0_blocks': [6], 'P1_blocks': [5], 'interpretation': 'Prescribed compound program substitution, not a defect'}; first retained full-array difference: token0/decision6/h_after, max|Δ|=0.14370501, actual-array L2(Δ)=1.11096855. Final same-P0 boundary first retained live-array difference: decision0/h_before, max|Δ|=0.291310847.
- **acquisition/L16/row02**: first prescribed action/block difference={'token': 0, 'decision': 6, 'P0_action': 6, 'P1_action': 8, 'P0_blocks': [6], 'P1_blocks': [5], 'interpretation': 'Prescribed compound program substitution, not a defect'}; first retained full-array difference: token0/decision6/h_after, max|Δ|=0.14370501, actual-array L2(Δ)=1.11096855. Final same-P0 boundary first retained live-array difference: decision0/h_before, max|Δ|=0.101203501.
- **acquisition/L16/row03**: first prescribed action/block difference={'token': 0, 'decision': 6, 'P0_action': 6, 'P1_action': 8, 'P0_blocks': [6], 'P1_blocks': [5], 'interpretation': 'Prescribed compound program substitution, not a defect'}; first retained full-array difference: token0/decision6/h_after, max|Δ|=0.596541882, actual-array L2(Δ)=4.60245388. Final same-P0 boundary first retained live-array difference: decision0/h_before, max|Δ|=2.80432951.

The differing prefix step executes A6/block6 for P0 versus REPEAT/block5 for P1. Later pointer/mask differences follow that prescribed substitution; they do not establish a defect. Prefix h0 and memory-read arrays and input summaries remain constant within each recorded token. Later tokens have differing incoming snapshots and preparation arrays. Full h, anchor, read and mean comparisons at every recorded decision are in array_differences.json/csv; all15-token paired timelines include logged output predictions/CE without turning them into retrospective gates.

At the final same-P0 boundary, differing incoming observations precede the first retained live-array difference. Corresponding actions, executed blocks, P/D, cache versions, window/count metadata and consumed flags are compared separately from numerical states. The first retained observation in semantic reporting order is not a first causal error; fine-grained execution timestamps or causal interventions are unavailable.

## Final-boundary state component summaries, all four cases

Signed Δ means P1-history minus P0-history, in the *published summary statistic*. Neither Δnorm nor Δmax is a distance between the underlying tensors. No component is selected or ranked for causality.

### acquisition/L16/row00

|Component|Incoming Δnorm|Incoming Δmax|Returned Δnorm|Returned Δmax|
|---|---:|---:|---:|---:|
|/s|-0.00181770325|0.0356456041|-0.000128746033|-0.00268685818|
|/memory|0.36973238|0.0250608325|0.358909607|0.0248527527|
|/ctx/kv/2/committed_keys|-0.000400543213|-0.0194482803|-0.000383377075|-0.0194482803|
|/ctx/kv/2/committed_values|-0.214887619|-0.00251758099|-0.207970619|-0.00251758099|
|/ctx/kv/3/committed_keys|-0.00236225128|0.30066669|-0.00229072571|0.30066669|
|/ctx/kv/3/committed_values|0.390561104|0.00501471758|0.379545212|0.00501471758|
|/ctx/swa/0/committed_keys|-5.7220459e-05|0.224504471|-6.29425049e-05|0.224504471|
|/ctx/swa/0/committed_values|0.125977516|0.101480842|0.122345924|0.101480842|
|/ctx/swa/1/committed_keys|0.000468254089|-0.139089823|0.0004529953|-0.139089823|
|/ctx/swa/1/committed_values|-0.17006588|0.000774860382|-0.164888382|5.14984131e-05|
|/ctx/swa/pre/committed_keys|0|0|0|0|
|/ctx/swa/pre/committed_values|0|0|0|0|
|/ctx/gdn/4/committed_state|0.0120309591|0.0154332072|0.00410223007|0.00599250197|
|/ctx/gdn/4/committed_conv_buffer|0.0174474716|-0.0016503334|0.0100870132|9.01222229e-05|
|/ctx/gdn/5/committed_state|0.00932592154|0.00481945276|0.003295362|0.00223568082|
|/ctx/gdn/5/committed_conv_buffer|0.0202608109|0.000234484673|0.0124540329|0.000193119049|
### acquisition/L16/row01

|Component|Incoming Δnorm|Incoming Δmax|Returned Δnorm|Returned Δmax|
|---|---:|---:|---:|---:|
|/s|0.00118160248|0.0497350693|-0.00149154663|0.027884841|
|/memory|0.55598855|0.0214785971|0.472798944|0.0146542862|
|/ctx/kv/2/committed_keys|2.67028809e-05|0.0231571198|-0.000122070312|0.0231571198|
|/ctx/kv/2/committed_values|-0.0952157974|0.210577607|-0.0831804276|0.20822823|
|/ctx/kv/3/committed_keys|-0.000309944153|0.525622129|-0.00032043457|0.525622129|
|/ctx/kv/3/committed_values|0.505669117|0.452634275|0.489253998|0.452634275|
|/ctx/swa/0/committed_keys|-0.00337505341|0.127969265|-0.00304985046|0.127969265|
|/ctx/swa/0/committed_values|0.269088268|0.44872725|0.259897709|0.44872725|
|/ctx/swa/1/committed_keys|-4.86373901e-05|0.169008017|-8.77380371e-05|0.169008017|
|/ctx/swa/1/committed_values|-0.188276291|0.158528686|-0.182711601|0.147848725|
|/ctx/swa/pre/committed_keys|0|0|0|0|
|/ctx/swa/pre/committed_values|0|0|0|0|
|/ctx/gdn/4/committed_state|0.00235760212|-0.0132770538|-0.00892925262|-0.0118428767|
|/ctx/gdn/4/committed_conv_buffer|-0.0989904404|-0.0627492666|-0.0738582611|-0.0236837864|
|/ctx/gdn/5/committed_state|0.0382540822|-0.00355868042|0.00400674343|-9.91299748e-05|
|/ctx/gdn/5/committed_conv_buffer|0.102367401|-0.00265550613|0.109144211|-0.00139266253|
### acquisition/L16/row02

|Component|Incoming Δnorm|Incoming Δmax|Returned Δnorm|Returned Δmax|
|---|---:|---:|---:|---:|
|/s|-0.00183486938|0.0409349203|-0.00237369537|-0.125760913|
|/memory|0.0182847977|0.00539875031|0.0159211159|0.0045517087|
|/ctx/kv/2/committed_keys|-8.96453857e-05|0.0176422596|-0.000160217285|0.0176422596|
|/ctx/kv/2/committed_values|-0.0762243271|-0.00187754631|-0.0654602051|0.0160769224|
|/ctx/kv/3/committed_keys|-0.000142097473|0.0112317801|-0.000183105469|0.0112317801|
|/ctx/kv/3/committed_values|0.0906772614|0.00501471758|0.0891876221|0.00501471758|
|/ctx/swa/0/committed_keys|-0.000918388367|-0.0114328861|-0.000774383545|-0.0114328861|
|/ctx/swa/0/committed_values|0.0525398254|0.042796731|0.0493946075|0.042796731|
|/ctx/swa/1/committed_keys|-0.000608444214|0.0327029228|-0.000733375549|0.0327029228|
|/ctx/swa/1/committed_values|0.00143051147|0.00106322765|-0.00596141815|0.00106322765|
|/ctx/swa/pre/committed_keys|0|0|0|0|
|/ctx/swa/pre/committed_values|0|0|0|0|
|/ctx/gdn/4/committed_state|0.00555235147|0.00687585771|-0.0110666752|0.00627534091|
|/ctx/gdn/4/committed_conv_buffer|-0.0573883057|0.00436544418|0.064658165|0.00436544418|
|/ctx/gdn/5/committed_state|0.00681215525|0.000566095114|-0.00240713358|-0.00358425081|
|/ctx/gdn/5/committed_conv_buffer|0.0139141083|-0.000116050243|0.0251436234|-0.000116050243|
### acquisition/L16/row03

|Component|Incoming Δnorm|Incoming Δmax|Returned Δnorm|Returned Δmax|
|---|---:|---:|---:|---:|
|/s|0.0298595428|0.66579175|0.0391168594|0.5849123|
|/memory|2.20402706|0.130578853|1.99042571|0.122925363|
|/ctx/kv/2/committed_keys|-0.00612163544|-0.107579708|-0.00580406189|-0.107579708|
|/ctx/kv/2/committed_values|-0.620463371|-0.130390525|-0.542431831|-0.130390525|
|/ctx/kv/3/committed_keys|0.00358581543|-0.235293627|0.00341415405|-0.235293627|
|/ctx/kv/3/committed_values|-0.310330391|-0.0633869171|-0.29143095|-0.0633869171|
|/ctx/swa/0/committed_keys|0.00340938568|-0.199324369|0.00295829773|-0.199324369|
|/ctx/swa/0/committed_values|0.0369563103|0.0226317644|0.0352253914|0.0226317644|
|/ctx/swa/1/committed_keys|-0.00521183014|0.236214399|-0.00398731232|0.236214399|
|/ctx/swa/1/committed_values|-0.0273008347|-0.139347315|-0.0620317459|-0.139347315|
|/ctx/swa/pre/committed_keys|0|0|0|0|
|/ctx/swa/pre/committed_values|0|0|0|0|
|/ctx/gdn/4/committed_state|-0.0338112116|-0.041880317|-0.0317599773|-0.126964308|
|/ctx/gdn/4/committed_conv_buffer|-0.135707855|-0.0516873598|-0.296524048|-0.0528322458|
|/ctx/gdn/5/committed_state|0.000689744949|0.00528907031|-0.0147231221|0.0268415958|
|/ctx/gdn/5/committed_conv_buffer|-0.0599861145|0.00146329403|-0.109686852|0.0101875067|

All stream, memory, GDN recurrence/buffer and attention cache summary fields are retained in component_summary_differences.json/csv for incoming, commit-pre and committed-return boundaries. Exact metadata is in recorded_metadata_comparisons.json and retained_protocol_consistency.json. If two summaries match, that is summary agreement only. GDN temporary states, complete memory slots, raw committed tensor values and separate rotary-position arrays are not available in the readable selected-case records.

## Ledger arithmetic

Archival16 + P0 prefixes1696 + P1 prefixes60 + current boundaries107 + supplied references100 =1979 completed tokens. Retained selected calls total15832 and commits1979. The67 P1 tokens comprise60 prefix tokens and7 boundary tokens. They execute block5 twice each:134 block5 calls,67 beyond one per token. Global block5 count2046 =1979+67; block6 count1912 =1979−67. REPEAT total67; RESTART0. This corroborates the logged program definitions, not an additional implementation audit. No future records/calls or final-joint bootstrap outputs exist.

## What the evidence cannot determine

The observed state fingerprints, summaries and array differences do not establish which component represents parity, whether information was irreversibly lost, or which component caused the row03 error. We cannot compute raw GDN/memory/cache distances from norm summaries, inspect ephemeral GDN state, establish equivalent future behavior, or predict the unexecuted row03 P1→P1 result. Serialized state files for the three controls exist but were not deserialized; row03 has no saved bundle. A prior runtime full-state comparison is not a new offline tensor comparison. No population estimates, confidence intervals, calibrated semantic probe or component-selection rule is inferred from these four ordered cases.

## Assessment: program-dependent readability

The retained case supports asking a separately preregistered question about whether a P1-generated committed state may yield different task-readable behavior under different boundary programs. It does not answer that question: row03’s P1→P1 boundary is unobserved, and present-output correctness elsewhere does not prove recoverable future information. The three controls show that switching failure is not universal in the observed cases. Readability mismatch, accumulated state error and other coupled effects remain unresolved. No new runner, component chosen by largest summary difference, or follow-up execution is proposed here. P1 is not reinterpreted as passing v4.

## Integrity and execution boundary

All23 v4 evidence files read had identical before/after SHA-256 hashes. Original tracked-diff identity also matches. V4’s broader finding that617 protected identities/checkpoints and historical bytecode exception were unchanged is carried forward, **not newly reverified** by this analysis. No checkpoints or state bundles were deserialized. All new files are confined to experiments/results/gatec_two_program_v4_forensics_5100_v1. No model import/construction, forward/backward, test, optimizer, training, selector fitting, noise, bootstrap, production edit or commit occurred.

## Additional descriptive prefix-readout qualification

These statements expose readouts already present in the complete paired timeline; they add no outcomes, analysis cases, gate, inferential procedure or execution.

|Case|P0 prefix wrong-readout indices (zero-based)|P1 prefix wrong-readout indices (zero-based)|Human token15: truth/P0/P1|
|---|---|---|---|
|row00|[]|[]|1/1/1|
|row01|[]|[]|0/0/0|
|row02|[]|[]|1/1/1|
|row03|[1]|[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]|1/1/0|

Primary retained input identity: `acquisition/L16/row03`; bits `0000101000110010`; bit-content SHA-256 `8ee3bd216c58a9e17d3403ec3f5c182a5823371fc064608119cf8d770b230e68`. This was copied from v4's existing current_screen_analysis.json, not regenerated. The retained canonical comparator is P0-history/P0-boundary for that same ID; its recorded CE is 0.0022292304784059525, versus 6.418639659881592 at the rejected P1-history/P0-boundary. The three controls are bound by explicit case IDs and execution order; their original encoded tensors were not read from outside v4.

Row03 has one logged wrong P0 prefix readout at internal token1, while P1 has logged wrong readouts at internal tokens1–14. The first *prediction disagreement* is internal token2; both paths are wrong at internal token1. These are recorded intermediate answers, not newly imposed token correctness criteria and not identified first causal errors. At human token15, P1 already predicts0 for truth1 before the final P0 boundary. The three controls have no logged prefix readout errors. Equal summaries in some cache components do not establish underlying tensor equality.

## Final assessment

The evidence motivates a separately preregistered program-dependent-readability **question**, but does not specifically establish a switch-only readability mismatch. Existing P1-prefix errors provide a competing accumulated-state/history explanation. The unexecuted row03 P1→P1 cell could not be inspected and cannot be predicted from the accepted controls. No component is nominated by summary magnitude or first divergence. P1 remains rejected under v4; no new runner, experimental design, execution or contract reinterpretation follows from this offline report.
