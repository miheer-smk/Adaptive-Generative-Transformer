# Retained-evidence interpretation: feature-complement pilot

These are descriptive summaries of the already retained prescribed evaluations. No model, tensor forward/backward, fit, test, sample or bootstrap is run here. No acceptance threshold or classification is changed.

The control final parameter/buffer identity is `7ab9239da5b1a4957cde546bb95e4c2961800b337f8bf677f8202e662e94f1a0`, exactly matching the historical failed training identity `7ab9239da5b1a4957cde546bb95e4c2961800b337f8bf677f8202e662e94f1a0`. All three control-panel native metrics also reproduce the archived reported values. This is a final control reconstruction check, not initialization from the failed checkpoint. The new serialized checkpoint SHA differs because its metadata and archive names differ.

## Main task finding

The locked result is PARTIAL_EFFECT_TASK_NON_SUCCESS. Treatment development accuracy10.9375% versus control3.125% gives+7.8125 percentage points, with adjusted lower bound+6.4404296875 points. Treatment minus its own query-independent component has point0 and lower bound0: no resolved query-conditioned advantage. Both arms have0% exact-array accuracy on all panels and fail all four competence thresholds. A zero-width observed bound is not a universal equivalence guarantee.

|Panel|Arm|Native single-candidate arrays|Native/common identical-prediction arrays|Mean common score norm|Mean interaction norm|
|---|---|---:|---:|---:|---:|
|training_subset|C|128/128|128/128|0.02406793649|8.023997294e-06|
|training_subset|T|128/128|128/128|1.887795054|0.004431137066|
|development|C|32/32|32/32|0.0227362361|7.671822188e-06|
|development|T|32/32|32/32|1.764024949|0.004134789061|
|size_transfer|C|32/32|32/32|0.09486866478|2.53367992e-05|
|size_transfer|T|32/32|32/32|7.725006418|0.01746226801|

The native predictions remain concentrated on one candidate per array. Interaction can change individual scores/margins without changing the selected candidate. Native/common accuracy equality is an observed finite-panel result; it does not imply tensor/state equality or absence of information in every persistent channel.

## Terminal geometry by role

|Panel|Role|Arm|Pre-norm centered energy|Committed centered energy|Pre-norm mean norm|Committed mean norm|Pre-norm angular dispersion|Committed angular dispersion|
|---|---|---|---:|---:|---:|---:|---:|---:|
|training_subset|candidates|C|69140.7462|0.001884852973|1990.926084|7.704378393|2.100790523e-06|2.116937975e-06|
|training_subset|candidates|T|74.60850432|0.5021619629|78.53101103|8.010370727|0.0005285672829|0.0005217686833|
|training_subset|queries|C|433.4801432|0.0001104357021|1893.184799|7.70437644|1.237369447e-07|1.240322926e-07|
|training_subset|queries|T|5.691097399|0.06154885765|75.04163713|8.012111477|6.664926923e-05|6.391651206e-05|
|development|candidates|C|62311.64701|0.001813061384|1988.939137|7.704379115|2.021581398e-06|2.036306428e-06|
|development|candidates|T|69.6298681|0.4481071202|78.54931263|8.010438159|0.0004717995958|0.000465601157|
|development|queries|C|433.4546887|0.0001104346787|1893.185068|7.704376458|1.237359277e-07|1.240311372e-07|
|development|queries|T|5.682884901|0.06147524211|75.04231754|8.012110271|6.657283717e-05|6.384009605e-05|
|size_transfer|candidates|C|109802.8705|0.003908962225|1976.706665|7.704387167|1.031345557e-06|1.045304864e-06|
|size_transfer|candidates|T|220.3791945|1.946466949|77.99203114|8.010601241|0.0004868052775|0.0004815179574|
|size_transfer|queries|C|1978.803113|0.0004467632253|1891.897331|7.704376373|1.191628724e-07|1.194684246e-07|
|size_transfer|queries|T|23.57621493|0.2518285727|75.10628663|8.011922862|6.494123763e-05|6.226830877e-05|

On development, treatment committed candidate centered energy is0.4481071202 versus control0.001813061384; query energy is0.06147524211 versus0.0001104346787. Committed mean angular dispersion is4.656011570e-4 versus2.036306428e-6 for candidates, and6.384009605e-5 versus1.240311372e-7 for queries. The terminal representations show substantially more absolute positional variation and directional dispersion in treatment on this panel. This did not establish useful query-conditioned task computation or competence. No binary collapse cutoff is invented.

## Observed difference transport at each development mixer

The entries below are RMS amplification of actual corresponding position differences: sqrt(mean output pair-distance-squared / mean input pair-distance-squared), with equal weighting of development arrays within each role. This is not a worst-case singular value and not the mean of individual pair ratios. Absolute input/output centered energies are retained in mechanistic_summary_v1.json, and each array retains full transport, cosine, radial/angular and whole-block residual summaries.

|Block|Role|C observed RMS amplification|T observed RMS amplification|
|---|---|---:|---:|
|0|candidates|0.8550363577|1.050240199|
|0|queries|0.3942016765|1.049756539|
|1|candidates|0.7420178825|1.03057983|
|1|queries|0.3884138424|1.020099367|
|2|candidates|0.7951316331|1.020521581|
|2|queries|0.6749797089|0.9844369065|
|3|candidates|0.4881033317|0.9954732858|
|3|queries|0.498799497|0.99833836|
|4|candidates|0.9938455426|0.9460784322|
|4|queries|0.9938626885|0.9462462336|
|5|candidates|0.9913561737|0.8869134297|
|5|queries|0.9908823182|0.9238224141|
|6|candidates|1.003828992|0.9775737453|
|6|queries|1.003826241|1.018862114|
|7|candidates|0.990004092|0.9865936955|
|7|queries|0.9870074515|1.011104212|

This compares two separately learned final arms. The mixer inputs, gains, constrained matrices and other learned parameters differ after training, so it is not a fixed-weight, same-input attribution to one module. Some treatment mixers preserve/amplify actual differences and others reduce them; the complete raw tensor records permit stage-accurate reconstruction. Whole-block residual identities were checked exactly. There is no inversion, omitted-activation reconstruction or claim that one mixer uniquely caused historical failure.

The task-level conclusion remains non-success despite a supported partial accuracy effect and changed geometry. Under the locked stop rule this supplies no authorization to strengthen the complement, alter normalization/interface, extend training or run another experiment. Await a new research decision.
