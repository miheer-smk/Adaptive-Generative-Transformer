# Gate-C paired full-ST gradient averaging4900

Manifest: `3ebaad9fdedd65d938dedbfb7eb097f4d36b178b8ffa037db7a00f074a236b3c`. Exactly520 updates and1040 training rollouts per arm; final520 only. No tuning, extensions or replacement seeds.

Intervention helped: **False**. Useful treatment policy: **False**. Combined useful-policy repair: **False**.

| Panel | Path | Deterministic accuracy/CE | Pooled stochastic accuracy/CE |
|---|---|---|---|
| val | F | 1.000000 / 0.00178035 | fixed canonical reference |
| val | U | 0.500000 / 1.47393 | 0.539062 / 2.81672 |
| val | C | 0.531250 / 0.969835 | 0.484375 / 2.93979 |
| val | T | 0.625000 / 0.951763 | 0.453125 / 2.72984 |
| ood_test | F | 1.000000 / 0.00178658 | fixed canonical reference |
| ood_test | U | 0.312500 / 1.95107 | 0.503906 / 3.0365 |
| ood_test | C | 0.562500 / 0.955672 | 0.476562 / 2.9174 |
| ood_test | T | 0.484375 / 1.55863 | 0.492188 / 2.36389 |

## Paired effects

val: pooled accuracy T-C={'mean': -0.03125, 'bootstrap_95': [-0.125, 0.0546875]}; CE T-C={'mean': -0.20995284898890532, 'bootstrap_95': [-0.779233149206948, 0.39871685834152665]}; individual tapes=[{'tape': 'tape0', 'accuracy_T_minus_C': -0.03125, 'CE_T_minus_C': -0.16177118803170742}, {'tape': 'tape1', 'accuracy_T_minus_C': -0.0625, 'CE_T_minus_C': -0.35055979182288866}, {'tape': 'tape2', 'accuracy_T_minus_C': 0.0625, 'CE_T_minus_C': -0.5904662728280528}, {'tape': 'tape3', 'accuracy_T_minus_C': -0.09375, 'CE_T_minus_C': 0.26298585672702757}]; deterministic T-C=0.09375.
ood_test: pooled accuracy T-C={'mean': 0.015625, 'bootstrap_95': [-0.03125, 0.06640625]}; CE T-C={'mean': -0.5535125354972479, 'bootstrap_95': [-0.8052384610171543, -0.3129349779685302]}; individual tapes=[{'tape': 'tape0', 'accuracy_T_minus_C': 0.03125, 'CE_T_minus_C': -0.6970523671279807}, {'tape': 'tape1', 'accuracy_T_minus_C': 0.046875, 'CE_T_minus_C': -0.4607946195610566}, {'tape': 'tape2', 'accuracy_T_minus_C': 0.03125, 'CE_T_minus_C': -0.8802209987552487}, {'tape': 'tape3', 'accuracy_T_minus_C': -0.046875, 'CE_T_minus_C': -0.1759821565447055}]; deterministic T-C=-0.078125.

All individual tape, length and route results are retained. No prediction ensembling. Paired10,000 sequence bootstrap resamples preserve all four outcomes together, stratified by parity and OOD length. Conditional uncertainty concerns one initialization and four tapes. Duplicate control equality verifies deterministic coupling, not zero sampling variance.

Gradient statistics: {'C': {'minimum_average_norm': 1.1964103578065988e-05, 'maximum_average_norm': 6656.6748046875, 'mean_pair_difference': 0.0, 'maximum_pair_difference': 0.0, 'nonzero_parameter_updates': 520}, 'T': {'minimum_average_norm': 1.5538176739937626e-05, 'maximum_average_norm': 21083.16015625, 'mean_pair_difference': 97.89630941038024, 'maximum_pair_difference': 42166.3203125, 'nonzero_parameter_updates': 520}}. Controller displacement: C=0.6724020838737488, T=1.485459327697754.

Both arms use original full-sequence BPTT through frozen computations, no added detach and no clipping. Each rollout uses an independent initial persistent state. Parameters remain fixed across a pair; two original gradients are averaged before one fresh AdamW step. No auxiliary/imitation loss.

Mechanical invariants and frozen/source/historical identities verified. Historical bytecode exception retained. Results do not establish general seed reproducibility, adaptive depth, useful diversity or superiority over canonical routing. Any chance-level result is a failure of this repair under the declared budget; ordinary poor performance was not used for stopping.
