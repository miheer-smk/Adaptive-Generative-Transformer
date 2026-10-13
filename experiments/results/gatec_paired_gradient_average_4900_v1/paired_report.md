# Gate-C: paired two-gradient intervention, seed 4900

Intervention helped under the declared rule: **False**. Treatment learned a useful policy under the additional accuracy/CE criteria: **False**. Combined useful-policy repair: **False**.

Both arms started from the saved seed-4700 update-0 controller and received exactly 520 fresh AdamW updates. C averaged duplicate-noise gradients; T averaged independent lane gradients. There were exactly 1,040 complete training rollouts per arm. Each pair used fixed parameters and separate normal sequence resets; only gradients were averaged. No clipping, scheduler, auxiliary loss, LR rescaling or estimator change occurred.

Manifest SHA-256: `3ebaad9fdedd65d938dedbfb7eb097f4d36b178b8ffa037db7a00f074a236b3c`. Initial controller: `ae8ff9256afd573331719cf15a3f82d21f70181891400640739a0415c82d6279`. Stage-1 backbone: `53b56e7e90f7eb7a6570107b6dfe3a297caac48d637fe27e5f00fbe94c6f09c3`. Revision: `75f138e6b909c5ca9884e9a7755de63489d3d6f2`. Tracked-diff identity: `45b3d093bd7fd7976019bf4572dc574784ed271b3294b86194f80c05826a8eac`.

The 104 raw sequences and the first five historical permutations were preserved, including duplicate content. Full-sequence BPTT through frozen computations was retained. Existing ST relaxed stop-gradient and discrete pointer/mask metadata boundaries were unchanged. Tau=2, budget feature/presence=[0,0], P=8, FP32 and batch=1 were fixed. HALT/convergence remained disabled.

## Confirmation outcomes

| Panel | Mode | U accuracy / CE | C accuracy / CE | T accuracy / CE | T−C accuracy | T−C CE |
|---|---|---|---|---|---:|---:|
| val | deterministic | 0.500000 / 1.47392711 | 0.531250 / 0.969834556 | 0.625000 / 0.951763468 | +0.093750 | -0.0180710875 |
| val | tape0 | 0.562500 / 2.57396557 | 0.437500 / 3.07420946 | 0.406250 / 2.91243827 | -0.031250 | -0.161771188 |
| val | tape1 | 0.500000 / 3.12619812 | 0.468750 / 3.1444784 | 0.406250 / 2.79391861 | -0.062500 | -0.350559792 |
| val | tape2 | 0.531250 / 2.90553262 | 0.437500 / 3.27883173 | 0.500000 / 2.68836545 | +0.062500 | -0.590466273 |
| val | tape3 | 0.562500 / 2.6612 | 0.593750 / 2.26165692 | 0.500000 / 2.52464278 | -0.093750 | +0.262985857 |
| val | pooled_stochastic | 0.539062 / 2.81672408 | 0.484375 / 2.93979413 | 0.453125 / 2.72984128 | -0.031250 | -0.209952849 |

val fixed canonical reference: accuracy=1.000000, CE=0.00178034819328. Absolute parity accuracy is separate from agreement with the reference.

| ood_test | deterministic | 0.312500 / 1.95107008 | 0.562500 / 0.955671624 | 0.484375 / 1.55863126 | -0.078125 | +0.602959637 |
| ood_test | tape0 | 0.546875 / 2.83375492 | 0.484375 / 2.85674871 | 0.515625 / 2.15969634 | +0.031250 | -0.697052367 |
| ood_test | tape1 | 0.468750 / 3.27395385 | 0.500000 / 2.80581211 | 0.546875 / 2.34501749 | +0.046875 | -0.46079462 |
| ood_test | tape2 | 0.546875 / 2.76083977 | 0.406250 / 3.33636285 | 0.437500 / 2.45614185 | +0.031250 | -0.880220999 |
| ood_test | tape3 | 0.453125 / 3.27746401 | 0.515625 / 2.67067618 | 0.468750 / 2.49469403 | -0.046875 | -0.175982157 |
| ood_test | pooled_stochastic | 0.503906 / 3.03650314 | 0.476562 / 2.91739996 | 0.492188 / 2.36388743 | +0.015625 | -0.553512535 |

ood_test fixed canonical reference: accuracy=1.000000, CE=0.00178657703691. Absolute parity accuracy is separate from agreement with the reference.

## Paired uncertainty and declared decision

Each stochastic contrast first averages four outcomes within a sequence; the 10,000 paired bootstrap resamples then resample sequences within parity and OOD-length strata. Every sequence’s four outcomes stay together. These are approximate intervals conditional on one initialization and the four locked tapes. No inference across controller-training seeds is made.

| Panel | Pooled accuracy T−C | Paired 95% interval | Pooled CE T−C | Paired 95% interval | Positive tapes /4 | Deterministic T−C |
|---|---:|---|---:|---|---:|---:|
| val | -0.031250 | [-0.125, 0.0546875] | -0.209952849 | [-0.779233149206948, 0.39871685834152665] | 1 | +0.093750 |
| ood_test | +0.015625 | [-0.03125, 0.06640625] | -0.553512535 | [-0.8052384610171543, -0.3129349779685302] | 3 | -0.078125 |

“Helped” requires ≥10 percentage-point stochastic accuracy gain on each panel, both intervals excluding zero, at least three positive tapes per panel, and no deterministic deterioration versus C. Useful-policy repair additionally requires T≥90% accuracy in deterministic and pooled stochastic evaluation on each panel and lower CE than U in both modes. CE-only or sub-threshold gains remain partial evidence.

## Training, gradients and displacement

| Pass | C training accuracy / CE | T training accuracy / CE | Updates per arm | Rollouts per arm |
|---:|---|---|---:|---:|
| 1 | 0.461538 / 3.22950244 | 0.480769 / 3.05270312 | 104 | 208 |
| 2 | 0.528846 / 2.7980723 | 0.562500 / 2.59383302 | 104 | 208 |
| 3 | 0.567308 / 2.46481138 | 0.581731 / 2.19668332 | 104 | 208 |
| 4 | 0.557692 / 2.58851502 | 0.634615 / 1.91495993 | 104 | 208 |
| 5 | 0.605769 / 2.28402584 | 0.644231 / 1.796877 | 104 | 208 |

Control’s duplicate rollouts are not independent observations. Its per-pass average is the same sample-weighted statistic as one observation per update. Treatment’s two noise lanes provide two conditional sampled trajectories per update, not independent training sequences.

Gradient statistics: `{"C": {"maximum_average_norm": 6656.6748046875, "maximum_pair_difference": 0.0, "mean_pair_difference": 0.0, "minimum_average_norm": 1.1964103578065988e-05, "nonzero_parameter_updates": 520}, "T": {"maximum_average_norm": 21083.16015625, "maximum_pair_difference": 42166.3203125, "mean_pair_difference": 97.89630941038024, "minimum_average_norm": 1.5538176739937626e-05, "nonzero_parameter_updates": 520}}`. Maximum duplicated-control pair difference: 0.0. Arithmetic averaging was checked within scheduled training computations. Zero within-pair control difference is not zero population sampling variance.

C: final displacement L2=0.672402084; changed scalar parameters=103306; optimizer steps=520.
T: final displacement L2=1.48545933; changed scalar parameters=103306; optimizer steps=520.

## Routing and execution

| Panel/mode | Arm | Canonical action fraction | Canonical block fraction | Exact sequence fraction | Mean/max D | RESTART | REPEAT | P distribution |
|---|---|---:|---:|---:|---|---:|---:|---|
| val/deterministic | U | 0.000000 | 0.000000 | 0.000000 | 11.839844/15 | 1966 | 207 | {'8': 512} |
| val/deterministic | C | 0.000000 | 0.000000 | 0.000000 | 13.562500/15 | 2848 | 736 | {'8': 512} |
| val/deterministic | T | 0.000000 | 0.000000 | 0.000000 | 14.005859/15 | 3075 | 490 | {'8': 512} |
| val/pooled_stochastic | U | 0.000000 | 0.000000 | 0.000000 | 10.284668/15 | 4679 | 4091 | {'8': 2048} |
| val/pooled_stochastic | C | 0.000000 | 0.000000 | 0.000000 | 10.299316/14 | 4709 | 4575 | {'8': 2048} |
| val/pooled_stochastic | T | 0.000000 | 0.000000 | 0.000000 | 10.457520/14 | 5033 | 4238 | {'8': 2048} |
| ood_test/deterministic | U | 0.000000 | 0.000000 | 0.000000 | 11.649235/15 | 5722 | 684 | {'8': 1568} |
| ood_test/deterministic | C | 0.000000 | 0.000000 | 0.000000 | 13.981505/15 | 9379 | 1597 | {'8': 1568} |
| ood_test/deterministic | T | 0.000000 | 0.000000 | 0.000000 | 14.217474/15 | 9749 | 1193 | {'8': 1568} |
| ood_test/pooled_stochastic | U | 0.000000 | 0.000000 | 0.000000 | 10.266103/14 | 14213 | 12432 | {'8': 6272} |
| ood_test/pooled_stochastic | C | 0.000000 | 0.000000 | 0.000000 | 10.305963/15 | 14463 | 13667 | {'8': 6272} |
| ood_test/pooled_stochastic | T | 0.000000 | 0.000000 | 0.000000 | 10.463967/15 | 15454 | 12802 | {'8': 6272} |

Complete per-length outcomes, route signatures/frequencies and executed-block histories are retained in results.json, per-sequence evaluation JSON and compressed decision/token logs. Noncanonical routing is not automatically a task failure, and diversity alone is not success.

Offline verification: 54080 completed tokens, 571731 decisions, 432640 actual selected-block executions, 4251888 candidate bank calls, 757120 cache-version checks. All completed tokens have P=8 and D=8+RESTART≤15; RESTART executes zero blocks. Legality, selected successor, in-flight candidate/GDN purity and exactly one seven-entry terminal commit were checked at runtime. Frozen non-controller values and historical file hashes were verified. Historical bytecode exception and failure artifacts remain preserved.

## Predetermined checkpoints

| Arm | Update | Path | SHA-256 |
|---|---:|---|---|
| C | 0 | experiments/runs/stage3_gradient_average_4900_C_update0000_locked_v1.pt | `c6cc4da2ff05f69e9d39de222b6c380ed509f885725878d255ffdf7a7720d812` |
| T | 0 | experiments/runs/stage3_gradient_average_4900_T_update0000_locked_v1.pt | `eaf8f98adec91fd87b007eaf9cb4d5f6b6cd12fae388352574f8984f72469335` |
| C | 104 | experiments/runs/stage3_gradient_average_4900_C_update0104_locked_v1.pt | `6b706a7c49d9faa7a940d0a4075a6e74e714df589169eac67f4a7b0903e9dd71` |
| T | 104 | experiments/runs/stage3_gradient_average_4900_T_update0104_locked_v1.pt | `c722c997c29f5b96ed34f688bedbc46a0077ddcfa9978c8fb8a8f87e98e911da` |
| C | 260 | experiments/runs/stage3_gradient_average_4900_C_update0260_locked_v1.pt | `59b115341180261d2752b5c7abee92a67934ebbfc431f140988bcd0fbd36f2e3` |
| T | 260 | experiments/runs/stage3_gradient_average_4900_T_update0260_locked_v1.pt | `7720e0ab6d8d33ff9dafc7cdb4131c8dd66254b4066a20cdcbc70ccb38604d01` |
| C | 520 | experiments/runs/stage3_gradient_average_4900_C_update0520_locked_v1.pt | `0b414c7bfa9fb5e12dd7fb81dcab21325dadf56cf131e051a76ed79f32e9e95d` |
| T | 520 | experiments/runs/stage3_gradient_average_4900_T_update0520_locked_v1.pt | `f2da58059806802eaf8e16ff958c95558b088c404a2e9cd7c79f866a40ab21ac` |

## Limits and conclusion

The independent-noise averaging intervention did not meet the declared paired help criterion. Any individual-panel or CE improvement should be reported separately and does not establish useful-policy repair.
Treatment did not meet the declared useful-policy accuracy/CE thresholds. This experiment does not establish reference-level task-loss routing discovery.

Confirmation inputs were generated with seeds4901/4902, balanced and distinct, excluding recoverable retained training/evaluation content. Unknown Stage-1 exposure and unretained historical inputs remain limitations. Four tapes replicate evaluation sampling, not training seeds. No outcome-based checkpoint selection, extension, replacement seed, new training variant, production edit or commit occurred. A useful policy in both arms would support task-loss learning but would not attribute that success to averaging absent the declared paired effect.

Elapsed scheduled experiment time: 5227.474 seconds. Focused tests: 10 passed. git diff --check: clean. No follow-up run is launched.
