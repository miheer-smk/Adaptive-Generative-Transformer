# Task-A retained-logit diagnostic — seed3401

Measurement only; no optimizer, parameter updates, or core changes.
Eight train examples of length eight, fresh D64 controller, frozen Stage-1 seed3305 backbone; tau=2.0, zero budget, eight decisions/token, training=True. Data uses Task-A's fixed FEATURE_SEED=3002 and train generator seed3001; controller and routing seed3401. Routing RNG is reset once before the eight-example stream, not per example.

Retained gradients: 512/512 present, 512/512 nonzero. Decision 7: 64/64 nonzero. HALT: 512/512 exactly zero; maximum absolute gradient 0.
All captured selection/pointer/mask metadata lacks requires_grad/grad_fn: True.

## gradient_l2 (means across eight examples)

| token | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.454e-05 | 0.00012928 | 0.00011985 | 0.00024813 | 0.00011775 | 0.00023195 | 0.00026257 | 0.00033457 |
| 1 | 0.0015549 | 0.0012795 | 0.0011392 | 0.00048663 | 0.00057934 | 0.0018006 | 0.00082283 | 0.0014139 |
| 2 | 2.6246e-05 | 3.6837e-05 | 8.6519e-05 | 7.8355e-05 | 8.132e-05 | 6.5109e-05 | 7.5459e-05 | 2.7264e-05 |
| 3 | 0.0028295 | 0.0034404 | 0.011948 | 0.010445 | 0.0068389 | 0.021025 | 0.032612 | 0.01642 |
| 4 | 0.0011938 | 0.0055451 | 0.0025832 | 0.0064593 | 0.0074704 | 0.0046904 | 0.0029044 | 0.0032302 |
| 5 | 0.00081185 | 0.00024999 | 0.00041974 | 0.00043434 | 0.00029127 | 0.00036996 | 0.0003472 | 0.00019082 |
| 6 | 0.0015742 | 0.002558 | 0.0026006 | 0.0034008 | 0.0028064 | 0.00087125 | 0.00077279 | 0.0012094 |
| 7 | 1.2244 | 0.89315 | 0.077881 | 0.43792 | 0.13367 | 0.080672 | 0.15322 | 0.17628 |

## legal_gradient_l2 (means across eight examples)

| token | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.454e-05 | 0.00012928 | 0.00011985 | 0.00024813 | 0.00011775 | 0.00023195 | 0.00026257 | 0.00033457 |
| 1 | 0.0015549 | 0.0012795 | 0.0011392 | 0.00048663 | 0.00057934 | 0.0018006 | 0.00082283 | 0.0014139 |
| 2 | 2.6246e-05 | 3.6837e-05 | 8.6519e-05 | 7.8355e-05 | 8.132e-05 | 6.5109e-05 | 7.5459e-05 | 2.7264e-05 |
| 3 | 0.0028295 | 0.0034404 | 0.011948 | 0.010445 | 0.0068389 | 0.021025 | 0.032612 | 0.01642 |
| 4 | 0.0011938 | 0.0055451 | 0.0025832 | 0.0064593 | 0.0074704 | 0.0046904 | 0.0029044 | 0.0032302 |
| 5 | 0.00081185 | 0.00024999 | 0.00041974 | 0.00043434 | 0.00029127 | 0.00036996 | 0.0003472 | 0.00019082 |
| 6 | 0.0015742 | 0.002558 | 0.0026006 | 0.0034008 | 0.0028064 | 0.00087125 | 0.00077279 | 0.0012094 |
| 7 | 1.2244 | 0.89315 | 0.077881 | 0.43792 | 0.13367 | 0.080672 | 0.15322 | 0.17628 |

## max_abs_gradient (means across eight examples)

| token | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.9327e-05 | 0.00011089 | 8.4917e-05 | 0.00017234 | 8.4635e-05 | 0.00017127 | 0.00019437 | 0.00025692 |
| 1 | 0.0013249 | 0.0009608 | 0.00086391 | 0.00039032 | 0.00047788 | 0.0014075 | 0.0006843 | 0.0012018 |
| 2 | 2.1145e-05 | 2.9531e-05 | 6.7303e-05 | 5.5993e-05 | 5.8498e-05 | 4.6334e-05 | 5.3458e-05 | 2.3551e-05 |
| 3 | 0.0024179 | 0.002959 | 0.010179 | 0.008248 | 0.0059544 | 0.017981 | 0.026813 | 0.013059 |
| 4 | 0.00099947 | 0.0042227 | 0.0023208 | 0.0054654 | 0.0062198 | 0.0042386 | 0.0024875 | 0.0028637 |
| 5 | 0.00069979 | 0.00021958 | 0.00031592 | 0.00030888 | 0.0002043 | 0.00026316 | 0.00024578 | 0.00016443 |
| 6 | 0.0013804 | 0.0021915 | 0.0022523 | 0.002748 | 0.0020147 | 0.00074943 | 0.0006513 | 0.0010029 |
| 7 | 1.1235 | 0.82731 | 0.065945 | 0.31105 | 0.11833 | 0.069809 | 0.14157 | 0.16285 |

## entropy (means across eight examples)

| token | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.0745 | 1.6965 | 1.3494 | 1.0703 | 1.3972 | 1.4027 | 1.2749 | 1.3247 |
| 1 | 2.0728 | 1.5405 | 1.5462 | 1.4747 | 1.4152 | 1.569 | 1.6079 | 1.2341 |
| 2 | 2.0717 | 1.5327 | 1.3277 | 1.5142 | 1.2175 | 1.293 | 0.82288 | 1.6909 |
| 3 | 2.0721 | 1.5535 | 1.3607 | 1.1342 | 1.0751 | 1.4935 | 1.4129 | 1.7336 |
| 4 | 2.0722 | 1.2651 | 1.6818 | 1.4782 | 1.1895 | 1.1704 | 1.2697 | 1.3935 |
| 5 | 2.0722 | 1.795 | 1.2858 | 0.97366 | 1.4283 | 1.5013 | 1.3258 | 1.7196 |
| 6 | 2.0728 | 1.6853 | 1.3157 | 1.0097 | 1.4131 | 1.1862 | 1.3783 | 1.3223 |
| 7 | 2.0723 | 1.8106 | 1.467 | 1.0979 | 1.2726 | 1.2701 | 1.6327 | 1.6547 |

## Decision summaries

| Decision | Mean logit L2 | Mean entropy | REPEAT count | RESTART count | Mean abs REPEAT grad | Mean abs RESTART grad |
|---|---|---|---|---|---|---|
| 0 | 0.154055 | 2.07257 | 0 | 0 | 0 | 0 |
| 1 | 0.113299 | 1.60989 | 14 | 13 | 0.0120457 | 0.00794723 |
| 2 | 0.0120972 | 1.41679 | 18 | 7 | 0.00333591 | 0.00169008 |
| 3 | 0.0574337 | 1.2191 | 26 | 14 | 0.0390604 | 0.0386309 |
| 4 | 0.0189818 | 1.30106 | 17 | 17 | 0.000829115 | 0.00107315 |
| 5 | 0.0137158 | 1.3608 | 21 | 14 | 0.00286434 | 0.00219179 |
| 6 | 0.0238768 | 1.34064 | 18 | 18 | 0.00498666 | 0.0022925 |
| 7 | 0.0248884 | 1.50918 | 19 | 7 | 0.00591633 | 0.00336044 |

Total selected actions: {"ADVANCE(0)": 19, "ADVANCE(1)": 19, "ADVANCE(2)": 30, "ADVANCE(3)": 24, "ADVANCE(4)": 24, "ADVANCE(5)": 47, "ADVANCE(6)": 50, "ADVANCE(7)": 76, "REPEAT": 133, "RESTART": 90, "HALT": 0}

HALT is never sampled before exclusion in this path. Its full-mask probability is recorded as a hypothetical categorical diagnostic; it is not an observed HALT frequency.

## One-decision sensitivity

Only token7/decision4 changes; all later scripted ADVANCE destinations remain exactly canonical and legal. These controls use deterministic inference and the same terminal commit. They jointly change the successor and its metadata, and do not isolate metadata-only effects.

| Action | CE | Delta CE | Prediction | Block calls/sequence |
|---|---|---|---|---|
| ADVANCE(4) | 0.00145460619 | 0 | 0 | 64 |
| REPEAT | 0.00145448709 | -1.19092874e-07 | 0 | 64 |
| RESTART | 0.00145198742 | -2.61876266e-06 | 0 | 63 |

## Interpretation boundary

Frozen parameters still pass gradients through activations and persistent token states. Retained logits have a rigorous loss cotangent; shared parameter gradients are reported per example, not attributed to individual decisions. Magnitudes do not imply an optimizer preference. No optimization direction was applied.

This single-seed, eight-example measurement does not establish the cause of the earlier training failure, estimator bias magnitude, or an exact hidden-state/metadata decomposition. Counterfactual route CE is a forward effect, not a gradient. Non-differentiable outgoing metadata at decision7 is unused by terminal commit; its action logits can nevertheless affect terminal h and receive gradient. No fix is proposed.

## Answers to the seven research questions

1. **Gradient availability:** all 512 decisions have nonzero retained-logit gradients, including all 64 at decision7. This rules out complete gradient disconnection on these trajectories; it does not establish useful credit.

2. **Within-token variation:** mean L2 by decision0–7 is 0.15405458, 0.11329916, 0.012097209, 0.057433678, 0.018981797, 0.013715751, 0.023876784, 0.024888366.

3. **Concentration:** gradients are not concentrated on late within-token decisions: decisions4–7 account for 19.4725% of the summed retained-logit norms. They are concentrated on the final token: 94.9334% of summed norms, and its mean is 131.1601 times the mean over earlier tokens. Final-token mean exceeds earlier-token mean in 6/8 examples. These norm sums are descriptive, not an additive decomposition of loss or shared-parameter gradients.

4. **RESTART magnitude:** overall mean absolute RESTART gradient is 0.0071482566, REPEAT 0.0086298118, ADVANCE(7) 0.04136678. RESTART is not the largest aggregate action gradient. Legal-conditioned statistics are retained in JSON because actions have unequal availability. At decision3, RESTART and REPEAT are comparable; this is not evidence of a unique restart-gradient anomaly. No update direction was tested.

5. **HALT:** exactly zero in all 512 measured unified-logit gradients. Actual HALT selection count is zero; pre-exclusion sampling never occurs.

6. **Single-decision sensitivity:** canonical CE 0.0014546061866; REPEAT CE 0.0014544870937, delta −1.1909287423e−7; RESTART CE 0.0014519874239, delta −2.6187626645e−6. Prediction remains class0 in all controls. This position/example exhibits negligible CE change, not material task disruption. The smallest delta is near float32 resolution. This cannot characterize other positions or multi-decision perturbations.

7. **Plausible failure mechanism:** temporal attenuation of task credit to early-token routing is the clearest measured candidate. The policy is not strongly saturated at decision0 (mean entropy 2.07257488 nats versus log(8)=2.07944154). Gradients exist throughout, but availability alone does not show that the biased ST signal ranks useful routes correctly. Hard-metadata credit remains absent and unquantified as a task-loss contribution by this measurement. Neither a sole cause nor a concrete implementation error is established.

This fixed draw contains seven class0 examples and one class1 example; fresh stochastic-route accuracy is 0.375 and mean CE 3.4203078. It is a mechanism probe, not an accuracy benchmark or multi-seed result. Weights were bitwise unchanged. No core code change, optimization change, or Gate-C success/failure declaration is justified by these measurements.
