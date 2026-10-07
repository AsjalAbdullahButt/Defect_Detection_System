# Evaluation

Both test sets were evaluated **once** (`make evaluate`), with the threshold, temperature and
review band chosen on validation. The run refuses a second evaluation, and the test-set
fingerprints match the leakage audit (`reports/metrics.json`). 95% CIs: percentile bootstrap,
1,000 resamples.

## Headline results (threshold 0.9036)

| Metric (defective = positive) | Official test (n = 715) | External test (n = 1,300) |
| --- | --- | --- |
| PR-AUC | 1.0000 [1.0000, 1.0000] | 0.978 [0.973, 0.983] |
| ROC-AUC | 1.0000 [0.9999, 1.0000] | 0.963 [0.955, 0.972] |
| Precision | 1.0000 [1.0000, 1.0000] | 0.889 [0.869, 0.913] |
| Recall | 0.9956 [0.9886, 1.0000] | 0.914 [0.894, 0.935] |
| F1 | 0.9978 [0.9943, 1.0000] | 0.902 [0.886, 0.918] |
| Macro F1 | 0.9970 [0.9923, 1.0000] | 0.874 [0.856, 0.893] |
| TN / FP / FN / TP | 262 / 0 / 2 / 451 | 430 / 89 / 67 / 714 |
| Normalised (recall normal / defective) | 1.000 / 0.996 | 0.829 / 0.914 |

![Confusion matrices](figures/test_confusion.png)
![PR and ROC curves](figures/test_curves.png)

**How to read this:** the official test is still very close to training parts (median
similarity 0.978), so it mostly measures recognising near-copies. The external set is a
different capture (median 0.880), and its numbers are the honest estimate for new parts or a
new camera setup.

## Review band (`needs_review`)

| | Official test | External test |
| --- | --- | --- |
| Images routed to a person | 4 (0.6%) | 282 (21.7%) |
| Errors at the threshold | 2 | 156 |
| Errors caught by review | 2 | 113 |
| Missed defects shipped automatically | 0 | 16 |
| False alarms rejected automatically | 0 | 27 |

The jump in review rate (0.6% → 21.7%) is itself a drift alarm worth monitoring.

## Where the errors are

**By similarity to the nearest training image (external test):**

| Similarity | Images | Error rate |
| --- | --- | --- |
| < 0.95 | 1,188 | 12.8% |
| 0.95 – 0.97 | 89 | 4.5% |
| 0.97 – 0.98 | 17 | 0% |
| 0.98 – 0.99 | 6 | 0% |

**Lighting shortcut:** within the external *defective* class, P(defective) falls as
brightness rises (Spearman −0.354). Errors per brightness quartile: 14 → 26 → 52 → 64. On the
official test the effect is negligible (−0.07). Bright defective parts are the main failure
mode.

## Error cases with Grad-CAM (`reports/error_analysis/`, shown on the UI's Explain page)

| Case | p(defective) | Review? | Category (human visual judgement) |
| --- | --- | --- | --- |
| FN official `cast_def_0_150` | 0.223 | yes | Tiny defect or possible mislabel: nothing visible at 224 px |
| FN official `cast_def_0_1591` | 0.345 | yes | Subtle defect; part of the attention is on background corners |
| FP external `cast_ok_0_7604` | 0.99999 | no | Capture shift: off-centre part, dark specks read as defects |
| FP external `cast_ok_0_4358` | 0.99994 | no | Capture shift: bright lighting and glare (matches the shortcut) |

## Runtime trade-offs (`reports/benchmark.md`, 4 CPU threads)

| Runtime | Batch 1 p50 / p95 / p99 | img/s (batch 1) | Val F1 | Decisions changed |
| --- | --- | --- | --- | --- |
| PyTorch fp32 | 31.65 / 42.89 / 53.69 ms | 31.6 | 0.9991 | — |
| **ONNX Runtime fp32 (served)** | **9.87 / 10.82 / 11.14 ms** | **101.3** | 0.9991 | 0 / 901 |
| ONNX Runtime INT8 (dynamic) | 184.21 / 235.91 / 247.95 ms | 5.4 | 0.2335 | 498 / 901 |

Dynamic INT8 is slower and breaks accuracy, so it is not shipped. End-to-end API latency
(auth, validation, decode, inference) measured 13–18 ms per image (`examples/`).

## Not evaluated

PatchCore (normal-only anomaly detection) is implemented (`make anomaly`) but was not run on
the real data in this round.
