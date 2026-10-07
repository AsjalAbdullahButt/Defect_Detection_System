# Model card — casting defect classifier

| Field | Value |
| --- | --- |
| Model version | `20261007T173514Z_c9fca37c51cb` (`<training start UTC>_<config hash>`) |
| Task | Binary image classification: `normal` (0) vs `defective` (1, positive class) |
| Architecture | EfficientNet-B0 (timm, ImageNet `ra_in1k` weights), 4.01 M parameters, 2-logit head |
| Input | RGB, EXIF-upright, direct resize to 224×224 (bilinear), ImageNet mean/std |
| Output | Logits; `P(defective) = softmax(logits / T)[1]` with **T = 1.298** |
| Decision | defective if P ≥ **0.9036**; `needs_review` if 0.1054 ≤ P < 0.9979 |
| Artifact | `models/<version>/model.onnx` (16.0 MB, opset 17), `model_meta.json`, `SHA256SUMS` |
| Provenance | config hash `c9fca37c51cb`; git commit recorded in `model_meta.json` |
| Licence | Code MIT. Data CC BY-NC-ND 4.0 (per Kaggle), so **non-commercial use only** |

## Intended use

Visual inspection of **top-view photographs of submersible-pump impeller castings** taken on a
fixed rig like the one in the training data. The model flags likely defects (blow holes,
burrs, shrinkage, edge damage) for removal or human review. It is decision support: images in
the review band, and any change of camera, lighting or part type, need a person.

**Out of scope:** other parts, other viewpoints, colour-critical inspection, defect
localisation/measurement, and any safety-critical use without human review.

## Training data

Kaggle "Real-life industrial dataset of casting product" (`data/README.md`). The official
300×300 train folder, cleaned for leakage (D-031–D-033):

| Split | Normal | Defective | Notes |
| --- | --- | --- | --- |
| train | 2,039 | 3,105 | 64 byte-identical and 524 same-part copies of test images removed |
| val | 353 | 548 | carved from official train, grouped by same-part clusters |
| test (official) | 262 | 453 | kept intact; median similarity to train 0.978 after cleaning |
| external test (512×512 release) | 519 | 781 | never trained on; median similarity to train 0.880 |

Training: two-stage fine-tuning (head 3 epochs, then all layers), AdamW, cosine schedule,
class-weighted cross-entropy (weights from train counts), early stopping on val PR-AUC. It ran
11 epochs on CPU; the best checkpoint was fine-tune epoch 4.

## Evaluation (each test set evaluated once; 95% bootstrap CIs, 1,000 resamples)

| Metric @ threshold 0.9036 | Official test (n = 715) | External test (n = 1,300) |
| --- | --- | --- |
| PR-AUC | 1.0000 [1.0000, 1.0000] | 0.978 [0.973, 0.983] |
| ROC-AUC | 1.0000 [0.9999, 1.0000] | 0.963 [0.955, 0.972] |
| Precision (defective) | 1.0000 [1.0000, 1.0000] | 0.889 [0.869, 0.913] |
| Recall (defective) | 0.9956 [0.9886, 1.0000] | 0.914 [0.894, 0.935] |
| F1 (defective) | 0.9978 [0.9943, 1.0000] | 0.902 [0.886, 0.918] |
| Macro F1 | 0.9970 [0.9923, 1.0000] | 0.874 [0.856, 0.893] |
| Confusion (TN / FP / FN / TP) | 262 / 0 / 2 / 451 | 430 / 89 / 67 / 714 |

**With the review band** (assuming reviewers decide correctly):

| | Official test | External test |
| --- | --- | --- |
| Images routed to review | 4 (0.6%) | 282 (21.7%) |
| Errors caught by review | 2 of 2 | 113 of 156 |
| Missed defects shipped automatically | 0 | 16 |
| False alarms rejected automatically | 0 | 27 |

Validation (used for calibration and threshold choice): PR-AUC 1.0, precision 1.0, recall
0.9982 (1 of 548 defects missed), NLL 0.0035 → 0.0033 after temperature scaling. ECE is
0.0009 / 0.0012 before / after, which isn't meaningful with a single validation error.

## Known limitations and weaknesses

1. **The official test set overstates performance.** After cleaning, its images are still very
   similar to training parts (median 0.978), and it scores almost perfectly. On the independent
   512×512 capture, F1 falls to 0.90 and 67 of 781 defects are missed at the threshold. The
   external number is the honest estimate for a new camera setup or new parts.
2. **Lighting shortcut.** In train/val, brightness alone separates the classes (AUC 0.883):
   defective photos are darker. On the external set, P(defective) falls as brightness rises
   *within* the defective class (Spearman −0.354), and errors grow with brightness quartile
   (14 → 26 → 52 → 64). Bright defective parts are the main failure mode.
3. **Unfamiliar parts.** External images without a close training relative (similarity < 0.95)
   have a 12.8% error rate, against 4.5% for those at 0.95–0.97.
4. **Label noise.** 6 near-duplicate clusters carry both labels (16 train images). One val
   defect (`cast_def_0_2838.jpeg`) scores 0.108; it may be mislabelled or an atypical defect.
5. **Unseen defect types** and other casting models are not represented.
6. **Single viewpoint**, top-down, fixed rig. Side-wall defects are invisible.
7. **Small clean test sets** relative to production volumes: a recall CI width of about 1 point
   on the official test means rare failure modes can hide.

## Recommendations / next steps

- Deploy **with the review band on**. Monitor the review rate and the `dd_defect_probability`
  histogram: the review rate rising from ~0.6% towards ~20% is the drift signal seen on the
  external capture.
- Before trusting a new rig: collect a few hundred labelled images, re-run `make evaluate` on
  them as a new external set, and recalibrate the threshold on a local val set.
- Reduce the lighting shortcut: stronger brightness/contrast augmentation or per-image
  normalisation, then compare on the external set (D-035).
- Add the reviewed images (especially false negatives) to training: active learning from the
  review queue.
- PatchCore (normal-only anomaly detection) is implemented (`make anomaly`) but was not
  evaluated on the real data in this round; it's the natural complement for unseen defect types.
