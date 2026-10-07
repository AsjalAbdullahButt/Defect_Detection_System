# Model: approach, training and operating point

## Approach

| Choice | Why (decision id in `DECISIONS.md`) |
| --- | --- |
| **EfficientNet-B0**, ImageNet weights (timm `ra_in1k`), 2-logit head | 4.0 M parameters, strong transfer, 16 MB ONNX, fast CPU inference (D-020) |
| Input 224×224, direct resize, ImageNet mean/std | No crop (could cut off a rim defect); nothing fitted to our data (D-017) |
| One preprocessing implementation, `core/preprocessing.py` | Training and serving produce bit-identical tensors (parity test) (D-018) |
| Mild augmentation, train only | ±15° rotation, flips, ±20% brightness/contrast, light blur/noise, ±5% shift/scale. No crop or cutout: they can erase the only defect (D-019) |
| Two-stage fine-tuning | Head only (3 epochs, LR 1e-3, BatchNorm frozen), then all layers (LR 1e-4, cosine), AdamW (D-021) |
| Imbalance: class-weighted cross-entropy | Weights from **train** counts (normal 1.26, defective 0.83); simpler than a sampler or focal loss (D-022) |
| Early stopping on val PR-AUC, val loss as tie-break | Threshold-free, focused on defects; PR-AUC saturates on this data (D-023) |
| Checkpoints `weights_only=True` | A tampered checkpoint can't run code (D-024) |

## Training run (`reports/train_summary.json`)

Seed 42, CPU, 11 epochs (3 head + 8 fine-tune). Early stopping kept **fine-tune epoch 4**.

| | PR-AUC | ROC-AUC | Precision | Recall | F1 @ 0.5 | Loss |
| --- | --- | --- | --- | --- | --- | --- |
| val | 1.0000 | 1.0000 | 1.0000 | 0.9982 | 0.9991 | 0.0035 |
| train (no augmentation) | 0.9999 | 0.9998 | 1.0000 | 0.9948 | 0.9974 | 0.0180 |

No overfitting: val is slightly *better* than un-augmented train. Train contains the noisier
images; val is close to train (median similarity 0.978).

![Training curves](figures/train_curves.png)

## Calibration and operating point (validation only, `reports/calibration.json`)

- **Temperature scaling:** T = 1.298. NLL 0.0035 → 0.0033. ECE 0.0009 → 0.0012, which isn't
  meaningful with a single val error (D-025, D-026).
- **Threshold = 0.9036:** the highest precision with defect recall ≥ 0.99, placed at the
  max-margin point of the empty score gap. Val defects score ≥ 0.998 except one outlier
  (0.108); normals score ≤ 0.157 (D-028, D-054).
- **Review band = [0.1054, 0.9979):** below it, ≥ 99.9% of val defects would still be caught;
  above it, flagged images are defective with ≥ 99.5% precision. The API returns
  `needs_review: true` inside it (D-029).

**Business rationale:** a missed defect ships a faulty pump to a customer; a false alarm
scraps or re-inspects a good part. Recall is the constraint and precision is optimised under
it. The uncertain middle goes to a person rather than being forced either way. The
illustrative cost table (missed defect 20 : false alarm 1 : review 0.25) is in
`reports/calibration.json`.

![Reliability diagram](figures/reliability.png)

## Export

ONNX opset 17, dynamic batch, `onnx.checker` passed. PyTorch ↔ ONNX parity on 32 val images:
max |Δlogit| 1.16e-4 with |logits| up to 37, which uses 9% of the `allclose` tolerance
(atol 1e-4 + rtol 1e-4). No decision changes (D-055).
