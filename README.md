# Visual Defect Detection

Classifies photos of cast impellers as **normal** or **defective** (the positive class). The
model is an ImageNet-pretrained EfficientNet-B0, fine-tuned and evaluated under a leakage-safe
protocol, and served by a hardened, torch-free ONNX Runtime API in a non-root, read-only
container.

**The main finding:** the public dataset's official test set shares physical parts with its
training set. Cleaned of those copies, the official test still scores near-perfectly (F1 0.998),
but an independent capture of the same product scores **F1 0.90**. The review band (human in the
loop) brings automatically shipped missed defects there from 67 down to 16.

![Serving request flow](docs/architecture_serving.png)

## Contents

- [Quickstart](#quickstart)
- [Results](#results)
- [Dataset strategy and leakage controls](#dataset-strategy-and-leakage-controls)
- [Model approach](#model-approach)
- [Error analysis](#error-analysis)
- [Accuracy, latency and size trade-offs](#accuracy-latency-and-size-trade-offs)
- [API reference](#api-reference)
- [Security summary](#security-summary)
- [Reproducing training](#reproducing-training)
- [Known limitations and future work](#known-limitations-and-future-work)
- [Repository layout](#repository-layout)

## Quickstart

### Docker (production-like)

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # put the key in DD_API_KEYS
# models/<MODEL_VERSION>/ must exist (produced by `make export`)
docker compose up -d --build
curl -s http://127.0.0.1:8000/ready
curl -s -H "X-API-Key: <your key>" -F "file=@examples/images/<image>.jpeg" \
     http://127.0.0.1:8000/v1/predict
```

### Local development

Prerequisites: [uv](https://docs.astral.sh/uv/) and GNU make (Windows:
`winget install ezwinports.make`).

```bash
make setup        # .venv (Python 3.12) from hash-pinned lockfiles
make lint test    # ruff, ruff format --check, mypy (strict on core/serving), pytest
DD_MODEL_DIR=models/<version> DD_ENVIRONMENT=development make serve   # http://127.0.0.1:8000/docs
make help         # every target
```

## Results

Model `20261007T173514Z_c9fca37c51cb`; operating threshold **0.9036** (chosen on validation:
best precision with defect recall ≥ 0.99); temperature **1.298**. Each test set was evaluated
**once**. 95% bootstrap CIs, 1,000 resamples.

| Metric (defective = positive) | Official test (n = 715) | External test, 512px capture (n = 1,300) |
| --- | --- | --- |
| PR-AUC | 1.0000 [1.0000, 1.0000] | 0.978 [0.973, 0.983] |
| ROC-AUC | 1.0000 [0.9999, 1.0000] | 0.963 [0.955, 0.972] |
| Precision | 1.0000 [1.0000, 1.0000] | 0.889 [0.869, 0.913] |
| Recall | 0.9956 [0.9886, 1.0000] | 0.914 [0.894, 0.935] |
| F1 | 0.9978 [0.9943, 1.0000] | 0.902 [0.886, 0.918] |
| Macro F1 | 0.9970 [0.9923, 1.0000] | 0.874 [0.856, 0.893] |

Confusion matrices (rows = true class, columns = predicted):

| | Official: pred normal | Official: pred defective | External: pred normal | External: pred defective |
| --- | --- | --- | --- | --- |
| **normal** | 262 | 0 | 430 | 89 |
| **defective** | 2 | 451 | 67 | 714 |

![Confusion matrices](docs/figures/test_confusion.png)

**Review band** (`0.1054 ≤ P(defective) < 0.9979` → `needs_review`), assuming reviewers decide
correctly:

| | Official test | External test |
| --- | --- | --- |
| Routed to a person | 4 images (0.6%) | 282 images (21.7%) |
| Errors caught by review | 2 of 2 | 113 of 156 |
| Missed defects shipped automatically | 0 | 16 |

The full results are in `reports/metrics.json` and `docs/MODEL_CARD.md`.

## Dataset strategy and leakage controls

Data: Kaggle *Real-life industrial dataset of casting product* (both releases; `data/README.md`).
Licence CC BY-NC-ND 4.0, so non-commercial use only.

| Control | What was done | Evidence |
| --- | --- | --- |
| Manifest first | SHA-256 for all 8,648 images before any split | `make manifest` |
| Exact duplicates | **64 official-test images were byte-identical to train images**; the train copies were removed | `split_report.json` |
| Same-part copies | Rotation/flip-aligned image correlation (32px shortlist, 128px re-score). 64-bit pHash was tried first and rejected: 95% of images were within 4 bits of another, including different parts (D-031) | DECISIONS D-031–D-033 |
| Threshold 0.99 | Chosen by rendering pairs at 0.94–0.9999 and checking they were the same part (matching burrs) | D-032 |
| Official test kept intact | 524 train images with their own match ≥ 0.99 to a test image removed (direct matches; chains of look-alike normals join different parts) | D-033 |
| Group-aware split | Val carved from train by same-part clusters, stratified (60% defective in both) | `tests/unit/test_split.py` |
| Second test set | The 512px release (no match ≥ 0.99 to train/val) is held out as `external_test` | D-034 |
| Independent audit | Re-derives cross-split similarity from pixels, not cluster ids; fails CI on any overlap. It caught a real clustering miss once (shortlist k = 10 → 30) | `make audit`, `tests/leakage/` |
| Test isolation | EDA loads only train/val; calibration/threshold use val only (asserted); `evaluate` refuses a second run and checks the audited test fingerprint | `one_shot_test.py` |
| Fit on train only | Class weights from train; ImageNet mean/std (nothing fitted on our data) | D-017, D-022 |
| Train/serve skew | Training and serving share `core/preprocessing.py`; dataset tensors equal serving tensors bit for bit; PyTorch ↔ ONNX parity checked at export | `tests/leakage/test_preprocessing_parity.py` |

EDA (train+val only) also found a **lighting shortcut**: brightness alone separates the classes
with AUC 0.883 (defective photos are darker). See the error analysis for its effect.

## Model approach

- **EfficientNet-B0** (timm, ImageNet weights): 4.0 M parameters, a 16 MB ONNX file and fast CPU
  inference. ResNet-18 and ConvNeXt-Tiny can be selected in `configs/train.yaml`.
- **Two-stage fine-tuning:** classifier head first (3 epochs, LR 1e-3, BatchNorm frozen), then all
  layers (LR 1e-4, cosine schedule), AdamW, early stopping on **val PR-AUC** with val loss as the
  tie-break.
- **Mild augmentation** (±15° rotation, flips, ±20% brightness/contrast, light blur/noise, ±5%
  shift/scale). No crops or cutout: they can delete the only defective pixels.
- **Imbalance:** about 60% defective. **Class-weighted cross-entropy** with weights from train
  counts, rather than a sampler (which repeats images) or focal loss (an extra γ to tune) (D-022).
- **Calibration and threshold on validation only:** temperature scaling (T = 1.298), then the
  highest-precision threshold with recall ≥ 0.99, placed at the max-margin point of the empty
  score gap (D-028, D-054).

## Error analysis

From `reports/metrics.json` (both test sets) and `reports/test_predictions.csv`:

1. **Official test:** 2 missed defects, 0 false alarms, and both errors fall in the review band.
2. **External test:** 156 errors at the threshold, concentrated on **unfamiliar parts**:
   12.8% error rate for images with similarity < 0.95 to any training image, against 4.5% at
   0.95–0.97.
3. **Lighting shortcut confirmed.** Within the external *defective* class, P(defective) falls as
   brightness rises (Spearman −0.354), and errors per brightness quartile go 14 → 26 → 52 → 64.
   Bright defective parts are the main failure mode. On the official test the effect is
   negligible (−0.07).
4. **Label noise:** 6 same-part clusters carry both labels; one val "defect" scores 0.108.

Grad-CAM galleries and confidence histograms are produced by `make error-analysis`
(`notebooks/02_error_analysis.ipynb`).

## Accuracy, latency and size trade-offs

CPU benchmark (`make benchmark`, 4 threads, 100 timed calls per cell, real val images; full
table in `reports/benchmark.md`):

| Runtime | Batch 1 p50 / p99 | Batch 1 img/s | Batch 32 img/s | Val F1 @ threshold | Decisions changed vs PyTorch | File |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch fp32 | 31.7 / 53.7 ms | 31.6 | 33.7 | 0.9991 | — | 16.3 MB (`best.pt`) |
| **ONNX Runtime fp32 (served)** | **9.9 / 11.1 ms** | **101.3** | 76.8 | 0.9991 | 0 of 901 | 16.0 MB |
| ONNX Runtime INT8 (dynamic) | 184 / 248 ms | 5.4 | 5.0 | **0.2335** | 498 of 901 | 4.3 MB |

ONNX Runtime is 3.2× faster than PyTorch at batch 1 with identical decisions. Dynamic INT8 is
both slower (quantised depthwise convolutions fall back to slow integer kernels) and broken
(F1 0.23), so it is **not shipped**. Static, calibrated INT8 or OpenVINO would be the path for
edge devices. End-to-end HTTP latency through the full API stack (auth, validation, decode,
inference) measured 13–18 ms per image.

| Choice | Effect |
| --- | --- |
| EfficientNet-B0 vs ConvNeXt-Tiny | ~7× fewer parameters and a much smaller artifact; accuracy on the official test is already saturated |
| Threshold 0.9036 vs 0.5 | Same decisions on val; max margin to both classes |
| Review band | ~0.6% of images to people in-distribution, ~22% under drift; catches 113 of 156 external errors |
| Serving image without torch | Smaller image and attack surface; inference via ONNX Runtime only |

## API reference

All `/v1/*` endpoints require `X-API-Key` when keys are configured (always in production).

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness (no auth) |
| `GET` | `/ready` | Model loaded and verified (no auth) |
| `GET` | `/v1/model` | Version, threshold, temperature, review band, val/test metrics |
| `POST` | `/v1/predict` | Multipart `file`: one JPEG/PNG/BMP/WebP image |
| `POST` | `/v1/predict/batch` | Multipart `files`: up to `DD_MAX_BATCH_FILES` images (all-or-nothing) |

A real response of `/v1/predict` (`examples/responses/3_official_missed_but_reviewed.json`: a
defect the model misses at the threshold, which the review band sends to a person):

```json
{
  "predicted_class": "normal",
  "confidence": 0.6552811397032885,
  "defect_probability": 0.3447188602967115,
  "threshold": 0.9036446967158055,
  "needs_review": true,
  "model_version": "20261007T173514Z_c9fca37c51cb",
  "request_id": "1636646fe2964c11a8f203102df0889e",
  "latency_ms": 13.68
}
```

`confidence` is the probability of the *predicted* class. Errors are always
`{"error", "detail", "request_id"}`: 400 invalid image, 401 bad key, 413 too large,
415 not an accepted image type, 422 missing field, 429 rate limited (with `Retry-After`),
503 overloaded / not ready, 504 inference timeout. Real request/response examples:
`examples/README.md`.

## Security summary

Threat model (STRIDE-lite, 19 threats, each mapped to its test) in `docs/SECURITY.md`.

- **Input:** size cap enforced while streaming; type from magic bytes; pixel cap checked before
  decoding (decompression bombs); `verify()` + re-decode; no multi-frame images; uploads stay
  in memory, and filenames are never used or logged.
- **API:** API keys compared in constant time against all keys; auth and rate limits run
  *before* the body is read; failed attempts throttled; strict security headers; CORS off by
  default; docs off in production; uniform errors with no stack traces.
- **Model:** `SHA256SUMS` + schema verified at startup, and the server refuses to start on any
  mismatch; no pickle in serving; `weights_only=True` for checkpoints.
- **Supply chain / container:** hash-pinned lockfiles with `--require-hashes`; base image pinned
  by digest; non-root UID 10001; read-only root FS; `cap_drop: ALL`; `no-new-privileges`;
  resource limits; metrics on an unpublished internal port.
- **Scanners:** bandit 0 issues; pip-audit 0 known vulnerabilities in all lockfiles; gitleaks
  and Trivy in CI.

## Reproducing training

```bash
# put the Kaggle archive's contents in data/raw/ (see data/README.md)
make data         # manifest -> split (dedupe, group-aware) -> leakage audit
make eda          # notebooks/01_eda.ipynb (train + val only)
make train        # ~3 h on a 4-core CPU; seed 42; runs/<stamp>_<config-hash>/
make calibrate    # temperature, threshold, review band on val
make evaluate     # ONE-SHOT test + external test (refuses a second run)
make export       # models/<version>/ after checker + parity
make benchmark    # reports/benchmark.md
```

Every artifact records the git commit and config hash. All hyperparameters are in
`configs/train.yaml`, with every decision explained in `docs/DECISIONS.md`.

## Known limitations and future work

**Limitations:** unseen defect types and other casting models; lighting/camera drift (the
external capture loses ~10 F1 points); the lighting shortcut (bright defective parts are
missed); a single top-down viewpoint; small clean test sets; CPU-only training; rate limits
per process; PatchCore implemented but not evaluated on the real data in this round.

**Future work:**

- **Drift monitoring:** alert on the review rate and the `dd_defect_probability` histogram.
- **Active learning:** retrain on images reviewed by people (especially false negatives).
- **Lighting robustness:** stronger photometric augmentation or per-image normalisation, judged
  on the external set.
- **Anomaly detection:** run the PatchCore baseline for defect types never seen in training.
- **Edge deployment:** static INT8 calibration, OpenVINO or TensorRT.
- **Model registry** (e.g. MLflow) instead of versioned folders; Redis-backed rate limits for
  multi-replica deployments.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/defect_detection/core/` | Torch-free code shared by training and serving: constants, preprocessing, model metadata, hashing |
| `src/defect_detection/data/` | Inventory, manifest, dedupe, split, leakage audit, EDA helpers, torch Dataset |
| `src/defect_detection/training/` | Model, trainer, calibration, thresholds, one-shot evaluation, error analysis, PatchCore, export, benchmark |
| `src/defect_detection/serving/` | FastAPI app factory, routes, service, ONNX engine, validation, security, middleware, observability |
| `configs/` | `train.yaml` (all hyperparameters), `serve.env.example` (all serving variables) |
| `docker/`, `docker-compose.yml` | Hardened serving image and stack; optional training image |
| `docs/` | Architecture, decisions, model card, security, runbook |
| `tests/` | `unit/`, `integration/` (real ONNX model), `security/`, `leakage/` |

Code licence: MIT.
