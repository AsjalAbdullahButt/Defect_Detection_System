# Defect Detection

Secure defect detection API with leakage controls and a simple Streamlit demo UI.

[![CI](https://github.com/AsjalAbdullahButt/Defect_Detection_System/actions/workflows/ci.yml/badge.svg)](https://github.com/AsjalAbdullahButt/Defect_Detection_System/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-087F8C)](LICENSE)

![Inspect page: upload an image and review its result](docs/images/inspect.png)

[View the screenshot and results gallery](docs/GALLERY.md)

## What it does

Upload a top-view photo of a cast impeller, individually or in a batch.
The model returns **normal** or **defective**, a calibrated confidence score, and a flag for human review.
A FastAPI service runs the exported ONNX model on CPU; the Streamlit UI calls that API with a server-side key.

## Quickstart

You need an exported model in `models/<version>/` containing `model.onnx`,
`model_meta.json`, and `SHA256SUMS`. **No pretrained download is published yet**;
use an existing local export or follow [Train from scratch](#train-from-scratch) first.
Images and model weights are not committed to this repository.

### Option 1 - Docker (recommended)

Requires Docker Engine with the Compose plugin.

```bash
git clone https://github.com/AsjalAbdullahButt/Defect_Detection_System.git
cd Defect_Detection_System
cp .env.example .env
# Edit .env: set MODEL_VERSION to your exported folder name.
# Set DD_API_KEYS to a random key of at least 32 characters.
# Set DD_UI_API_KEY to that same key.
docker compose up --build
```

- UI: <http://localhost:8501>
- API: <http://localhost:8000> (interactive `/docs` when `DD_ENVIRONMENT=development`)
- Both containers run as non-root with a read-only root filesystem. Ports bind to localhost.
- Docker's UI starts without local reports or sample images; Inspect and Batch use the API.
  See the [UI guide](ui/README.md) for optional read-only report mounts.

### Option 2 - Local with Bash

Requires [uv](https://docs.astral.sh/uv/) and Bash (Git Bash on Windows).
Run the clone and `.env` steps above, then:

```bash
bash scripts/setup.sh serve ui
source .venv/bin/activate         # Git Bash on Windows: source .venv/Scripts/activate
bash scripts/run_api.sh           # terminal 1
bash scripts/run_ui.sh            # terminal 2, from the repository root
```

PowerShell activation is `.venv\Scripts\Activate.ps1`; run the `.sh` scripts through Git Bash.
The scripts also support Python 3.12 with pip when uv is unavailable.

### Try it

Use an image you have locally; the dataset download is described in the [dataset guide](data/README.md).

```bash
read -r -s -p "API key: " API_KEY; echo
curl -s -H "X-API-Key: $API_KEY" -F "file=@path/to/your/impeller.jpeg" \
  http://localhost:8000/v1/predict
```

Actual response from the evaluated model on an official-test defective image
([saved response](examples/responses/2_official_defective.json)); your result and latency will vary:

```json
{
  "predicted_class": "defective",
  "confidence": 0.999999999999915,
  "defect_probability": 0.999999999999915,
  "threshold": 0.9036446967158055,
  "needs_review": false,
  "model_version": "20261007T173514Z_c9fca37c51cb",
  "request_id": "a7aeff7cb2fd4512bcbcf37a88611ac4",
  "latency_ms": 13.27
}
```

## Train from scratch

Clone the repository first, then download and extract the dataset into `data/raw/`
as described in the [data setup](data/README.md). Training requires Python 3.12
and can take several hours on CPU.

```bash
bash scripts/setup.sh train
bash scripts/prepare_data.sh --data-dir data/raw
bash scripts/train_pipeline.sh --config configs/train.yaml
# Optional: generate the Explain page's saved heatmaps.
uv run defect-detection explain
```

The pipeline deduplicates images, audits split overlap, trains, calibrates on validation,
evaluates the test sets once, and exports a checksummed model to `models/<version>/`.
Set `MODEL_VERSION` in `.env` to that folder name. Repeating evaluation on the same run is refused.

## Results

Model `20261007T173514Z_c9fca37c51cb`. Defective is the positive class;
brackets show 95% bootstrap confidence intervals from the saved `reports/metrics.json`.
The report stays local; the [evaluation guide](docs/evaluation.md) records the published results.

| Metric | Official test (715 images) | External test (1,300 images) |
| --- | --- | --- |
| Precision | 1.0000 [1.0000, 1.0000] | 0.8892 [0.8689, 0.9128] |
| Recall | 0.9956 [0.9886, 1.0000] | 0.9142 [0.8944, 0.9348] |
| F1 (defective) | 0.9978 [0.9943, 1.0000] | 0.9015 [0.8862, 0.9183] |
| PR-AUC | 1.0000 [1.0000, 1.0000] | 0.9780 [0.9730, 0.9834] |
| CPU inference p95 | 10.82 ms | Same runtime benchmark; not measured separately |

Latency is from [reports/benchmark.md](reports/benchmark.md): ONNX FP32, batch 1,
four CPU threads, 100 timed calls on validation images; it excludes HTTP overhead.

![Confusion matrices for the official and external test sets](docs/figures/test_confusion.png)

The threshold **0.9036** was selected on validation to maximize precision subject to
at least 99% defect recall; uncertain scores in **[0.1054, 0.9979)** request human review.

## How it works

![Serving architecture: validation, preprocessing, ONNX inference and response](docs/architecture_serving.png)

- **Data:** hash images, remove exact and same-part duplicates, then audit the splits.
- **Model:** fine-tune EfficientNet-B0; choose calibration and the decision threshold on validation only.
- **ONNX:** export FP32 weights, verify prediction parity, and checksum the artifacts.
- **API/UI:** validate uploads, authenticate and rate-limit requests, return predictions to the browser.

## Project structure

<details>
<summary>Browse the top two levels</summary>

```text
.
|-- configs/              # Training and serving configuration
|-- data/
|   |-- raw/              # Local dataset (ignored)
|   |-- processed/        # Generated manifests and splits (ignored)
|-- docker/               # API, UI and training Dockerfiles
|-- docs/
|   |-- figures/          # Evaluation plots
|   |-- images/           # UI screenshots
|-- examples/
|   |-- responses/        # Real saved API responses
|-- models/               # Local model exports (ignored)
|-- notebooks/            # EDA and error analysis
|-- reports/              # Generated metrics and analysis
|-- requirements/         # Dependency inputs and hash-pinned lockfiles
|-- scripts/              # Setup, training and launch commands
|-- src/
|   |-- defect_detection/ # Data, training, core and serving code
|-- tests/                # Unit, integration, leakage, security and UI tests
|-- ui/
    |-- assets/           # Logo and optional local samples
    |-- components/       # Shared display components
    |-- pages/            # Inspect, Batch, Model and Explain
    |-- services/         # Typed HTTP client and session helpers
    |-- styles/           # Shared theme and design tokens
```

</details>

## Documentation

| Topic | Link |
| --- | --- |
| Screenshots and figures | [Gallery](docs/GALLERY.md) |
| UI setup and controls | [UI guide](ui/README.md) |
| Dataset and leakage controls | [Dataset](docs/dataset.md) |
| Model and training | [Model](docs/model.md) |
| Evaluation and failure cases | [Evaluation](docs/evaluation.md) |
| Endpoints and responses | [API reference](docs/api.md) |
| Threat model and protections | [Security](docs/SECURITY.md) |
| Deployment and operations | [Runbook](docs/RUNBOOK.md) |
| Intended use and performance | [Model card](docs/MODEL_CARD.md) |
| Design decisions | [Decision log](docs/DECISIONS.md) |

## Known limitations

- **Capture shift matters:** F1 drops from 0.998 on the official test to 0.902 on the external capture.
- **Bright defects can be missed.** Brightness is a learned shortcut; human review catches some errors, not all.
- **Narrow scope:** evaluated on cast impellers, not other parts, cameras or production lines.
- **Heatmaps are explanatory, not segmentation.** Grad-CAM does not establish the cause of a defect.
- **Dataset terms differ from the code license.** The code is MIT; the dataset is CC BY-NC-ND 4.0 and is not bundled.
- **Local demo by default:** models and private reports must be supplied locally; public deployment needs TLS, access control and validation on target data.
