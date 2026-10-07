# Visual Defect Detection

Classifies product images as **normal** or **defective** (the positive class) using a
transfer-learned CNN. Evaluation follows a leakage-safe protocol, and the model is served by a
hardened ONNX Runtime API that doesn't depend on torch.

> Status: **V3 — calibration & operating point.** Real dataset (Kaggle casting, both releases) processed and leakage-audited; training in progress. Later sections are added as each version lands.

## Quickstart (development)

Prerequisites: [uv](https://docs.astral.sh/uv/) and GNU make (Windows: `winget install ezwinports.make`).

```bash
make setup      # .venv on Python 3.12 from hash-pinned lockfiles
make lint       # ruff + ruff format --check + mypy
make test       # pytest with coverage
make inspect    # inventory of data/raw (see data/README.md)
make data       # manifest -> dedupe + group-aware split -> leakage audit
make eda        # run notebooks/01_eda.ipynb (train + val only)
make train      # two-stage fine-tuning -> runs/<stamp>_<config-hash>/best.pt
make calibrate  # temperature + threshold + review band on val -> model_meta.candidate.json
make help       # all targets
```

## Layout

| Path | Purpose |
| --- | --- |
| `src/defect_detection/core/` | Framework-agnostic code shared by training and serving (no torch) |
| `src/defect_detection/data/` | Inventory, manifest, dedupe, split, leakage audit, torch Dataset |
| `src/defect_detection/training/` | Model, training, calibration, evaluation, export, benchmark |
| `src/defect_detection/serving/` | FastAPI + ONNX Runtime inference service (torch forbidden) |
| `requirements/*.in` → `*.txt` | Dependency declarations → hash-pinned lockfiles (`make lock`) |
| `docs/DECISIONS.md` | Every non-trivial design decision, with alternatives and trade-offs |

## Roadmap

| Version | Scope |
| --- | --- |
| V0 | Foundation, tooling, dataset inventory |
| V1 | Manifest, dedupe, group-aware split, leakage audit, EDA |
| V2 | Shared preprocessing, two-stage fine-tuning |
| V3 | Calibration and operating point (validation only) |
| V4 | One-shot test evaluation, error analysis |
| V5 | ONNX export, parity, benchmark |
| V6 | Inference API |
| V7 | Security hardening |
| V8 | Docker, CI/CD, docs, demo |

## Licence

Code: MIT (see `LICENSE`). The dataset has its own licence, recorded in `data/README.md`.
