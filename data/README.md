# Data

Nothing under `data/raw/` or `data/processed/` is committed (see `.gitignore`).

## Placing the dataset

Copy or unzip the dataset into `data/raw/` **without modifying it**. Any layout works: the
inventory treats the immediate parent folder of each image as its class folder and recognises
`train` / `val|valid|validation` / `test` folders as an existing split. For example:

```text
data/raw/
└── casting_data/
    ├── train/{def_front,ok_front}/*.jpeg
    └── test/{def_front,ok_front}/*.jpeg
```

Then run:

```bash
make inspect          # = defect-detection inspect data/raw --json-out reports/inventory.json
```

## Pipeline

```bash
make data    # manifest -> split -> audit (audit exits 1 on any leak)
make eda     # executes notebooks/01_eda.ipynb (train + val only)
```

Raw class folders map to `normal` / `defective` through `data.class_aliases` in
`configs/train.yaml`. An unmapped folder stops the manifest step; it is never guessed.

## Folders

| Path | Contents | Written by |
| --- | --- | --- |
| `raw/` | Original dataset, read-only by convention | you |
| `processed/manifest.csv` | One row per decodable image: SHA-256, 8 rotation/flip pHashes, size, label | `make manifest` |
| `processed/manifest_report.json` | Counts, corrupt files left out, provenance | `make manifest` |
| `processed/splits.csv` | Image → `train`/`val`/`test`/`excluded` (+ reason), cluster id | `make split` |
| `processed/split_report.json` | Duplicate statistics, achieved ratios, exclusions, provenance | `make split` |
| `processed/leakage_audit.json` | Pass/fail checks, cross-split near-duplicate counts, test-set hash | `make audit` |

## The dataset used

**Real-life industrial dataset of casting product** (submersible pump impellers, top view),
Kaggle: <https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product>.
`archive.zip` is extracted unmodified into `data/raw/`:

| Folder | Images | Size | Role in this project |
| --- | --- | --- | --- |
| `casting_data/casting_data/train/{def_front,ok_front}` | 3758 / 2875 | 300×300 | train + val (after cleaning) |
| `casting_data/casting_data/test/{def_front,ok_front}` | 453 / 262 | 300×300 | official test, kept intact |
| `casting_512x512/casting_512x512/{def_front,ok_front}` | 781 / 519 | 512×512 | external test (`data.external_test_dirs`) |

All 8,648 files decode as RGB JPEG; none are corrupt.

**Leakage found in the official split** (details in `docs/DECISIONS.md`, D-031 to D-034):

- 64 test images are byte-identical to train images;
- 524 more train images are same-part copies (rotated/flipped/re-lit) of test images at
  aligned correlation ≥ 0.99. All of these are excluded from train;
- the 512×512 release has no same-part match in train/val (median nearest similarity 0.88),
  so it serves as an independent second test set.

## Licence

Listed on Kaggle as **CC BY-NC-ND 4.0** (to be confirmed on the dataset page before any reuse).
Non-commercial: fine for this assessment, but a model trained on it must not be deployed
commercially, and the data itself must not be redistributed (it is never committed here).
