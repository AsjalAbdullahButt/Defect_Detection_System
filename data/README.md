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

## Licence

Record the dataset's licence here once `make inspect` has identified the dataset and the
licence has been confirmed on its source page. Non-commercial licences (e.g. CC BY-NC-*)
allow this assessment but not commercial deployment of models trained on the data.
