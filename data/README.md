# Data

Nothing under `data/raw/` or `data/processed/` is committed (see `.gitignore`).

## Placing the dataset

Copy or unzip the dataset into `data/raw/` **without modifying it**. Any layout works: the
inventory treats the immediate parent folder of each image as its class folder and recognises
`train` / `val|valid|validation` / `test` folders as an existing split. For example:

```
data/raw/
└── casting_data/
    ├── train/{def_front,ok_front}/*.jpeg
    └── test/{def_front,ok_front}/*.jpeg
```

Then run:

```bash
make inspect          # = defect-detection inspect data/raw --json-out reports/inventory.json
```

## Folders

| Path | Contents | Written by |
|---|---|---|
| `raw/` | Original dataset, read-only by convention | you |
| `processed/manifest.csv` | One row per image: SHA-256, pHash, size, label | `defect-detection manifest` (V1) |
| `processed/splits.csv` | Image → train/val/test assignment | `defect-detection split` (V1) |
| `processed/leakage_audit.json` | Cross-split overlap checks + test-set hash | `defect-detection audit` (V1) |

## Licence

Record the dataset's licence here once `make inspect` has identified the dataset and the
licence has been confirmed on its source page. Non-commercial licences (e.g. CC BY-NC-*)
allow this assessment but not commercial deployment of models trained on the data.
