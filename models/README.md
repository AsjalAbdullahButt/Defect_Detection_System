# models/

Exported, servable model versions. Everything here except this README is gitignored. Model
files stay local and are never committed.

## Layout

```text
models/
└── <version>/                 # e.g. 20261007T173514Z_c9fca37c51cb
    ├── model.onnx             # logits for (normal, defective), dynamic batch, opset 17
    ├── model_meta.json        # threshold, temperature, review band, input spec, metrics
    └── SHA256SUMS             # checksums of the two files above
```

`<version>` = `<training start UTC>_<config hash>`, so every artifact traces back to its
config (`configs/train.yaml`) and git commit (inside `model_meta.json`).

## Rules

- **Immutable.** `make export` refuses to overwrite an existing version. Never edit files
  inside a version folder: the API refuses to start if a checksum no longer matches.
- **Select by name.** The API serves `models/$MODEL_VERSION` (set in `.env`). Rolling back
  means changing that one variable (see `docs/RUNBOOK.md`).

## Verify a version by hand

```bash
cd models/<version> && sha256sum -c SHA256SUMS
```

```powershell
Get-Content models\<version>\SHA256SUMS | ForEach-Object {
  $hash, $name = $_ -split '  '
  if ((Get-FileHash "models\<version>\$name" -Algorithm SHA256).Hash.ToLower() -ne $hash) { "MISMATCH $name" }
}
```

## Get a model

- Train one: `bash scripts/prepare_data.sh` then `bash scripts/train_pipeline.sh`.
