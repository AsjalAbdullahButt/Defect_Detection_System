# API examples (real responses)

Every file in `responses/` was produced by `run_examples.sh` against the production-mode server
(`DD_ENVIRONMENT=production`, API key required, docs off), serving model
`20261007T173514Z_c9fca37c51cb`. The images live in `data/raw/`: the dataset licence
(CC BY-NC-ND 4.0) does not allow redistributing them here.

```bash
# server (any port; 8000 by default)
DD_MODEL_DIR=models/20261007T173514Z_c9fca37c51cb DD_API_KEYS=<key> make serve
# examples
KEY=<key> BASE_URL=http://127.0.0.1:8000 sh examples/run_examples.sh
```

Single prediction:

```bash
curl -s -H "X-API-Key: $KEY" \
     -F "file=@data/raw/casting_data/casting_data/test/def_front/cast_def_0_1647.jpeg" \
     http://127.0.0.1:8000/v1/predict
```

## The six images

Decision threshold 0.9036; review band [0.1054, 0.9979).

| # | Image | True class | p(defective) | Predicted | needs_review | What it shows |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | official test `cast_ok_0_1210` | normal | 0.0000018 | normal | no | Typical clean part: confident and correct |
| 2 | official test `cast_def_0_1647` | defective | ≈ 1.0 | defective | no | Typical defect: confident and correct |
| 3 | official test `cast_def_0_1591` | defective | 0.3447 | normal | **yes** | One of the 2 official-test misses: wrong at the threshold, but routed to a person |
| 4 | external `cast_def_0_9435` | defective | ≈ 1.0 | defective | no | The model also works on the new capture for clear defects |
| 5 | external `cast_ok_0_2819` | normal | 0.9805 | defective | **yes** | False alarm on the new capture, caught by the review band |
| 6 | external `cast_def_0_335` | defective | 0.0201 | normal | no | **Failure mode**: a defect on the new capture scored confidently normal and shipped. 16 of the 67 external misses look like this |

Each of these probabilities matches `reports/test_predictions.csv` (offline evaluation) to the
printed precision, which confirms serving uses exactly the evaluated model and preprocessing.
End-to-end HTTP latency was 13–18 ms per image on a 4-thread CPU.

## Other files

| File | Shows |
| --- | --- |
| `7_batch.json` | `/v1/predict/batch` with 3 images; results keep upload order (`index`) |
| `8_model_info.json` | `/v1/model`: version, threshold, temperature, review band, val and test metrics |
| `9_error_no_api_key.json` | 401, uniform error body |
| `10_error_fake_png.json` | A shell script named `.png` with `image/png` MIME type gets 415: magic bytes win |
| `11_error_docs_disabled.json` | `/docs` returns 404 in production |
| `12_response_headers.txt` | Security headers and `X-Request-ID` (no `server` header) |
