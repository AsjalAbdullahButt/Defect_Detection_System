# API reference

Base URL: `http://127.0.0.1:8000` (local or compose). Interactive docs at `/docs` only when
`DD_ENVIRONMENT=development`.

## Authentication

Send `X-API-Key: <key>` on every `/v1/*` request. Keys come from `DD_API_KEYS`
(comma-separated, ≥ 32 characters each). `/health` and `/ready` are public. A missing or wrong
key gets 401; repeated failures from one IP get 429.

## Endpoints

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/health` | — | `{"status":"ok"}`: the process is up (liveness) |
| GET | `/ready` | — | `{"status":"ready"}`: the verified model is loaded (readiness); 503 otherwise |
| GET | `/v1/model` | — | version, backbone, input spec, temperature, threshold, review band, val/test metrics |
| POST | `/v1/predict` | multipart `file` | one prediction |
| POST | `/v1/predict/batch` | multipart `files` (≤ `DD_MAX_BATCH_FILES`, total ≤ `DD_MAX_BATCH_BYTES`) | predictions in upload order; any invalid file fails the whole batch |

Accepted images: JPEG, PNG, BMP, WebP. The type is decided by the file's bytes, never by the
filename or `Content-Type`.

## Prediction response

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

| Field | Meaning |
| --- | --- |
| `defect_probability` | Calibrated P(defective) |
| `predicted_class` | `defective` if `defect_probability ≥ threshold`, else `normal` |
| `confidence` | Probability of the **predicted** class (can be < 0.5 just above the threshold) |
| `needs_review` | `true` inside the review band: send the part to a person |
| `request_id` | Also in the `X-Request-ID` header and in every log line for this request |

## Errors

Every error has the same body: `{"error": "<code>", "detail": "<text>", "request_id": "<id>"}`.
No stack traces, paths or library versions are ever returned.

| Status | `error` | When |
| --- | --- | --- |
| 400 | `invalid_image` | Not decodable, truncated, too many pixels, animated, too small/large, aspect > 4:1 |
| 400 | `too_many_files` | Batch over `DD_MAX_BATCH_FILES` |
| 401 | `unauthorized` | Missing or wrong API key |
| 404 / 405 | `not_found` / `method_not_allowed` | Unknown path or method |
| 413 | `payload_too_large` | Body over the limit (checked while streaming) |
| 415 | `unsupported_media_type` | Content is not JPEG/PNG/BMP/WebP |
| 422 | `validation_error` | Missing form field (field names only, never values) |
| 429 | `rate_limited` | Over `DD_RATE_LIMIT`; see the `Retry-After` header |
| 500 | `internal_error` | Unexpected; details only in server logs |
| 503 | `overloaded` / `model_not_ready` | All inference slots busy / model not loaded |
| 504 | `inference_timeout` | Over `DD_INFERENCE_TIMEOUT_S` |

## curl

```bash
KEY=<your key>
curl -s http://127.0.0.1:8000/ready
curl -s -H "X-API-Key: $KEY" http://127.0.0.1:8000/v1/model
curl -s -H "X-API-Key: $KEY" -F "file=@path/to/image.jpeg" http://127.0.0.1:8000/v1/predict
curl -s -H "X-API-Key: $KEY" -F "files=@a.jpeg" -F "files=@b.jpeg" http://127.0.0.1:8000/v1/predict/batch
```

PowerShell (7+):

```powershell
$KEY = "<your key>"
curl.exe -s -H "X-API-Key: $KEY" -F "file=@path\to\image.jpeg" http://127.0.0.1:8000/v1/predict
```

Real responses for six images and every error case: `examples/example_predictions.md`.

## Response headers

Every response carries `X-Request-ID`, `X-Content-Type-Options: nosniff`,
`X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store`,
`Content-Security-Policy: default-src 'none'; frame-ancestors 'none'`. There is no `server`
header.
