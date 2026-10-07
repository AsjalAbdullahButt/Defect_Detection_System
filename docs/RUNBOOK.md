# Runbook — defect-detection API

## 1. Deploy (Docker Compose)

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # paste into DD_API_KEYS
# set MODEL_VERSION to a folder under ./models (an exported, checksummed version)
docker compose up -d --build
docker compose ps                     # STATUS must become "healthy"
```

Prerequisites: the model folder `models/<MODEL_VERSION>/` with `model.onnx`,
`model_meta.json` and `SHA256SUMS` (produced by `make export`), and a TLS reverse proxy in
front of `127.0.0.1:8000` for anything beyond localhost.

Smoke test:

```bash
curl -s http://127.0.0.1:8000/health                  # {"status":"ok"}
curl -s http://127.0.0.1:8000/ready                   # {"status":"ready"}
curl -s -H "X-API-Key: $KEY" http://127.0.0.1:8000/v1/model
curl -s -H "X-API-Key: $KEY" -F "file=@examples/images/<image>.jpeg" \
     http://127.0.0.1:8000/v1/predict
```

## 2. Health checks

| Endpoint | Meaning | Use as |
| --- | --- | --- |
| `GET /health` | Process is up (no dependencies checked) | liveness probe |
| `GET /ready` | Verified model loaded, accepting traffic | readiness probe |
| Docker `HEALTHCHECK` | `/health` every 30 s, 3 retries | restart policy |

If the container never becomes healthy: `docker compose logs api`. The most common causes:

| Log message | Cause | Fix |
| --- | --- | --- |
| `model directory failed verification: [... checksum mismatch]` | Model files changed after export, or a partial copy | Re-copy the folder; never edit files inside a version |
| `SHA256SUMS missing` / `has no 'onnx' section` | Not an exported version (e.g. a `runs/` folder) | Point `MODEL_VERSION` at `models/<version>` from `make export` |
| `production requires DD_API_KEYS` | No keys in production | Set `DD_API_KEYS` (≥ 32 chars each) |
| `every API key must be at least 32 characters` | Weak key | Generate with `secrets.token_urlsafe(32)` |

## 3. Roll back to the previous model

Model versions are immutable folders, so rollback is a configuration change:

```bash
ls models/                                   # pick the previous version
sed -i 's/^MODEL_VERSION=.*/MODEL_VERSION=<previous>/' .env
docker compose up -d                         # recreates the container, verifies checksums
curl -s -H "X-API-Key: $KEY" http://127.0.0.1:8000/v1/model | grep model_version
```

## 4. Ship a new model

1. `make data train calibrate`: train and choose the operating point on validation.
2. `make evaluate`: one-shot test evaluation. Review `reports/metrics.json` and **especially
   the external test set** and the similarity buckets before going further.
3. `make export`: writes `models/<new version>/` only if `onnx.checker` and the PyTorch/ONNX
   parity check pass.
4. `make benchmark`: latency must still fit the line's cycle time.
5. Copy the folder to the server, set `MODEL_VERSION`, `docker compose up -d`, then run the smoke
   test. Keep the previous folder for rollback.

## 5. Rotate API keys

1. Add the new key: `DD_API_KEYS=<old>,<new>`, then `docker compose up -d`.
2. Move clients to the new key.
3. Remove the old key from `DD_API_KEYS`, then `docker compose up -d`.

## 6. Reading logs

Every line is JSON on stdout (`docker compose logs api`). Access lines look like:

```json
{"ts":"...","level":"INFO","logger":"defect_detection.serving.access","message":"request",
 "request_id":"9f13...","method":"POST","route":"/v1/predict","status":200,"duration_ms":38.2}
```

- Clients get the same `request_id` in the `X-Request-ID` header and in every error body, so a
  support ticket that quotes it maps to exactly one log line.
- `level: ERROR` with `message: unhandled error` comes with the stack trace (server-side only).
- Image bytes, filenames and API keys are never logged.

## 7. Reading metrics (Prometheus, internal port 9100)

| Metric | Watch for |
| --- | --- |
| `dd_http_requests_total{route,status}` | Rising `401`/`429` (bad clients or brute force), `503` (capacity), `504` (slow inference) |
| `dd_http_request_duration_seconds` | p95 above the line's cycle-time budget |
| `dd_inference_duration_seconds` | CPU contention (threads × concurrency > cores) |
| `dd_predictions_total{predicted_class,needs_review}` | **Review rate**: ~0.6% on in-distribution data; a sustained rise (≈ 20% on a new capture) means drift; recalibrate or retrain |
| `dd_defect_probability` (histogram) | Shift of the score distribution: early drift warning before labels exist |
| `dd_errors_total{error}` | `invalid_image` / `unsupported_media_type` spikes = a client sending the wrong files |
| `dd_model_info{model_version}` | Which version is live (confirm after deploy/rollback) |

## 8. Capacity and tuning

- One Uvicorn worker per container; scale with replicas.
- Keep `DD_INFERENCE_THREADS × DD_MAX_CONCURRENT_INFERENCES ≤` the container's CPU limit
  (`cpus` in compose). Oversubscribing makes every request slower.
- Rate limits are per process: with N replicas behind a load balancer, the effective limit per
  key is N × `DD_RATE_LIMIT`.
- Behind a reverse proxy, run Uvicorn with `--proxy-headers --forwarded-allow-ips=<proxy IP>`
  so per-IP limits see real client addresses.
