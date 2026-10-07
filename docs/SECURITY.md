# Security

Scope: the inference API (`src/defect_detection/serving`), the model artifact it loads, and the
build/supply chain. Training runs offline on trusted data and is out of scope except where it
produces the artifact.

## Assets and trust boundaries

| Asset | Why it matters |
| --- | --- |
| Model artifact (`model.onnx`, `model_meta.json`) | A swapped or altered model silently changes every decision (threshold, temperature included). |
| API keys | Grant use of the API and consume its CPU budget. |
| Service availability | A line-side inspection service that is down or slow stops the line or lets parts through unchecked. |
| Uploaded images | Customer production data: not stored, not logged. |
| Logs and metrics | Reveal traffic patterns and model behaviour. |

Boundaries: **client → API** (untrusted bytes, untrusted headers), **API → model directory**
(trusted only after checksum verification), **build → dependencies** (pinned with hashes).

## Threat model (STRIDE-lite)

| # | STRIDE | Threat | Mitigation | Verified by |
| --- | --- | --- | --- | --- |
| 1 | Spoofing | Anonymous use of `/v1/*` | `X-API-Key` required whenever keys are configured; production refuses to start without keys unless `DD_ALLOW_UNAUTHENTICATED=true` is set explicitly | `test_missing_or_invalid_key_is_401`, `test_production_without_keys_refuses_to_start` |
| 2 | Spoofing | Guessing a key (timing or brute force) | `hmac.compare_digest` against **every** key (no early exit); keys ≥ 32 chars; identical 401 for missing/wrong keys; failed attempts limited per IP (default 10/min) → 429 | `test_failed_auth_attempts_are_throttled`, `test_short_keys_and_bad_limits_rejected` |
| 3 | Tampering | Modified or swapped model files | `SHA256SUMS` verified at startup; schema validation of `model_meta.json`; graph input checked against metadata; the server refuses to start on any mismatch | `test_startup_refuses_tampered_model`, `test_startup_refuses_missing_checksums`, `test_sha256sums_*` |
| 4 | Tampering | Path traversal via a crafted `SHA256SUMS` or filenames | Manifest lines with `/`, `\` or a leading `.` are rejected; client filenames are never used | `test_sha256sums_rejects_malformed_or_unsafe_lines` |
| 5 | Tampering / Elevation | Malicious upload: fake extension, script disguised as an image, polyglot | Format from magic bytes only (JPEG/PNG/BMP/WebP), Pillow restricted to that format, `verify()` then full decode, bytes stay in memory | `test_magic_bytes_win_over_extension_and_mime`, `test_header_only_polyglot_is_rejected`, `test_polyglot_png_*` |
| 6 | Tampering / Elevation | Code execution through model loading | Serving loads ONNX only (no pickle anywhere in serving); training checkpoints load with `torch.load(weights_only=True)` | `test_checkpoint_loads_weights_only_*`; ruff bans torch in serving |
| 7 | Repudiation | No trace of who did what | Every request gets a request id (returned in the header and every error body) and one JSON access-log line: id, method, route, status, duration; caller identified by a key fingerprint, never the key | `test_logs_are_json_and_never_contain_filenames`, `test_request_id_kept_only_when_safe` |
| 8 | Information disclosure | Stack traces, paths or versions in responses | Uniform `{error, detail, request_id}` with fixed details; generic 500; 422 lists field names only, never values; OpenAPI docs off in production | `test_error_bodies_contain_no_stack_traces`, `test_error_bodies_never_leak_internals`, `test_missing_file_gives_422_without_echo`, `test_docs_disabled_*` |
| 9 | Information disclosure | Sensitive data in logs | No image bytes, no client filenames, no API keys (only a SHA-256 fingerprint) in logs; `SecretStr` keeps keys out of reprs | `test_logs_are_json_and_never_contain_filenames`, `test_short_keys_and_bad_limits_rejected` |
| 10 | Information disclosure | Metrics exposure | Prometheus served on a separate port bound to 127.0.0.1 by default and never published (V8 compose) | design; see `serve.env.example` |
| 11 | Information disclosure | Log injection via `X-Request-ID` | Client ids kept only if they match `[A-Za-z0-9._-]{1,64}`, otherwise replaced; logs are JSON-encoded | `test_request_id_kept_only_when_safe` |
| 12 | Denial of service | Huge bodies | Size limit enforced while streaming (declared `Content-Length` and chunked) → 413; batch has its own count and byte limits | `test_oversize_declared_body_is_413_before_reading`, `test_oversize_streamed_body_without_length_is_413`, `test_batch_file_limit` |
| 13 | Denial of service | Decompression bombs | Header pixel count checked against `DD_MAX_IMAGE_PIXELS` before decoding; Pillow's bomb warning escalated to an error; min/max side and aspect-ratio limits | `test_decompression_bomb_header_*`, `test_pixel_count_between_1x_and_2x_cap_*`, `test_dimension_limits` |
| 14 | Denial of service | Request floods, slow inference | Per-caller rate limit → 429 + `Retry-After`; bounded thread pool, fail fast with 503 when full; per-request timeout → 504 (slot held until work really ends) | `test_rate_limit_per_caller`, `test_gate_rejects_when_full_and_times_out` |
| 15 | Denial of service | Unauthenticated body parsing | Auth and rate limit run as ASGI middleware **before** the body is read (FastAPI parses multipart before route dependencies) | `test_auth_rejects_before_reading_the_body` |
| 16 | Denial of service | Animated / multi-frame images | Rejected (multi-frame) or not accepted at all (GIF) | `test_animated_image_is_rejected`, `test_animated_gif_is_rejected_by_media_type` |
| 17 | Elevation | Browser-based abuse (clickjacking, MIME sniffing, cross-origin calls) | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `CSP: default-src 'none'`, `Referrer-Policy`, `Cache-Control: no-store`; CORS off unless origins are allow-listed, no credentials | `test_security_headers_on_success_and_error`, `test_cors_*` |
| 18 | Tampering (supply chain) | Malicious or vulnerable dependency | Lockfiles with SHA-256 hashes, `--require-hashes` installs, pip-audit in CI, serving image without torch/training packages (smaller attack surface); GitHub Actions pinned to commit SHAs | `pip-audit` (below), `make setup` |
| 19 | Information disclosure | Secrets committed to git | gitleaks in pre-commit and CI; `.gitignore` covers `.env`, keys and credentials; `.dockerignore` is an allow-list | pre-commit run (V0) |

## Scanner results (V7)

| Tool | Command | Result |
| --- | --- | --- |
| bandit 1.9.4 | `bandit -r src` | **0 issues.** 3 low-severity findings explicitly skipped with `# nosec` and a reason (below). |
| pip-audit | `pip-audit -r requirements/{serve,train,dev}.txt --require-hashes --disable-pip` | **No known vulnerabilities** in any lockfile. |
| ruff `S` rules | `ruff check .` | Clean. |
| Tests | `pytest tests/security tests/integration tests/unit` | 171 passed; coverage of `core/` + `serving/` **97%**. |

Skipped bandit findings (all in `provenance.py`, a training-side helper that records the git
commit in artifacts; never reachable from the API):

- **B404** (`import subprocess`) and **B603 × 2** (`subprocess.run` without a shell): the argument
  lists are fixed (`git rev-parse HEAD`, `git status --porcelain`), the executable path comes
  from `shutil.which`, and no user input is involved.

## Residual risks (accepted, documented)

- **Rate limits are per worker process** (in-memory storage). With N Uvicorn workers the effective
  limit is N × the configured one. A shared store (Redis) is the fix if a global limit is required.
- **Client IP behind a proxy.** Without `--proxy-headers` and a trusted proxy list, every caller
  behind a load balancer shares one IP for the unauthenticated limits. Configure the proxy
  explicitly in deployment (RUNBOOK).
- **API keys are static shared secrets.** Rotation is supported (several keys at once), but there
  is no per-client scoping or expiry; an identity provider (OAuth2/mTLS) is the upgrade path.
- **TLS is terminated outside the app** (ingress / reverse proxy). The API must not be exposed over
  plain HTTP beyond a trusted network.
- **Adversarial images** (inputs crafted to flip the classifier) are not defended against; the
  review band and human review are the operational safeguard.
- **Uvicorn's `server` header** is removed at the container level (V8: `--no-server-header`).
