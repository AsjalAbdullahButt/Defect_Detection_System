# Production inference image: ONNX Runtime + FastAPI only (no torch, no training code).
# Build from the repo root:  docker build -f docker/serve.Dockerfile -t defect-detection-api .
#
# - base image pinned by digest (python:3.12-slim-bookworm, linux/amd64 + arm64 index)
# - every dependency installed from requirements/serve.txt with --require-hashes, wheels only
# - runs as a non-root user; works on a read-only root filesystem (no .pyc writes)
# - the model is NOT baked in: mount models/<version> read-only and set DD_MODEL_DIR

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

# ---------------------------------------------------------------- builder
FROM ${PYTHON_IMAGE} AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
COPY requirements/serve.txt /tmp/serve.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --require-hashes --no-deps --only-binary=:all: -r /tmp/serve.txt

# ---------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="defect-detection-api" \
      org.opencontainers.image.description="Visual defect detection inference API (ONNX Runtime)" \
      org.opencontainers.image.licenses="MIT"

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DD_METRICS_HOST=0.0.0.0 \
    DD_METRICS_PORT=9100

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv
# Only the code serving needs: the package root, core/ and serving/ (no training/, no data/).
COPY src/defect_detection/__init__.py /app/src/defect_detection/__init__.py
COPY src/defect_detection/core /app/src/defect_detection/core
COPY src/defect_detection/serving /app/src/defect_detection/serving

USER 10001:10001
WORKDIR /app
EXPOSE 8000 9100

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

# One worker per container: scale with replicas, and keep threads x concurrency <= CPU limit.
CMD ["uvicorn", "--factory", "defect_detection.serving.main:create_app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1", \
     "--no-server-header", "--timeout-graceful-shutdown", "20"]
