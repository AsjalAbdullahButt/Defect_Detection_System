# Optional reproducible training image (CPU). Data and outputs are mounted, never baked in.
# Build:  docker build -f docker/train.Dockerfile -t defect-detection-train .
# Run:    docker run --rm -v "$PWD/data:/work/data" -v "$PWD/runs:/work/runs" \
#                    -v "$PWD/models:/work/models" -v "$PWD/reports:/work/reports" \
#                    defect-detection-train train
# Note: requirements/train.txt is a universal lock; on Linux it resolves the CUDA build of torch,
# which makes this image large. Swap to the CPU wheel index if image size matters.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
FROM ${PYTHON_IMAGE}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONPATH=/work/src
WORKDIR /work
COPY requirements/ requirements/
RUN pip install --require-hashes --no-deps --only-binary=:all: \
        -r requirements/train.txt -r requirements/serve.txt \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home app
COPY configs/ configs/
COPY src/ src/
USER 10001:10001
ENTRYPOINT ["python", "-m", "defect_detection"]
CMD ["--help"]
