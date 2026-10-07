"""HTTP layer only: parse the request, call the service, shape the response.

Validation and inference run inside the InferenceGate thread pool, so CPU-heavy decoding never
blocks the event loop and every request is bounded in concurrency and time.
"""

import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile

from defect_detection.serving.api.schemas import (
    BatchItem,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    InputInfo,
    ModelInfoResponse,
    PredictionResponse,
)
from defect_detection.serving.errors import ModelNotReadyError, TooManyFilesError, request_id_of
from defect_detection.serving.inference.gate import InferenceGate
from defect_detection.serving.observability import DEFECT_PROBABILITY, INFERENCE, PREDICTIONS
from defect_detection.serving.services.prediction import Prediction, PredictionService
from defect_detection.serving.settings import Settings
from defect_detection.serving.validation import decode_image

ERRORS_DOC: dict[int | str, dict[str, object]] = {
    code: {"model": ErrorResponse} for code in (400, 413, 415, 422, 500, 503, 504)
}

health_router = APIRouter(tags=["health"])
api_router = APIRouter(prefix="/v1", tags=["inference"], responses=ERRORS_DOC)


def get_settings(request: Request) -> Settings:
    """Settings the app was built with."""
    settings: Settings = request.app.state.settings
    return settings


def get_service(request: Request) -> PredictionService:
    """The loaded prediction service, or 503 if the model is not ready."""
    service: PredictionService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise ModelNotReadyError("The model is not loaded.")
    return service


def get_gate(request: Request) -> InferenceGate:
    """Shared bounded executor for inference work."""
    gate: InferenceGate = request.app.state.gate
    return gate


ServiceDep = Annotated[PredictionService, Depends(get_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
GateDep = Annotated[InferenceGate, Depends(get_gate)]


def _record(predictions: list[Prediction]) -> None:
    for p in predictions:
        PREDICTIONS.labels(
            model_version=p.model_version,
            predicted_class=p.predicted_class,
            needs_review=str(p.needs_review).lower(),
        ).inc()
        DEFECT_PROBABILITY.observe(p.defect_probability)


async def _infer(
    payloads: list[bytes], service: PredictionService, settings: Settings, gate: InferenceGate
) -> list[Prediction]:
    def work() -> list[Prediction]:
        start = time.perf_counter()
        images = [decode_image(data, settings) for data in payloads]
        predictions = service.predict(images)
        INFERENCE.observe(time.perf_counter() - start)
        return predictions

    predictions = await gate.run(work)
    _record(predictions)
    return predictions


@health_router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness: the process is up (no dependencies checked)."""
    return HealthResponse(status="ok")


@health_router.get(
    "/ready", response_model=HealthResponse, responses={503: {"model": ErrorResponse}}
)
async def ready(service: ServiceDep) -> HealthResponse:
    """Readiness: the verified model is loaded and can take traffic."""
    return HealthResponse(status="ready")


@api_router.get("/model", response_model=ModelInfoResponse)
async def model_info(service: ServiceDep) -> ModelInfoResponse:
    """Version, operating point and validation/test metrics of the loaded model."""
    meta = service.meta
    return ModelInfoResponse(
        model_version=meta.model_version,
        backbone=meta.backbone,
        class_names=meta.class_names,
        positive_class=meta.positive_class,
        input=InputInfo(image_size=meta.input.image_size, mean=meta.input.mean, std=meta.input.std),
        temperature=meta.temperature,
        threshold=meta.threshold,
        review_band=(meta.review_band.low, meta.review_band.high),
        threshold_policy=meta.threshold_policy,
        val_metrics=meta.val_metrics,
        test_metrics=meta.test_metrics,
    )


@api_router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    file: Annotated[UploadFile, File(description="JPEG, PNG, BMP or WebP image.")],
    service: ServiceDep,
    settings: SettingsDep,
    gate: GateDep,
) -> PredictionResponse:
    """Classify one image as normal or defective."""
    start = time.perf_counter()
    (prediction,) = await _infer([await file.read()], service, settings, gate)
    return PredictionResponse(
        request_id=request_id_of(request),
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
        **prediction.__dict__,
    )


@api_router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: Request,
    files: Annotated[list[UploadFile], File(description="Up to DD_MAX_BATCH_FILES images.")],
    service: ServiceDep,
    settings: SettingsDep,
    gate: GateDep,
) -> BatchPredictionResponse:
    """Classify several images in one model call; any invalid file fails the whole request."""
    start = time.perf_counter()
    if len(files) > settings.max_batch_files:
        raise TooManyFilesError(f"At most {settings.max_batch_files} files per batch.")
    payloads = [await f.read() for f in files]
    predictions = await _infer(payloads, service, settings, gate)
    return BatchPredictionResponse(
        request_id=request_id_of(request),
        model_version=service.meta.model_version,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
        results=[BatchItem(index=i, **p.__dict__) for i, p in enumerate(predictions)],
    )
