"""Application factory and lifespan.

``create_app(settings)`` builds a fully configured FastAPI app; nothing happens at import time.
On startup the lifespan verifies and loads the model ONCE (refusing to start if SHA256SUMS or
the metadata do not check out), creates the bounded inference pool and starts the internal
metrics server. On shutdown it lets in-flight inference finish.

Run: ``uvicorn --factory defect_detection.serving.main:create_app`` with DD_* variables set.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import start_http_server

from defect_detection import __version__
from defect_detection.serving.api.routes import api_router, health_router
from defect_detection.serving.errors import register_error_handlers
from defect_detection.serving.inference.engine import OnnxPredictor, load_verified_meta
from defect_detection.serving.inference.gate import InferenceGate
from defect_detection.serving.middleware import (
    AccessLogMiddleware,
    BodySizeLimitMiddleware,
    RequestIdMiddleware,
)
from defect_detection.serving.observability import MODEL_INFO, configure_logging
from defect_detection.serving.services.prediction import PredictionService
from defect_detection.serving.settings import Settings
from defect_detection.serving.validation import configure_pillow

log = logging.getLogger("defect_detection.serving")
MULTIPART_OVERHEAD = 64 * 1024  # boundaries and part headers around the file bytes


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app; reads DD_* environment variables when ``settings`` is not given."""
    settings = settings if settings is not None else Settings()  # type: ignore[call-arg]
    configure_logging(settings.log_level)
    configure_pillow(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        meta = load_verified_meta(settings.model_dir)  # raises -> the server does not start
        predictor = OnnxPredictor(settings.model_dir, meta, settings.inference_threads)
        app.state.service = PredictionService(predictor, meta)
        app.state.gate = InferenceGate(
            settings.max_concurrent_inferences, settings.inference_timeout_s
        )
        MODEL_INFO.labels(model_version=meta.model_version, backbone=meta.backbone).set(1)
        metrics_server = None
        if settings.metrics_port is not None:
            metrics_server, _ = start_http_server(settings.metrics_port, addr=settings.metrics_host)
        log.info("model loaded", extra={"model_version": meta.model_version})
        try:
            yield
        finally:
            app.state.service = None
            app.state.gate.shutdown()
            if metrics_server is not None:
                metrics_server.shutdown()
            log.info("shutdown complete")

    docs = settings.docs_enabled
    app = FastAPI(
        title="Visual Defect Detection API",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs else None,
    )
    app.state.settings = settings
    app.state.service = None
    register_error_handlers(app)
    app.include_router(health_router)
    app.include_router(api_router)
    # add_middleware wraps: the LAST added is the OUTERMOST.
    app.add_middleware(
        BodySizeLimitMiddleware,
        default_limit=settings.max_upload_bytes + MULTIPART_OVERHEAD,
        limits={"/v1/predict/batch": settings.max_batch_bytes + MULTIPART_OVERHEAD},
    )
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestIdMiddleware)
    return app
