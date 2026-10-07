"""Typed API errors and handlers that always return ``{error, detail, request_id}``.

Clients never see stack traces, file paths, library versions or their own input echoed back:
known errors carry a fixed, human-written ``detail``; anything unexpected becomes a generic
500 whose cause is logged server-side only.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from defect_detection.serving.observability import ERRORS

log = logging.getLogger("defect_detection.serving")


class ApiError(Exception):
    """Base class: an error that maps to a specific HTTP status and stable error code."""

    status_code = 500
    error = "internal_error"

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class InvalidImageError(ApiError):
    """The upload is not a decodable, acceptable image."""

    status_code = 400
    error = "invalid_image"


class TooManyFilesError(ApiError):
    """Batch request with more files than allowed."""

    status_code = 400
    error = "too_many_files"


class PayloadTooLargeError(ApiError):
    """Request body over the configured limit."""

    status_code = 413
    error = "payload_too_large"


class UnsupportedMediaTypeError(ApiError):
    """File content is not JPEG, PNG, BMP or WebP (checked by magic bytes)."""

    status_code = 415
    error = "unsupported_media_type"


class OverloadedError(ApiError):
    """All inference slots are busy; the client should retry later."""

    status_code = 503
    error = "overloaded"


class ModelNotReadyError(ApiError):
    """The model is not loaded (startup failed or still in progress)."""

    status_code = 503
    error = "model_not_ready"


class InferenceTimeoutError(ApiError):
    """Inference did not finish within the configured time."""

    status_code = 504
    error = "inference_timeout"


def request_id_of(request: Request) -> str:
    """The id assigned by RequestIdMiddleware (or 'unknown' if it did not run)."""
    return str(getattr(request.state, "request_id", "unknown"))


def error_response(request: Request, status: int, error: str, detail: str) -> JSONResponse:
    """The single error body shape used by every handler."""
    ERRORS.labels(error=error).inc()
    return JSONResponse(
        status_code=status,
        content={"error": error, "detail": detail, "request_id": request_id_of(request)},
    )


async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
    log.error(
        "unhandled error",
        exc_info=exc,
        extra={"request_id": request_id_of(request), "path": request.url.path},
    )
    return error_response(request, 500, "internal_error", "An internal error occurred.")


async def _api_error(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, ApiError):
        return await _unexpected(request, exc)
    return error_response(request, exc.status_code, exc.error, exc.detail)


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, StarletteHTTPException):
        return await _unexpected(request, exc)
    names = {404: "not_found", 405: "method_not_allowed", 413: "payload_too_large"}
    error = names.get(exc.status_code, "http_error")
    # Our own 401/403/413/429 details are fixed strings; anything else gets a generic text.
    known = exc.status_code in (401, 403, 413, 429)
    detail = str(exc.detail) if known else error.replace("_", " ")
    return error_response(request, exc.status_code, error, detail)


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):
        return await _unexpected(request, exc)
    # Only field locations are returned, never the submitted values.
    fields = sorted({".".join(str(p) for p in e.get("loc", ())) for e in exc.errors()})
    detail = f"invalid or missing: {', '.join(fields)}"
    return error_response(request, 422, "validation_error", detail)


def register_error_handlers(app: FastAPI) -> None:
    """Attach all handlers; order matters only in that subclasses are matched first."""
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unexpected)
