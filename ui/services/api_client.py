"""Typed HTTP client for the defect-detection API (the UI's only link to the model).

* Every call has a timeout; a timeout or HTTP error becomes an ``ApiError`` with a friendly
  message and the API's ``request_id`` when there is one (never a traceback).
* One retry, and only for connection errors (the request never reached the server), so a
  prediction is never silently sent twice after the server received it.
* Images are sent as the original bytes: the UI never preprocesses for inference.
"""

from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

UPLOAD_NAME = "image"  # client filenames are never forwarded


class _ApiModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, protected_namespaces=())


class Prediction(_ApiModel):
    """Response of POST /v1/predict."""

    predicted_class: Literal["normal", "defective"]
    confidence: float
    defect_probability: float
    threshold: float
    needs_review: bool
    model_version: str
    request_id: str
    latency_ms: float


class BatchItem(_ApiModel):
    """One entry of a batch response."""

    index: int
    predicted_class: Literal["normal", "defective"]
    confidence: float
    defect_probability: float
    needs_review: bool


class BatchResult(_ApiModel):
    """Response of POST /v1/predict/batch."""

    request_id: str
    model_version: str
    latency_ms: float
    results: list[BatchItem]


class ModelInfo(_ApiModel):
    """Response of GET /v1/model."""

    model_version: str
    backbone: str
    temperature: float
    threshold: float
    review_band: tuple[float, float]
    threshold_policy: str
    val_metrics: dict[str, float]
    test_metrics: dict[str, dict[str, float]] | None = None


class ApiError(Exception):
    """A failed call, reduced to what is safe and useful to show a user."""

    def __init__(self, status: int | None, error: str, detail: str, request_id: str | None) -> None:
        super().__init__(detail)
        self.status = status
        self.error = error
        self.detail = detail
        self.request_id = request_id


class ApiClient:
    """Small synchronous client; one instance is cached per Streamlit session."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_s: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"X-API-Key": api_key} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url, headers=headers, timeout=timeout_s, transport=transport
        )

    def _send(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        for attempt in (1, 2):
            try:
                return self._client.request(method, path, **kwargs)  # type: ignore[arg-type]
            except httpx.ConnectError as exc:
                if attempt == 2:
                    raise ApiError(None, "unreachable", "The API is not reachable.", None) from exc
            except httpx.TimeoutException as exc:
                raise ApiError(None, "timeout", "The API did not answer in time.", None) from exc
        raise AssertionError("unreachable")  # pragma: no cover - loop always returns or raises

    @staticmethod
    def _error(response: httpx.Response) -> ApiError:
        try:
            body = response.json()
            return ApiError(
                response.status_code,
                str(body.get("error", "http_error")),
                str(body.get("detail", f"HTTP {response.status_code}")),
                body.get("request_id"),
            )
        except (ValueError, AttributeError):
            return ApiError(
                response.status_code, "http_error", f"HTTP {response.status_code}", None
            )

    def _get_json(self, response: httpx.Response) -> object:
        if not response.is_success:
            raise self._error(response)
        try:
            return response.json()
        except ValueError as exc:
            raise ApiError(
                response.status_code, "bad_response", "Unexpected API response.", None
            ) from exc

    def is_ready(self) -> bool:
        """True when the API reports a verified model loaded."""
        try:
            return self._send("GET", "/ready").status_code == 200
        except ApiError:
            return False

    def model_info(self) -> ModelInfo:
        """GET /v1/model."""
        return self._validate(ModelInfo, self._get_json(self._send("GET", "/v1/model")))

    def predict(self, image: bytes) -> Prediction:
        """POST /v1/predict with the original image bytes."""
        response = self._send("POST", "/v1/predict", files={"file": (UPLOAD_NAME, image)})
        return self._validate(Prediction, self._get_json(response))

    def predict_batch(self, images: list[bytes]) -> BatchResult:
        """POST /v1/predict/batch; results come back in upload order."""
        files = [("files", (f"{UPLOAD_NAME}-{i}", data)) for i, data in enumerate(images)]
        response = self._send("POST", "/v1/predict/batch", files=files)
        return self._validate(BatchResult, self._get_json(response))

    @staticmethod
    def _validate[T: BaseModel](model: type[T], payload: object) -> T:
        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            raise ApiError(None, "bad_response", "Unexpected API response.", None) from exc
