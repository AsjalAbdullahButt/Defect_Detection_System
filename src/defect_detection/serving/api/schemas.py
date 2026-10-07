"""Request/response models (also the OpenAPI contract when docs are enabled)."""

from typing import Literal

from pydantic import BaseModel, Field

ClassName = Literal["normal", "defective"]


class PredictionFields(BaseModel):
    """Outcome for one image."""

    predicted_class: ClassName
    confidence: float = Field(ge=0, le=1, description="Probability of the predicted class.")
    defect_probability: float = Field(ge=0, le=1, description="Calibrated P(defective).")
    threshold: float = Field(description="Decision threshold on defect_probability.")
    needs_review: bool = Field(description="True inside the uncertainty band: route to a person.")
    model_version: str


class PredictionResponse(PredictionFields):
    """Response of POST /v1/predict."""

    request_id: str
    latency_ms: float


class BatchItem(PredictionFields):
    """One entry of a batch response, in upload order."""

    index: int


class BatchPredictionResponse(BaseModel):
    """Response of POST /v1/predict/batch."""

    request_id: str
    model_version: str
    latency_ms: float
    results: list[BatchItem]


class InputInfo(BaseModel):
    """What the model expects."""

    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


class ModelInfoResponse(BaseModel):
    """Response of GET /v1/model (no file paths or internals)."""

    model_version: str
    backbone: str
    class_names: tuple[str, str]
    positive_class: str
    input: InputInfo
    temperature: float
    threshold: float
    review_band: tuple[float, float]
    threshold_policy: str
    val_metrics: dict[str, float]
    test_metrics: dict[str, dict[str, float]] | None


class HealthResponse(BaseModel):
    """Response of GET /health and GET /ready."""

    status: Literal["ok", "ready"]


class ErrorResponse(BaseModel):
    """Body of every error response."""

    error: str
    detail: str
    request_id: str
