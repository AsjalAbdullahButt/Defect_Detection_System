"""Pydantic schema for model_meta.json: everything serving needs besides the weights.

V3 writes a *candidate* (calibration + operating point from validation); V5 completes it
with the ONNX graph description and the V4 test-metrics summary, and writes SHA256SUMS.
Serving (V6) refuses to start if the file fails validation.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from defect_detection.core.constants import CLASS_NAMES, POSITIVE_CLASS


class InputSpec(BaseModel):
    """How to turn an image into the model's input tensor (mirrors core.preprocessing)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    image_size: int = Field(ge=32, le=1024)
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    layout: Literal["NCHW"] = "NCHW"
    resize: Literal["direct_bilinear"] = "direct_bilinear"


class ReviewBand(BaseModel):
    """P(defective) interval routed to a human: [low, high)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    low: float = Field(ge=0, le=1)
    high: float = Field(ge=0, le=1)


class OnnxSpec(BaseModel):
    """How to run the exported graph. Added by the export step (V5)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str = "model.onnx"
    opset: int = Field(ge=13)
    input_name: str
    output_name: str
    output: Literal["logits"] = "logits"  # P(defective) = softmax(logits / temperature)[:, 1]
    dynamic_batch: bool = True


class ModelMeta(BaseModel):
    """Contents of model_meta.json."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    schema_version: Literal[1] = 1
    model_version: str = Field(min_length=1)
    backbone: str
    input: InputSpec
    class_names: tuple[str, str] = CLASS_NAMES
    positive_class: str = POSITIVE_CLASS
    temperature: float = Field(gt=0)
    threshold: float = Field(gt=0, lt=1)
    review_band: ReviewBand
    threshold_policy: str
    val_metrics: dict[str, float]
    git_commit: str
    config_hash: str
    created_at: str
    onnx: OnnxSpec | None = None  # None in the V3 candidate; required by serving
    test_metrics: dict[str, dict[str, float]] | None = None  # one-shot V4 results, per test set

    @model_validator(mode="after")
    def _consistent(self) -> "ModelMeta":
        if self.class_names != CLASS_NAMES or self.positive_class != POSITIVE_CLASS:
            raise ValueError(f"class names must be {CLASS_NAMES} with positive {POSITIVE_CLASS!r}")
        if not self.review_band.low <= self.threshold <= self.review_band.high:
            raise ValueError("threshold must lie inside the review band")
        return self
