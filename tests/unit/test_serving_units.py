"""Serving building blocks without HTTP: settings, gate, prediction maths, JSON logs."""

import asyncio
import json
import logging
import threading

import numpy as np
import pytest
from pydantic import ValidationError

from defect_detection.core.model_meta import InputSpec, ModelMeta, ReviewBand
from defect_detection.serving.errors import InferenceTimeoutError, OverloadedError
from defect_detection.serving.inference.gate import InferenceGate
from defect_detection.serving.observability import JsonFormatter
from defect_detection.serving.services.prediction import PredictionService, defect_probabilities
from defect_detection.serving.settings import Settings
from tests.fixtures.synthetic import smooth_image


def test_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DD_MODEL_DIR", "models/v1")
    monkeypatch.setenv("DD_MAX_BATCH_FILES", "4")
    settings = Settings()  # type: ignore[call-arg]
    assert settings.max_batch_files == 4
    assert settings.environment == "production"
    assert not settings.docs_enabled


def test_settings_fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DD_MODEL_DIR", raising=False)
    with pytest.raises(ValidationError):
        Settings()  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="min_image_side"):
        Settings(model_dir="m", min_image_side=500, max_image_side=100)  # type: ignore[arg-type]


def test_softmax_with_temperature() -> None:
    logits = np.array([[0.0, 0.0], [0.0, 2.0], [1000.0, -1000.0]], dtype=np.float32)
    probs = defect_probabilities(logits, temperature=2.0)
    assert probs[0] == pytest.approx(0.5)
    assert probs[1] == pytest.approx(1 / (1 + np.exp(-1.0)))
    assert probs[2] == pytest.approx(0.0)  # no overflow


class FixedPredictor:
    def __init__(self, logits: np.ndarray) -> None:
        self.logits = logits.astype(np.float32)

    def predict_logits(self, batch: np.ndarray) -> np.ndarray:
        assert batch.shape[1:] == (3, 32, 32)
        return self.logits[: len(batch)]


def meta(threshold: float = 0.4, band: tuple[float, float] = (0.2, 0.7)) -> ModelMeta:
    return ModelMeta(
        model_version="v",
        backbone="b",
        input=InputSpec(image_size=32, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        temperature=1.0,
        threshold=threshold,
        review_band=ReviewBand(low=band[0], high=band[1]),
        threshold_policy="test",
        val_metrics={},
        git_commit="c",
        config_hash="h",
        created_at="t",
    )


def test_service_decision_confidence_and_review_flag() -> None:
    p = np.array([0.1, 0.3, 0.5, 0.9])
    logits = np.stack([np.zeros(4), np.log(p / (1 - p))], axis=1)
    service = PredictionService(FixedPredictor(logits), meta())
    results = service.predict([smooth_image(s, size=40) for s in range(4)])
    assert [r.predicted_class for r in results] == ["normal", "normal", "defective", "defective"]
    assert [r.needs_review for r in results] == [False, True, True, False]
    assert [round(r.confidence, 4) for r in results] == [0.9, 0.7, 0.5, 0.9]


def test_gate_rejects_when_full_and_times_out() -> None:
    async def scenario() -> None:
        gate = InferenceGate(max_concurrent=1, timeout_s=0.2)
        release = threading.Event()
        slow = asyncio.create_task(gate.run(lambda: release.wait(5)))
        await asyncio.sleep(0.05)
        with pytest.raises(OverloadedError):
            await gate.run(lambda: 1)
        with pytest.raises(InferenceTimeoutError):
            await slow
        assert gate.active == 1  # slot held until the work really finishes
        release.set()
        await asyncio.sleep(0.1)
        assert gate.active == 0
        assert await gate.run(lambda: 42) == 42
        gate.shutdown()

    asyncio.run(scenario())


def test_json_formatter_includes_extra_fields() -> None:
    record = logging.makeLogRecord({"msg": "hello", "levelname": "INFO", "request_id": "r1"})
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "hello"
    assert payload["request_id"] == "r1"
