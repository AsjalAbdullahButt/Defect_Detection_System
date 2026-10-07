"""End-to-end API tests against a real exported ONNX model."""

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from prometheus_client import generate_latest

from defect_detection.core.preprocessing import preprocess
from defect_detection.serving.inference.engine import ModelIntegrityError
from defect_detection.serving.main import create_app
from defect_detection.training.calibration import defect_probability
from defect_detection.training.trainer import load_trained_model, resolve_device
from tests.fixtures.serving import ExportedModel, make_settings
from tests.fixtures.synthetic import smooth_image


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_and_ready(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}


def test_model_info_has_no_paths(client: TestClient, exported_model: ExportedModel) -> None:
    body = client.get("/v1/model").json()
    assert body["model_version"] == exported_model.model_dir.name
    assert body["class_names"] == ["normal", "defective"]
    assert body["review_band"][0] <= body["threshold"] <= body["review_band"][1]
    assert str(exported_model.model_dir.parent) not in json.dumps(body)


def test_predict_matches_training_pipeline(
    client: TestClient, exported_model: ExportedModel
) -> None:
    """Train/serve skew end to end: API probability == PyTorch checkpoint + calibration."""
    img = smooth_image(7, size=120)
    response = client.post("/v1/predict", files={"file": ("x.png", png_bytes(img), "image/png")})
    assert response.status_code == 200, response.text
    body = response.json()

    trained = load_trained_model(exported_model.run_dir, resolve_device("cpu"))
    meta = client.get("/v1/model").json()
    with torch.no_grad():
        logits = trained.model(torch.from_numpy(preprocess(img, trained.spec))[None]).numpy()
    expected = float(defect_probability(logits.astype(np.float64), meta["temperature"])[0])
    assert body["defect_probability"] == pytest.approx(expected, abs=1e-4)
    defective = expected >= meta["threshold"]
    assert body["predicted_class"] == ("defective" if defective else "normal")
    # Confidence = probability of the PREDICTED class. With a threshold below 0.5 it can be
    # < 0.5 for images just above the threshold (those are inside the review band).
    assert body["confidence"] == pytest.approx(expected if defective else 1 - expected, abs=1e-4)
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert body["latency_ms"] > 0


def test_batch_preserves_order(client: TestClient) -> None:
    images = [png_bytes(smooth_image(s, size=80)) for s in (1, 2, 3)]
    files = [("files", (f"{i}.png", data, "image/png")) for i, data in enumerate(images)]
    batch = client.post("/v1/predict/batch", files=files).json()
    assert [r["index"] for r in batch["results"]] == [0, 1, 2]
    for i, data in enumerate(images):
        single = client.post("/v1/predict", files={"file": ("x.png", data, "image/png")}).json()
        assert batch["results"][i]["defect_probability"] == pytest.approx(
            single["defect_probability"], abs=1e-5
        )


def test_batch_file_limit(exported_model: ExportedModel) -> None:
    settings = make_settings(exported_model.model_dir, max_batch_files=2)
    with TestClient(create_app(settings)) as client:
        data = png_bytes(smooth_image(1, size=64))
        files = [("files", (f"{i}.png", data, "image/png")) for i in range(3)]
        response = client.post("/v1/predict/batch", files=files)
    assert response.status_code == 400
    assert response.json()["error"] == "too_many_files"


def test_request_id_kept_only_when_safe(client: TestClient) -> None:
    assert (
        client.get("/health", headers={"X-Request-ID": "abc-123"}).headers["X-Request-ID"]
        == "abc-123"
    )
    injected = client.get("/health", headers={"X-Request-ID": 'x"\n{"level":"ERROR"}'})
    assert injected.headers["X-Request-ID"] != 'x"\n{"level":"ERROR"}'
    assert len(injected.headers["X-Request-ID"]) == 32


@pytest.mark.parametrize(
    ("method", "path", "status", "error"),
    [("get", "/nope", 404, "not_found"), ("delete", "/v1/predict", 405, "method_not_allowed")],
)
def test_uniform_error_body(
    client: TestClient, method: str, path: str, status: int, error: str
) -> None:
    response = getattr(client, method)(path)
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error", "detail", "request_id"}
    assert body["error"] == error
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_missing_file_gives_422_without_echo(client: TestClient) -> None:
    response = client.post("/v1/predict", data={"other": "SECRET-VALUE"})
    assert response.status_code == 422
    assert "SECRET-VALUE" not in response.text
    assert response.json()["error"] == "validation_error"


def test_docs_disabled_in_production(exported_model: ExportedModel) -> None:
    prod = make_settings(exported_model.model_dir, environment="production")
    with TestClient(create_app(prod)) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404
    dev = make_settings(exported_model.model_dir)
    with TestClient(create_app(dev)) as client:
        assert client.get("/openapi.json").status_code == 200


def _copy_model(exported_model: ExportedModel, tmp_path: Path) -> Path:
    target = tmp_path / exported_model.model_dir.name
    shutil.copytree(exported_model.model_dir, target)
    return target


def test_startup_refuses_tampered_model(exported_model: ExportedModel, tmp_path: Path) -> None:
    model_dir = _copy_model(exported_model, tmp_path)
    onnx_file = model_dir / "model.onnx"
    data = bytearray(onnx_file.read_bytes())
    data[-1] ^= 0xFF
    onnx_file.write_bytes(bytes(data))
    with (
        pytest.raises(ModelIntegrityError, match="checksum mismatch"),
        TestClient(create_app(make_settings(model_dir))),
    ):
        pass


def test_startup_refuses_missing_checksums(exported_model: ExportedModel, tmp_path: Path) -> None:
    model_dir = _copy_model(exported_model, tmp_path)
    (model_dir / "SHA256SUMS").unlink()
    with (
        pytest.raises(ModelIntegrityError, match="SHA256SUMS missing"),
        TestClient(create_app(make_settings(model_dir))),
    ):
        pass


def test_metrics_record_predictions(client: TestClient) -> None:
    client.post("/v1/predict", files={"file": ("x.png", png_bytes(smooth_image(3)), "image/png")})
    exposition = generate_latest().decode()
    assert "dd_predictions_total{" in exposition
    assert 'dd_http_requests_total{method="POST",route="/v1/predict",status="200"}' in exposition


def test_logs_are_json_and_never_contain_filenames(
    exported_model: ExportedModel, capsys: pytest.CaptureFixture[str]
) -> None:
    # The app is built here, after capsys swapped sys.stdout: the JSON handler binds stdout
    # when logging is configured.
    with TestClient(create_app(make_settings(exported_model.model_dir))) as client:
        client.post(
            "/v1/predict",
            files={"file": ("customer-secret-name.png", png_bytes(smooth_image(4)), "image/png")},
        )
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    access = [r for r in records if r.get("message") == "request"]
    assert access
    assert access[-1]["route"] == "/v1/predict"
    assert access[-1]["status"] == 200
    assert "customer-secret-name" not in "\n".join(lines)


def test_serving_never_imports_torch_or_training_stack() -> None:
    code = (
        "import sys, defect_detection.serving.main as m; m.create_app; "
        "bad = [n for n in ('torch', 'torchvision', 'timm', 'pandas', 'yaml', 'sklearn') "
        "if n in sys.modules]; print(bad); sys.exit(1 if bad else 0)"
    )
    # Fixed interpreter and code string: no untrusted input.
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout
