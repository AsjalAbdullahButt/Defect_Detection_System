"""V7: API-key auth, rate limiting, security headers, CORS and fail-fast production config."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from defect_detection.serving.main import create_app
from defect_detection.serving.security import STRICT_HEADERS
from defect_detection.serving.settings import Settings
from tests.fixtures.serving import ExportedModel, make_settings
from tests.fixtures.synthetic import smooth_image

KEY_A = "a" * 40
KEY_B = "b" * 40


def png() -> bytes:
    buf = io.BytesIO()
    smooth_image(1, size=64).save(buf, format="PNG")
    return buf.getvalue()


def protected_app(exported_model: ExportedModel, **overrides: object) -> TestClient:
    settings = make_settings(
        exported_model.model_dir, api_keys=f"{KEY_A},{KEY_B}", environment="production", **overrides
    )
    return TestClient(create_app(settings))


def predict(client: TestClient, key: str | None = None) -> object:
    headers = {"X-API-Key": key} if key is not None else {}
    return client.post(
        "/v1/predict", files={"file": ("a.png", png(), "image/png")}, headers=headers
    )


@pytest.mark.parametrize("key", [None, "", "wrong" * 10, KEY_A[:-1], KEY_A + "x"])
def test_missing_or_invalid_key_is_401(exported_model: ExportedModel, key: str | None) -> None:
    with protected_app(exported_model) as client:
        response = predict(client, key)
    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"
    assert response.json()["detail"] == "Missing or invalid API key."  # same for every failure
    assert response.headers["WWW-Authenticate"] == "ApiKey"


@pytest.mark.parametrize("key", [KEY_A, KEY_B])
def test_every_configured_key_works(exported_model: ExportedModel, key: str) -> None:
    with protected_app(exported_model) as client:
        assert predict(client, key).status_code == 200
        assert client.get("/v1/model", headers={"X-API-Key": key}).status_code == 200


def test_probes_stay_open(exported_model: ExportedModel) -> None:
    with protected_app(exported_model) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200


def test_auth_rejects_before_reading_the_body(exported_model: ExportedModel) -> None:
    """An anonymous oversized upload gets 401, not 413: the body is never parsed."""
    with protected_app(exported_model, max_upload_bytes=10_000, max_batch_bytes=10_000) as client:
        big = b"\x89PNG\r\n\x1a\n" + b"\x00" * 500_000
        response = client.post("/v1/predict", files={"file": ("a.png", big, "image/png")})
    assert response.status_code == 401


def test_failed_auth_attempts_are_throttled(exported_model: ExportedModel) -> None:
    with protected_app(exported_model, auth_failure_limit="3/minute") as client:
        statuses = [predict(client, "guess-" + "x" * 40).status_code for _ in range(5)]
        assert statuses == [401, 401, 401, 429, 429]
        assert predict(client, KEY_A).status_code == 200  # a valid key is not locked out


def test_rate_limit_per_caller(exported_model: ExportedModel) -> None:
    with protected_app(exported_model, rate_limit="2/minute") as client:
        first = [predict(client, KEY_A).status_code for _ in range(3)]
        assert first == [200, 200, 429]
        limited = predict(client, KEY_A)
        assert int(limited.headers["Retry-After"]) >= 1
        assert limited.json()["error"] == "rate_limited"
        assert predict(client, KEY_B).status_code == 200  # other key, own budget


def test_security_headers_on_success_and_error(exported_model: ExportedModel) -> None:
    with protected_app(exported_model) as client:
        for response in (client.get("/health"), predict(client, None), client.get("/nope")):
            for name, value in STRICT_HEADERS.items():
                assert response.headers[name] == value


def test_docs_disabled_in_production_even_with_auth(exported_model: ExportedModel) -> None:
    with protected_app(exported_model) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


def test_cors_disabled_by_default(exported_model: ExportedModel) -> None:
    with protected_app(exported_model) as client:
        response = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_cors_allow_list(exported_model: ExportedModel) -> None:
    with protected_app(exported_model, cors_origins="https://qa.example.com") as client:
        allowed = client.options(
            "/v1/predict",
            headers={"Origin": "https://qa.example.com", "Access-Control-Request-Method": "POST"},
        )
        denied = client.get("/health", headers={"Origin": "https://evil.example"})
    assert allowed.headers["access-control-allow-origin"] == "https://qa.example.com"
    assert "access-control-allow-origin" not in denied.headers


def test_production_without_keys_refuses_to_start(exported_model: ExportedModel) -> None:
    with pytest.raises(ValidationError, match="production requires DD_API_KEYS"):
        Settings(model_dir=exported_model.model_dir, environment="production")  # type: ignore[arg-type]
    explicit = Settings(  # type: ignore[call-arg]
        model_dir=exported_model.model_dir, environment="production", allow_unauthenticated=True
    )
    assert not explicit.auth_enabled


def test_short_keys_and_bad_limits_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError, match="at least 32"):
        Settings(model_dir="m", api_keys="short", environment="development")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Settings(model_dir="m", rate_limit="lots", environment="development")  # type: ignore[arg-type]
    monkeypatch.setenv("DD_MODEL_DIR", "m")
    monkeypatch.setenv("DD_API_KEYS", f" {KEY_A} , {KEY_B} ")
    settings = Settings()  # type: ignore[call-arg]
    assert [k.get_secret_value() for k in settings.api_keys] == [KEY_A, KEY_B]
    assert KEY_A not in repr(settings)  # SecretStr never prints


def test_animated_gif_is_rejected_by_media_type(exported_model: ExportedModel) -> None:
    frames = [Image.new("RGB", (64, 64), c) for c in ("red", "blue")]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:])
    with protected_app(exported_model) as client:
        response = client.post(
            "/v1/predict",
            files={"file": ("a.gif", buf.getvalue(), "image/gif")},
            headers={"X-API-Key": KEY_A},
        )
    assert response.status_code == 415


def test_error_bodies_contain_no_stack_traces(exported_model: ExportedModel) -> None:
    with protected_app(exported_model) as client:
        bad = client.post(
            "/v1/predict",
            files={"file": ("a.png", b"\x89PNG\r\n\x1a\nbroken", "image/png")},
            headers={"X-API-Key": KEY_A},
        )
    assert bad.status_code == 400
    assert set(bad.json()) == {"error", "detail", "request_id"}
    for leak in ("Traceback", 'File "', "line ", "PIL", "site-packages"):
        assert leak not in bad.text
