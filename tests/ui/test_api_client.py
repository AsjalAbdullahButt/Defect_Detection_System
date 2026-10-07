"""ApiClient against mocked transports: success, 4xx, 5xx, timeout, connect retry."""

import json
from pathlib import Path

import httpx
import pytest

from services.api_client import UPLOAD_NAME, ApiClient, ApiError

PREDICTION = {
    "predicted_class": "defective",
    "confidence": 0.99,
    "defect_probability": 0.99,
    "threshold": 0.9,
    "needs_review": False,
    "model_version": "v1",
    "request_id": "rid-1",
    "latency_ms": 12.5,
}


def client_for(handler, key: str = "k" * 32) -> ApiClient:  # type: ignore[no-untyped-def]
    return ApiClient("http://api", key, timeout_s=1.0, transport=httpx.MockTransport(handler))


def test_predict_success_sends_key_and_original_bytes() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("x-api-key")
        seen["body"] = request.content
        return httpx.Response(200, json=PREDICTION)

    result = client_for(handler).predict(b"\xff\xd8\xffRAWBYTES")
    assert result.predicted_class == "defective"
    assert result.request_id == "rid-1"
    assert seen["key"] == "k" * 32
    assert b"RAWBYTES" in seen["body"]  # type: ignore[operator]
    assert f'filename="{UPLOAD_NAME}"'.encode() in seen["body"]  # type: ignore[operator]


def test_4xx_becomes_api_error_with_request_id() -> None:
    body = {"error": "unsupported_media_type", "detail": "Only JPEG...", "request_id": "rid-415"}
    client = client_for(lambda r: httpx.Response(415, json=body))
    with pytest.raises(ApiError) as caught:
        client.predict(b"x")
    assert (caught.value.status, caught.value.error, caught.value.request_id) == (
        415,
        "unsupported_media_type",
        "rid-415",
    )


def test_5xx_without_json_body() -> None:
    client = client_for(lambda r: httpx.Response(502, text="<html>bad gateway</html>"))
    with pytest.raises(ApiError) as caught:
        client.model_info()
    assert caught.value.status == 502
    assert caught.value.detail == "HTTP 502"


def test_timeout_is_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ApiError, match="did not answer in time"):
        client_for(handler).predict(b"x")
    assert calls == 1  # the server may have received it: never resend a prediction


def test_connect_error_is_retried_once() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json=PREDICTION)

    assert client_for(handler).predict(b"x").latency_ms == 12.5
    assert calls == 2


def test_connect_error_twice_gives_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ApiError) as caught:
        client_for(handler).predict(b"x")
    assert caught.value.error == "unreachable"
    assert client_for(handler).is_ready() is False


def test_unexpected_success_payload_is_rejected() -> None:
    client = client_for(lambda r: httpx.Response(200, json={"surprise": True}))
    with pytest.raises(ApiError, match="Unexpected API response"):
        client.predict(b"x")


def test_batch_keeps_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/predict/batch"
        results = [
            {
                "index": i,
                "predicted_class": "normal",
                "confidence": 0.9,
                "defect_probability": 0.1,
                "needs_review": False,
            }
            for i in range(3)
        ]
        return httpx.Response(
            200, json={"request_id": "r", "model_version": "v", "latency_ms": 5, "results": results}
        )

    batch = client_for(handler).predict_batch([b"a", b"b", b"c"])
    assert [r.index for r in batch.results] == [0, 1, 2]


def test_api_key_is_never_in_error_text() -> None:
    key = "s3cret-" + "x" * 30
    client = client_for(
        lambda r: httpx.Response(
            401,
            json={
                "error": "unauthorized",
                "detail": "Missing or invalid API key.",
                "request_id": "r",
            },
        ),
        key,
    )
    with pytest.raises(ApiError) as caught:
        client.predict(b"x")
    assert key not in json.dumps(caught.value.__dict__, default=str)


def test_ui_imports_nothing_from_src() -> None:
    ui = Path(__file__).resolve().parents[2] / "ui"
    offenders = [
        p.name for p in ui.rglob("*.py") if "defect_detection" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []
