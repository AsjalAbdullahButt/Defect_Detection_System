"""Hostile and malformed uploads: every one must be rejected with the right status code."""

import io
import struct
import zlib

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from defect_detection.serving.errors import InvalidImageError, UnsupportedMediaTypeError
from defect_detection.serving.settings import Settings
from defect_detection.serving.validation import decode_image, sniff_format
from tests.fixtures.serving import ExportedModel, make_settings
from tests.fixtures.synthetic import smooth_image

SETTINGS = Settings(
    model_dir="unused", metrics_port=None, min_image_side=16, max_image_pixels=4_000_000
)  # type: ignore[arg-type]


def encode(img: Image.Image, fmt: str, **kwargs: object) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def png_with_header_size(width: int, height: int) -> bytes:
    """A tiny PNG whose IHDR claims huge dimensions (a decompression-bomb header)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b"\x00"))
        + chunk(b"IEND", b"")
    )


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "BMP", "WEBP"])
def test_accepted_formats_decode(fmt: str) -> None:
    data = encode(smooth_image(1, size=64), fmt)
    assert sniff_format(data) == fmt
    assert decode_image(data, SETTINGS).size == (64, 64)


@pytest.mark.parametrize(
    "data",
    [
        b"GIF89a" + b"\x00" * 64,  # real format, not accepted
        b"%PDF-1.7 not an image",
        b"<?php system($_GET['c']); ?>",
        b"",
    ],
)
def test_unaccepted_content_is_415(data: bytes) -> None:
    with pytest.raises(UnsupportedMediaTypeError):
        decode_image(data, SETTINGS)


def test_magic_bytes_win_over_extension_and_mime(exported_model: ExportedModel) -> None:
    """A script named .png with an image/png MIME type is still rejected (415)."""
    with TestClient(create_app_for(exported_model)) as client:
        response = client.post(
            "/v1/predict", files={"file": ("cat.png", b"#!/bin/sh\nrm -rf /\n", "image/png")}
        )
    assert response.status_code == 415
    assert response.json()["error"] == "unsupported_media_type"


def test_polyglot_png_with_appended_payload_is_only_an_image() -> None:
    """Bytes after IEND are ignored by the decoder; the file is treated purely as pixels."""
    data = encode(smooth_image(2, size=64), "PNG") + b"PK\x03\x04<script>alert(1)</script>"
    assert decode_image(data, SETTINGS).size == (64, 64)


def test_header_only_polyglot_is_rejected() -> None:
    with pytest.raises(InvalidImageError):
        decode_image(b"\xff\xd8\xff" + b"<html>not really a jpeg</html>", SETTINGS)


def test_truncated_image_is_400() -> None:
    data = encode(smooth_image(3, size=128), "JPEG")
    with pytest.raises(InvalidImageError, match="could not be decoded"):
        decode_image(data[: len(data) // 2], SETTINGS)


def test_decompression_bomb_header_is_rejected_before_decoding() -> None:
    with pytest.raises(InvalidImageError, match="too many pixels"):
        decode_image(png_with_header_size(30_000, 30_000), SETTINGS)


def test_pixel_count_between_1x_and_2x_cap_is_also_rejected() -> None:
    with pytest.raises(InvalidImageError, match="too many pixels"):
        decode_image(png_with_header_size(2500, 2500), SETTINGS)  # 6.25 MP vs 4 MP cap


def test_animated_image_is_rejected() -> None:
    frames = [smooth_image(s, size=64) for s in range(3)]
    data = encode(frames[0], "WEBP", save_all=True, append_images=frames[1:])
    with pytest.raises(InvalidImageError, match="Multi-frame"):
        decode_image(data, SETTINGS)


@pytest.mark.parametrize(("size", "message"), [((8, 8), "at least"), ((300, 30), "Aspect ratio")])
def test_dimension_limits(size: tuple[int, int], message: str) -> None:
    data = encode(Image.new("RGB", size), "PNG")
    with pytest.raises(InvalidImageError, match=message):
        decode_image(data, SETTINGS)


def create_app_for(exported_model: ExportedModel, **overrides: object):  # type: ignore[no-untyped-def]
    from defect_detection.serving.main import create_app

    return create_app(make_settings(exported_model.model_dir, **overrides))


def test_oversize_declared_body_is_413_before_reading(exported_model: ExportedModel) -> None:
    with TestClient(
        create_app_for(exported_model, max_upload_bytes=10_000, max_batch_bytes=10_000)
    ) as client:
        big = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200_000
        response = client.post("/v1/predict", files={"file": ("a.png", big, "image/png")})
    assert response.status_code == 413
    body = response.json()
    assert body["error"] == "payload_too_large"
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_oversize_streamed_body_without_length_is_413(exported_model: ExportedModel) -> None:
    """Chunked upload with no Content-Length: the guard counts bytes as they stream."""
    boundary = "xyz"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode()

    def body():  # type: ignore[no-untyped-def]
        yield head
        for _ in range(50):
            yield b"\x00" * 10_000
        yield f"\r\n--{boundary}--\r\n".encode()

    with TestClient(
        create_app_for(exported_model, max_upload_bytes=10_000, max_batch_bytes=10_000)
    ) as client:
        response = client.post(
            "/v1/predict",
            content=body(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
    assert response.status_code == 413
    assert response.json()["error"] == "payload_too_large"


def test_error_bodies_never_leak_internals(exported_model: ExportedModel) -> None:
    with TestClient(create_app_for(exported_model)) as client:
        response = client.post(
            "/v1/predict", files={"file": ("a.jpg", b"\xff\xd8\xff garbage", "image/jpeg")}
        )
    text = response.text
    assert response.status_code == 400
    for leak in ("Traceback", "PIL", "site-packages", ".py", str(exported_model.model_dir)):
        assert leak not in text
