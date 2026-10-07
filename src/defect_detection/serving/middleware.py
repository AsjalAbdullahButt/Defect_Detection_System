"""Pure-ASGI middleware: request id, body-size guard, access log + HTTP metrics.

Order (outermost first): RequestIdMiddleware -> AccessLogMiddleware -> BodySizeLimitMiddleware
-> FastAPI. Pure ASGI (rather than BaseHTTPMiddleware) so the body is checked while it
streams and is never buffered by the middleware itself.
"""

import json
import logging
import re
import time
import uuid

from fastapi import HTTPException
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from defect_detection.serving.observability import ERRORS, LATENCY, REQUESTS

log = logging.getLogger("defect_detection.serving.access")
REQUEST_ID_HEADER = "X-Request-ID"
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
BODY_TOO_LARGE = "Request body exceeds the size limit."


class RequestIdMiddleware:
    """Assign every request an id (a safe client-supplied one is kept) and echo it back."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        # Untrusted header: only short, plain ids are kept, so it can't inject into logs.
        request_id = incoming if _SAFE_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        await self.app(scope, receive, send_with_id)


class BodySizeLimitMiddleware:
    """Reject bodies over the limit with 413, both declared (Content-Length) and streamed."""

    def __init__(self, app: ASGIApp, default_limit: int, limits: dict[str, int]) -> None:
        self.app = app
        self.default_limit = default_limit
        self.limits = limits  # per exact path, e.g. the batch endpoint allows more

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection."""
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return
        limit = self.limits.get(scope["path"], self.default_limit)
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > limit):
            await _send_413(scope, send)  # refused before reading a single body byte
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    # fastapi.HTTPException is re-raised untouched by body parsing -> 413.
                    raise HTTPException(status_code=413, detail=BODY_TOO_LARGE)
            return message

        await self.app(scope, limited_receive, send)


async def _send_413(scope: Scope, send: Send) -> None:
    request_id = scope.get("state", {}).get("request_id", "unknown")
    body = json.dumps(
        {"error": "payload_too_large", "detail": BODY_TOO_LARGE, "request_id": request_id}
    ).encode()
    ERRORS.labels(error="payload_too_large").inc()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [(b"content-type", b"application/json"), (b"connection", b"close")],
        }
    )
    await send({"type": "http.response.body", "body": body})


class AccessLogMiddleware:
    """One structured log line and HTTP metrics per request (no bodies, no filenames)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        status = 500  # if the app dies before sending a response

        async def capture(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, capture)
        finally:
            elapsed = time.perf_counter() - start
            route = getattr(scope.get("route"), "path", "unmatched")  # template, bounded labels
            REQUESTS.labels(method=scope["method"], route=route, status=str(status)).inc()
            LATENCY.labels(route=route).observe(elapsed)
            log.info(
                "request",
                extra={
                    "request_id": scope.get("state", {}).get("request_id"),
                    "method": scope["method"],
                    "route": route,
                    "status": status,
                    "duration_ms": round(elapsed * 1000, 2),
                },
            )
