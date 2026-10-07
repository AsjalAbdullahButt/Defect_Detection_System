"""API-key authentication, rate limiting and security headers (pure ASGI middleware).

All three run BEFORE the request body is read. FastAPI parses multipart bodies before route
dependencies run, so dependency-based auth would let an anonymous client make the server
parse a full upload before being told 401.

* Auth: ``X-API-Key`` on ``/v1/*`` compared in constant time against EVERY configured key
  (no early exit). Missing and wrong keys get the identical 401. Failed attempts are rate
  limited per client IP, which blunts key guessing. ``/health`` and ``/ready`` stay open for
  orchestrator probes.
* Rate limit: moving window per caller. The caller is a fingerprint of the API key when
  authenticated, else the client IP. 429 with ``Retry-After``.
* Security headers on every response, errors included.

Limiter state is in-process memory, so limits apply per worker process (documented in
SECURITY.md; a shared store such as Redis would be needed for a global limit).
"""

import hashlib
import hmac
import json
import math
import time

from limits import RateLimitItem, parse
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter
from pydantic import SecretStr
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from defect_detection.serving.observability import ERRORS

PROTECTED_PREFIX = "/v1/"
API_KEY_HEADER = b"x-api-key"
UNAUTHORIZED = "Missing or invalid API key."
TOO_MANY = "Rate limit exceeded; retry later."

STRICT_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
}
DOCS_PATHS = ("/docs", "/openapi.json")  # the docs UI needs scripts; only exists outside prod


def _client_ip(scope: Scope) -> str:
    client = scope.get("client")
    return str(client[0]) if client else "unknown"


async def send_json_error(
    scope: Scope,
    send: Send,
    status: int,
    error: str,
    detail: str,
    headers: dict[str, str] | None = None,
) -> None:
    """Send the standard ``{error, detail, request_id}`` body directly from middleware."""
    request_id = scope.get("state", {}).get("request_id", "unknown")
    body = json.dumps({"error": error, "detail": detail, "request_id": request_id}).encode()
    raw_headers = [(b"content-type", b"application/json")]
    raw_headers += [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    ERRORS.labels(error=error).inc()
    await send({"type": "http.response.start", "status": status, "headers": raw_headers})
    await send({"type": "http.response.body", "body": body})


class _Limiter:
    """Moving-window limiter keyed by caller identity."""

    def __init__(self, limit: str) -> None:
        self.item: RateLimitItem = parse(limit)
        self.limiter = MovingWindowRateLimiter(MemoryStorage())

    def hit(self, key: str) -> int | None:
        """Record a hit; return None if allowed, else seconds until the client may retry."""
        if self.limiter.hit(self.item, key):
            return None
        reset = self.limiter.get_window_stats(self.item, key).reset_time
        return max(1, math.ceil(reset - time.time()))


class ApiKeyAuthMiddleware:
    """Require a valid X-API-Key on /v1/*; throttle failed attempts per client IP."""

    def __init__(self, app: ASGIApp, api_keys: tuple[SecretStr, ...], failure_limit: str) -> None:
        self.app = app
        self.keys = [k.get_secret_value().encode() for k in api_keys]
        self.failures = _Limiter(failure_limit)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection."""
        if scope["type"] != "http" or not scope["path"].startswith(PROTECTED_PREFIX):
            await self.app(scope, receive, send)
            return
        presented = dict(scope["headers"]).get(API_KEY_HEADER, b"")
        # Compare against every key so timing does not reveal which one (or whether) matched.
        matches = [hmac.compare_digest(presented, key) for key in self.keys]
        if not any(matches):
            retry = self.failures.hit(_client_ip(scope))
            if retry is not None:
                await send_json_error(
                    scope, send, 429, "rate_limited", TOO_MANY, {"Retry-After": str(retry)}
                )
                return
            await send_json_error(
                scope, send, 401, "unauthorized", UNAUTHORIZED, {"WWW-Authenticate": "ApiKey"}
            )
            return
        # Identify the caller by a non-reversible fingerprint, never the key itself.
        scope.setdefault("state", {})["caller"] = (
            "key:" + hashlib.sha256(presented).hexdigest()[:12]
        )
        await self.app(scope, receive, send)


class RateLimitMiddleware:
    """Moving-window rate limit on /v1/* per caller (API-key fingerprint or client IP)."""

    def __init__(self, app: ASGIApp, limit: str) -> None:
        self.app = app
        self.limiter = _Limiter(limit)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection."""
        if scope["type"] != "http" or not scope["path"].startswith(PROTECTED_PREFIX):
            await self.app(scope, receive, send)
            return
        caller = scope.get("state", {}).get("caller") or "ip:" + _client_ip(scope)
        retry = self.limiter.hit(caller)
        if retry is not None:
            await send_json_error(
                scope, send, 429, "rate_limited", TOO_MANY, {"Retry-After": str(retry)}
            )
            return
        await self.app(scope, receive, send)


class SecurityHeadersMiddleware:
    """Add defensive headers to every HTTP response (the docs UI gets a relaxed CSP)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        is_docs = scope["path"].startswith(DOCS_PATHS)

        async def add_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in STRICT_HEADERS.items():
                    if not (is_docs and name == "Content-Security-Policy"):
                        headers[name] = value
            await send(message)

        await self.app(scope, receive, add_headers)
