"""Structured JSON logging and Prometheus metrics.

Logs: one JSON object per line on stdout (container-friendly), with ``request_id`` and other
fields passed via ``extra``. Image bytes and client filenames are never logged.

Metrics: label values are bounded (route templates, not raw paths; class names; status codes)
so a client cannot create unbounded time series. They are served on a separate internal port.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

_STANDARD = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Render a log record as a single JSON line, including ``extra`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise the record."""
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str) -> None:
    """Route the app's and uvicorn's loggers through the JSON formatter on stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    for name in ("defect_detection", "uvicorn", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers = [handler]
        logger.setLevel(level)
        logger.propagate = False
    logging.getLogger("uvicorn.access").disabled = True  # replaced by our access log


REQUESTS = Counter("dd_http_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram(
    "dd_http_request_duration_seconds",
    "HTTP request latency",
    ["route"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
INFERENCE = Histogram(
    "dd_inference_duration_seconds",
    "Decode + preprocess + model time per request",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
ERRORS = Counter("dd_errors_total", "Error responses by error code", ["error"])
PREDICTIONS = Counter(
    "dd_predictions_total",
    "Predictions by outcome",
    ["model_version", "predicted_class", "needs_review"],
)
DEFECT_PROBABILITY = Histogram(
    "dd_defect_probability",
    "Distribution of P(defective); a shift suggests data drift",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99),
)
MODEL_INFO = Gauge(
    "dd_model_info", "Loaded model (value is always 1)", ["model_version", "backbone"]
)
