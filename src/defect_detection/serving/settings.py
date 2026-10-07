"""Serving configuration from environment variables only (12-factor), validated at startup.

Every variable is prefixed ``DD_`` (e.g. ``DD_MODEL_DIR``). Invalid or missing values make
``Settings()`` raise, so the process fails fast instead of starting half-configured.
See ``configs/serve.env.example`` for the documented list.
"""

from pathlib import Path
from typing import Annotated, Literal

from limits import parse
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

MIB = 1024 * 1024
MIN_API_KEY_LENGTH = 32


class Settings(BaseSettings):
    """All serving knobs. Defaults are the safe production choice."""

    model_config = SettingsConfigDict(env_prefix="DD_", extra="forbid", frozen=True)

    # Model artifact: models/<version>/ with model.onnx, model_meta.json, SHA256SUMS.
    model_dir: Path
    environment: Literal["production", "development"] = "production"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Upload limits.
    max_upload_bytes: int = Field(default=5 * MIB, ge=1024)
    max_batch_files: int = Field(default=16, ge=1, le=256)
    max_batch_bytes: int = Field(default=32 * MIB, ge=1024)
    min_image_side: int = Field(default=32, ge=1)
    max_image_side: int = Field(default=4096, ge=32)
    max_image_pixels: int = Field(default=16_000_000, ge=1024)  # PIL decompression-bomb cap
    max_aspect_ratio: float = Field(default=4.0, ge=1.0)  # direct resize distorts extreme shapes

    # Inference resources. threads * max_concurrent_inferences should not exceed the CPU cores.
    inference_threads: int = Field(default=2, ge=1)
    max_concurrent_inferences: int = Field(default=2, ge=1)
    inference_timeout_s: float = Field(default=10.0, gt=0)

    # API-key auth on /v1/*. Comma-separated in DD_API_KEYS (several keys allow rotation).
    # Production refuses to start without keys unless DD_ALLOW_UNAUTHENTICATED=true.
    api_keys: Annotated[tuple[SecretStr, ...], NoDecode] = ()
    allow_unauthenticated: bool = False
    # Rate limits (limits-library syntax, e.g. "60/minute"), per API key or client IP.
    rate_limit: str = "120/minute"
    auth_failure_limit: str = "10/minute"
    # CORS is off unless origins are listed (comma-separated in DD_CORS_ORIGINS).
    cors_origins: Annotated[tuple[str, ...], NoDecode] = ()

    # Prometheus metrics on a separate port that is not published outside the container network.
    metrics_port: int | None = Field(default=9100, ge=1024, le=65535)
    metrics_host: str = "127.0.0.1"

    @field_validator("api_keys", "cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

    @field_validator("api_keys")
    @classmethod
    def _keys_long_enough(cls, keys: tuple[SecretStr, ...]) -> tuple[SecretStr, ...]:
        if any(len(k.get_secret_value()) < MIN_API_KEY_LENGTH for k in keys):
            raise ValueError(f"every API key must be at least {MIN_API_KEY_LENGTH} characters")
        return keys

    @field_validator("rate_limit", "auth_failure_limit")
    @classmethod
    def _valid_limit(cls, value: str) -> str:
        parse(value)  # raises ValueError on bad syntax
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "Settings":
        if self.min_image_side > self.max_image_side:
            raise ValueError("min_image_side must be <= max_image_side")
        if self.max_batch_bytes < self.max_upload_bytes:
            raise ValueError("max_batch_bytes must be >= max_upload_bytes")
        if (
            self.environment == "production"
            and not self.api_keys
            and not self.allow_unauthenticated
        ):
            raise ValueError(
                "production requires DD_API_KEYS (or explicitly DD_ALLOW_UNAUTHENTICATED=true)"
            )
        return self

    @property
    def auth_enabled(self) -> bool:
        """API-key auth is on whenever keys are configured."""
        return bool(self.api_keys)

    @property
    def docs_enabled(self) -> bool:
        """OpenAPI docs are served only outside production."""
        return self.environment != "production"
