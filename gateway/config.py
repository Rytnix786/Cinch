"""Configuration settings for Cinch FastAPI Gateway."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    """Gateway operational configuration loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vllm_base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of upstream vLLM inference server",
    )
    gateway_host: str = Field(
        default="0.0.0.0",
        description="Bind host for FastAPI gateway server",
    )
    gateway_port: int = Field(
        default=8080,
        description="Bind port for FastAPI gateway server",
    )
    gateway_api_key: str | None = Field(
        default=None,
        description="Shared secret API key for gateway authentication. If None/empty, auth is disabled.",
    )
    rate_limit_rpm: int = Field(
        default=60,
        description="Max allowed requests per minute per client IP",
        gt=0,
    )
    request_timeout_seconds: float = Field(
        default=60.0,
        description="Timeout in seconds for upstream vLLM requests",
        gt=0.0,
    )


# Default singleton instance
settings = GatewaySettings()


def get_settings() -> GatewaySettings:
    """Dependency provider for GatewaySettings."""
    return settings
