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
    rate_limit_tpm: int = Field(
        default=50000,
        description="Max allowed tokens per minute per client IP",
        gt=0,
    )
    max_concurrent_interactive_requests: int = Field(
        default=8,
        description="Max concurrent active requests processed before queue buffering",
        gt=0,
    )
    max_queue_size: int = Field(
        default=64,
        description="Maximum capacity of priority request queue",
        gt=0,
    )
    queue_timeout_seconds: float = Field(
        default=30.0,
        description="Timeout in seconds for queued requests before rejection",
        gt=0.0,
    )
    prefix_cache_routing_enabled: bool = Field(
        default=True,
        description="Enable prefix hashing and KV-cache affinity routing",
    )
    cache_router_capacity: int = Field(
        default=1024,
        description="Maximum capacity of LRU prefix cache registry",
        gt=0,
    )
    prefix_min_chars: int = Field(
        default=32,
        description="Minimum character length for prefix extraction and hashing",
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
