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
    circuit_breaker_enabled: bool = Field(
        default=True,
        description="Enable automated circuit breaking for upstream fault resilience",
    )
    circuit_failure_threshold: int = Field(
        default=3,
        description="Number of consecutive failures before tripping circuit breaker to OPEN",
        gt=0,
    )
    circuit_recovery_timeout_seconds: float = Field(
        default=10.0,
        description="Cooldown period in seconds before testing recovery in HALF_OPEN state",
        gt=0.0,
    )
    request_timeout_seconds: float = Field(
        default=60.0,
        description="Timeout in seconds for upstream vLLM requests",
        gt=0.0,
    )
    semantic_cache_enabled: bool = Field(
        default=True,
        description="Enable semantic vector cache for paraphrase deduplication at gateway ingress",
    )
    semantic_cache_capacity: int = Field(
        default=512,
        description="Maximum number of prompt/response pairs in the semantic LRU cache",
        gt=0,
    )
    semantic_cache_similarity_threshold: float = Field(
        default=0.92,
        description="Minimum cosine similarity score for a cache hit (0.0 to 1.0)",
        gt=0.0,
        le=1.0,
    )
    lora_routing_enabled: bool = Field(
        default=True,
        description="Enable dynamic Multi-LoRA adapter resolution and virtual model synthesis",
    )
    lora_default_base_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct-AWQ",
        description="Default base model used for bare LoRA adapter aliases",
    )
    grammar_guard_enabled: bool = Field(
        default=True,
        description="Enable guided structured output validation and automatic syntax repair",
    )
    grammar_guard_auto_repair: bool = Field(
        default=True,
        description="Automatically sanitize markdown fences and syntax errors in structured outputs",
    )
    guardrails_enabled: bool = Field(
        default=True,
        description="Enable ingress security scanning, prompt injection defense, and PII redaction",
    )
    guardrails_injection_defense_enabled: bool = Field(
        default=True,
        description="Block adversarial prompt injections, DAN jailbreaks, and delimiter escapes",
    )
    guardrails_pii_redaction_enabled: bool = Field(
        default=True,
        description="Automatically redact SSNs, credit cards, API keys, and phone numbers in-place",
    )
    guardrails_system_prompt_leak_defense: bool = Field(
        default=True,
        description="Filter egress completions to prevent system prompt leakage",
    )
    cascade_routing_enabled: bool = Field(
        default=True,
        description="Enable intelligent complexity evaluation and model tier cascading",
    )
    cascade_small_model: str = Field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        description="Small model tier for low-complexity queries (greetings, simple QA, classification)",
    )
    cascade_large_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct-AWQ",
        description="Large model tier for complex reasoning (code, SQL, multi-step math)",
    )
    cascade_complexity_threshold: float = Field(
        default=0.50,
        description="Complexity score threshold (0.0 to 1.0) above which queries route to the large model tier",
        gt=0.0,
        le=1.0,
    )
    compressor_enabled: bool = Field(
        default=True,
        description="Enable lexical entropy prompt compaction and filler stripping",
    )
    compressor_min_tokens: int = Field(
        default=50,
        description="Minimum token length required before prompt compaction is triggered",
        ge=1,
    )
    compressor_target_ratio: float = Field(
        default=0.60,
        description="Target token compression ratio (0.60 = 40% reduction)",
        gt=0.1,
        le=1.0,
    )
    compressor_preserve_code_blocks: bool = Field(
        default=True,
        description="Ensure code blocks within triple backticks are preserved byte-for-byte",
    )
    tool_engine_enabled: bool = Field(
        default=True,
        description="Enable native server-side agentic tool execution sandboxes (calculator, sql_runner, python_repl)",
    )
    tool_engine_max_iterations: int = Field(
        default=3,
        description="Maximum closed-loop agentic tool execution iterations per request",
        ge=1,
        le=10,
    )
    tool_engine_sandbox_timeout_seconds: float = Field(
        default=2.0,
        description="Execution timeout in seconds for sandboxed tools",
        gt=0.1,
    )
    finops_enabled: bool = Field(
        default=True,
        description="Enable real-time multi-tenant FinOps cost metering and budget enforcement",
    )
    finops_default_budget_usd: float = Field(
        default=100.0,
        description="Default budget allocation in USD for newly discovered tenants",
        ge=0.0,
    )
    finops_enforce_budgets: bool = Field(
        default=True,
        description="Reject requests with HTTP 402 when tenant spend exceeds budget limit",
    )
    finops_prompt_rate_per_1k: float = Field(
        default=0.00015,
        description="Dollar cost per 1,000 prompt tokens ($0.15 / 1M)",
        ge=0.0,
    )
    finops_completion_rate_per_1k: float = Field(
        default=0.00060,
        description="Dollar cost per 1,000 completion tokens ($0.60 / 1M)",
        ge=0.0,
    )
    shadow_replayer_enabled: bool = Field(
        default=True,
        description="Enable asynchronous background shadow traffic replication for regression detection",
    )
    shadow_backend_url: str = Field(
        default="http://host.k3d.internal:8000",
        description="Target candidate backend URL for mirrored shadow traffic",
    )
    shadow_sample_rate: float = Field(
        default=0.10,
        description="Sampling probability (0.0 to 1.0) of production traffic to duplicate to shadow backend",
        ge=0.0,
        le=1.0,
    )
    shadow_max_traces: int = Field(
        default=100,
        description="Maximum shadow comparison trace records to retain in memory",
        ge=10,
        le=1000,
    )


# Default singleton instance
settings = GatewaySettings()


def get_settings() -> GatewaySettings:
    """Dependency provider for GatewaySettings."""
    return settings
