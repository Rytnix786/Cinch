"""Smoke test client and verification script for local vLLM serving.

Validates:
1. GET /health: server liveness
2. GET /v1/models: registered model ID verification
3. POST /v1/chat/completions: single-request inference, token metrics, and schema validity
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx


@dataclass
class HealthResult:
    is_healthy: bool
    status_code: int
    details: str


@dataclass
class ModelsResult:
    models: list[str]
    has_expected_model: bool
    raw_response: dict[str, Any]


@dataclass
class CompletionResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    raw_response: dict[str, Any]


@dataclass
class SmokeTestSummary:
    success: bool
    health: HealthResult
    models: Optional[ModelsResult] = None
    completion: Optional[CompletionResult] = None
    error: Optional[str] = None


class VLLMSmokeClient:
    """Client for testing vLLM OpenAI-compatible serving endpoints."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 60.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = client

    @contextlib.contextmanager
    def _client_session(self):
        if self._client is not None:
            yield self._client
        else:
            with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
                yield client

    def check_health(self) -> HealthResult:
        """Check the /health endpoint."""
        try:
            with self._client_session() as client:
                resp = client.get(f"{self.base_url}/health" if self._client is None else "/health")
                is_healthy = resp.status_code == 200
                return HealthResult(
                    is_healthy=is_healthy,
                    status_code=resp.status_code,
                    details="OK" if is_healthy else f"Unexpected status code: {resp.status_code}",
                )
        except Exception as exc:
            return HealthResult(
                is_healthy=False,
                status_code=0,
                details=f"Connection error: {exc}",
            )

    def get_models(self, expected_model: Optional[str] = None) -> ModelsResult:
        """Query /v1/models and verify available model IDs."""
        with self._client_session() as client:
            resp = client.get(f"{self.base_url}/v1/models" if self._client is None else "/v1/models")
            resp.raise_for_status()
            data = resp.json()

            models: list[str] = []
            for item in data.get("data", []):
                if isinstance(item, dict) and "id" in item:
                    models.append(item["id"])

            has_expected = True
            if expected_model:
                has_expected = any(m == expected_model or expected_model in m for m in models)

            return ModelsResult(
                models=models,
                has_expected_model=has_expected,
                raw_response=data,
            )

    def chat_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> CompletionResult:
        """Send a chat completion request to /v1/chat/completions."""
        target_model = model or "Qwen/Qwen2.5-7B-Instruct-AWQ"
        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        start_time = time.perf_counter()

        with self._client_session() as client:
            resp = client.post(
                f"{self.base_url}/v1/chat/completions" if self._client is None else "/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            latency = time.perf_counter() - start_time
            data = resp.json()

            choices = data.get("choices", [])
            if not choices or not isinstance(choices, list):
                raise ValueError("Response missing choices list")

            first_choice = choices[0]
            message = first_choice.get("message", {})
            text = message.get("content", "").strip()

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            return CompletionResult(
                text=text,
                model=data.get("model", target_model),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_seconds=latency,
                raw_response=data,
            )

    def run_all(
        self,
        expected_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
        prompt: str = "Explain latency vs throughput in one concise sentence.",
        max_tokens: int = 128,
    ) -> SmokeTestSummary:
        """Run end-to-end smoke verification pipeline."""
        # 1. Health check
        health = self.check_health()
        if not health.is_healthy:
            return SmokeTestSummary(
                success=False,
                health=health,
                error=f"Health check failed (status {health.status_code}): {health.details}",
            )

        # 2. Model list check
        try:
            models_result = self.get_models(expected_model=expected_model)
        except Exception as exc:
            return SmokeTestSummary(
                success=False,
                health=health,
                error=f"Failed to query /v1/models: {exc}",
            )

        if expected_model and not models_result.has_expected_model:
            return SmokeTestSummary(
                success=False,
                health=health,
                models=models_result,
                error=f"Expected model '{expected_model}' not found in registered models: {models_result.models}",
            )

        # 3. Chat completion check
        try:
            target_model = models_result.models[0] if models_result.models else expected_model
            completion = self.chat_completion(
                prompt=prompt,
                model=target_model,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            return SmokeTestSummary(
                success=False,
                health=health,
                models=models_result,
                error=f"Chat completion failed: {exc}",
            )

        if not completion.text:
            return SmokeTestSummary(
                success=False,
                health=health,
                models=models_result,
                completion=completion,
                error="Chat completion returned empty output text.",
            )

        return SmokeTestSummary(
            success=True,
            health=health,
            models=models_result,
            completion=completion,
        )


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke test against local vLLM serving container.")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="vLLM server base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct-AWQ",
        help="Expected model identifier (default: Qwen/Qwen2.5-7B-Instruct-AWQ)",
    )
    parser.add_argument(
        "--prompt",
        default="Explain latency vs throughput in one concise sentence.",
        help="Test prompt for chat completion",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum generation tokens (default: 128)",
    )
    return parser.parse_args(args)


def main(args: Optional[list[str]] = None) -> int:
    parsed = parse_args(args)
    print(f"Connecting to vLLM at {parsed.base_url}...")
    client = VLLMSmokeClient(base_url=parsed.base_url, timeout=parsed.timeout)

    summary = client.run_all(
        expected_model=parsed.model,
        prompt=parsed.prompt,
        max_tokens=parsed.max_tokens,
    )

    print("\n--- Smoke Test Results ---")
    print(f"Health check: {'[PASS]' if summary.health.is_healthy else '[FAIL]'} (status {summary.health.status_code})")

    if summary.models:
        print(f"Registered models: {summary.models.models}")
        print(f"Model match ('{parsed.model}'): {'[PASS]' if summary.models.has_expected_model else '[FAIL]'}")

    if summary.completion:
        comp = summary.completion
        print("Chat completion: [PASS]")
        print(f"Model used: {comp.model}")
        print(f"Latency: {comp.latency_seconds:.3f}s")
        print(f"Tokens: prompt={comp.prompt_tokens}, completion={comp.completion_tokens}, total={comp.total_tokens}")
        print(f"\nGenerated response:\n{comp.text}\n")

    if not summary.success:
        print(f"\n[ERROR] Smoke test failed: {summary.error}", file=sys.stderr)
        return 1

    print("[SUCCESS] Milestone 1 verification passed: single request served successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
