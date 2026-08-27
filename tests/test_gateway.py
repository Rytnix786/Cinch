"""Comprehensive automated test suite for Cinch FastAPI Gateway."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator
from fastapi.testclient import TestClient
import httpx
import pytest

from gateway.app import app, gateway_state, get_client
from gateway.config import GatewaySettings, get_settings
from gateway.limiter import rate_limiter


@pytest.fixture(autouse=True)
def reset_gateway_state() -> None:
    """Reset rate limiter and gateway state metrics between tests."""
    rate_limiter.reset()
    gateway_state.total_requests = 0
    gateway_state.rate_limited_requests = 0
    gateway_state.error_requests = 0
    gateway_state.recent_latencies.clear()


def make_mock_client(handler: Any) -> httpx.AsyncClient:
    """Create an AsyncClient with a MockTransport handler."""
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, base_url="http://mock-vllm:8000")


def test_health_healthy() -> None:
    """Verify /health returns 200 when upstream vLLM is healthy."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"})

    mock_client = make_mock_client(mock_handler)
    app.dependency_overrides[get_client] = lambda: mock_client

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["gateway"] == "ok"
        assert data["vllm"] == "ok"

    app.dependency_overrides.clear()


def test_health_upstream_unreachable() -> None:
    """Verify /health returns 503 degraded when upstream vLLM is down."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    mock_client = make_mock_client(mock_handler)
    app.dependency_overrides[get_client] = lambda: mock_client

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["gateway"] == "ok"
        assert "unreachable" in data["vllm"]

    app.dependency_overrides.clear()


def test_health_upstream_500() -> None:
    """Verify /health returns 503 degraded when upstream returns 500."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal engine failure")

    mock_client = make_mock_client(mock_handler)
    app.dependency_overrides[get_client] = lambda: mock_client

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "degraded"
        assert "500" in data["vllm"]

    app.dependency_overrides.clear()


def test_metrics_endpoint() -> None:
    """Verify /metrics tracks requests and average latency."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    mock_client = make_mock_client(mock_handler)
    app.dependency_overrides[get_client] = lambda: mock_client

    with TestClient(app) as client:
        client.get("/health")
        client.get("/health")
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] >= 2
        assert data["error_requests"] == 0
        assert data["rate_limited_requests"] == 0
        assert "uptime_seconds" in data

    app.dependency_overrides.clear()


def test_auth_disabled_by_default() -> None:
    """Verify requests pass without auth headers when GATEWAY_API_KEY is not set."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen2.5-7B-Instruct-AWQ"}]})

    mock_client = make_mock_client(mock_handler)
    app.dependency_overrides[get_client] = lambda: mock_client

    with TestClient(app) as client:
        resp = client.get("/v1/models")
        assert resp.status_code == 200

    app.dependency_overrides.clear()


def test_auth_bearer_token_success() -> None:
    """Verify Bearer token authentication."""
    custom_settings = GatewaySettings(gateway_api_key="cinch-secret-key", vllm_base_url="http://mock-vllm:8000")

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen2.5-7B-Instruct-AWQ"}]})

    mock_client = make_mock_client(mock_handler)
    app.dependency_overrides[get_client] = lambda: mock_client
    app.dependency_overrides[get_settings] = lambda: custom_settings

    with TestClient(app) as client:
        # Missing key -> 401
        resp = client.get("/v1/models")
        assert resp.status_code == 401

        # Invalid key -> 401
        resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401

        # Valid Bearer key -> 200
        resp = client.get("/v1/models", headers={"Authorization": "Bearer cinch-secret-key"})
        assert resp.status_code == 200

        # Valid X-API-Key -> 200
        resp = client.get("/v1/models", headers={"X-API-Key": "cinch-secret-key"})
        assert resp.status_code == 200

    app.dependency_overrides.clear()


def test_rate_limiter_enforcement() -> None:
    """Verify sliding-window rate limit triggers 429 when quota exceeded."""
    custom_settings = GatewaySettings(rate_limit_rpm=3, vllm_base_url="http://mock-vllm:8000")

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    mock_client = make_mock_client(mock_handler)
    app.dependency_overrides[get_client] = lambda: mock_client
    app.dependency_overrides[get_settings] = lambda: custom_settings

    with TestClient(app) as client:
        # First 3 requests should pass
        r1 = client.get("/v1/models")
        assert r1.status_code == 200
        assert r1.headers["X-RateLimit-Remaining"] == "2"

        r2 = client.get("/v1/models")
        assert r2.status_code == 200
        assert r2.headers["X-RateLimit-Remaining"] == "1"

        r3 = client.get("/v1/models")
        assert r3.status_code == 200
        assert r3.headers["X-RateLimit-Remaining"] == "0"

        # 4th request must be rejected with 429
        r4 = client.get("/v1/models")
        assert r4.status_code == 429
        assert "Retry-After" in r4.headers
        assert "Rate limit exceeded" in r4.json()["detail"]

    app.dependency_overrides.clear()


def test_chat_completions_non_streaming() -> None:
    """Verify non-streaming chat completion proxying."""
    expected_response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "Latency measures delay."}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "Qwen/Qwen2.5-7B-Instruct-AWQ"
        assert not body.get("stream", False)
        return httpx.Response(200, json=expected_response)

    mock_client = make_mock_client(mock_handler)
    app.dependency_overrides[get_client] = lambda: mock_client

    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert resp.json() == expected_response

    app.dependency_overrides.clear()


def test_chat_completions_streaming_sse() -> None:
    """Verify SSE streaming chat completion proxying."""
    sse_chunks = [
        b"data: {\"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}\n\n",
        b"data: {\"choices\": [{\"delta\": {\"content\": \" world\"}}]}\n\n",
        b"data: [DONE]\n\n",
    ]

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body.get("stream") is True

        async def chunk_stream() -> AsyncGenerator[bytes, None]:
            for chunk in sse_chunks:
                yield chunk

        return httpx.Response(200, content=chunk_stream(), headers={"content-type": "text/event-stream"})

    mock_client = make_mock_client(mock_handler)
    app.dependency_overrides[get_client] = lambda: mock_client

    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [{"role": "user", "content": "Stream test"}],
        "stream": True,
    }

    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body_text = resp.text
        assert "Hello" in body_text
        assert "world" in body_text
        assert "[DONE]" in body_text

    app.dependency_overrides.clear()


def test_chat_completions_upstream_error() -> None:
    """Verify upstream connection errors return 503."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("Upstream timed out")

    mock_client = make_mock_client(mock_handler)
    app.dependency_overrides[get_client] = lambda: mock_client

    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [{"role": "user", "content": "Timeout test"}],
    }

    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 503
        assert "Upstream vLLM error" in resp.json()["detail"]

    app.dependency_overrides.clear()


def test_chat_completions_malformed_json() -> None:
    """Verify malformed body returns 400."""
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            content="not a valid json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        assert "Malformed JSON" in resp.json()["detail"]
