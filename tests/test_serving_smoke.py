"""Unit and contract tests for the vLLM smoke test client and verification suite."""

from __future__ import annotations

import httpx
import pytest

from scripts.smoke_test import VLLMSmokeClient, main, parse_args


# ---------------------------------------------------------------------------
# Fixtures & Mock Responses
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_models_payload() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                "object": "model",
                "created": 1700000000,
                "owned_by": "vllm",
            }
        ],
    }


@pytest.fixture
def mock_completion_payload() -> dict:
    return {
        "id": "cmpl-123456",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Latency measures the delay in processing a single request, whereas throughput measures the total requests processed over time.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 22,
            "total_tokens": 37,
        },
    }


# ---------------------------------------------------------------------------
# Health Check Tests
# ---------------------------------------------------------------------------


def test_check_health_success():
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"status": "ok"}))
    client = httpx.Client(transport=transport, base_url="http://testserver")
    smoke = VLLMSmokeClient(base_url="http://testserver", client=client)

    result = smoke.check_health()
    assert result.is_healthy is True
    assert result.status_code == 200
    assert result.details == "OK"


def test_check_health_non_200():
    transport = httpx.MockTransport(lambda req: httpx.Response(503, text="Service Unavailable"))
    client = httpx.Client(transport=transport, base_url="http://testserver")
    smoke = VLLMSmokeClient(base_url="http://testserver", client=client)

    result = smoke.check_health()
    assert result.is_healthy is False
    assert result.status_code == 503
    assert "Unexpected status code: 503" in result.details


def test_check_health_connection_error():
    def raise_connect_error(req):
        raise httpx.ConnectError("Connection refused")

    transport = httpx.MockTransport(raise_connect_error)
    client = httpx.Client(transport=transport, base_url="http://testserver")
    smoke = VLLMSmokeClient(base_url="http://testserver", client=client)

    result = smoke.check_health()
    assert result.is_healthy is False
    assert result.status_code == 0
    assert "Connection error" in result.details


# ---------------------------------------------------------------------------
# Model Registry Tests
# ---------------------------------------------------------------------------


def test_get_models_success(mock_models_payload):
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=mock_models_payload))
    client = httpx.Client(transport=transport, base_url="http://testserver")
    smoke = VLLMSmokeClient(base_url="http://testserver", client=client)

    result = smoke.get_models(expected_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    assert result.models == ["Qwen/Qwen2.5-7B-Instruct-AWQ"]
    assert result.has_expected_model is True


def test_get_models_missing_expected(mock_models_payload):
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=mock_models_payload))
    client = httpx.Client(transport=transport, base_url="http://testserver")
    smoke = VLLMSmokeClient(base_url="http://testserver", client=client)

    result = smoke.get_models(expected_model="NonExistentModel")
    assert result.has_expected_model is False


def test_get_models_http_error():
    transport = httpx.MockTransport(lambda req: httpx.Response(500, text="Internal Server Error"))
    client = httpx.Client(transport=transport, base_url="http://testserver")
    smoke = VLLMSmokeClient(base_url="http://testserver", client=client)

    with pytest.raises(httpx.HTTPStatusError):
        smoke.get_models()


# ---------------------------------------------------------------------------
# Chat Completion Tests
# ---------------------------------------------------------------------------


def test_chat_completion_success(mock_completion_payload):
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=mock_completion_payload))
    client = httpx.Client(transport=transport, base_url="http://testserver")
    smoke = VLLMSmokeClient(base_url="http://testserver", client=client)

    result = smoke.chat_completion(prompt="Hello", model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    assert "Latency measures" in result.text
    assert result.model == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert result.prompt_tokens == 15
    assert result.completion_tokens == 22
    assert result.total_tokens == 37
    assert result.latency_seconds >= 0.0


def test_chat_completion_missing_choices():
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"choices": []}))
    client = httpx.Client(transport=transport, base_url="http://testserver")
    smoke = VLLMSmokeClient(base_url="http://testserver", client=client)

    with pytest.raises(ValueError, match="missing choices list"):
        smoke.chat_completion(prompt="Hello")


def test_chat_completion_http_error():
    transport = httpx.MockTransport(lambda req: httpx.Response(422, text="Unprocessable Entity"))
    client = httpx.Client(transport=transport, base_url="http://testserver")
    smoke = VLLMSmokeClient(base_url="http://testserver", client=client)

    with pytest.raises(httpx.HTTPStatusError):
        smoke.chat_completion(prompt="Hello")


# ---------------------------------------------------------------------------
# Full Pipeline (run_all) Tests
# ---------------------------------------------------------------------------


def test_run_all_success(mock_models_payload, mock_completion_payload):
    def router(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        elif request.url.path == "/v1/models":
            return httpx.Response(200, json=mock_models_payload)
        elif request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json=mock_completion_payload)
        return httpx.Response(404)

    transport = httpx.MockTransport(router)
    client = httpx.Client(transport=transport, base_url="http://testserver")
    smoke = VLLMSmokeClient(base_url="http://testserver", client=client)

    summary = smoke.run_all(expected_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    assert summary.success is True
    assert summary.health.is_healthy is True
    assert summary.models is not None
    assert summary.completion is not None
    assert summary.error is None


def test_run_all_health_failure():
    transport = httpx.MockTransport(lambda req: httpx.Response(500))
    client = httpx.Client(transport=transport, base_url="http://testserver")
    smoke = VLLMSmokeClient(base_url="http://testserver", client=client)

    summary = smoke.run_all()
    assert summary.success is False
    assert summary.health.is_healthy is False
    assert "Health check failed" in (summary.error or "")


def test_run_all_model_mismatch(mock_models_payload):
    def router(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        elif request.url.path == "/v1/models":
            return httpx.Response(200, json=mock_models_payload)
        return httpx.Response(404)

    transport = httpx.MockTransport(router)
    client = httpx.Client(transport=transport, base_url="http://testserver")
    smoke = VLLMSmokeClient(base_url="http://testserver", client=client)

    summary = smoke.run_all(expected_model="llama-3-8b")
    assert summary.success is False
    assert "not found in registered models" in (summary.error or "")


def test_run_all_empty_completion(mock_models_payload):
    empty_comp = {
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "choices": [{"message": {"content": "   "}}],
        "usage": {},
    }

    def router(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        elif request.url.path == "/v1/models":
            return httpx.Response(200, json=mock_models_payload)
        elif request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json=empty_comp)
        return httpx.Response(404)

    transport = httpx.MockTransport(router)
    client = httpx.Client(transport=transport, base_url="http://testserver")
    smoke = VLLMSmokeClient(base_url="http://testserver", client=client)

    summary = smoke.run_all(expected_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    assert summary.success is False
    assert "empty output text" in (summary.error or "")


# ---------------------------------------------------------------------------
# CLI Argument Parsing & Entrypoint Tests
# ---------------------------------------------------------------------------


def test_parse_args_defaults():
    args = parse_args([])
    assert args.base_url == "http://localhost:8000"
    assert args.model == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert args.timeout == 60.0
    assert args.max_tokens == 128


def test_main_cli_success(monkeypatch, mock_models_payload, mock_completion_payload):
    def router(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        elif request.url.path == "/v1/models":
            return httpx.Response(200, json=mock_models_payload)
        elif request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json=mock_completion_payload)
        return httpx.Response(404)

    original_init = VLLMSmokeClient.__init__

    def mock_init(self, base_url="http://localhost:8000", timeout=60.0, client=None):
        mock_client = httpx.Client(transport=httpx.MockTransport(router), base_url=base_url)
        original_init(self, base_url=base_url, timeout=timeout, client=mock_client)

    monkeypatch.setattr(VLLMSmokeClient, "__init__", mock_init)
    exit_code = main(["--base-url", "http://testserver"])
    assert exit_code == 0


def test_main_cli_failure(monkeypatch):
    def router(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    original_init = VLLMSmokeClient.__init__

    def mock_init(self, base_url="http://localhost:8000", timeout=60.0, client=None):
        mock_client = httpx.Client(transport=httpx.MockTransport(router), base_url=base_url)
        original_init(self, base_url=base_url, timeout=timeout, client=mock_client)

    monkeypatch.setattr(VLLMSmokeClient, "__init__", mock_init)
    exit_code = main(["--base-url", "http://testserver"])
    assert exit_code == 1
