"""Automated unit test suite for Cinch Quality Evaluation Harness."""

from __future__ import annotations

import pathlib
import tempfile
from typing import Any
from unittest.mock import patch
import pytest
import httpx

from evals.metrics import (
    extract_python_code,
    score_bullet_count_item,
    score_code_item,
    score_json_validity_item,
    score_keyword_item,
    score_math_item,
    score_sentence_count_item,
    verify_python_syntax,
)
from evals.runner import (
    evaluate_single_item,
    load_eval_prompts,
    run_quality_evaluation,
)

_real_async_client = httpx.AsyncClient


def test_extract_python_code() -> None:
    """Verify code block extraction from markdown."""
    fenced = "Here is the code:\n```python\ndef test():\n    return 42\n```\nHope this helps!"
    assert extract_python_code(fenced) == "def test():\n    return 42"

    raw = "def simple(): return 1"
    assert extract_python_code(raw) == "def simple(): return 1"


def test_verify_python_syntax() -> None:
    """Verify AST syntax validation."""
    valid_code = "def add(a, b):\n    return a + b\n"
    is_valid, err = verify_python_syntax(valid_code)
    assert is_valid is True
    assert err is None

    invalid_code = "def broken(a, b\n    return a +"
    is_valid, err = verify_python_syntax(invalid_code)
    assert is_valid is False
    assert err is not None


def test_score_code_item() -> None:
    """Verify code evaluation scoring."""
    valid_completion = "```python\ndef fibonacci(n):\n    return n\n```"
    score, details = score_code_item(valid_completion, ["def fibonacci", "return"])
    assert score == 1.0
    assert details["is_valid_syntax"] is True

    syntax_err_completion = "def fib(n return"
    score, details = score_code_item(syntax_err_completion, ["return"])
    assert score == 0.0
    assert details["is_valid_syntax"] is False


def test_score_math_item() -> None:
    """Verify mathematical numerical matching."""
    completion_correct = "After calculating the distance, the speed is 80 mph."
    score, details = score_math_item(completion_correct, 80.0)
    assert score == 1.0
    assert details["found_match"] is True

    completion_wrong = "The speed is 75.5 mph."
    score, details = score_math_item(completion_wrong, 80.0)
    assert score == 0.0
    assert details["found_match"] is False


def test_score_keyword_item() -> None:
    """Verify keyword recall scoring."""
    completion = "PagedAttention solves fragmentation by organizing the KV cache into virtual memory blocks."
    score, details = score_keyword_item(completion, ["kv cache", "fragmentation", "virtual memory", "blocks"])
    assert score == 1.0
    assert len(details["found_keywords"]) == 4

    partial_completion = "It manages the KV cache efficiently."
    score, details = score_keyword_item(partial_completion, ["kv cache", "fragmentation", "virtual memory", "blocks"])
    assert score == 0.25


def test_score_sentence_count_item() -> None:
    """Verify sentence constraint scoring."""
    two_sentences = "An API Gateway is a reverse proxy for microservices. It handles authentication and rate limiting."
    score, details = score_sentence_count_item(two_sentences, 2)
    assert score == 1.0
    assert details["actual_count"] == 2

    three_sentences = "This is sentence one. This is sentence two! This is sentence three?"
    score, details = score_sentence_count_item(three_sentences, 2)
    assert score == 0.5


def test_score_json_validity_item() -> None:
    """Verify JSON validity scoring."""
    valid_json = '```json\n{"framework": "vLLM", "quantization": "AWQ", "vram_gb": 8}\n```'
    score, details = score_json_validity_item(valid_json, ["framework", "quantization", "vram_gb"])
    assert score == 1.0
    assert details["is_valid_json"] is True

    invalid_json = '{"framework": "vLLM", broken}'
    score, details = score_json_validity_item(invalid_json, ["framework"])
    assert score == 0.0
    assert details["is_valid_json"] is False


def test_score_bullet_count_item() -> None:
    """Verify bullet count constraint scoring."""
    bullets = "- Point 1\n- Point 2\n- Point 3"
    score, details = score_bullet_count_item(bullets, 3)
    assert score == 1.0
    assert details["actual_bullets"] == 3


def test_load_eval_prompts() -> None:
    """Verify loading prompts_quality.json."""
    prompts = load_eval_prompts()
    assert len(prompts) >= 5
    for p in prompts:
        assert "id" in p
        assert "category" in p
        assert "eval_type" in p


@pytest.mark.asyncio
async def test_evaluate_single_item_mock() -> None:
    """Verify single item evaluation mock."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "```python\ndef fibonacci(n):\n    return n\n```"}}]},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        item = {
            "id": "test-code",
            "category": "code",
            "eval_type": "code_syntax",
            "prompt": "Write fibonacci",
            "expected_keywords": ["def fibonacci", "return"],
        }
        res = await evaluate_single_item(
            client=client,
            target_url="http://mock:8000",
            model_name="test-model",
            item=item,
        )
        assert res["score"] == 1.0
        assert res["is_error"] is False


@pytest.mark.asyncio
async def test_run_quality_evaluation_mock() -> None:
    """Verify complete quality evaluation mock run."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "The speed is 80 mph. Also ```python\ndef f(): return 1\n``` and {\"framework\": \"vLLM\"}"}}]},
        )

    transport = httpx.MockTransport(handler)

    def client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        return _real_async_client(transport=transport, **kwargs)

    with patch("evals.runner.httpx.AsyncClient", side_effect=client_factory):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = str(pathlib.Path(tmpdir) / "quality_results.json")
            res = await run_quality_evaluation(
                target_url="http://mock:8000",
                model_name="test-model",
                output_path=out_file,
            )
            assert res["overall_quality_score"] > 0.0
            assert "category_scores" in res
            assert pathlib.Path(out_file).exists()
