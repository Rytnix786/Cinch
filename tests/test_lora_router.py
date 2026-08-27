"""Unit tests for Multi-LoRA Dynamic Adapter Router (gateway/lora_router.py)."""

from __future__ import annotations

from gateway.lora_router import LoRAAdapterInfo, LoRARouter


def test_parse_compound_model_identifier() -> None:
    router = LoRARouter(default_base_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    base, adapter = router.parse_model_identifier("Qwen/Qwen2.5-7B-Instruct-AWQ:sql-coder")
    assert base == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert adapter == "sql-coder"


def test_parse_bare_adapter_alias() -> None:
    router = LoRARouter(default_base_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    base, adapter = router.parse_model_identifier("sql-coder")
    assert base == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert adapter == "sql-coder"


def test_parse_base_model_only() -> None:
    router = LoRARouter(default_base_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    base, adapter = router.parse_model_identifier("Qwen/Qwen2.5-7B-Instruct-AWQ")
    assert base == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert adapter is None


def test_parse_custom_compound_identifier() -> None:
    router = LoRARouter(default_base_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    base, adapter = router.parse_model_identifier("meta-llama/Llama-3.1-8B-Instruct:custom-lora")
    assert base == "meta-llama/Llama-3.1-8B-Instruct"
    assert adapter == "custom-lora"


def test_resolve_request_transformation() -> None:
    router = LoRARouter(default_base_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    body = {
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ:python-agent",
        "messages": [{"role": "user", "content": "Write a quicksort function"}],
        "temperature": 0.2,
    }
    transformed, adapter_name, base_model = router.resolve_request(body)
    assert base_model == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert adapter_name == "python-agent"
    assert transformed["model"] == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert transformed["lora_adapter"] == "python-agent"
    assert transformed["messages"] == body["messages"]


def test_resolve_request_bare_alias() -> None:
    router = LoRARouter(default_base_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    body = {
        "model": "medical-expert",
        "messages": [{"role": "user", "content": "Explain metformin mechanism"}],
    }
    transformed, adapter_name, base_model = router.resolve_request(body)
    assert base_model == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert adapter_name == "medical-expert"
    assert transformed["model"] == "Qwen/Qwen2.5-7B-Instruct-AWQ"


def test_resolve_request_standard_base_model() -> None:
    router = LoRARouter(default_base_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    body = {
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    transformed, adapter_name, base_model = router.resolve_request(body)
    assert base_model == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert adapter_name is None
    assert transformed["model"] == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert "lora_adapter" not in transformed


def test_synthesize_models_response() -> None:
    router = LoRARouter(default_base_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    upstream_json = {
        "object": "list",
        "data": [
            {
                "id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                "object": "model",
                "created": 1710000000,
                "owned_by": "vllm",
                "root": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                "parent": None,
                "permission": [],
            }
        ],
    }
    synthesized = router.synthesize_models_response(upstream_json)
    ids = [m["id"] for m in synthesized["data"]]

    assert "Qwen/Qwen2.5-7B-Instruct-AWQ" in ids
    assert "Qwen/Qwen2.5-7B-Instruct-AWQ:sql-coder" in ids
    assert "sql-coder" in ids
    assert "python-agent" in ids
    assert "medical-expert" in ids
    assert "legal-analyst" in ids

    # Check structure of a synthesized entry
    sql_compound = next(m for m in synthesized["data"] if m["id"] == "Qwen/Qwen2.5-7B-Instruct-AWQ:sql-coder")
    assert sql_compound["owned_by"] == "cinch-lora-router"
    assert sql_compound["adapter"] == "sql-coder"
    assert sql_compound["rank"] == 16


def test_dynamic_register_and_unregister_adapter() -> None:
    router = LoRARouter(default_base_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    custom_adapter = LoRAAdapterInfo(
        name="finance-analyst",
        base_model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        adapter_path="/models/loras/finance-lora",
        description="SEC filings parsing and DCF model valuation",
        rank=16,
    )
    router.register_adapter(custom_adapter)
    assert router.get_adapter("finance-analyst") is not None

    base, adapter = router.parse_model_identifier("finance-analyst")
    assert adapter == "finance-analyst"

    unreg_ok = router.unregister_adapter("finance-analyst")
    assert unreg_ok is True
    assert router.get_adapter("finance-analyst") is None


def test_get_metrics_tracking() -> None:
    router = LoRARouter(default_base_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    router.resolve_request({"model": "sql-coder"})
    router.resolve_request({"model": "sql-coder"})
    router.resolve_request({"model": "python-agent"})
    router.resolve_request({"model": "Qwen/Qwen2.5-7B-Instruct-AWQ"})

    metrics = router.get_metrics()
    assert metrics["enabled"] is True
    assert metrics["total_routing_requests"] == 4
    assert metrics["total_lora_invocations"] == 3
    assert metrics["invocations_by_adapter"]["sql-coder"] == 2
    assert metrics["invocations_by_adapter"]["python-agent"] == 1
