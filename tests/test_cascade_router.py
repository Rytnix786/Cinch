"""Unit tests for Smart Model Cascading & Complexity Routing (gateway/cascade_router.py)."""

from __future__ import annotations

from gateway.cascade_router import CascadeRouter, CascadeTier


def test_simple_greetings_routed_to_small() -> None:
    router = CascadeRouter(small_model="Qwen/Qwen2.5-0.5B-Instruct", large_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    greetings = ["Hello!", "Hi there", "Good morning", "Hey", "Thank you"]
    for g in greetings:
        analysis = router.analyze_complexity(g)
        assert analysis.tier == CascadeTier.SMALL
        assert analysis.score < 0.40
        assert analysis.selected_model == "Qwen/Qwen2.5-0.5B-Instruct"


def test_text_classification_routed_to_small() -> None:
    router = CascadeRouter(small_model="Qwen/Qwen2.5-0.5B-Instruct", large_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    prompts = [
        "Classify sentiment of this review: 'Great product!'",
        "What is the capital of France?",
        "Translate to French: 'Good evening'",
    ]
    for p in prompts:
        analysis = router.analyze_complexity(p)
        assert analysis.tier == CascadeTier.SMALL
        assert analysis.score < 0.50


def test_python_code_routed_to_large() -> None:
    router = CascadeRouter(small_model="Qwen/Qwen2.5-0.5B-Instruct", large_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    code_prompts = [
        "Write a Python function using async def to fetch data concurrently.",
        "```python\ndef quicksort(arr):\n    pass\n``` Explain how this works.",
        "Implement a Redis-backed token bucket rate limiter in Python.",
    ]
    for p in code_prompts:
        analysis = router.analyze_complexity(p)
        assert analysis.tier == CascadeTier.LARGE
        assert analysis.score >= 0.50
        assert analysis.has_code_syntax or analysis.has_reasoning_keywords
        assert analysis.selected_model == "Qwen/Qwen2.5-7B-Instruct-AWQ"


def test_sql_generation_routed_to_large() -> None:
    router = CascadeRouter(small_model="Qwen/Qwen2.5-0.5B-Instruct", large_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    prompt = "Write a SQL query: SELECT department_id, avg(salary) FROM employees GROUP BY department_id HAVING avg(salary) > 80000"
    analysis = router.analyze_complexity(prompt)
    assert analysis.tier == CascadeTier.LARGE
    assert analysis.score >= 0.50


def test_math_reasoning_routed_to_large() -> None:
    router = CascadeRouter(small_model="Qwen/Qwen2.5-0.5B-Instruct", large_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    prompt = "Derive the mathematical proof for gradient descent convergence on convex functions."
    analysis = router.analyze_complexity(prompt)
    assert analysis.tier == CascadeTier.LARGE
    assert analysis.score >= 0.50
    assert analysis.has_reasoning_keywords is True


def test_length_threshold_routing() -> None:
    router = CascadeRouter(small_model="Qwen/Qwen2.5-0.5B-Instruct", large_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    long_prompt = "Document analysis context: " + ("The quick brown fox jumps over the lazy dog. " * 25)
    analysis = router.analyze_complexity(long_prompt)
    assert analysis.tier == CascadeTier.LARGE
    assert analysis.token_length > 150
    assert analysis.score >= 0.50


def test_structured_schema_escalates_complexity() -> None:
    router = CascadeRouter(small_model="Qwen/Qwen2.5-0.5B-Instruct", large_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    prompt = "Extract employee data."
    analysis_no_schema = router.analyze_complexity(prompt, has_schema=False)
    analysis_with_schema = router.analyze_complexity(prompt, has_schema=True)

    assert analysis_with_schema.score > analysis_no_schema.score


def test_auto_model_resolution() -> None:
    router = CascadeRouter(small_model="Qwen/Qwen2.5-0.5B-Instruct", large_model="Qwen/Qwen2.5-7B-Instruct-AWQ")

    # Simple prompt with model="auto"
    model_resolved, a1 = router.resolve_model("auto", "Hello there!")
    assert model_resolved == "Qwen/Qwen2.5-0.5B-Instruct"
    assert a1.tier == CascadeTier.SMALL

    # Complex prompt with model="auto:cascade"
    model_resolved2, a2 = router.resolve_model("auto:cascade", "Write an algorithm in Python to balance an AVL tree.")
    assert model_resolved2 == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert a2.tier == CascadeTier.LARGE


def test_explicit_model_override() -> None:
    router = CascadeRouter(small_model="Qwen/Qwen2.5-0.5B-Instruct", large_model="Qwen/Qwen2.5-7B-Instruct-AWQ")

    # Even though prompt is simple, explicit model selection is honored
    model_resolved, analysis = router.resolve_model("Qwen/Qwen2.5-7B-Instruct-AWQ", "Hello there!")
    assert model_resolved == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert analysis.tier == CascadeTier.SMALL  # analysis still records true complexity


def test_energy_savings_metric_calculation() -> None:
    router = CascadeRouter(small_model="Qwen/Qwen2.5-0.5B-Instruct", large_model="Qwen/Qwen2.5-7B-Instruct-AWQ")

    # 3 simple requests, 1 complex request
    router.resolve_model("auto", "Hi")
    router.resolve_model("auto", "Good morning")
    router.resolve_model("auto", "What is the capital of Japan?")
    router.resolve_model("auto", "Write a Python script to compute Fibonacci numbers.")

    metrics = router.get_metrics()
    assert metrics["enabled"] is True
    assert metrics["total_routed_requests"] == 4
    assert metrics["small_tier_routed"] == 3
    assert metrics["large_tier_routed"] == 1
    assert metrics["small_tier_ratio"] == 0.75
    # 75% small requests * 93% compute reduction = 69.8%
    assert metrics["estimated_gpu_energy_saved_pct"] > 50.0
