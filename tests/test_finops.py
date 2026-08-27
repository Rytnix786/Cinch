"""Unit tests for Multi-Tenant FinOps Cost Metering Engine (gateway/finops.py)."""

from __future__ import annotations

from gateway.finops import FinOpsEngine


def test_cost_calculation_accuracy() -> None:
    engine = FinOpsEngine(
        prompt_rate_per_1k=0.00015,
        completion_rate_per_1k=0.00060,
    )

    # 1,000 prompt tokens + 500 completion tokens
    cost_info = engine.record_usage(
        tenant_id="team-nlp",
        team_id="ai-research",
        prompt_tokens=1000,
        completion_tokens=500,
    )

    expected_p = (1000 / 1000.0) * 0.00015  # 0.00015
    expected_c = (500 / 1000.0) * 0.00060   # 0.00030
    expected_tot = expected_p + expected_c  # 0.00045

    assert abs(cost_info.prompt_cost_usd - expected_p) < 1e-7
    assert abs(cost_info.completion_cost_usd - expected_c) < 1e-7
    assert abs(cost_info.total_cost_usd - expected_tot) < 1e-7
    assert abs(cost_info.total_spend_usd - expected_tot) < 1e-7


def test_multi_tenant_isolation() -> None:
    engine = FinOpsEngine(default_budget_usd=50.0)

    engine.record_usage("tenant-a", "sales", 2000, 400)
    engine.record_usage("tenant-b", "marketing", 5000, 1000)

    usage_a = engine.get_tenant_usage("tenant-a")["tenant"]
    usage_b = engine.get_tenant_usage("tenant-b")["tenant"]

    assert usage_a["tenant_id"] == "tenant-a"
    assert usage_a["team_id"] == "sales"
    assert usage_a["prompt_tokens_total"] == 2000
    assert usage_a["completion_tokens_total"] == 400

    assert usage_b["tenant_id"] == "tenant-b"
    assert usage_b["team_id"] == "marketing"
    assert usage_b["prompt_tokens_total"] == 5000
    assert usage_b["completion_tokens_total"] == 1000


def test_budget_enforcement_blocking() -> None:
    engine = FinOpsEngine(
        default_budget_usd=0.0001,
        enforce_budgets=True,
        prompt_rate_per_1k=0.001,
        completion_rate_per_1k=0.001,
    )

    # Initial check passes
    ok, reason, rem = engine.check_budget("budget-test")
    assert ok is True

    # Spend $0.002 (exceeds $0.0001 limit)
    engine.record_usage("budget-test", "dev", 1000, 1000)

    # Next pre-flight check should be blocked
    ok, reason, rem = engine.check_budget("budget-test")
    assert ok is False
    assert "budget limit exceeded" in reason.lower()
    assert rem == 0.0


def test_dynamic_budget_update() -> None:
    engine = FinOpsEngine(default_budget_usd=10.0)

    tenant = engine.get_or_create_tenant("dynamic-team")
    assert tenant.budget_limit_usd == 10.0

    updated = engine.set_budget("dynamic-team", 250.0)
    assert updated.budget_limit_usd == 250.0

    usage = engine.get_tenant_usage("dynamic-team")["tenant"]
    assert usage["budget_limit_usd"] == 250.0


def test_usage_report_structure() -> None:
    engine = FinOpsEngine()
    engine.record_usage("team-x", "core", 500, 200)
    engine.record_usage("team-y", "infra", 1200, 800)

    report = engine.get_tenant_usage()
    assert report["total_tenants"] == 2
    assert report["total_requests"] == 2
    assert report["total_platform_spend_usd"] > 0.0
    assert len(report["tenants"]) == 2


def test_finops_metrics_tracking() -> None:
    engine = FinOpsEngine()
    engine.record_usage("team-m", "ops", 100, 50)

    metrics = engine.get_metrics()
    assert metrics["enabled"] is True
    assert metrics["enforce_budgets"] is True
    assert metrics["total_registered_tenants"] == 1
    assert metrics["total_tracked_requests"] == 1
    assert metrics["total_platform_spend_usd"] > 0.0
