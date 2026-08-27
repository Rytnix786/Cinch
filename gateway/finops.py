"""Multi-Tenant FinOps Cost Metering Engine (gateway/finops.py).

Provides real-time token-level micro-dollar cost attribution, tenant budget limit enforcement (HTTP 402),
usage accounting APIs, and dollar-denominated Prometheus metrics.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, Optional, Tuple


@dataclasses.dataclass
class CostBreakdown:
    """Cost breakdown for an individual completed inference request."""

    tenant_id: str
    team_id: str
    prompt_tokens: int
    completion_tokens: int
    prompt_cost_usd: float
    completion_cost_usd: float
    total_cost_usd: float
    total_spend_usd: float
    budget_limit_usd: float
    budget_remaining_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "team_id": self.team_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prompt_cost_usd": round(self.prompt_cost_usd, 6),
            "completion_cost_usd": round(self.completion_cost_usd, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_spend_usd": round(self.total_spend_usd, 4),
            "budget_limit_usd": round(self.budget_limit_usd, 2),
            "budget_remaining_usd": round(self.budget_remaining_usd, 4),
        }


@dataclasses.dataclass
class TenantRecord:
    """Ledger record for an individual tenant team."""

    tenant_id: str
    team_id: str
    budget_limit_usd: float
    total_spend_usd: float = 0.0
    request_count: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    created_at: float = dataclasses.field(default_factory=time.time)
    last_active: float = dataclasses.field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        remaining = max(0.0, self.budget_limit_usd - self.total_spend_usd)
        utilization_pct = round(
            (self.total_spend_usd / max(self.budget_limit_usd, 0.0001)) * 100.0, 1
        )
        return {
            "tenant_id": self.tenant_id,
            "team_id": self.team_id,
            "budget_limit_usd": round(self.budget_limit_usd, 2),
            "total_spend_usd": round(self.total_spend_usd, 6),
            "budget_remaining_usd": round(remaining, 6),
            "budget_utilization_pct": utilization_pct,
            "request_count": self.request_count,
            "prompt_tokens_total": self.prompt_tokens_total,
            "completion_tokens_total": self.completion_tokens_total,
            "total_tokens": self.prompt_tokens_total + self.completion_tokens_total,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.created_at)),
            "last_active": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.last_active)),
        }


class FinOpsEngine:
    """
    Real-time multi-tenant cost attribution and budget enforcement engine.
    """

    def __init__(
        self,
        enabled: bool = True,
        default_budget_usd: float = 100.0,
        enforce_budgets: bool = True,
        prompt_rate_per_1k: float = 0.00015,
        completion_rate_per_1k: float = 0.00060,
    ) -> None:
        self.enabled = enabled
        self.default_budget_usd = default_budget_usd
        self.enforce_budgets = enforce_budgets
        self.prompt_rate_per_1k = prompt_rate_per_1k
        self.completion_rate_per_1k = completion_rate_per_1k

        self._tenants: Dict[str, TenantRecord] = {}
        self._total_platform_spend_usd: float = 0.0
        self._total_requests: int = 0
        self._budget_breaches_blocked: int = 0

    def get_or_create_tenant(self, tenant_id: str, team_id: str = "engineering") -> TenantRecord:
        """Fetch tenant record or create with default budget allocation."""
        clean_tenant = tenant_id.strip().lower() or "default"
        clean_team = team_id.strip().lower() or "engineering"

        if clean_tenant not in self._tenants:
            self._tenants[clean_tenant] = TenantRecord(
                tenant_id=clean_tenant,
                team_id=clean_team,
                budget_limit_usd=self.default_budget_usd,
            )
        return self._tenants[clean_tenant]

    def check_budget(self, tenant_id: str) -> Tuple[bool, str, float]:
        """
        Pre-flight check verifying if tenant has remaining spend capacity.

        Returns:
            (is_allowed, reason_message, budget_remaining_usd)
        """
        if not self.enabled or not self.enforce_budgets:
            return True, "FinOps budget enforcement disabled", 999999.0

        tenant = self.get_or_create_tenant(tenant_id)
        remaining = tenant.budget_limit_usd - tenant.total_spend_usd

        if remaining <= 0.0:
            self._budget_breaches_blocked += 1
            reason = (
                f"Tenant '{tenant.tenant_id}' budget limit exceeded: "
                f"${tenant.total_spend_usd:.4f} spend >= ${tenant.budget_limit_usd:.2f} limit"
            )
            return False, reason, 0.0

        return True, "OK", round(remaining, 6)

    def record_usage(
        self,
        tenant_id: str,
        team_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> CostBreakdown:
        """
        Calculate micro-dollar cost and record consumption in tenant ledger.
        """
        tenant = self.get_or_create_tenant(tenant_id, team_id)

        # Micro-dollar cost calculation
        p_cost = (prompt_tokens / 1000.0) * self.prompt_rate_per_1k
        c_cost = (completion_tokens / 1000.0) * self.completion_rate_per_1k
        req_cost = p_cost + c_cost

        tenant.total_spend_usd += req_cost
        tenant.request_count += 1
        tenant.prompt_tokens_total += prompt_tokens
        tenant.completion_tokens_total += completion_tokens
        tenant.last_active = time.time()

        self._total_platform_spend_usd += req_cost
        self._total_requests += 1

        remaining = max(0.0, tenant.budget_limit_usd - tenant.total_spend_usd)

        return CostBreakdown(
            tenant_id=tenant.tenant_id,
            team_id=tenant.team_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_cost_usd=p_cost,
            completion_cost_usd=c_cost,
            total_cost_usd=req_cost,
            total_spend_usd=tenant.total_spend_usd,
            budget_limit_usd=tenant.budget_limit_usd,
            budget_remaining_usd=remaining,
        )

    def set_budget(self, tenant_id: str, budget_limit_usd: float) -> TenantRecord:
        """Dynamically adjust tenant budget limit."""
        tenant = self.get_or_create_tenant(tenant_id)
        tenant.budget_limit_usd = max(0.0, float(budget_limit_usd))
        return tenant

    def get_tenant_usage(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Return usage report for a specific tenant or all registered tenants."""
        if tenant_id:
            tenant = self.get_or_create_tenant(tenant_id)
            return {"tenant": tenant.to_dict()}

        tenants_list = [t.to_dict() for t in self._tenants.values()]
        return {
            "total_tenants": len(self._tenants),
            "total_platform_spend_usd": round(self._total_platform_spend_usd, 6),
            "total_requests": self._total_requests,
            "budget_breaches_blocked": self._budget_breaches_blocked,
            "tenants": sorted(tenants_list, key=lambda x: x["total_spend_usd"], reverse=True),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational metrics for FinOps engine."""
        return {
            "enabled": self.enabled,
            "enforce_budgets": self.enforce_budgets,
            "total_registered_tenants": len(self._tenants),
            "total_platform_spend_usd": round(self._total_platform_spend_usd, 6),
            "total_tracked_requests": self._total_requests,
            "budget_breaches_blocked": self._budget_breaches_blocked,
            "prompt_rate_per_1k": self.prompt_rate_per_1k,
            "completion_rate_per_1k": self.completion_rate_per_1k,
        }
