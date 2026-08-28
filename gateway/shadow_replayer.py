"""Production Shadow Traffic Replayer (gateway/shadow_replayer.py).

Provides non-blocking background duplication of live production traffic to an experimental
candidate backend to evaluate latency regressions, token acceptance rates, and semantic divergence.
"""

from __future__ import annotations

import collections
import dataclasses
import random
import re
import time
from typing import Any, Dict, List, Optional
import httpx


@dataclasses.dataclass
class ShadowTraceRecord:
    """Record of a shadow traffic comparison between production and candidate backends."""

    trace_id: str
    model: str
    prompt: str
    prod_status: int
    shadow_status: int
    prod_latency_ms: float
    shadow_latency_ms: float
    latency_delta_ms: float
    prod_tokens: int
    shadow_tokens: int
    token_count_ratio: float
    lexical_similarity_score: float
    divergence_detected: bool
    shadow_error: Optional[str] = None
    timestamp: float = dataclasses.field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "model": self.model,
            "prompt_preview": (self.prompt[:60] + "...") if len(self.prompt) > 60 else self.prompt,
            "prod_status": self.prod_status,
            "shadow_status": self.shadow_status,
            "prod_latency_ms": round(self.prod_latency_ms, 2),
            "shadow_latency_ms": round(self.shadow_latency_ms, 2),
            "latency_delta_ms": round(self.latency_delta_ms, 2),
            "prod_tokens": self.prod_tokens,
            "shadow_tokens": self.shadow_tokens,
            "token_count_ratio": round(self.token_count_ratio, 3),
            "lexical_similarity_score": round(self.lexical_similarity_score, 3),
            "divergence_detected": self.divergence_detected,
            "shadow_error": self.shadow_error,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)),
        }


def compute_lexical_similarity(text1: str, text2: str) -> float:
    """
    Compute lexical token overlap and length consistency between two strings.
    Returns float in range [0.0, 1.0].
    """
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0

    words1 = set(re.findall(r"\w+", text1.lower()))
    words2 = set(re.findall(r"\w+", text2.lower()))

    if not words1 and not words2:
        return 1.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)
    jaccard = intersection / max(union, 1)

    len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2), 1)
    return round(jaccard * (0.5 + 0.5 * len_ratio), 4)


class ShadowTrafficReplayer:
    """
    Asynchronous shadow traffic replayer and divergence analyzer.
    """

    def __init__(
        self,
        enabled: bool = True,
        shadow_backend_url: str = "http://localhost:8000",
        sample_rate: float = 0.10,
        max_traces: int = 100,
    ) -> None:
        self.enabled = enabled
        self.shadow_backend_url = shadow_backend_url.rstrip("/")
        self.sample_rate = sample_rate
        self.max_traces = max_traces

        self._traces: collections.deque[ShadowTraceRecord] = collections.deque(maxlen=max_traces)
        self._total_sampled: int = 0
        self._successful_replays: int = 0
        self._failed_replays: int = 0
        self._divergences_detected: int = 0
        self._total_latency_delta_ms: float = 0.0

    def should_sample(self, force_header: Optional[str] = None) -> bool:
        """Evaluate if incoming request should be mirrored to shadow backend."""
        if not self.enabled:
            return False

        if force_header is not None:
            val = force_header.strip().lower()
            if val in ("true", "1", "yes"):
                return True
            elif val in ("false", "0", "no"):
                return False

        if self.sample_rate <= 0.0:
            return False
        if self.sample_rate >= 1.0:
            return True

        return random.random() < self.sample_rate

    async def replay_shadow(
        self,
        client: httpx.AsyncClient,
        request_body: Dict[str, Any],
        prod_resp_json: Dict[str, Any],
        prod_latency_ms: float,
        prod_status: int,
        api_key: Optional[str] = None,
    ) -> Optional[ShadowTraceRecord]:
        """
        Asynchronously dispatch duplicate request to candidate backend and record divergence metrics.
        """
        self._total_sampled += 1
        trace_id = f"trace_{int(time.time() * 1000)}_{random.randint(100, 999)}"

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        messages = request_body.get("messages", [])
        prompt_text = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        model = request_body.get("model", "unknown")

        shadow_url = f"{self.shadow_backend_url}/v1/chat/completions"
        shadow_status = 500
        shadow_latency_ms = 0.0
        shadow_error = None
        shadow_resp_json: Dict[str, Any] = {}

        t0 = time.perf_counter()
        try:
            resp = await client.post(shadow_url, json=request_body, headers=headers, timeout=15.0)
            shadow_latency_ms = (time.perf_counter() - t0) * 1000.0
            shadow_status = resp.status_code
            if resp.status_code == 200:
                shadow_resp_json = resp.json()
                self._successful_replays += 1
            else:
                self._failed_replays += 1
                shadow_error = f"Shadow HTTP {resp.status_code}"
        except Exception as exc:
            shadow_latency_ms = (time.perf_counter() - t0) * 1000.0
            self._failed_replays += 1
            shadow_error = f"Shadow dispatch error: {exc}"

        # Extract textual contents and token counts
        prod_choices = prod_resp_json.get("choices", [])
        prod_content = prod_choices[0].get("message", {}).get("content", "") if prod_choices else ""
        prod_tokens = prod_resp_json.get("usage", {}).get("completion_tokens", len(prod_content.split()))

        shadow_choices = shadow_resp_json.get("choices", [])
        shadow_content = shadow_choices[0].get("message", {}).get("content", "") if shadow_choices else ""
        shadow_tokens = shadow_resp_json.get("usage", {}).get("completion_tokens", len(shadow_content.split()))

        token_ratio = round(shadow_tokens / max(prod_tokens, 1), 3)
        similarity = compute_lexical_similarity(prod_content, shadow_content)
        lat_delta = shadow_latency_ms - prod_latency_ms

        # Detect divergence: status mismatch, severe length shift, or low similarity
        divergence = (
            (prod_status != shadow_status)
            or (similarity < 0.50 and prod_status == 200 and shadow_status == 200)
            or (token_ratio < 0.40 or token_ratio > 2.50)
        )

        if divergence:
            self._divergences_detected += 1

        self._total_latency_delta_ms += lat_delta

        record = ShadowTraceRecord(
            trace_id=trace_id,
            model=model,
            prompt=prompt_text,
            prod_status=prod_status,
            shadow_status=shadow_status,
            prod_latency_ms=prod_latency_ms,
            shadow_latency_ms=shadow_latency_ms,
            latency_delta_ms=lat_delta,
            prod_tokens=prod_tokens,
            shadow_tokens=shadow_tokens,
            token_count_ratio=token_ratio,
            lexical_similarity_score=similarity,
            divergence_detected=divergence,
            shadow_error=shadow_error,
        )

        self._traces.append(record)
        return record

    def get_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return list of recent shadow comparison trace records."""
        traces = list(self._traces)[-limit:]
        return [t.to_dict() for t in reversed(traces)]

    def set_config(
        self,
        sample_rate: Optional[float] = None,
        shadow_backend_url: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Dynamically update replayer configuration."""
        if sample_rate is not None:
            self.sample_rate = max(0.0, min(1.0, float(sample_rate)))
        if shadow_backend_url is not None:
            self.shadow_backend_url = shadow_backend_url.rstrip("/")
        if enabled is not None:
            self.enabled = bool(enabled)
        return self.get_metrics()

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational summary metrics for shadow traffic replayer."""
        total = max(self._total_sampled, 1)
        avg_delta = self._total_latency_delta_ms / total
        return {
            "enabled": self.enabled,
            "shadow_backend_url": self.shadow_backend_url,
            "sample_rate": self.sample_rate,
            "total_sampled_requests": self._total_sampled,
            "successful_replays": self._successful_replays,
            "failed_replays": self._failed_replays,
            "divergences_detected": self._divergences_detected,
            "divergence_rate_pct": round((self._divergences_detected / total) * 100.0, 1),
            "average_latency_delta_ms": round(avg_delta, 2),
            "retained_traces_count": len(self._traces),
        }
