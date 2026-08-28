"""Unit test suite for Prometheus metrics, OpenTelemetry tracing, and Grafana dashboard specs."""

from __future__ import annotations

import json
import pathlib
from gateway.telemetry import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    OpenTelemetrySpan,
)


def test_prometheus_counter() -> None:
    """Verify Prometheus counter increments and text exposition format."""
    c = Counter("test_requests_total", "Test counter")
    c.inc(1.0, labels={"status": "200", "model": "qwen"})
    c.inc(2.0, labels={"status": "200", "model": "qwen"})

    text = c.to_prometheus()
    assert "# HELP test_requests_total Test counter" in text
    assert "# TYPE test_requests_total counter" in text
    assert 'test_requests_total{model="qwen",status="200"} 3.0' in text


def test_prometheus_gauge() -> None:
    """Verify Prometheus gauge updates."""
    g = Gauge("test_queue_depth", "Test queue gauge")
    g.set(5.0, labels={"priority": "high"})
    g.set(2.0, labels={"priority": "high"})

    text = g.to_prometheus()
    assert "# TYPE test_queue_depth gauge" in text
    assert 'test_queue_depth{priority="high"} 2.0' in text


def test_prometheus_histogram() -> None:
    """Verify Prometheus histogram bucket aggregation and sum/count."""
    h = Histogram("test_latency_seconds", "Test histogram", buckets=[0.1, 0.5, 1.0])
    h.observe(0.05, labels={"endpoint": "/v1/chat"})
    h.observe(0.3, labels={"endpoint": "/v1/chat"})
    h.observe(0.8, labels={"endpoint": "/v1/chat"})

    text = h.to_prometheus()
    assert "# TYPE test_latency_seconds histogram" in text
    assert 'test_latency_seconds_bucket{endpoint="/v1/chat",le="0.1"} 1' in text
    assert 'test_latency_seconds_bucket{endpoint="/v1/chat",le="0.5"} 2' in text
    assert 'test_latency_seconds_bucket{endpoint="/v1/chat",le="1"} 3' in text
    assert 'test_latency_seconds_bucket{endpoint="/v1/chat",le="+Inf"} 3' in text
    assert 'test_latency_seconds_count{endpoint="/v1/chat"} 3' in text


def test_metrics_registry_generation() -> None:
    """Verify full registry exposition contains core Cinch platform metrics."""
    reg = MetricsRegistry()
    reg.requests_total.inc(1, labels={"status": "200", "endpoint": "/health"})
    reg.tokens_total.inc(50, labels={"type": "prompt", "model": "qwen"})
    reg.ttft.observe(0.15, labels={"model": "qwen"})
    reg.queue_depth.set(0, labels={"priority": "all"})

    text = reg.generate_exposition()
    assert "cinch_requests_total" in text
    assert "cinch_tokens_total" in text
    assert "cinch_time_to_first_token_seconds_bucket" in text
    assert "cinch_queue_depth" in text


def test_opentelemetry_span() -> None:
    """Verify OpenTelemetry span creation and W3C traceparent formatting."""
    span = OpenTelemetrySpan("test_span", trace_id="4bf92f3577b34da6a3ce929d0e0e4736")
    span.set_attribute("model", "qwen2.5")
    duration = span.finish()

    assert duration >= 0.0
    assert span.attributes["model"] == "qwen2.5"
    traceparent = span.get_w3c_traceparent()
    assert traceparent.startswith("00-4bf92f3577b34da6a3ce929d0e0e4736-")
    assert traceparent.endswith("-01")
    assert len(traceparent) == 55


def test_grafana_dashboard_json_validity() -> None:
    """Verify Grafana dashboard manifest exists, parses, and contains essential metrics."""
    dashboard_path = (
        pathlib.Path(__file__).parent.parent / "k8s" / "observability" / "grafana-dashboard.json"
    )
    assert dashboard_path.exists()

    with open(dashboard_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["title"] == "Cinch LLM Platform — Enterprise Observability"
    assert len(data["panels"]) >= 5

    expressions = []
    for panel in data["panels"]:
        for target in panel.get("targets", []):
            expressions.append(target.get("expr", ""))

    all_expr = " ".join(expressions)
    assert "cinch_requests_total" in all_expr
    assert "cinch_tokens_total" in all_expr
    assert "cinch_time_to_first_token_seconds" in all_expr
    assert "cinch_prefix_cache_hits_total" in all_expr
