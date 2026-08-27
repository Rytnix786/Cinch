# Milestone 13: Full Observability Stack (Prometheus, Grafana, OpenTelemetry)

This document details the architecture, metric specifications, OpenTelemetry span hierarchy, and Grafana dashboard layout for the Cinch LLM Serving Platform.

---

## 1. Observability Architecture

The Cinch Gateway acts as the central telemetry aggregation and trace-propagation hub:

```
[ Client Ingress ]
        | (Injects / Propagates W3C `traceparent` header)
        v
+-----------------------------------------------------------------------------------------+
| Cinch Gateway (FastAPI + OpenTelemetry + Prometheus Instrumentation)                     |
|                                                                                         |
|  1. Ingress Span       -> [ Track Request Counters, Token Estimates, Rate Limits ]      |
|  2. Queue Span         -> [ Track Queue Depth & Wait Duration ]                         |
|  3. Dispatch Span      -> [ Observe Time-To-First-Token (TTFT) & TPOT Latency Buckets ] |
|  4. Prometheus (/metrics) -> Standard exposition (text/plain; version=0.0.4)            |
+-----------------------------------------------------------------------------------------+
        |                                                 |
        v                                                 v
[ Prometheus Scraper (k8s ConfigMap) ]         [ Grafana Dashboard (Panel Visualizer) ]
```

---

## 2. Prometheus Metrics Catalog

The gateway exposes operational metrics on `/metrics`:

### Metric Definitions & Dimensions

| Metric Name | Type | Dimensions / Labels | Description |
|---|---|---|---|
| `cinch_requests_total` | Counter | `status`, `endpoint` | Total HTTP requests handled by status code and path. |
| `cinch_tokens_total` | Counter | `type`, `model` | Cumulative tokens processed (prompt estimates and completions). |
| `cinch_request_duration_seconds` | Histogram | `endpoint`, `le` | Total request latency distribution across 11 predefined buckets ($0.025\text{s} \to 60.0\text{s}$). |
| `cinch_time_to_first_token_seconds` | Histogram | `model`, `le` | Time-To-First-Token for streaming chat completions. |
| `cinch_queue_depth` | Gauge | `priority` | Real-time pending requests buffered in the priority queue. |
| `cinch_active_gpu_slots` | Gauge | — | Real-time concurrent GPU execution slots utilized. |
| `cinch_prefix_cache_hits_total` | Counter | `model` | Total vLLM PagedAttention prefix cache hits. |
| `cinch_prefix_cache_misses_total` | Counter | `model` | Total vLLM PagedAttention prefix cache misses. |

---

## 3. OpenTelemetry Distributed Tracing

Every request propagates a W3C trace context:
- Header: `traceparent: 00-{trace_id}-{span_id}-01`
- Format:
  - Version: `00`
  - Trace ID: 32-character hexadecimal string
  - Span ID: 16-character hexadecimal string
  - Trace Flags: `01` (Recorded / Sampled)

Spans capture critical LLM serving milestones:
- `gateway.ingress`: Auth, IP rate checking, and token cost budgeting.
- `gateway.priority_queue`: Time spent waiting for an available GPU slot.
- `gateway.upstream_dispatch`: Time to first token stream chunk (TTFT) and decode stream assembly.

---

## 4. Kubernetes Manifests & Grafana Dashboard

### Scrape Configuration ([k8s/observability/prometheus-config.yaml](file:///H:/Projects/Cinch/k8s/observability/prometheus-config.yaml))
Configures $5\text{-second}$ scrape intervals targeting `cinch-gateway.cinch.svc.cluster.local:8080/metrics`.

### Dashboard Visualizations ([k8s/observability/grafana-dashboard.json](file:///H:/Projects/Cinch/k8s/observability/grafana-dashboard.json))
The Grafana dashboard contains 5 dedicated operational panels:
1. **Inference Request Throughput (RPS)**: `sum(rate(cinch_requests_total[1m])) by (status)`
2. **Token Throughput (Tokens/sec)**: `sum(rate(cinch_tokens_total[1m])) by (type)`
3. **Prefix Cache Hit Ratio (%)**: Gauge derived from hits vs misses.
4. **TTFT Latency Percentiles (p50 / p95 / p99)**: `histogram_quantile(0.95, sum(rate(cinch_time_to_first_token_seconds_bucket[1m])) by (le))`
5. **Priority Queue Depth & Active GPU Slots**: Real-time timeseries of queue buffers and GPU concurrency.
