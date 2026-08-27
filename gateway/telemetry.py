"""Prometheus metrics collection and OpenTelemetry distributed tracing for Cinch Gateway."""

from __future__ import annotations

import collections
import os
import time
from typing import Any, Dict, List, Optional, Tuple


class Histogram:
    """In-memory Prometheus Histogram metric with predefined latency buckets."""

    def __init__(self, name: str, doc: str, buckets: Optional[List[float]] = None) -> None:
        self.name = name
        self.doc = doc
        self.buckets = buckets if buckets else [0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
        # map label_tuple -> dict of bucket_counts, count, sum
        self._data: Dict[Tuple[Tuple[str, str], ...], Dict[str, Any]] = collections.defaultdict(
            lambda: {
                "bucket_counts": [0] * len(self.buckets),
                "count": 0,
                "sum": 0.0,
            }
        )

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record an observation in the histogram."""
        label_key = tuple(sorted(labels.items())) if labels else ()
        entry = self._data[label_key]
        entry["count"] += 1
        entry["sum"] += value
        for i, b in enumerate(self.buckets):
            if value <= b:
                entry["bucket_counts"][i] += 1

    def to_prometheus(self) -> str:
        """Format histogram data in Prometheus exposition text format."""
        lines = [f"# HELP {self.name} {self.doc}", f"# TYPE {self.name} histogram"]
        for label_key, entry in self._data.items():
            base_labels = [f'{k}="{v}"' for k, v in label_key]
            for i, b in enumerate(self.buckets):
                count_le = entry["bucket_counts"][i]
                b_str = f"{b:g}" if b != float("inf") else "+Inf"
                all_labels = [*base_labels, f'le="{b_str}"']
                lbl_str = f"{{{','.join(all_labels)}}}" if all_labels else ""
                lines.append(f"{self.name}_bucket{lbl_str} {count_le}")
            # +Inf bucket
            inf_labels = [*base_labels, 'le="+Inf"']
            lines.append(f"{self.name}_bucket{{{','.join(inf_labels)}}} {entry['count']}")
            lbl_str = f"{{{','.join(base_labels)}}}" if base_labels else ""
            lines.append(f"{self.name}_sum{lbl_str} {round(entry['sum'], 4)}")
            lines.append(f"{self.name}_count{lbl_str} {entry['count']}")
        return "\n".join(lines)



class Counter:
    """In-memory Prometheus Counter metric."""

    def __init__(self, name: str, doc: str) -> None:
        self.name = name
        self.doc = doc
        self._counts: Dict[Tuple[Tuple[str, str], ...], float] = collections.defaultdict(float)

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter by amount."""
        label_key = tuple(sorted(labels.items())) if labels else ()
        self._counts[label_key] += amount

    def to_prometheus(self) -> str:
        """Format counter data in Prometheus exposition text format."""
        lines = [f"# HELP {self.name} {self.doc}", f"# TYPE {self.name} counter"]
        for label_key, val in self._counts.items():
            if label_key:
                lbl_str = "{" + ",".join(f'{k}="{v}"' for k, v in label_key) + "}"
            else:
                lbl_str = ""
            lines.append(f"{self.name}{lbl_str} {val}")
        return "\n".join(lines)


class Gauge:
    """In-memory Prometheus Gauge metric."""

    def __init__(self, name: str, doc: str) -> None:
        self.name = name
        self.doc = doc
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = collections.defaultdict(float)

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge to value."""
        label_key = tuple(sorted(labels.items())) if labels else ()
        self._values[label_key] = value

    def to_prometheus(self) -> str:
        """Format gauge data in Prometheus exposition text format."""
        lines = [f"# HELP {self.name} {self.doc}", f"# TYPE {self.name} gauge"]
        for label_key, val in self._values.items():
            if label_key:
                lbl_str = "{" + ",".join(f'{k}="{v}"' for k, v in label_key) + "}"
            else:
                lbl_str = ""
            lines.append(f"{self.name}{lbl_str} {val}")
        return "\n".join(lines)


class MetricsRegistry:
    """Central Prometheus Metrics Registry for the Cinch platform."""

    def __init__(self) -> None:
        self.requests_total = Counter("cinch_requests_total", "Total inference requests processed")
        self.tokens_total = Counter("cinch_tokens_total", "Total tokens processed (prompt and completion)")
        self.request_duration = Histogram("cinch_request_duration_seconds", "Total end-to-end request duration")
        self.ttft = Histogram("cinch_time_to_first_token_seconds", "Time-To-First-Token for streaming requests")
        self.queue_depth = Gauge("cinch_queue_depth", "Current number of requests waiting in priority queue")
        self.active_gpu_slots = Gauge("cinch_active_gpu_slots", "Current number of active GPU execution slots")
        self.prefix_cache_hits = Counter("cinch_prefix_cache_hits_total", "Total prefix cache hits")
        self.prefix_cache_misses = Counter("cinch_prefix_cache_misses_total", "Total prefix cache misses")

    def generate_exposition(self) -> str:
        """Generate complete Prometheus exposition text."""
        sections = [
            self.requests_total.to_prometheus(),
            self.tokens_total.to_prometheus(),
            self.request_duration.to_prometheus(),
            self.ttft.to_prometheus(),
            self.queue_depth.to_prometheus(),
            self.active_gpu_slots.to_prometheus(),
            self.prefix_cache_hits.to_prometheus(),
            self.prefix_cache_misses.to_prometheus(),
        ]
        return "\n\n".join(s for s in sections if s) + "\n"


metrics_registry = MetricsRegistry()


# --- OpenTelemetry Distributed Tracing Helpers ---

class OpenTelemetrySpan:
    """Lightweight W3C-compliant OpenTelemetry trace span."""

    def __init__(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ) -> None:
        self.name = name
        self.trace_id = trace_id or os.urandom(16).hex()
        self.span_id = os.urandom(8).hex()
        self.parent_span_id = parent_span_id
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        self.attributes: Dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        """Add attribute metadata to span."""
        self.attributes[key] = value

    def finish(self) -> float:
        """Mark span as finished and return duration in seconds."""
        self.end_time = time.time()
        return self.end_time - self.start_time

    def get_w3c_traceparent(self) -> str:
        """Format W3C standard traceparent header: 00-{trace_id}-{span_id}-01."""
        return f"00-{self.trace_id}-{self.span_id}-01"
