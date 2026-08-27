"""Prefix hashing and cache affinity routing for Cinch FastAPI Gateway."""

from __future__ import annotations

import collections
import hashlib
from typing import Any, Dict, List, Optional, Tuple


def extract_prompt_prefix(body: Dict[str, Any], min_chars: int = 32) -> Tuple[str, str]:
    """Extract static prompt prefix (system message / few-shot header) and compute SHA-256 hash.

    Returns:
        Tuple of (normalized_prefix_text, sha256_hash_digest_16_chars).
        Returns ("", "") if no valid prefix meets min_chars threshold.
    """
    raw_prefix = ""

    if "messages" in body and isinstance(body["messages"], list) and body["messages"]:
        messages = body["messages"]
        # 1. Prefer explicit system message
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    raw_prefix = content
                    break
        # 2. Fallback to first message if long enough
        if not raw_prefix and len(messages) > 0:
            first_msg = messages[0]
            if isinstance(first_msg, dict):
                content = first_msg.get("content", "")
                if isinstance(content, str) and len(content) >= min_chars:
                    raw_prefix = content

    elif "prompt" in body:
        prompt = body["prompt"]
        if isinstance(prompt, str):
            raw_prefix = prompt[:512]
        elif isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
            raw_prefix = prompt[0][:512]

    # Normalize whitespace for robust hash stability
    normalized = " ".join(raw_prefix.split())
    if len(normalized) < min_chars:
        return "", ""

    prefix_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return normalized, prefix_hash


class PrefixCacheRouter:
    """LRU Prefix-Affinity Router mapping shared prompt hashes to backend instances."""

    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = capacity
        # _cache maps prefix_hash -> dict(target_url, hits, timestamp)
        self._cache: collections.OrderedDict[str, Dict[str, Any]] = collections.OrderedDict()
        self._total_hits = 0
        self._total_misses = 0

    def route(
        self,
        prefix_hash: str,
        available_targets: Optional[List[str]] = None,
        default_target: str = "http://localhost:8000",
    ) -> Tuple[str, bool]:
        """Route request by prefix hash to preferred backend target.

        Returns:
            Tuple of (target_url, is_cache_hit).
        """
        targets = available_targets if available_targets else [default_target]

        if not prefix_hash:
            return targets[0], False

        # Cache HIT: Known prefix
        if prefix_hash in self._cache:
            entry = self._cache[prefix_hash]
            entry["hits"] += 1
            self._cache.move_to_end(prefix_hash)
            self._total_hits += 1
            return entry["target_url"], True

        # Cache MISS: New prefix -> Consistent hashing across available targets
        target_idx = int(prefix_hash, 16) % len(targets)
        selected_target = targets[target_idx]

        # Enforce LRU capacity limit
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)

        self._cache[prefix_hash] = {
            "target_url": selected_target,
            "hits": 0,
        }
        self._total_misses += 1
        return selected_target, False

    def reset(self) -> None:
        """Clear cache state and counters."""
        self._cache.clear()
        self._total_hits = 0
        self._total_misses = 0

    def get_metrics(self) -> Dict[str, Any]:
        """Return cache affinity statistics."""
        total_queries = self._total_hits + self._total_misses
        hit_ratio = (self._total_hits / total_queries) if total_queries > 0 else 0.0
        return {
            "cached_prefixes": len(self._cache),
            "max_capacity": self.capacity,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "hit_ratio": round(hit_ratio, 4),
        }
