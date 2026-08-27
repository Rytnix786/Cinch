"""Multi-LoRA Dynamic Adapter Router and Model Multiplexer.

Enables compound model resolution (base_model:adapter_name), virtual adapter multiplexing,
and dynamic /v1/models OpenAI-compatible discovery without duplicate VRAM allocations.
"""

from __future__ import annotations

import copy
import dataclasses
import time
from typing import Any, Dict, List, Optional, Tuple


@dataclasses.dataclass
class LoRAAdapterInfo:
    """Metadata describing a registered LoRA adapter module."""

    name: str
    base_model: str
    adapter_path: str
    description: str
    rank: int = 16
    max_lora_rank: int = 32
    created_at: float = dataclasses.field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert adapter info to dictionary."""
        return {
            "name": self.name,
            "base_model": self.base_model,
            "adapter_path": self.adapter_path,
            "description": self.description,
            "rank": self.rank,
            "max_lora_rank": self.max_lora_rank,
            "created_at": self.created_at,
        }


class LoRARouter:
    """
    Router and registry for dynamic Multi-LoRA adapter multiplexing.

    Parses compound model identifiers ('base_model:adapter_name' or bare aliases),
    transforms request payloads for the upstream inference engine, and synthesizes
    virtual LoRA models into /v1/models listings.
    """

    def __init__(
        self,
        default_base_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
        enabled: bool = True,
    ) -> None:
        self.default_base_model = default_base_model
        self.enabled = enabled
        self._adapters: Dict[str, LoRAAdapterInfo] = {}
        self._total_requests: int = 0
        self._adapter_invocations: Dict[str, int] = {}

        # Pre-register standard enterprise domain adapters
        self._register_default_adapters()

    def _register_default_adapters(self) -> None:
        """Register default domain-specific LoRA adapters."""
        defaults = [
            LoRAAdapterInfo(
                name="sql-coder",
                base_model=self.default_base_model,
                adapter_path="/models/loras/sql-coder-lora",
                description="Specialized text-to-SQL generation and query optimization",
                rank=16,
            ),
            LoRAAdapterInfo(
                name="python-agent",
                base_model=self.default_base_model,
                adapter_path="/models/loras/python-agent-lora",
                description="Python coding, AST transformation, and tool execution",
                rank=16,
            ),
            LoRAAdapterInfo(
                name="medical-expert",
                base_model=self.default_base_model,
                adapter_path="/models/loras/medical-expert-lora",
                description="Clinical terminology extraction and biomedical reasoning",
                rank=32,
            ),
            LoRAAdapterInfo(
                name="legal-analyst",
                base_model=self.default_base_model,
                adapter_path="/models/loras/legal-analyst-lora",
                description="Contract analysis, statutory interpretation, and compliance",
                rank=32,
            ),
        ]
        for adapter in defaults:
            self.register_adapter(adapter)

    def register_adapter(self, adapter: LoRAAdapterInfo) -> None:
        """Register a new LoRA adapter into the router."""
        self._adapters[adapter.name] = adapter
        if adapter.name not in self._adapter_invocations:
            self._adapter_invocations[adapter.name] = 0

    def unregister_adapter(self, name: str) -> bool:
        """Remove a LoRA adapter from the router registry."""
        if name in self._adapters:
            del self._adapters[name]
            return True
        return False

    def get_adapter(self, name: str) -> Optional[LoRAAdapterInfo]:
        """Retrieve adapter metadata by name."""
        return self._adapters.get(name)

    def list_adapters(self) -> List[LoRAAdapterInfo]:
        """Return all currently registered LoRA adapters."""
        return list(self._adapters.values())

    def parse_model_identifier(self, model_id: str) -> Tuple[str, Optional[str]]:
        """
        Parse a model identifier into (base_model, adapter_name).

        Handles:
        1. Compound: 'Qwen/Qwen2.5-7B-Instruct-AWQ:sql-coder' -> ('Qwen/Qwen2.5-7B-Instruct-AWQ', 'sql-coder')
        2. Alias: 'sql-coder' -> (default_base_model, 'sql-coder')
        3. Base only: 'Qwen/Qwen2.5-7B-Instruct-AWQ' -> ('Qwen/Qwen2.5-7B-Instruct-AWQ', None)
        """
        if not model_id:
            return self.default_base_model, None

        if ":" in model_id:
            parts = model_id.split(":", 1)
            base_model = parts[0].strip() or self.default_base_model
            adapter_name = parts[1].strip()
            return base_model, adapter_name

        if model_id in self._adapters:
            return self._adapters[model_id].base_model, model_id

        return model_id, None

    def resolve_request(self, body: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str], str]:
        """
        Resolve incoming request payload, resolving virtual LoRA identifiers.

        Returns:
            (transformed_body, adapter_name, base_model)
        """
        self._total_requests += 1
        raw_model = body.get("model", self.default_base_model)
        base_model, adapter_name = self.parse_model_identifier(raw_model)

        transformed_body = copy.copy(body)
        # Point upstream inference engine to the base model weights
        transformed_body["model"] = base_model

        if adapter_name and adapter_name in self._adapters:
            self._adapter_invocations[adapter_name] += 1
            # Propagate adapter metadata for upstream LoRA-aware engines if applicable
            transformed_body["lora_adapter"] = adapter_name

        return transformed_body, adapter_name, base_model

    def synthesize_models_response(self, upstream_models_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize virtual LoRA models into standard OpenAI /v1/models response.

        Combines upstream physical base models with compound virtual models and aliases.
        """
        result = copy.deepcopy(upstream_models_json)
        existing_data = result.get("data", [])
        existing_ids = {m.get("id") for m in existing_data if isinstance(m, dict)}

        synthesized_entries: List[Dict[str, Any]] = []
        created_timestamp = int(time.time())

        for name, adapter in self._adapters.items():
            compound_id = f"{adapter.base_model}:{name}"
            # 1. Add compound model identifier (e.g. Qwen2.5-7B:sql-coder)
            if compound_id not in existing_ids:
                synthesized_entries.append({
                    "id": compound_id,
                    "object": "model",
                    "created": created_timestamp,
                    "owned_by": "cinch-lora-router",
                    "root": adapter.base_model,
                    "parent": adapter.base_model,
                    "adapter": name,
                    "description": adapter.description,
                    "rank": adapter.rank,
                    "permission": [
                        {
                            "id": f"modelperm-{compound_id}",
                            "object": "model_permission",
                            "created": created_timestamp,
                            "allow_create_engine": False,
                            "allow_sampling": True,
                            "allow_logprobs": True,
                            "allow_search_indices": False,
                            "allow_view": True,
                            "allow_fine_tuning": False,
                            "organization": "*",
                            "group": None,
                            "is_blocking": False,
                        }
                    ],
                })
                existing_ids.add(compound_id)

            # 2. Add bare alias (e.g. sql-coder)
            if name not in existing_ids:
                synthesized_entries.append({
                    "id": name,
                    "object": "model",
                    "created": created_timestamp,
                    "owned_by": "cinch-lora-router",
                    "root": adapter.base_model,
                    "parent": adapter.base_model,
                    "adapter": name,
                    "description": adapter.description,
                    "rank": adapter.rank,
                    "permission": [
                        {
                            "id": f"modelperm-{name}",
                            "object": "model_permission",
                            "created": created_timestamp,
                            "allow_create_engine": False,
                            "allow_sampling": True,
                            "allow_logprobs": True,
                            "allow_search_indices": False,
                            "allow_view": True,
                            "allow_fine_tuning": False,
                            "organization": "*",
                            "group": None,
                            "is_blocking": False,
                        }
                    ],
                })
                existing_ids.add(name)

        result["data"] = existing_data + synthesized_entries
        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational metrics for LoRA adapter router."""
        total_lora_invocations = sum(self._adapter_invocations.values())
        return {
            "enabled": self.enabled,
            "default_base_model": self.default_base_model,
            "registered_adapters_count": len(self._adapters),
            "registered_adapters": list(self._adapters.keys()),
            "total_routing_requests": self._total_requests,
            "total_lora_invocations": total_lora_invocations,
            "invocations_by_adapter": dict(self._adapter_invocations),
        }
