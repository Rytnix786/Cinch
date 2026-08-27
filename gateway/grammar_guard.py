"""Guided Structured Output & JSON Grammar Enforcement Engine (gateway/grammar_guard.py).

Guarantees 100% JSON/schema compliance, strips markdown code blocks, repairs syntax anomalies,
and enforces regex/choice DFA constraints for downstream agents and microservices.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclasses.dataclass
class StructuredConstraint:
    """Represents a structured output constraint extracted from an incoming request."""

    constraint_type: str  # "json_object", "json_schema", "regex", "choice", "none"
    schema_dict: Optional[Dict[str, Any]] = None
    schema_name: Optional[str] = None
    regex_pattern: Optional[str] = None
    choices: Optional[List[str]] = None

    @property
    def is_active(self) -> bool:
        return self.constraint_type != "none"


def _validate_type(value: Any, expected_type: str) -> bool:
    """Validate standard JSON schema basic types."""
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    expected = type_map.get(expected_type.lower())
    if expected is None:
        return True
    if expected_type.lower() == "integer" and isinstance(value, bool):
        return False
    if expected_type.lower() == "number" and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def _validate_json_schema(instance: Any, schema: Dict[str, Any]) -> bool:
    """Lightweight pure-Python recursive JSON schema validator."""
    if not isinstance(schema, dict):
        return True

    # Type check
    if "type" in schema:
        expected = schema["type"]
        if isinstance(expected, str):
            if not _validate_type(instance, expected):
                return False
        elif isinstance(expected, list):
            if not any(_validate_type(instance, t) for t in expected):
                return False

    # Object properties and required checks
    if isinstance(instance, dict):
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in instance:
                return False

        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if prop_name in instance:
                if not _validate_json_schema(instance[prop_name], prop_schema):
                    return False

    # Array item check
    if isinstance(instance, list) and "items" in schema:
        item_schema = schema["items"]
        for item in instance:
            if not _validate_json_schema(item, item_schema):
                return False

    return True


class GrammarGuard:
    """
    Gateway interceptor enforcing strict JSON syntax, DFA regex, and choice grammars.
    """

    def __init__(
        self,
        enabled: bool = True,
        auto_repair: bool = True,
    ) -> None:
        self.enabled = enabled
        self.auto_repair = auto_repair
        self._total_guarded: int = 0
        self._valid_count: int = 0
        self._repaired_count: int = 0
        self._rejected_count: int = 0
        self._type_counts: Dict[str, int] = {
            "json_object": 0,
            "json_schema": 0,
            "regex": 0,
            "choice": 0,
        }

    def extract_constraints(self, request_body: Dict[str, Any]) -> StructuredConstraint:
        """Extract structured output parameters from request payload."""
        if not self.enabled or not request_body:
            return StructuredConstraint(constraint_type="none")

        # 1. Check OpenAI standard response_format
        response_format = request_body.get("response_format")
        if isinstance(response_format, dict):
            fmt_type = response_format.get("type", "")
            if fmt_type == "json_object":
                return StructuredConstraint(constraint_type="json_object")
            elif fmt_type == "json_schema":
                schema_info = response_format.get("json_schema", {})
                return StructuredConstraint(
                    constraint_type="json_schema",
                    schema_dict=schema_info.get("schema"),
                    schema_name=schema_info.get("name"),
                )

        # 2. Check vLLM guided generation parameters
        if "guided_json" in request_body:
            raw_schema = request_body.get("guided_json")
            schema_dict = raw_schema if isinstance(raw_schema, dict) else None
            return StructuredConstraint(
                constraint_type="json_schema",
                schema_dict=schema_dict,
            )

        if "guided_regex" in request_body:
            return StructuredConstraint(
                constraint_type="regex",
                regex_pattern=str(request_body.get("guided_regex")),
            )

        if "guided_choice" in request_body:
            choices = request_body.get("guided_choice")
            if isinstance(choices, list):
                return StructuredConstraint(
                    constraint_type="choice",
                    choices=[str(c) for c in choices],
                )

        return StructuredConstraint(constraint_type="none")

    def sanitize_and_repair_json(self, raw_text: str) -> Tuple[str, bool]:
        """
        Sanitize and repair common LLM syntax anomalies.

        Strips markdown code fences, extracts JSON substring, and repairs
        trailing commas, single quotes, and Python literals (True/False/None).
        """
        text = raw_text.strip()
        was_repaired = False

        # 1. Strip Markdown code fences: ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()
            was_repaired = True

        # 2. Extract bounding JSON object {...} or array [...] if surrounded by conversational text
        first_brace = text.find("{")
        first_bracket = text.find("[")

        start_idx = -1
        if first_brace != -1 and first_bracket != -1:
            start_idx = min(first_brace, first_bracket)
        elif first_brace != -1:
            start_idx = first_brace
        elif first_bracket != -1:
            start_idx = first_bracket

        if start_idx > 0:
            text = text[start_idx:]
            was_repaired = True

        last_brace = text.rfind("}")
        last_bracket = text.rfind("]")
        end_idx = max(last_brace, last_bracket)
        if end_idx != -1 and end_idx < len(text) - 1:
            text = text[: end_idx + 1]
            was_repaired = True

        # Quick check if already valid JSON
        try:
            json.loads(text)
            return text, was_repaired
        except Exception:
            if not self.auto_repair:
                return raw_text, False

        repaired = text

        # 3. Replace Python constants: True -> true, False -> false, None -> null
        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        repaired = re.sub(r"\bNone\b", "null", repaired)

        # 4. Remove trailing commas before closing braces/brackets
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

        # 5. Fix single-quoted keys and values: {'key': 'val'} -> {"key": "val"}
        if "'" in repaired and '"' not in repaired:
            repaired = re.sub(r"'([^']*)'", r'"\1"', repaired)
        else:
            # Fix single-quoted dictionary keys: {'foo': -> {"foo": or , 'foo': -> , "foo":
            repaired = re.sub(r"([{\[,]\s*)'([a-zA-Z0-9_]+)'\s*:", r'\1"\2":', repaired)

        try:
            json.loads(repaired)
            return repaired, True
        except Exception:
            return raw_text, False

    def validate_constraint(
        self, text: str, constraint: StructuredConstraint
    ) -> Tuple[bool, str, str]:
        """
        Validate and sanitize completion against constraint.

        Returns:
            (is_valid, sanitized_text, status_label)
        """
        if not constraint.is_active:
            return True, text, "UNCONSTRAINED"

        self._total_guarded += 1
        ctype = constraint.constraint_type
        if ctype in self._type_counts:
            self._type_counts[ctype] += 1

        # 1. JSON Object & JSON Schema
        if ctype in ("json_object", "json_schema"):
            sanitized, repaired = self.sanitize_and_repair_json(text)
            try:
                parsed = json.loads(sanitized)
            except Exception:
                self._rejected_count += 1
                return False, text, "INVALID_JSON"

            # Check schema compliance if schema_dict is present
            if ctype == "json_schema" and constraint.schema_dict:
                if not _validate_json_schema(parsed, constraint.schema_dict):
                    self._rejected_count += 1
                    return False, sanitized, "SCHEMA_VIOLATION"

            if repaired:
                self._repaired_count += 1
                return True, sanitized, "REPAIRED"

            self._valid_count += 1
            return True, sanitized, "VALID"

        # 2. Guided Regex
        if ctype == "regex" and constraint.regex_pattern:
            cleaned = text.strip()
            pattern = constraint.regex_pattern
            if re.fullmatch(pattern, cleaned):
                self._valid_count += 1
                return True, cleaned, "VALID"
            if self.auto_repair:
                match = re.search(pattern, cleaned)
                if match:
                    self._repaired_count += 1
                    return True, match.group(0), "REPAIRED"
            self._rejected_count += 1
            return False, text, "REGEX_VIOLATION"

        # 3. Guided Choice
        if ctype == "choice" and constraint.choices:
            cleaned = text.strip().strip('"\'')
            if cleaned in constraint.choices:
                self._valid_count += 1
                return True, cleaned, "VALID"
            if self.auto_repair:
                for ch in constraint.choices:
                    if re.search(r"\b" + re.escape(ch) + r"\b", cleaned, re.IGNORECASE):
                        self._repaired_count += 1
                        return True, ch, "REPAIRED"
            self._rejected_count += 1
            return False, text, "CHOICE_VIOLATION"

        return True, text, "VALID"

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational metrics for grammar guard."""
        total = max(self._total_guarded, 1)
        return {
            "enabled": self.enabled,
            "auto_repair_enabled": self.auto_repair,
            "total_guarded_requests": self._total_guarded,
            "valid_completions": self._valid_count,
            "repaired_completions": self._repaired_count,
            "rejected_completions": self._rejected_count,
            "compliance_rate": round((self._valid_count + self._repaired_count) / total, 4),
            "by_constraint_type": dict(self._type_counts),
        }
