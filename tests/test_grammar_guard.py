"""Unit tests for Guided Structured Output & JSON Grammar Enforcement (gateway/grammar_guard.py)."""

from __future__ import annotations

import json
from gateway.grammar_guard import GrammarGuard, StructuredConstraint


def test_extract_constraints_none() -> None:
    guard = GrammarGuard()
    constraint = guard.extract_constraints({"messages": [{"role": "user", "content": "hello"}]})
    assert not constraint.is_active
    assert constraint.constraint_type == "none"


def test_extract_constraints_response_format_json_object() -> None:
    guard = GrammarGuard()
    body = {"response_format": {"type": "json_object"}}
    constraint = guard.extract_constraints(body)
    assert constraint.is_active
    assert constraint.constraint_type == "json_object"


def test_extract_constraints_response_format_json_schema() -> None:
    guard = GrammarGuard()
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    body = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "person_schema", "schema": schema},
        }
    }
    constraint = guard.extract_constraints(body)
    assert constraint.is_active
    assert constraint.constraint_type == "json_schema"
    assert constraint.schema_name == "person_schema"
    assert constraint.schema_dict == schema


def test_extract_constraints_guided_regex_and_choice() -> None:
    guard = GrammarGuard()

    # Regex
    body_regex = {"guided_regex": r"[A-Z]{3}-\d{4}"}
    c_regex = guard.extract_constraints(body_regex)
    assert c_regex.constraint_type == "regex"
    assert c_regex.regex_pattern == r"[A-Z]{3}-\d{4}"

    # Choice
    body_choice = {"guided_choice": ["RED", "GREEN", "BLUE"]}
    c_choice = guard.extract_constraints(body_choice)
    assert c_choice.constraint_type == "choice"
    assert c_choice.choices == ["RED", "GREEN", "BLUE"]


def test_strip_markdown_code_fences() -> None:
    guard = GrammarGuard()
    raw = 'Here is your JSON output:\n```json\n{\n  "status": "ok",\n  "count": 42\n}\n```\nHope that helps!'
    sanitized, repaired = guard.sanitize_and_repair_json(raw)
    assert repaired is True
    parsed = json.loads(sanitized)
    assert parsed == {"status": "ok", "count": 42}


def test_repair_trailing_commas() -> None:
    guard = GrammarGuard()
    raw = '{"name": "Alice", "skills": ["python", "k8s",], "active": true,}'
    sanitized, repaired = guard.sanitize_and_repair_json(raw)
    assert repaired is True
    parsed = json.loads(sanitized)
    assert parsed["name"] == "Alice"
    assert parsed["skills"] == ["python", "k8s"]
    assert parsed["active"] is True


def test_repair_single_quotes_and_python_literals() -> None:
    guard = GrammarGuard()
    raw = "{'status': 'success', 'code': 200, 'verified': True, 'details': None}"
    sanitized, repaired = guard.sanitize_and_repair_json(raw)
    assert repaired is True
    parsed = json.loads(sanitized)
    assert parsed == {"status": "success", "code": 200, "verified": True, "details": None}


def test_json_schema_validation_success() -> None:
    guard = GrammarGuard()
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "views": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "views"],
    }
    constraint = StructuredConstraint(constraint_type="json_schema", schema_dict=schema)

    raw = '```json\n{"title": "Cinch Guide", "views": 1500, "tags": ["llm", "k8s"]}\n```'
    is_valid, sanitized, status = guard.validate_constraint(raw, constraint)
    assert is_valid is True
    assert status == "REPAIRED"
    assert json.loads(sanitized)["title"] == "Cinch Guide"


def test_json_schema_validation_type_mismatch() -> None:
    guard = GrammarGuard()
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }
    constraint = StructuredConstraint(constraint_type="json_schema", schema_dict=schema)

    # count is string instead of integer
    raw = '{"count": "forty-two"}'
    is_valid, _, status = guard.validate_constraint(raw, constraint)
    assert is_valid is False
    assert status == "SCHEMA_VIOLATION"


def test_regex_and_choice_validation() -> None:
    guard = GrammarGuard()

    # Regex test
    c_regex = StructuredConstraint(constraint_type="regex", regex_pattern=r"^[A-Z]{3}-\d{4}$")
    ok_res, _, st1 = guard.validate_constraint("ABC-1234", c_regex)
    assert ok_res is True
    assert st1 == "VALID"

    bad_res, _, st2 = guard.validate_constraint("abc-1234-xyz", c_regex)
    assert bad_res is False
    assert st2 == "REGEX_VIOLATION"

    # Choice test
    c_choice = StructuredConstraint(constraint_type="choice", choices=["LOW", "MEDIUM", "HIGH"])
    ok_ch, _, st3 = guard.validate_constraint('"HIGH"', c_choice)
    assert ok_ch is True
    assert st3 == "VALID"

    bad_ch, _, st4 = guard.validate_constraint("EXTREME", c_choice)
    assert bad_ch is False
    assert st4 == "CHOICE_VIOLATION"


def test_metrics_tracking() -> None:
    guard = GrammarGuard()
    c_json = StructuredConstraint(constraint_type="json_object")

    guard.validate_constraint('{"valid": true}', c_json)
    guard.validate_constraint('```json\n{"repaired": true,}\n```', c_json)
    guard.validate_constraint("not a json at all", c_json)

    metrics = guard.get_metrics()
    assert metrics["total_guarded_requests"] == 3
    assert metrics["valid_completions"] == 1
    assert metrics["repaired_completions"] == 1
    assert metrics["rejected_completions"] == 1
    assert metrics["compliance_rate"] == 0.6667
