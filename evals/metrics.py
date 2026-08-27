"""Domain-specific evaluation metric calculators for LLM output quality."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple


def extract_python_code(text: str) -> str:
    """Extract Python code block from markdown fences or return raw text."""
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    return text.strip()


def verify_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Parse Python code using AST to verify syntactic correctness."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as err:
        return False, str(err)


def score_code_item(completion: str, expected_keywords: Optional[List[str]] = None) -> Tuple[float, Dict[str, Any]]:
    """Score code generation item on syntax validity and keyword adherence."""
    code = extract_python_code(completion)
    is_valid_syntax, syntax_err = verify_python_syntax(code)

    if not is_valid_syntax:
        return 0.0, {
            "is_valid_syntax": False,
            "syntax_error": syntax_err,
            "found_keywords": [],
            "extracted_code": code[:200],
        }

    kw_score = 0.0
    found_kws = []
    if expected_keywords:
        for kw in expected_keywords:
            if kw.lower() in completion.lower():
                found_kws.append(kw)
        kw_score = 0.4 * (len(found_kws) / len(expected_keywords))
    else:
        kw_score = 0.4

    total_score = round(0.6 + kw_score, 4)
    details = {
        "is_valid_syntax": True,
        "syntax_error": None,
        "found_keywords": found_kws,
        "extracted_code": code[:200],
    }
    return total_score, details



def score_math_item(completion: str, expected_number: float, tolerance: float = 0.1) -> Tuple[float, Dict[str, Any]]:
    """Score math reasoning item on whether expected numerical value is present."""
    # Find all float / integer numbers in text
    nums = re.findall(r"[-+]?\d*\.?\d+", completion.replace(",", ""))
    found_match = False
    closest_num = None

    for n_str in nums:
        try:
            val = float(n_str)
            if closest_num is None or abs(val - expected_number) < abs(closest_num - expected_number):
                closest_num = val
            if abs(val - expected_number) <= tolerance:
                found_match = True
                break
        except ValueError:
            continue

    score = 1.0 if found_match else 0.0
    details = {
        "expected_number": expected_number,
        "found_match": found_match,
        "closest_number": closest_num,
    }
    return score, details


def score_keyword_item(completion: str, expected_keywords: List[str]) -> Tuple[float, Dict[str, Any]]:
    """Score factual recall on presence of expected conceptual keywords."""
    found = []
    lower_comp = completion.lower()
    for kw in expected_keywords:
        if kw.lower() in lower_comp:
            found.append(kw)

    ratio = (len(found) / len(expected_keywords)) if expected_keywords else 1.0
    score = round(ratio, 4)
    details = {
        "expected_keywords": expected_keywords,
        "found_keywords": found,
        "match_ratio": ratio,
    }
    return score, details


def score_sentence_count_item(completion: str, expected_count: int) -> Tuple[float, Dict[str, Any]]:
    """Score constraint adherence for exact sentence count."""
    cleaned = re.sub(r"\s+", " ", completion.strip())
    # Split sentences by . ! ? followed by space or end of string
    sentences = [s.strip() for s in re.split(r"[.!?]+(?:\s+|$)", cleaned) if s.strip()]
    count = len(sentences)

    if count == expected_count:
        score = 1.0
    elif abs(count - expected_count) == 1:
        score = 0.5
    else:
        score = 0.0

    details = {
        "expected_count": expected_count,
        "actual_count": count,
        "sentences": sentences,
    }
    return score, details


def score_json_validity_item(completion: str, required_keys: List[str]) -> Tuple[float, Dict[str, Any]]:
    """Score constraint adherence for valid JSON format and expected keys."""
    cleaned = completion.strip()
    match = re.search(r"```(?:json)?\s*\n(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    raw_json = match.group(1).strip() if match else cleaned

    try:
        data = json.loads(raw_json)
        is_json = isinstance(data, dict)
        found_keys = [k for k in required_keys if k in data] if is_json else []
        key_ratio = (len(found_keys) / len(required_keys)) if required_keys else 1.0
        score = round(0.5 + 0.5 * key_ratio, 4) if is_json else 0.0
        details = {
            "is_valid_json": True,
            "parsed_keys": list(data.keys()) if is_json else [],
            "found_required_keys": found_keys,
        }
    except Exception as exc:
        score = 0.0
        details = {
            "is_valid_json": False,
            "json_error": str(exc),
        }
    return score, details


def score_bullet_count_item(completion: str, expected_count: int) -> Tuple[float, Dict[str, Any]]:
    """Score constraint adherence for bullet point count."""
    lines = [line.strip() for line in completion.splitlines() if line.strip()]
    bullet_lines = [
        bline for bline in lines if bline.startswith(("*", "-", "•")) or re.match(r"^\d+\.\s+", bline)
    ]
    actual_count = len(bullet_lines)

    if actual_count == expected_count:
        score = 1.0
    elif abs(actual_count - expected_count) == 1:
        score = 0.5
    else:
        score = 0.0

    details = {
        "expected_bullets": expected_count,
        "actual_bullets": actual_count,
        "bullet_lines": bullet_lines,
    }
    return score, details
