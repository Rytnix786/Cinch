"""Ingress Security Guardrails, Jailbreak Defense & PII Redaction Engine (gateway/guardrails.py).

Provides < 1ms CPU scanning for prompt injection, delimiter exploits, DAN jailbreaks,
bidirectional PII redaction (SSN, credit cards, API keys, phone numbers), and system prompt non-leakage.
"""

from __future__ import annotations

import dataclasses
import re
import time
from typing import Any, Dict, List, Optional, Tuple


@dataclasses.dataclass
class IngressScanResult:
    """Result of an ingress security and PII scan."""

    is_safe: bool
    violation_type: Optional[str] = None  # "PROMPT_INJECTION", "DELIMITER_ATTACK", "JAILBREAK"
    redacted_text: str = ""
    pii_types_found: List[str] = dataclasses.field(default_factory=list)
    scan_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Compiled Security & PII Patterns
# ---------------------------------------------------------------------------

INJECTION_PATTERNS = [
    (
        "PROMPT_INJECTION",
        re.compile(
            r"\b(?:ignore|disregard|bypass|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|rules?|commands?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "PROMPT_INJECTION",
        re.compile(
            r"\b(?:system\s+override|override\s+system\s+prompt|new\s+system\s+instructions?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "JAILBREAK",
        re.compile(
            r"\b(?:you\s+are\s+now\s+(?:dan|unrestricted|god\s+mode)|do\s+anything\s+now|developer\s+mode\s+enabled)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "JAILBREAK",
        re.compile(
            r"\b(?:jailbreak|bypass\s+all\s+(?:safety|filters?|rules?)|disregard\s+ethical\s+guidelines)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "DELIMITER_ATTACK",
        re.compile(
            r"(?:<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>|<\|system\|>)",
            re.IGNORECASE,
        ),
    ),
]

PII_PATTERNS: List[Tuple[str, re.Pattern[str], str]] = [
    (
        "ssn",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED_SSN]",
    ),
    (
        "api_key_openai",
        re.compile(r"\bsk-[a-zA-Z0-9_-]{20,}\b"),
        "[REDACTED_API_KEY]",
    ),
    (
        "api_key_github",
        re.compile(r"\bgh[pousr][-_][a-zA-Z0-9]{20,}\b"),
        "[REDACTED_API_KEY]",
    ),
    (
        "api_key_aws",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "[REDACTED_API_KEY]",
    ),
    (
        "jwt_token",
        re.compile(r"\beyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b"),
        "[REDACTED_API_KEY]",
    ),
    (
        "credit_card",
        re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),
        "[REDACTED_CREDIT_CARD]",
    ),
    (
        "phone_number",
        re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"),
        "[REDACTED_PHONE]",
    ),
]


class GuardrailsScanner:
    """
    Sub-millisecond ingress security and PII redaction engine.
    """

    def __init__(
        self,
        enabled: bool = True,
        injection_defense_enabled: bool = True,
        pii_redaction_enabled: bool = True,
        system_prompt_leak_defense: bool = True,
    ) -> None:
        self.enabled = enabled
        self.injection_defense_enabled = injection_defense_enabled
        self.pii_redaction_enabled = pii_redaction_enabled
        self.system_prompt_leak_defense = system_prompt_leak_defense

        self._total_scans: int = 0
        self._injections_blocked: int = 0
        self._pii_redactions: int = 0
        self._leaks_blocked: int = 0
        self._total_latency_ms: float = 0.0

    def redact_pii(self, text: str) -> Tuple[str, List[str]]:
        """Redact sensitive PII entities from text and return (redacted_text, detected_types)."""
        if not self.pii_redaction_enabled or not text:
            return text, []

        redacted = text
        found_types: List[str] = []

        for pii_name, pattern, placeholder in PII_PATTERNS:
            if pattern.search(redacted):
                redacted = pattern.sub(placeholder, redacted)
                found_types.append(pii_name)

        return redacted, found_types

    def scan_ingress(self, text: str) -> IngressScanResult:
        """
        Scan incoming user prompt for adversarial injections and redact PII.

        Completes in < 1ms on CPU.
        """
        if not self.enabled or not text:
            return IngressScanResult(
                is_safe=True,
                redacted_text=text,
                pii_types_found=[],
                scan_latency_ms=0.0,
            )

        t0 = time.perf_counter()
        self._total_scans += 1

        # 1. Adversarial Prompt Injection & Jailbreak Defense
        if self.injection_defense_enabled:
            for violation_type, pattern in INJECTION_PATTERNS:
                if pattern.search(text):
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    self._total_latency_ms += elapsed_ms
                    self._injections_blocked += 1
                    return IngressScanResult(
                        is_safe=False,
                        violation_type=violation_type,
                        redacted_text=text,
                        pii_types_found=[],
                        scan_latency_ms=round(elapsed_ms, 3),
                    )

        # 2. PII Redaction
        redacted_text, pii_types = self.redact_pii(text)
        if pii_types:
            self._pii_redactions += len(pii_types)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._total_latency_ms += elapsed_ms

        return IngressScanResult(
            is_safe=True,
            violation_type=None,
            redacted_text=redacted_text,
            pii_types_found=pii_types,
            scan_latency_ms=round(elapsed_ms, 3),
        )

    def sanitize_egress(
        self, completion_text: str, system_prompt: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Sanitize generated completion before returning to client.

        Applies egress PII masking and prevents system prompt regurgitation.
        """
        if not self.enabled or not completion_text:
            return completion_text, False

        sanitized = completion_text
        modified = False

        # 1. Egress PII Masking
        if self.pii_redaction_enabled:
            sanitized, pii_found = self.redact_pii(sanitized)
            if pii_found:
                modified = True

        # 2. System Prompt Non-Leakage Filter
        if self.system_prompt_leak_defense and system_prompt and len(system_prompt.strip()) > 20:
            cleaned_sys = system_prompt.strip()
            # If the completion regurgitates significant chunks of the system prompt
            if cleaned_sys in sanitized or (
                len(cleaned_sys) > 50 and cleaned_sys[:50] in sanitized
            ):
                sanitized = re.sub(
                    re.escape(cleaned_sys),
                    "[SYSTEM_INSTRUCTION_REDACTED]",
                    sanitized,
                    flags=re.IGNORECASE,
                )
                self._leaks_blocked += 1
                modified = True

        return sanitized, modified

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational metrics for guardrails scanner."""
        total = max(self._total_scans, 1)
        return {
            "enabled": self.enabled,
            "injection_defense_enabled": self.injection_defense_enabled,
            "pii_redaction_enabled": self.pii_redaction_enabled,
            "system_prompt_leak_defense": self.system_prompt_leak_defense,
            "total_scanned_requests": self._total_scans,
            "injections_blocked": self._injections_blocked,
            "pii_redacted_count": self._pii_redactions,
            "system_prompt_leaks_blocked": self._leaks_blocked,
            "average_scan_latency_ms": round(self._total_latency_ms / total, 3),
        }
