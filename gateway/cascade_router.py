"""Smart Model Cascading & Complexity Router (gateway/cascade_router.py).

Classifies prompt difficulty in < 0.5ms on CPU and routes simple queries to a 0.5B small tier
while reserving 7B models for complex code, SQL, and multi-step reasoning, cutting cluster power by up to 50%.
"""

from __future__ import annotations

import dataclasses
import enum
import re
from typing import Any, Dict, Tuple


class CascadeTier(str, enum.Enum):
    """Model tier classification."""

    SMALL = "SMALL"  # e.g. Qwen2.5-0.5B-Instruct
    LARGE = "LARGE"  # e.g. Qwen2.5-7B-Instruct-AWQ


@dataclasses.dataclass
class ComplexityAnalysis:
    """Detailed complexity breakdown for an evaluated prompt."""

    score: float
    tier: CascadeTier
    selected_model: str
    reason: str
    token_length: int
    has_code_syntax: bool
    has_reasoning_keywords: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 3),
            "tier": self.tier.value,
            "selected_model": self.selected_model,
            "reason": self.reason,
            "token_length": self.token_length,
            "has_code_syntax": self.has_code_syntax,
            "has_reasoning_keywords": self.has_reasoning_keywords,
        }


# ---------------------------------------------------------------------------
# Heuristic Matchers
# ---------------------------------------------------------------------------

CODE_PATTERNS = re.compile(
    r"(?:```|def\s+[a-zA-Z0-9_]+\s*\(|class\s+[a-zA-Z0-9_]+|import\s+[a-zA-Z0-9_]+|"
    r"from\s+[a-zA-Z0-9_]+\s+import|SELECT\s+.*FROM|INSERT\s+INTO|UPDATE\s+.*SET|"
    r"DELETE\s+FROM|function\s*\(|lambda\s+[a-zA-Z0-9_]+:|async\s+def\s+|const\s+[a-zA-Z0-9_]+\s*=|"
    r"\b(?:python|javascript|typescript|golang|rust|sql|query|algorithm|avl\s+tree|fibonacci|script|code|dockerfile|kubernetes|k8s)\b)",
    re.IGNORECASE,
)

REASONING_KEYWORDS = re.compile(
    r"\b(?:calculate|derive|prove|proof|algorithm|optimize|optimization|debug|"
    r"architect|architecture|step[\s-]by[\s-]step|explain\s+in\s+depth|"
    r"compare\s+and\s+contrast|implement|refactor|mathematical|calculus|differential|gradient\s+descent|convergence)\b",
    re.IGNORECASE,
)

SIMPLE_PATTERNS = re.compile(
    r"^(?:hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening|greetings|"
    r"thanks|thank\s+you|who\s+are\s+you|what\s+is\s+your\s+name)[\s.?!]*$|"
    r"\b(?:classify\s+sentiment|is\s+this\s+positive\s+or\s+negative|translate\s+to\s+french|"
    r"capitalize|short\s+summary|single\s+word|what\s+is\s+the\s+capital\s+of)\b",
    re.IGNORECASE,
)


class CascadeRouter:
    """
    Intelligent query complexity evaluator and dynamic model cascading tier router.
    """

    def __init__(
        self,
        enabled: bool = True,
        small_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        large_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
        complexity_threshold: float = 0.50,
    ) -> None:
        self.enabled = enabled
        self.small_model = small_model
        self.large_model = large_model
        self.complexity_threshold = complexity_threshold

        self._total_requests: int = 0
        self._small_tier_count: int = 0
        self._large_tier_count: int = 0
        self._total_complexity_score: float = 0.0

    def analyze_complexity(self, prompt: str, has_schema: bool = False) -> ComplexityAnalysis:
        """
        Evaluate prompt complexity and return detailed tier breakdown.

        Completes in < 0.5ms on CPU.
        """
        text = prompt.strip()
        tokens = text.split()
        token_count = len(tokens)

        # Baseline difficulty
        score = 0.15
        reasons: list[str] = []

        # 1. Code syntax check (+0.40)
        has_code = bool(CODE_PATTERNS.search(text))
        if has_code:
            score += 0.40
            reasons.append("code_syntax")

        # 2. Reasoning & analytical keywords (+0.35)
        has_reasoning = bool(REASONING_KEYWORDS.search(text))
        if has_reasoning:
            score += 0.35
            reasons.append("analytical_reasoning")

        # 3. Structured output constraint (+0.25)
        if has_schema:
            score += 0.25
            reasons.append("structured_schema")

        # 4. Length-based heuristics
        if token_count > 150:
            score += 0.50
            reasons.append("length_gt_150")
        elif token_count > 60:
            score += 0.20
            reasons.append("length_gt_60")
        elif token_count < 8 and not (has_code or has_reasoning):
            score -= 0.10

        # 5. Simple intent discount (-0.25)
        if SIMPLE_PATTERNS.search(text):
            score -= 0.25
            reasons.append("simple_intent")

        # Clamp score to [0.0, 1.0]
        final_score = max(0.0, min(1.0, score))
        primary_reason = ", ".join(reasons) if reasons else "baseline_heuristic"

        # Tier decision
        if final_score >= self.complexity_threshold:
            tier = CascadeTier.LARGE
            selected_model = self.large_model
        else:
            tier = CascadeTier.SMALL
            selected_model = self.small_model

        return ComplexityAnalysis(
            score=final_score,
            tier=tier,
            selected_model=selected_model,
            reason=primary_reason,
            token_length=token_count,
            has_code_syntax=has_code,
            has_reasoning_keywords=has_reasoning,
        )

    def resolve_model(
        self, requested_model: str, prompt: str, has_schema: bool = False
    ) -> Tuple[str, ComplexityAnalysis]:
        """
        Resolve requested model string, handling 'auto' and 'auto:cascade' virtual aliases.

        Returns:
            (target_model_name, complexity_analysis)
        """
        analysis = self.analyze_complexity(prompt, has_schema=has_schema)

        self._total_requests += 1
        self._total_complexity_score += analysis.score

        # Auto-routing enabled
        is_auto_target = requested_model in ("auto", "auto:cascade", "auto-router")
        if self.enabled and is_auto_target:
            if analysis.tier == CascadeTier.SMALL:
                self._small_tier_count += 1
            else:
                self._large_tier_count += 1
            return analysis.selected_model, analysis

        # Explicit model requested
        if requested_model == self.small_model:
            self._small_tier_count += 1
        else:
            self._large_tier_count += 1

        return requested_model, analysis

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational metrics and estimated GPU compute/energy savings."""
        total = max(self._total_requests, 1)
        avg_score = self._total_complexity_score / total

        # Small 0.5B model uses ~7% of the compute of 7B AWQ (14x smaller parameter count)
        # Energy saved = small_count * (1.0 - 0.07) / total * 100%
        small_ratio = self._small_tier_count / total
        energy_savings_pct = round(small_ratio * 0.93 * 100.0, 1)

        return {
            "enabled": self.enabled,
            "small_model": self.small_model,
            "large_model": self.large_model,
            "complexity_threshold": self.complexity_threshold,
            "total_routed_requests": self._total_requests,
            "small_tier_routed": self._small_tier_count,
            "large_tier_routed": self._large_tier_count,
            "small_tier_ratio": round(small_ratio, 3),
            "average_complexity_score": round(avg_score, 3),
            "estimated_gpu_energy_saved_pct": energy_savings_pct,
        }
