"""Fast token counter and estimation utilities for Cinch Gateway."""

from __future__ import annotations

from typing import Any, Dict, List, Union


def estimate_text_tokens(text: Union[str, None]) -> int:
    """Estimate token count for a raw text string.

    Heuristic approximation aligned with modern BPE tokenizers (Qwen / LLaMA / TikToken):
    - ~4 characters per token on average for English / code.
    - Adds word-boundary bias for short phrases.
    """
    if not text:
        return 0
    # Minimum 1 token per non-empty string
    char_estimate = max(1, len(text) // 4)
    word_estimate = max(1, int(len(text.split()) * 1.33))
    return max(char_estimate, word_estimate)


def estimate_chat_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate token footprint for an OpenAI-compatible messages array.

    Includes per-message formatting overhead (~4 tokens for <|im_start|>, role, newline, <|im_end|>).
    """
    if not messages:
        return 0

    total_tokens = 3  # Initial conversation priming overhead
    for msg in messages:
        total_tokens += 4  # Message framing tokens
        content = msg.get("content", "")
        if isinstance(content, str):
            total_tokens += estimate_text_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total_tokens += estimate_text_tokens(part.get("text", ""))
        name = msg.get("name")
        if name:
            total_tokens += estimate_text_tokens(str(name))

    return total_tokens


def estimate_request_tokens(body: Dict[str, Any], default_max_tokens: int = 128) -> int:
    """Compute estimated total token consumption for an inference request.

    Total Cost = Prompt Tokens + Max Output Tokens.
    """
    prompt_tokens = 0
    if "messages" in body and isinstance(body["messages"], list):
        prompt_tokens = estimate_chat_tokens(body["messages"])
    elif "prompt" in body:
        prompt = body["prompt"]
        if isinstance(prompt, str):
            prompt_tokens = estimate_text_tokens(prompt)
        elif isinstance(prompt, list):
            prompt_tokens = sum(estimate_text_tokens(p) for p in prompt if isinstance(p, str))

    max_tokens = body.get("max_tokens")
    if max_tokens is None or not isinstance(max_tokens, int) or max_tokens <= 0:
        max_tokens = default_max_tokens

    return prompt_tokens + max_tokens
