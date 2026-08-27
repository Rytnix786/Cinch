"""CLI Evaluation runner for scoring model outputs on quality benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional
import httpx

from evals.metrics import (
    score_bullet_count_item,
    score_code_item,
    score_json_validity_item,
    score_keyword_item,
    score_math_item,
    score_sentence_count_item,
)


def load_eval_prompts(prompts_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load evaluation prompt dataset from JSON."""
    if prompts_path is None:
        prompts_path = str(pathlib.Path(__file__).parent / "prompts_quality.json")
    with open(prompts_path, "r", encoding="utf-8") as f:
        return json.load(f)


async def evaluate_single_item(
    client: httpx.AsyncClient,
    target_url: str,
    model_name: str,
    item: Dict[str, Any],
    api_key: Optional[str] = None,
    timeout_seconds: float = 60.0,
) -> Dict[str, Any]:
    """Execute model generation for a single prompt and compute its evaluation score."""
    endpoint = f"{target_url.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "max_tokens": item.get("target_max_tokens", 160),
        "temperature": 0.0,
    }

    start_time = time.perf_counter()
    try:
        resp = await client.post(endpoint, json=payload, headers=headers, timeout=timeout_seconds)
        latency = time.perf_counter() - start_time
        if resp.status_code != 200:
            return {
                "id": item["id"],
                "category": item["category"],
                "eval_type": item["eval_type"],
                "score": 0.0,
                "latency_seconds": round(latency, 4),
                "is_error": True,
                "error_message": resp.text,
                "completion": "",
            }

        data = resp.json()
        completion = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        eval_type = item["eval_type"]
        if eval_type == "code_syntax":
            score, details = score_code_item(completion, item.get("expected_keywords"))
        elif eval_type == "math_numeric":
            score, details = score_math_item(completion, item.get("expected_number", 0.0))
        elif eval_type == "keywords":
            score, details = score_keyword_item(completion, item.get("expected_keywords", []))
        elif eval_type == "sentence_count":
            score, details = score_sentence_count_item(completion, item.get("expected_sentence_count", 2))
        elif eval_type == "json_validity":
            score, details = score_json_validity_item(completion, item.get("required_json_keys", []))
        elif eval_type == "bullet_count":
            score, details = score_bullet_count_item(completion, item.get("expected_bullet_count", 3))
        else:
            score, details = 1.0, {"info": "default_pass"}

        return {
            "id": item["id"],
            "category": item["category"],
            "eval_type": eval_type,
            "score": score,
            "latency_seconds": round(latency, 4),
            "is_error": False,
            "completion": completion,
            "details": details,
        }

    except Exception as exc:
        latency = time.perf_counter() - start_time
        return {
            "id": item["id"],
            "category": item["category"],
            "eval_type": item.get("eval_type", "unknown"),
            "score": 0.0,
            "latency_seconds": round(latency, 4),
            "is_error": True,
            "error_message": str(exc),
            "completion": "",
        }


async def run_quality_evaluation(
    target_url: str = "http://localhost:8000",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
    prompts_path: Optional[str] = None,
    api_key: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the complete quality evaluation suite across all held-out prompts."""
    prompts = load_eval_prompts(prompts_path)
    print(f"=== Running Model Quality Evaluation against {target_url} ===")
    print(f"Model: {model_name} | Total Prompts: {len(prompts)}\n")

    results: List[Dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        for item in prompts:
            print(f"--> Evaluating '{item['id']}' ({item['category']})... ", end="", flush=True)
            res = await evaluate_single_item(
                client=client,
                target_url=target_url,
                model_name=model_name,
                item=item,
                api_key=api_key,
            )
            results.append(res)
            status_tag = f"Score: {res['score']:.2f}" if not res["is_error"] else f"ERROR: {res['error_message']}"
            print(f"{status_tag} ({res['latency_seconds']:.2f}s)")

    # Aggregate by category
    categories: Dict[str, List[float]] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r["score"])

    category_scores = {
        cat: round(sum(scores) / len(scores), 4) for cat, scores in categories.items()
    }
    overall_quality_score = round(sum(r["score"] for r in results) / max(1, len(results)), 4)

    summary_payload = {
        "model": model_name,
        "target_url": target_url,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_quality_score": overall_quality_score,
        "category_scores": category_scores,
        "total_prompts": len(results),
        "successful_evaluations": len([r for r in results if not r["is_error"]]),
        "items": results,
    }

    print("\n=== Quality Evaluation Summary ===")
    print(f"Overall Quality Index: {overall_quality_score * 100:.1f}%\n")
    for cat, score in category_scores.items():
        print(f"  - {cat.capitalize():<12}: {score * 100:.1f}%")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"\n[SAVED] Quality evaluation report saved to {output_path}")

    return summary_payload


def main() -> None:
    """CLI entrypoint for running quality evaluations."""
    parser = argparse.ArgumentParser(description="Cinch Model Quality & Quantization Evaluation")
    parser.add_argument("--target-url", type=str, default="http://localhost:8000", help="Target API base URL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct-AWQ", help="Model name")
    parser.add_argument("--prompts", type=str, default=None, help="Path to prompts JSON")
    parser.add_argument("--api-key", type=str, default=None, help="API Key if required")
    parser.add_argument("--output", type=str, default="benchmarks/results/quality_eval.json", help="Output JSON path")
    args = parser.parse_args()

    asyncio.run(
        run_quality_evaluation(
            target_url=args.target_url,
            model_name=args.model,
            prompts_path=args.prompts,
            api_key=args.api_key,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
