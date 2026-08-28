"""Live end-to-end verification script for Cinch FastAPI Gateway."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional
import httpx


def run_gateway_live_tests(
    gateway_url: str = "http://localhost:8080",
    api_key: Optional[str] = None,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
) -> bool:
    """Execute live verification suite against running FastAPI gateway."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print(f"Connecting to Gateway at {gateway_url}...")
    client = httpx.Client(base_url=gateway_url, timeout=30.0)

    # 1. Health check
    try:
        health_resp = client.get("/health")
        if health_resp.status_code != 200:
            print(f"[FAIL] /health returned {health_resp.status_code}: {health_resp.text}")
            return False
        health_data = health_resp.json()
        print(f"Health check: [PASS] (gateway={health_data.get('gateway')}, vllm={health_data.get('vllm')})")
    except Exception as e:
        print(f"[FAIL] Could not connect to gateway: {e}")
        return False

    # 2. Model Discovery
    try:
        models_resp = client.get("/v1/models", headers=headers)
        if models_resp.status_code != 200:
            print(f"[FAIL] /v1/models returned {models_resp.status_code}: {models_resp.text}")
            return False
        models_data = models_resp.json()
        model_ids = [m["id"] for m in models_data.get("data", [])]
        print(f"Models discovered: [PASS] ({model_ids})")
    except Exception as e:
        print(f"[FAIL] /v1/models failed: {e}")
        return False

    # 3. Non-streaming Chat Completion
    prompt = "In one sentence, explain what an API gateway does."
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "temperature": 0.0,
    }

    try:
        start = time.perf_counter()
        chat_resp = client.post("/v1/chat/completions", json=payload, headers=headers)
        latency = time.perf_counter() - start

        if chat_resp.status_code != 200:
            print(f"[FAIL] /v1/chat/completions returned {chat_resp.status_code}: {chat_resp.text}")
            return False

        chat_data = chat_resp.json()
        content = chat_data["choices"][0]["message"]["content"]
        tokens = chat_data.get("usage", {})
        print("Chat completion (non-streaming): [PASS]")
        print(f"  Latency: {latency:.3f}s")
        print(f"  Tokens: prompt={tokens.get('prompt_tokens')}, completion={tokens.get('completion_tokens')}")
        print(f"  Response: {content.strip()}")
    except Exception as e:
        print(f"[FAIL] Chat completion failed: {e}")
        return False

    # 4. Streaming SSE Chat Completion
    stream_payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "max_tokens": 30,
        "temperature": 0.0,
        "stream": True,
    }

    try:
        start = time.perf_counter()
        stream_chunks = []
        with client.stream("POST", "/v1/chat/completions", json=stream_payload, headers=headers) as stream_resp:
            if stream_resp.status_code != 200:
                print(f"[FAIL] SSE stream returned {stream_resp.status_code}")
                return False
            for line in stream_resp.iter_lines():
                if line.startswith("data: ") and not line.endswith("[DONE]"):
                    chunk_json = json.loads(line[6:])
                    delta = chunk_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    stream_chunks.append(delta)
        stream_latency = time.perf_counter() - start
        assembled_stream = "".join(stream_chunks)
        print("Chat completion (streaming SSE): [PASS]")
        print(f"  Latency: {stream_latency:.3f}s")
        print(f"  Streamed text: {assembled_stream.strip()}")
    except Exception as e:
        print(f"[FAIL] Streaming completion failed: {e}")
        return False

    # 5. Metrics Endpoint
    try:
        metrics_resp = client.get("/metrics")
        if metrics_resp.status_code == 200:
            m = metrics_resp.json()
            print(
                f"Gateway metrics: [PASS] (requests={m.get('total_requests')}, avg_latency={m.get('average_latency_seconds')}s)"
            )
    except Exception as e:
        print(f"[WARN] Metrics query failed: {e}")

    print("\n[SUCCESS] Milestone 3 live verification passed: FastAPI Gateway operational.")
    return True


def main() -> None:
    """CLI entrypoint for gateway verification."""
    parser = argparse.ArgumentParser(description="Live test client for Cinch FastAPI Gateway")
    parser.add_argument("--gateway-url", type=str, default="http://localhost:8080", help="Gateway URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key if authentication is enabled")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct-AWQ", help="Model name")
    args = parser.parse_args()

    success = run_gateway_live_tests(
        gateway_url=args.gateway_url,
        api_key=args.api_key,
        model_name=args.model,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
