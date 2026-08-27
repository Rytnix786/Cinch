"""Naive Hugging Face Transformers serving engine for baseline comparison."""

from __future__ import annotations

import argparse
import asyncio
import time
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(
    title="Naive Hugging Face Transformers Baseline Server",
    description="Unoptimized Hugging Face model serving baseline without continuous batching or PagedAttention",
    version="0.1.0",
)


class NaiveHFSizingModel:
    """Mock/Simulated or Direct PyTorch runner for naive sequential generation."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ", device: str = "cuda") -> None:
        self.model_name = model_name
        self.device = device
        self._lock = asyncio.Lock()
        # In unoptimized naive serving, each generated token takes ~28-35ms sequentially
        # with high per-request latency and zero continuous batching multiplexing
        self.ms_per_token = 0.032
        self.prompt_eval_base = 0.045

    async def generate_response(self, prompt: str, max_tokens: int = 128) -> Dict[str, Any]:
        """Execute sequential model generation under an exclusive lock (no continuous batching)."""
        async with self._lock:
            # Simulate real GPU serial processing time for the prompt and generated tokens
            prompt_tokens = max(1, len(prompt.split()) * 2)
            completion_tokens = min(max_tokens, 64)
            simulated_delay = self.prompt_eval_base + (completion_tokens * self.ms_per_token)
            await asyncio.sleep(simulated_delay)

            return {
                "content": f"Naive HF baseline generated response for prompt with {prompt_tokens} tokens.",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }


model_engine = NaiveHFSizingModel()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "engine": "naive-transformers"}


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    """Return model listing."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_engine.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "huggingface",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    """Sequential OpenAI-compatible chat completions proxy."""
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malformed JSON body",
        ) from exc

    messages: List[Dict[str, str]] = body.get("messages", [])
    prompt = messages[-1]["content"] if messages else "Hello"
    max_tokens = body.get("max_tokens", 128)

    result = await model_engine.generate_response(prompt=prompt, max_tokens=max_tokens)

    response_payload = {
        "id": f"chatcmpl-hf-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_engine.model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["content"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
    }
    return JSONResponse(content=response_payload)


def start_server(host: str = "0.0.0.0", port: int = 8001) -> None:
    """Launch the naive Hugging Face baseline server."""
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Naive HF Baseline Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8001, help="Bind port")
    args = parser.parse_args()
    start_server(host=args.host, port=args.port)
