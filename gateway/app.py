"""FastAPI Application for Cinch LLM Gateway."""

from __future__ import annotations

import collections
import contextlib
import json
import time
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

from gateway.auth import get_api_key
from gateway.config import GatewaySettings, get_settings, settings
from gateway.limiter import enforce_rate_limit


class GatewayState:
    """State and metrics tracker for the gateway instance."""

    def __init__(self) -> None:
        self.start_time: float = time.time()
        self.total_requests: int = 0
        self.rate_limited_requests: int = 0
        self.error_requests: int = 0
        self.recent_latencies: collections.deque[float] = collections.deque(maxlen=1000)
        self.http_client: Optional[httpx.AsyncClient] = None


gateway_state = GatewayState()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager initializing HTTP client connection pool."""
    gateway_state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.request_timeout_seconds, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
    )
    try:
        yield
    finally:
        if gateway_state.http_client:
            await gateway_state.http_client.aclose()


app = FastAPI(
    title="Cinch LLM Gateway",
    description="Stateless authentication, rate limiting, and request proxying layer for vLLM",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_client() -> httpx.AsyncClient:
    """Dependency to access the pooled upstream HTTP client."""
    if gateway_state.http_client is None:
        raise RuntimeError("Gateway HTTP client is not initialized")
    return gateway_state.http_client


@app.middleware("http")
async def track_metrics_middleware(request: Request, call_next: Any) -> Response:
    """Middleware tracking request counts, latencies, and status metrics."""
    gateway_state.total_requests += 1
    start_time = time.time()
    try:
        response: Response = await call_next(request)
        elapsed = time.time() - start_time
        gateway_state.recent_latencies.append(elapsed)
        if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            gateway_state.rate_limited_requests += 1
        elif response.status_code >= 500:
            gateway_state.error_requests += 1
        return response
    except HTTPException as exc:
        elapsed = time.time() - start_time
        gateway_state.recent_latencies.append(elapsed)
        if exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            gateway_state.rate_limited_requests += 1
        elif exc.status_code >= 500:
            gateway_state.error_requests += 1
        raise
    except Exception:
        gateway_state.error_requests += 1
        raise


@app.get("/health")
async def health_check(
    client: httpx.AsyncClient = Depends(get_client),
    current_settings: GatewaySettings = Depends(get_settings),
) -> JSONResponse:
    """Health check validating gateway status and upstream vLLM reachability."""
    uptime = time.time() - gateway_state.start_time
    vllm_url = f"{current_settings.vllm_base_url.rstrip('/')}/health"
    vllm_status = "unknown"
    is_healthy = True

    try:
        resp = await client.get(vllm_url, timeout=5.0)
        if resp.status_code == 200:
            vllm_status = "ok"
        else:
            vllm_status = f"unhealthy (status {resp.status_code})"
            is_healthy = False
    except Exception as e:
        vllm_status = f"unreachable ({type(e).__name__})"
        is_healthy = False

    payload = {
        "status": "healthy" if is_healthy else "degraded",
        "gateway": "ok",
        "vllm": vllm_status,
        "uptime_seconds": round(uptime, 2),
    }
    status_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=payload, status_code=status_code)


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    """Observability endpoint returning operational counters and latency statistics."""
    latencies = list(gateway_state.recent_latencies)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    uptime = time.time() - gateway_state.start_time

    return {
        "uptime_seconds": round(uptime, 2),
        "total_requests": gateway_state.total_requests,
        "rate_limited_requests": gateway_state.rate_limited_requests,
        "error_requests": gateway_state.error_requests,
        "average_latency_seconds": round(avg_latency, 4),
        "sample_count": len(latencies),
    }


@app.get("/v1/models")
async def list_models(
    client: httpx.AsyncClient = Depends(get_client),
    current_settings: GatewaySettings = Depends(get_settings),
    _api_key: Optional[str] = Depends(get_api_key),
    rate_headers: Dict[str, str] = Depends(enforce_rate_limit),
) -> Response:
    """Authenticated and rate-limited proxy to vLLM model discovery."""
    upstream_url = f"{current_settings.vllm_base_url.rstrip('/')}/v1/models"
    try:
        resp = await client.get(upstream_url)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers={**rate_headers, "Content-Type": resp.headers.get("content-type", "application/json")},
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Upstream vLLM error: {exc}",
        ) from exc


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    client: httpx.AsyncClient = Depends(get_client),
    current_settings: GatewaySettings = Depends(get_settings),
    _api_key: Optional[str] = Depends(get_api_key),
    rate_headers: Dict[str, str] = Depends(enforce_rate_limit),
) -> Response:
    """Authenticated and rate-limited proxy for OpenAI-compatible chat completions.

    Supports both regular JSON responses and chunked Server-Sent Events (SSE) streaming.
    """
    raw_body = await request.body()
    try:
        body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malformed JSON request body",
        ) from exc

    is_streaming = bool(body.get("stream", False))
    upstream_url = f"{current_settings.vllm_base_url.rstrip('/')}/v1/chat/completions"

    if is_streaming:
        async def stream_generator() -> AsyncGenerator[bytes, None]:
            try:
                async with client.stream("POST", upstream_url, json=body) as upstream_resp:
                    if upstream_resp.status_code != 200:
                        error_body = await upstream_resp.aread()
                        yield error_body
                        return
                    async for chunk in upstream_resp.aiter_bytes():
                        yield chunk
            except httpx.RequestError as exc:
                yield f"data: {{\"error\": \"Upstream connection error: {exc}\"}}\n\n".encode("utf-8")

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers=rate_headers,
        )

    # Non-streaming request
    try:
        upstream_resp = await client.post(upstream_url, json=body)
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers={
                **rate_headers,
                "Content-Type": upstream_resp.headers.get("content-type", "application/json"),
            },
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Upstream vLLM error: {exc}",
        ) from exc
