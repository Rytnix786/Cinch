"""FastAPI Application for Cinch LLM Gateway with Token Limiting, Priority Queues, Prefix Routing, Telemetry & Circuit Breaking."""

from __future__ import annotations

import collections
import contextlib
import json
import time
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
import httpx

from gateway.auth import get_api_key
from gateway.cache_router import PrefixCacheRouter, extract_prompt_prefix
from gateway.circuit_breaker import CircuitBreaker
from gateway.config import GatewaySettings, get_settings, settings
from gateway.limiter import enforce_rate_limit, rate_limiter
from gateway.priority_queue import PriorityRequestQueue, RequestPriority
from gateway.telemetry import OpenTelemetrySpan, metrics_registry
from gateway.token_counter import estimate_request_tokens


class GatewayState:
    """State and metrics tracker for the gateway instance."""

    def __init__(self) -> None:
        self.start_time: float = time.time()
        self.total_requests: int = 0
        self.rate_limited_requests: int = 0
        self.error_requests: int = 0
        self.high_priority_requests: int = 0
        self.low_priority_requests: int = 0
        self.total_tokens_processed: int = 0
        self.recent_latencies: collections.deque[float] = collections.deque(maxlen=1000)
        self.http_client: Optional[httpx.AsyncClient] = None
        self.priority_queue: PriorityRequestQueue = PriorityRequestQueue(
            max_active=settings.max_concurrent_interactive_requests,
            max_queue=settings.max_queue_size,
        )
        self.cache_router: PrefixCacheRouter = PrefixCacheRouter(
            capacity=settings.cache_router_capacity,
        )
        self.circuit_breaker: CircuitBreaker = CircuitBreaker(
            failure_threshold=settings.circuit_failure_threshold,
            recovery_timeout_seconds=settings.circuit_recovery_timeout_seconds,
        )


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
    description="Stateless token rate limiting, priority scheduling, prefix cache routing, circuit breaking, and telemetry for vLLM",
    version="0.5.0",
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
    """Middleware tracking request counts, latencies, and Prometheus status metrics."""
    gateway_state.total_requests += 1
    start_time = time.time()
    endpoint = request.url.path
    status_code = 200
    try:
        response: Response = await call_next(request)
        elapsed = time.time() - start_time
        status_code = response.status_code
        gateway_state.recent_latencies.append(elapsed)

        if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            gateway_state.rate_limited_requests += 1
        elif response.status_code >= 500:
            gateway_state.error_requests += 1

        metrics_registry.requests_total.inc(labels={"status": str(status_code), "endpoint": endpoint})
        metrics_registry.request_duration.observe(elapsed, labels={"endpoint": endpoint})
        return response
    except HTTPException as exc:
        elapsed = time.time() - start_time
        status_code = exc.status_code
        gateway_state.recent_latencies.append(elapsed)
        if exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            gateway_state.rate_limited_requests += 1
        elif exc.status_code >= 500:
            gateway_state.error_requests += 1

        metrics_registry.requests_total.inc(labels={"status": str(status_code), "endpoint": endpoint})
        metrics_registry.request_duration.observe(elapsed, labels={"endpoint": endpoint})
        raise
    except Exception:
        elapsed = time.time() - start_time
        gateway_state.error_requests += 1
        metrics_registry.requests_total.inc(labels={"status": "500", "endpoint": endpoint})
        metrics_registry.request_duration.observe(elapsed, labels={"endpoint": endpoint})
        raise


@app.get("/health")
async def health_check(
    client: httpx.AsyncClient = Depends(get_client),
    current_settings: GatewaySettings = Depends(get_settings),
) -> JSONResponse:
    """Health check validating gateway status, circuit breaker, and upstream vLLM reachability."""
    uptime = time.time() - gateway_state.start_time
    vllm_url = f"{current_settings.vllm_base_url.rstrip('/')}/health"
    vllm_status = "unknown"
    is_healthy = True

    try:
        resp = await client.get(vllm_url, timeout=5.0)
        if resp.status_code == 200:
            vllm_status = "ok"
            gateway_state.circuit_breaker.record_success()
        else:
            vllm_status = f"unhealthy (status {resp.status_code})"
            is_healthy = False
            gateway_state.circuit_breaker.record_failure()
    except Exception as e:
        vllm_status = f"unreachable ({type(e).__name__})"
        is_healthy = False
        gateway_state.circuit_breaker.record_failure()

    payload = {
        "status": "healthy" if is_healthy else "degraded",
        "gateway": "ok",
        "vllm": vllm_status,
        "uptime_seconds": round(uptime, 2),
        "circuit_breaker": gateway_state.circuit_breaker.get_metrics(),
        "queue": gateway_state.priority_queue.get_metrics(),
        "prefix_cache": gateway_state.cache_router.get_metrics(),
    }
    status_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=payload, status_code=status_code)


@app.get("/metrics")
async def metrics(request: Request) -> Response:
    """Observability endpoint returning Prometheus exposition format or JSON based on Accept header/format param."""
    accept = request.headers.get("accept", "").lower()
    format_param = request.query_params.get("format", "").lower()

    if "text/plain" in accept or "prometheus" in accept or format_param == "prometheus":
        # Standard Prometheus text format
        q_metrics = gateway_state.priority_queue.get_metrics()
        metrics_registry.queue_depth.set(q_metrics.get("total_queue_depth", 0), labels={"priority": "all"})
        metrics_registry.active_gpu_slots.set(q_metrics.get("active_requests", 0))

        text_output = metrics_registry.generate_exposition()
        return PlainTextResponse(
            content=text_output,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    # Default JSON observability summary
    latencies = list(gateway_state.recent_latencies)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    uptime = time.time() - gateway_state.start_time
    return JSONResponse(
        content={
            "uptime_seconds": round(uptime, 2),
            "total_requests": gateway_state.total_requests,
            "rate_limited_requests": gateway_state.rate_limited_requests,
            "error_requests": gateway_state.error_requests,
            "high_priority_requests": gateway_state.high_priority_requests,
            "low_priority_requests": gateway_state.low_priority_requests,
            "total_tokens_processed": gateway_state.total_tokens_processed,
            "average_latency_seconds": round(avg_latency, 4),
            "sample_count": len(latencies),
            "circuit_breaker": gateway_state.circuit_breaker.get_metrics(),
            "queue_metrics": gateway_state.priority_queue.get_metrics(),
            "prefix_cache_metrics": gateway_state.cache_router.get_metrics(),
        }
    )


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
        if resp.status_code == 200:
            gateway_state.circuit_breaker.record_success()
        else:
            gateway_state.circuit_breaker.record_failure()
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers={**rate_headers, "Content-Type": resp.headers.get("content-type", "application/json")},
        )
    except httpx.RequestError as exc:
        gateway_state.circuit_breaker.record_failure()
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
    x_priority: Optional[str] = Header(None, alias="X-Priority"),
    traceparent: Optional[str] = Header(None, alias="traceparent"),
) -> Response:
    """Authenticated, rate-limited, priority-scheduled, circuit-broken chat completions proxy."""
    # Initialize OpenTelemetry Span
    parent_id = traceparent.split("-")[2] if traceparent and len(traceparent.split("-")) >= 3 else None
    trace_id = traceparent.split("-")[1] if traceparent and len(traceparent.split("-")) >= 2 else None
    span = OpenTelemetrySpan("gateway.chat_completions", trace_id=trace_id, parent_span_id=parent_id)

    cb_state = gateway_state.circuit_breaker.state.value

    # 1. Circuit Breaker Fast-Fail Protection
    if current_settings.circuit_breaker_enabled:
        cb_allowed, cb_reason, cb_retry_after = gateway_state.circuit_breaker.can_execute()
        if not cb_allowed:
            headers = {
                "X-Circuit-Breaker-State": cb_state,
                "Retry-After": str(int(cb_retry_after or 10.0)),
            }
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=cb_reason,
                headers=headers,
            )

    raw_body = await request.body()
    try:
        body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malformed JSON request body",
        ) from exc

    model_name = body.get("model", "unknown")

    # 2. Parse Priority
    raw_prio = (x_priority or body.get("priority", "high")).lower()
    if raw_prio in ("low", "batch", "1"):
        priority = RequestPriority.LOW
        gateway_state.low_priority_requests += 1
    else:
        priority = RequestPriority.HIGH
        gateway_state.high_priority_requests += 1

    # 3. Token Estimation & Rate Limiting
    estimated_tokens = estimate_request_tokens(body)
    request.state.estimated_tokens = estimated_tokens
    gateway_state.total_tokens_processed += estimated_tokens
    metrics_registry.tokens_total.inc(estimated_tokens, labels={"type": "estimated_total", "model": model_name})

    client_ip = request.client.host if request.client else "unknown"
    allowed, rem_rpm, rem_tpm, retry_after, reason = rate_limiter.check(
        client_ip,
        max_requests=current_settings.rate_limit_rpm,
        max_tokens=current_settings.rate_limit_tpm,
        requested_tokens=estimated_tokens,
    )

    rate_headers = {
        "X-RateLimit-Limit": str(current_settings.rate_limit_rpm),
        "X-RateLimit-Remaining": str(rem_rpm),
        "X-RateLimit-Limit-Requests": str(current_settings.rate_limit_rpm),
        "X-RateLimit-Remaining-Requests": str(rem_rpm),
        "X-RateLimit-Limit-Tokens": str(current_settings.rate_limit_tpm),
        "X-RateLimit-Remaining-Tokens": str(rem_tpm),
        "X-RateLimit-Reset": str(int(time.time() + 60.0)),
        "X-Request-Estimated-Tokens": str(estimated_tokens),
        "X-Request-Priority": "high" if priority == RequestPriority.HIGH else "low",
        "X-Circuit-Breaker-State": cb_state,
        "traceparent": span.get_w3c_traceparent(),
    }

    if not allowed:
        rate_headers["Retry-After"] = str(int(retry_after))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {reason}",
            headers=rate_headers,
        )

    # 4. Prefix Cache Affinity Routing
    target_backend = current_settings.vllm_base_url
    if current_settings.prefix_cache_routing_enabled:
        _, prefix_hash = extract_prompt_prefix(body, min_chars=current_settings.prefix_min_chars)
        if prefix_hash:
            target_backend, is_hit = gateway_state.cache_router.route(
                prefix_hash=prefix_hash,
                default_target=current_settings.vllm_base_url,
            )
            cache_status = "HIT" if is_hit else "MISS"
            if is_hit:
                metrics_registry.prefix_cache_hits.inc(labels={"model": model_name})
            else:
                metrics_registry.prefix_cache_misses.inc(labels={"model": model_name})
            rate_headers["X-Cache-Prefix-Hash"] = prefix_hash
            rate_headers["X-Cache-Status"] = cache_status
            rate_headers["X-Cache-Hit-Ratio"] = str(gateway_state.cache_router.get_metrics()["hit_ratio"])

    # 5. Schedule via Priority Queue
    try:
        req_id = await gateway_state.priority_queue.acquire(
            priority=priority,
            timeout=current_settings.queue_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request queue wait time exceeded threshold.",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    rate_headers["X-Request-ID"] = req_id

    # Strip gateway routing directives before forwarding to upstream vLLM
    upstream_body = {k: v for k, v in body.items() if k not in ("priority",)}

    # 6. Proxy to upstream inference backend
    is_streaming = bool(body.get("stream", False))
    upstream_url = f"{target_backend.rstrip('/')}/v1/chat/completions"
    t_ingress = time.time()

    if is_streaming:
        async def stream_generator() -> AsyncGenerator[bytes, None]:
            ttft_recorded = False
            try:
                async with client.stream("POST", upstream_url, json=upstream_body) as upstream_resp:
                    if upstream_resp.status_code != 200:
                        gateway_state.circuit_breaker.record_failure()
                        error_body = await upstream_resp.aread()
                        yield error_body
                        return
                    gateway_state.circuit_breaker.record_success()
                    async for chunk in upstream_resp.aiter_bytes():
                        if not ttft_recorded and len(chunk) > 0:
                            ttft_val = time.time() - t_ingress
                            metrics_registry.ttft.observe(ttft_val, labels={"model": model_name})
                            ttft_recorded = True
                        yield chunk
            except httpx.RequestError as exc:
                gateway_state.circuit_breaker.record_failure()
                yield f"data: {{\"error\": \"Upstream connection error: {exc}\"}}\n\n".encode("utf-8")
            finally:
                await gateway_state.priority_queue.release()
                span.finish()

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers=rate_headers,
        )

    # Non-streaming request
    try:
        upstream_resp = await client.post(upstream_url, json=upstream_body)
        if upstream_resp.status_code == 200:
            gateway_state.circuit_breaker.record_success()
        else:
            gateway_state.circuit_breaker.record_failure()
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers={
                **rate_headers,
                "Content-Type": upstream_resp.headers.get("content-type", "application/json"),
            },
        )
    except httpx.RequestError as exc:
        gateway_state.circuit_breaker.record_failure()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Upstream vLLM error: {exc}",
        ) from exc
    finally:
        await gateway_state.priority_queue.release()
        span.finish()
