"""FastAPI Application for Cinch LLM Gateway with Token Limiting, Priority Queues, Prefix Routing, Telemetry & Circuit Breaking."""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import os
import time
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx

from gateway.auth import get_api_key
from gateway.cache_router import PrefixCacheRouter, extract_prompt_prefix
from gateway.cascade_router import CascadeRouter
from gateway.circuit_breaker import CircuitBreaker
from gateway.compressor import PromptCompressor
from gateway.config import GatewaySettings, get_settings, settings
from gateway.finops import FinOpsEngine
from gateway.grammar_guard import GrammarGuard
from gateway.guardrails import GuardrailsScanner
from gateway.limiter import enforce_rate_limit, rate_limiter
from gateway.lora_router import LoRARouter
from gateway.priority_queue import PriorityRequestQueue, RequestPriority
from gateway.semantic_cache import SemanticCache
from gateway.shadow_replayer import ShadowTrafficReplayer
from gateway.telemetry import OpenTelemetrySpan, metrics_registry
from gateway.token_counter import estimate_request_tokens
from gateway.tool_engine import ToolEngine


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
        self.semantic_cache: SemanticCache = SemanticCache(
            capacity=settings.semantic_cache_capacity,
            threshold=settings.semantic_cache_similarity_threshold,
        )
        self.lora_router: LoRARouter = LoRARouter(
            default_base_model=settings.lora_default_base_model,
            enabled=settings.lora_routing_enabled,
        )
        self.grammar_guard: GrammarGuard = GrammarGuard(
            enabled=settings.grammar_guard_enabled,
            auto_repair=settings.grammar_guard_auto_repair,
        )
        self.guardrails: GuardrailsScanner = GuardrailsScanner(
            enabled=settings.guardrails_enabled,
            injection_defense_enabled=settings.guardrails_injection_defense_enabled,
            pii_redaction_enabled=settings.guardrails_pii_redaction_enabled,
            system_prompt_leak_defense=settings.guardrails_system_prompt_leak_defense,
        )
        self.cascade_router: CascadeRouter = CascadeRouter(
            enabled=settings.cascade_routing_enabled,
            small_model=settings.cascade_small_model,
            large_model=settings.cascade_large_model,
            complexity_threshold=settings.cascade_complexity_threshold,
        )
        self.compressor: PromptCompressor = PromptCompressor(
            enabled=settings.compressor_enabled,
            min_tokens=settings.compressor_min_tokens,
            target_ratio=settings.compressor_target_ratio,
            preserve_code_blocks=settings.compressor_preserve_code_blocks,
        )
        self.tool_engine: ToolEngine = ToolEngine(
            enabled=settings.tool_engine_enabled,
            max_iterations=settings.tool_engine_max_iterations,
            sandbox_timeout_seconds=settings.tool_engine_sandbox_timeout_seconds,
        )
        self.finops: FinOpsEngine = FinOpsEngine(
            enabled=settings.finops_enabled,
            default_budget_usd=settings.finops_default_budget_usd,
            enforce_budgets=settings.finops_enforce_budgets,
            prompt_rate_per_1k=settings.finops_prompt_rate_per_1k,
            completion_rate_per_1k=settings.finops_completion_rate_per_1k,
        )
        self.shadow_replayer: ShadowTrafficReplayer = ShadowTrafficReplayer(
            enabled=settings.shadow_replayer_enabled,
            shadow_backend_url=settings.shadow_backend_url,
            sample_rate=settings.shadow_sample_rate,
            max_traces=settings.shadow_max_traces,
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

# Mount interactive WebUI serving console assets
ui_path = "ui" if os.path.isdir("ui") else ("/app/ui" if os.path.isdir("/app/ui") else None)
if ui_path:
    app.mount("/ui", StaticFiles(directory=ui_path, html=True), name="ui")


@app.get("/console")
async def console_redirect() -> RedirectResponse:
    """Redirect /console to /ui/ interactive serving console."""
    return RedirectResponse(url="/ui/")


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
        "semantic_cache": gateway_state.semantic_cache.get_metrics(),
        "lora_router": gateway_state.lora_router.get_metrics(),
        "grammar_guard": gateway_state.grammar_guard.get_metrics(),
        "guardrails": gateway_state.guardrails.get_metrics(),
        "cascade_router": gateway_state.cascade_router.get_metrics(),
        "compressor": gateway_state.compressor.get_metrics(),
        "tool_engine": gateway_state.tool_engine.get_metrics(),
        "finops": gateway_state.finops.get_metrics(),
        "shadow_replayer": gateway_state.shadow_replayer.get_metrics(),
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
            models_data = resp.json()
            if current_settings.lora_routing_enabled:
                models_data = gateway_state.lora_router.synthesize_models_response(models_data)
            if current_settings.cascade_routing_enabled:
                created_ts = int(time.time())
                for auto_id, desc in [
                    ("auto", "Smart Model Cascading auto-router (0.5B small tier vs. 7B large tier)"),
                    ("auto:cascade", "Smart Model Cascading tier selector"),
                ]:
                    if not any(m.get("id") == auto_id for m in models_data.get("data", [])):
                        models_data.setdefault("data", []).append({
                            "id": auto_id,
                            "object": "model",
                            "created": created_ts,
                            "owned_by": "cinch-cascade-router",
                            "description": desc,
                            "root": "auto",
                            "parent": None,
                            "permission": [],
                        })
            return JSONResponse(
                content=models_data,
                status_code=200,
                headers=rate_headers,
            )
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
    x_compaction: Optional[str] = Header(None, alias="X-Prompt-Compaction"),
    x_server_tools: Optional[str] = Header(None, alias="X-Server-Tool-Execution"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_team_id: Optional[str] = Header(None, alias="X-Team-ID"),
    x_shadow_replay: Optional[str] = Header(None, alias="X-Shadow-Replay"),
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

    # 1. Multi-Tenant FinOps Pre-Flight Budget Check
    tenant_id = (x_tenant_id or body.get("tenant_id", "default")).lower()
    team_id = (x_team_id or body.get("team_id", "engineering")).lower()
    if current_settings.finops_enabled:
        budget_ok, budget_reason, _ = gateway_state.finops.check_budget(tenant_id)
        if not budget_ok:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=budget_reason,
                headers={
                    "X-FinOps-Tenant-ID": tenant_id,
                    "X-FinOps-Budget-Remaining-USD": "0.000000",
                    "X-Tenant-Budget-Exceeded": "true",
                },
            )

    # 2. Ingress Security Guardrails & PII Anonymization
    pii_found_total: list[str] = []
    if current_settings.guardrails_enabled:
        messages = body.get("messages", [])
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str) and content:
                scan_res = gateway_state.guardrails.scan_ingress(content)
                if not scan_res.is_safe:
                    headers = {
                        "X-Guardrails-Status": "BLOCKED",
                        "X-Guardrails-Violation": scan_res.violation_type or "SECURITY_VIOLATION",
                    }
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Security Guardrail Violation: {scan_res.violation_type}",
                        headers=headers,
                    )
                if scan_res.pii_types_found:
                    msg["content"] = scan_res.redacted_text
                    pii_found_total.extend(scan_res.pii_types_found)

    # 3. Context & Prompt Compaction (LLMLingua Heuristic)
    compaction_enabled = current_settings.compressor_enabled
    if x_compaction is not None:
        compaction_enabled = x_compaction.lower() not in ("false", "0", "no", "off")

    compaction_res = None
    if compaction_enabled:
        messages = body.get("messages", [])
        compacted_messages, compaction_res = gateway_state.compressor.compress_messages(messages)
        if compaction_res.is_compacted:
            body["messages"] = compacted_messages

    # 4. Extract Structured Output & JSON Grammar Constraints
    grammar_constraint = gateway_state.grammar_guard.extract_constraints(body)

    # 5. Smart Model Cascading & Complexity Routing
    full_prompt_text = " ".join(
        m.get("content", "") for m in body.get("messages", []) if isinstance(m.get("content"), str)
    )
    raw_model_requested = body.get("model", "auto")
    selected_model, cascade_analysis = gateway_state.cascade_router.resolve_model(
        requested_model=raw_model_requested,
        prompt=full_prompt_text,
        has_schema=grammar_constraint.is_active,
    )
    body["model"] = selected_model

    # 6. Resolve Multi-LoRA Compound Model Identifiers
    adapter_name: Optional[str] = None
    if current_settings.lora_routing_enabled:
        body, adapter_name, model_name = gateway_state.lora_router.resolve_request(body)
    else:
        model_name = body.get("model", "unknown")

    # 3. Parse Priority
    raw_prio = (x_priority or body.get("priority", "high")).lower()
    if raw_prio in ("low", "batch", "1"):
        priority = RequestPriority.LOW
        gateway_state.low_priority_requests += 1
    else:
        priority = RequestPriority.HIGH
        gateway_state.high_priority_requests += 1

    # 4. Token Estimation & Rate Limiting
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
        "X-LoRA-Adapter-Active": adapter_name or "none",
        "X-LoRA-Base-Model": model_name,
        "X-Grammar-Guard-Type": grammar_constraint.constraint_type,
        "X-Guardrails-Status": "PASSED",
        "X-Guardrails-PII-Redacted": "true" if pii_found_total else "false",
        "X-Cascade-Routing-Tier": cascade_analysis.tier.value,
        "X-Cascade-Complexity-Score": str(round(cascade_analysis.score, 3)),
        "X-Cascade-Selected-Model": selected_model,
        "X-Cascade-Reason": cascade_analysis.reason,
        "X-Prompt-Compacted": "true" if (compaction_res and compaction_res.is_compacted) else "false",
        "X-Prompt-Original-Tokens": str(compaction_res.original_tokens if compaction_res else estimated_tokens),
        "X-Prompt-Compacted-Tokens": str(compaction_res.compacted_tokens if compaction_res else estimated_tokens),
        "X-Prompt-Compaction-Ratio": str(compaction_res.compression_ratio if compaction_res else 1.0),
        "X-Tool-Engine-Executed": "false",
        "X-Tool-Engine-Iterations": "0",
        "X-Tool-Engine-Tools-Used": "none",
        "X-FinOps-Tenant-ID": tenant_id,
        "X-FinOps-Request-Cost-USD": "0.000000",
        "X-FinOps-Tenant-Spend-USD": "0.000000",
        "X-FinOps-Budget-Remaining-USD": "100.000000",
        "X-Shadow-Replay-Sampled": "true" if gateway_state.shadow_replayer.should_sample(x_shadow_replay) else "false",
        "traceparent": span.get_w3c_traceparent(),
    }

    if not allowed:
        rate_headers["Retry-After"] = str(int(retry_after))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {reason}",
            headers=rate_headers,
        )

    # 4. Semantic Vector Cache Lookup (Zero-GPU Fast Path)
    if current_settings.semantic_cache_enabled and not bool(body.get("stream", False)):
        prompt_text = " ".join(
            m.get("content", "") for m in body.get("messages", []) if isinstance(m.get("content"), str)
        )
        cached_response, similarity = gateway_state.semantic_cache.lookup(prompt_text)
        rate_headers["X-Semantic-Cache-Similarity"] = str(similarity)
        if cached_response is not None:
            rate_headers["X-Semantic-Cache-Status"] = "HIT"
            if current_settings.finops_enabled:
                cost_rec = gateway_state.finops.record_usage(
                    tenant_id=tenant_id,
                    team_id=team_id,
                    prompt_tokens=estimated_tokens,
                    completion_tokens=20,
                )
                rate_headers["X-FinOps-Tenant-ID"] = tenant_id
                rate_headers["X-FinOps-Request-Cost-USD"] = f"{cost_rec.total_cost_usd:.6f}"
                rate_headers["X-FinOps-Tenant-Spend-USD"] = f"{cost_rec.total_spend_usd:.6f}"
                rate_headers["X-FinOps-Budget-Remaining-USD"] = f"{cost_rec.budget_remaining_usd:.6f}"
            span.finish()
            return Response(
                content=json.dumps(cached_response),
                status_code=200,
                headers={**rate_headers, "Content-Type": "application/json"},
            )
        rate_headers["X-Semantic-Cache-Status"] = "MISS"
    else:
        prompt_text = ""

    # 5. Prefix Cache Affinity Routing
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
    if "max_tokens" not in upstream_body and "max_completion_tokens" not in upstream_body:
        upstream_body["max_tokens"] = 1024
    if current_settings.tool_engine_enabled and "tools" in upstream_body:
        upstream_body, _ = gateway_state.tool_engine.prepare_upstream_request(upstream_body)

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
        content_bytes = upstream_resp.content

        if upstream_resp.status_code == 200:
            gateway_state.circuit_breaker.record_success()

            # Asynchronous Shadow Traffic Replay Trigger
            if rate_headers.get("X-Shadow-Replay-Sampled") == "true":
                try:
                    prod_resp_obj = json.loads(content_bytes.decode("utf-8"))
                except Exception:
                    prod_resp_obj = {}
                prod_lat_ms = (time.time() - t_ingress) * 1000.0
                asyncio.create_task(
                    gateway_state.shadow_replayer.replay_shadow(
                        client=client,
                        request_body=body,
                        prod_resp_json=prod_resp_obj,
                        prod_latency_ms=prod_lat_ms,
                        prod_status=upstream_resp.status_code,
                        api_key=_api_key,
                    )
                )

            # Server-Side Agentic Tool Execution Loop
            server_tools_active = current_settings.tool_engine_enabled and (
                (x_server_tools and x_server_tools.lower() in ("true", "1", "yes"))
                or body.get("server_tool_execution") is True
                or bool(body.get("tools"))
            )

            tools_executed_list: list[str] = []
            tool_iterations = 0

            if server_tools_active:
                try:
                    curr_resp_json = upstream_resp.json()
                    tool_calls = gateway_state.tool_engine.extract_tool_calls(curr_resp_json)

                    while tool_calls and tool_iterations < current_settings.tool_engine_max_iterations:
                        tool_iterations += 1
                        asst_msg = curr_resp_json["choices"][0]["message"]
                        upstream_body.setdefault("messages", []).append(asst_msg)

                        for tc in tool_calls:
                            fn = tc.get("function", {})
                            fn_name = fn.get("name", "")
                            fn_args = fn.get("arguments", {})
                            call_id = tc.get("id", f"call_{tool_iterations}")

                            tool_res = gateway_state.tool_engine.execute_tool_call(fn_name, fn_args, call_id)
                            tools_executed_list.append(fn_name)
                            upstream_body["messages"].append(tool_res.to_tool_message())

                        iter_resp = await client.post(upstream_url, json=upstream_body)
                        if iter_resp.status_code == 200:
                            curr_resp_json = iter_resp.json()
                            content_bytes = iter_resp.content
                            tool_calls = gateway_state.tool_engine.extract_tool_calls(curr_resp_json)
                        else:
                            break
                except Exception:
                    pass

            rate_headers["X-Tool-Engine-Executed"] = "true" if tools_executed_list else "false"
            rate_headers["X-Tool-Engine-Iterations"] = str(tool_iterations)
            rate_headers["X-Tool-Engine-Tools-Used"] = ",".join(set(tools_executed_list)) if tools_executed_list else "none"

            # Enforce and sanitize structured outputs if constraint is active
            if grammar_constraint.is_active:
                try:
                    resp_json = upstream_resp.json()
                    choices = resp_json.get("choices", [])
                    if choices and isinstance(choices[0], dict):
                        msg = choices[0].get("message", {})
                        raw_msg_content = msg.get("content", "")
                        is_valid, sanitized_content, status_label = gateway_state.grammar_guard.validate_constraint(
                            raw_msg_content, grammar_constraint
                        )
                        rate_headers["X-Grammar-Guard-Status"] = status_label
                        if sanitized_content != raw_msg_content:
                            msg["content"] = sanitized_content
                            content_bytes = json.dumps(resp_json).encode("utf-8")
                except Exception:
                    rate_headers["X-Grammar-Guard-Status"] = "ERROR"
            else:
                rate_headers["X-Grammar-Guard-Status"] = "UNCONSTRAINED"

            # Apply egress PII & system prompt leakage defense
            if current_settings.guardrails_enabled:
                try:
                    sys_prompt = next(
                        (m.get("content", "") for m in body.get("messages", []) if m.get("role") == "system"),
                        None,
                    )
                    resp_json = json.loads(content_bytes.decode("utf-8"))
                    choices = resp_json.get("choices", [])
                    if choices and isinstance(choices[0], dict):
                        msg = choices[0].get("message", {})
                        curr_content = msg.get("content", "")
                        sanitized_content, egress_modified = gateway_state.guardrails.sanitize_egress(
                            curr_content, system_prompt=sys_prompt
                        )
                        if egress_modified:
                            msg["content"] = sanitized_content
                            content_bytes = json.dumps(resp_json).encode("utf-8")
                except Exception:
                    pass

            # Record multi-tenant FinOps cost attribution
            if current_settings.finops_enabled:
                try:
                    resp_json_obj = json.loads(content_bytes.decode("utf-8"))
                    usage_obj = resp_json_obj.get("usage", {})
                    p_toks = usage_obj.get("prompt_tokens", estimated_tokens)
                    c_toks = usage_obj.get("completion_tokens", 30)
                except Exception:
                    p_toks = estimated_tokens
                    c_toks = 30

                cost_rec = gateway_state.finops.record_usage(
                    tenant_id=tenant_id,
                    team_id=team_id,
                    prompt_tokens=p_toks,
                    completion_tokens=c_toks,
                )
                rate_headers["X-FinOps-Tenant-ID"] = tenant_id
                rate_headers["X-FinOps-Request-Cost-USD"] = f"{cost_rec.total_cost_usd:.6f}"
                rate_headers["X-FinOps-Tenant-Spend-USD"] = f"{cost_rec.total_spend_usd:.6f}"
                rate_headers["X-FinOps-Budget-Remaining-USD"] = f"{cost_rec.budget_remaining_usd:.6f}"

            # Store successful response in semantic cache for future paraphrase hits
            if current_settings.semantic_cache_enabled and prompt_text:
                try:
                    resp_json = upstream_resp.json()
                    gateway_state.semantic_cache.store(prompt_text, resp_json)
                except Exception:
                    pass  # Never let cache writes degrade the serving path
        else:
            gateway_state.circuit_breaker.record_failure()

        return Response(
            content=content_bytes,
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


@app.get("/v1/tenants/usage")
async def get_tenants_usage(
    tenant_id: Optional[str] = None,
    _api_key: Optional[str] = Depends(get_api_key),
) -> JSONResponse:
    """Authenticated endpoint returning real-time multi-tenant FinOps usage ledgers."""
    usage_data = gateway_state.finops.get_tenant_usage(tenant_id=tenant_id)
    return JSONResponse(content=usage_data, status_code=status.HTTP_200_OK)


@app.post("/v1/tenants/budget")
async def set_tenant_budget(
    request: Request,
    _api_key: Optional[str] = Depends(get_api_key),
) -> JSONResponse:
    """Authenticated endpoint to dynamically adjust tenant budget allocations."""
    body = await request.json()
    tenant_id = body.get("tenant_id", "default")
    budget_limit = float(body.get("budget_limit_usd", 100.0))
    tenant_record = gateway_state.finops.set_budget(tenant_id, budget_limit)
    return JSONResponse(content=tenant_record.to_dict(), status_code=status.HTTP_200_OK)


@app.get("/v1/shadow/metrics")
async def get_shadow_metrics(
    _api_key: Optional[str] = Depends(get_api_key),
) -> JSONResponse:
    """Authenticated endpoint returning shadow traffic summary metrics and divergence statistics."""
    metrics_data = gateway_state.shadow_replayer.get_metrics()
    return JSONResponse(content=metrics_data, status_code=status.HTTP_200_OK)


@app.get("/v1/shadow/traces")
async def get_shadow_traces(
    limit: int = 50,
    _api_key: Optional[str] = Depends(get_api_key),
) -> JSONResponse:
    """Authenticated endpoint returning recent shadow comparison traces."""
    traces = gateway_state.shadow_replayer.get_traces(limit=limit)
    return JSONResponse(content={"traces": traces, "count": len(traces)}, status_code=status.HTTP_200_OK)


@app.post("/v1/shadow/config")
async def update_shadow_config(
    request: Request,
    _api_key: Optional[str] = Depends(get_api_key),
) -> JSONResponse:
    """Authenticated endpoint dynamically configuring shadow replayer sampling and target URL."""
    body = await request.json()
    cfg = gateway_state.shadow_replayer.set_config(
        sample_rate=body.get("sample_rate"),
        shadow_backend_url=body.get("shadow_backend_url"),
        enabled=body.get("enabled"),
    )
    return JSONResponse(content=cfg, status_code=status.HTTP_200_OK)


@app.get("/v1/console/state")
async def get_console_state(
    _api_key: Optional[str] = Depends(get_api_key),
) -> JSONResponse:
    """Consolidated real-time serving console state for WebUI dashboards."""
    return JSONResponse(
        content={
            "status": "healthy",
            "uptime_seconds": round(time.time() - gateway_state.start_time, 2),
            "total_requests": gateway_state.total_requests,
            "queue": gateway_state.priority_queue.get_metrics(),
            "prefix_cache": gateway_state.cache_router.get_metrics(),
            "semantic_cache": gateway_state.semantic_cache.get_metrics(),
            "finops": gateway_state.finops.get_metrics(),
            "shadow_replayer": gateway_state.shadow_replayer.get_metrics(),
            "guardrails": gateway_state.guardrails.get_metrics(),
            "tool_engine": gateway_state.tool_engine.get_metrics(),
            "compressor": gateway_state.compressor.get_metrics(),
        },
        status_code=status.HTTP_200_OK,
    )
