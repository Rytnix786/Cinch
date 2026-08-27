# Milestone 3: FastAPI Gateway Layer

This document details the architecture, authentication, rate limiting, and request routing implementation for the Cinch FastAPI Gateway layer in front of vLLM.

---

## 1. Architecture Overview

The FastAPI gateway acts as the single ingress point for client traffic. It decouples client-facing operational concerns (authentication, rate limiting, and telemetry) from the GPU-bound vLLM inference backend.

```
+------------------+          +------------------------+          +-------------------------+
|                  |          |                        |          |                         |
|   Client / Test  | -------> |   FastAPI Gateway      | -------> |   vLLM Server (Docker)  |
|   Harness        |  :8080   |   - Bearer/X-API-Key   |  :8000   |   - Qwen2.5-7B AWQ      |
|                  |          |   - Sliding Window RPM |          |   - FlashAttention-2    |
|                  |          |   - SSE Stream Proxy   |          |   - Eager Mode          |
+------------------+          +------------------------+          +-------------------------+
```

### Stateless Design Principle
The gateway stores no persistent session state. It tracks rate limits in a memory-bounded sliding window and uses non-blocking HTTP connection pooling (`httpx.AsyncClient`). This stateless property allows horizontal scaling under Kubernetes Horizontal Pod Autoscalers (HPA) in subsequent milestones.

---

## 2. Security and Authentication

Authentication supports standard API key headers and uses constant-time byte comparisons to prevent timing attacks.

- **Header Schemes**:
  1. `Authorization: Bearer <GATEWAY_API_KEY>`
  2. `X-API-Key: <GATEWAY_API_KEY>`
- **Validation**: Implemented with Python's `secrets.compare_digest`.
- **Status Codes**:
  - Missing or invalid key: `401 Unauthorized` with `WWW-Authenticate: Bearer`
  - Unset `GATEWAY_API_KEY`: Authentication bypasses automatically for local testing.

---

## 3. Sliding-Window Rate Limiting

The gateway enforces per-client IP request rate limits via `gateway/limiter.py`.

### Algorithm
1. Maintains a double-ended queue (`collections.deque`) of request epoch timestamps per client IP.
2. Evicts timestamps older than the 60-second sliding window ($t - 60.0$).
3. If active timestamps exceed `RATE_LIMIT_RPM`, computes `retry_after = ceil(oldest_timestamp + 60.0 - now)`.
4. Returns `429 Too Many Requests` with rate-limit headers:
   - `Retry-After: <seconds>`
   - `X-RateLimit-Limit: <rpm>`
   - `X-RateLimit-Remaining: 0`
   - `X-RateLimit-Reset: <epoch_timestamp>`

---

## 4. Endpoints and Routing Matrix

| Endpoint | Method | Auth Required | Description |
|---|---|---|---|
| `/health` | `GET` | No | Probes gateway status and upstream vLLM health. Returns `200` if healthy, `503` if degraded. |
| `/metrics` | `GET` | No | Exposes total requests, 429 rate limit rejections, error counts, and average latency. |
| `/v1/models` | `GET` | Yes | Proxies model discovery directly to vLLM. |
| `/v1/chat/completions` | `POST` | Yes | Proxies chat requests. Supports both standard JSON payloads and Server-Sent Events (SSE) streaming (`stream: true`). |

---

## 5. Quickstart & Local Execution

### 1. Launch Gateway Locally (connecting to running vLLM on :8000)
```powershell
# Set optional auth key and run uvicorn
$env:VLLM_BASE_URL="http://localhost:8000"
$env:GATEWAY_API_KEY="cinch-dev-key"
python -m uvicorn gateway.app:app --host 0.0.0.0 --port 8080
```

### 2. Run Gateway Verification Suite
```powershell
python scripts/test_gateway_live.py --gateway-url http://localhost:8080 --api-key cinch-dev-key
```

### 3. Run Automated Tests
```powershell
python -m pytest tests/ -v
python -m ruff check .
```
