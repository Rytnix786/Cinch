# Milestone 25: Interactive Real-Time Serving Console (WebUI)

## 1. Overview and Problem Statement

Black-box inference servers lack an intuitive visual interface for developers to test prompts, inspect KV-cache memory heatmaps, monitor live streaming tokens, and audit FinOps costs in real time.

Milestone 25 implements a lightweight, zero-dependency, dark-mode real-time serving console (`ui/` mounted at `/ui` and `/console`).

Features:
1. **Interactive Prompt Playground**: Live SSE token streaming client, real-time TTFT and TPS calculation, parameter sliders, model selector, grammar schema toggle, and server-side tool execution.
2. **KV-Cache Memory Heatmap**: 64-block visual matrix of shared prefix and active context allocations, prefix hit ratios, and LRU cache statistics.
3. **FinOps Cost Center**: Live micro-dollar platform spend meters, team budget progress bars, dynamic budget limit configuration, and tenant usage ledgers.
4. **Security & Guardrails Audit Stream**: Real-time event log of verified clean queries, blocked prompt injections, and redacted PII entities.
5. **Shadow Replayer & Tools Inspector**: Production vs candidate model comparison diff viewer with latency deltas ($\Delta \text{ms}$) and sandboxed tool execution results (`calculator`, `sql_runner`, `python_repl`).

---

## 2. Technical Architecture

### WebUI Delivery and SSE Stream Processing

```
[ Browser Client (/ui/) ]
         │
         ├── GET /ui/style.css, /ui/app.js (Sub-5ms static delivery)
         ├── GET /v1/console/state (3s interval telemetry polling)
         │
         ▼
[ POST /v1/chat/completions (stream: true) ]
         │
         ▼
[ Gateway SSE Proxy Pipeline ]
         │
         ▼
[ Browser ReadableStreamReader ]
         ├── Real-time token append to DOM
         ├── TTFT Stopwatch: t_first_chunk - t_dispatch
         └── TPS Rate: total_tokens / (t_end - t_first_chunk)
```

---

## 3. Empirical Console Evaluation

Evaluation executed against the live k3d Kubernetes gateway and cluster backend.

Source dataset: `benchmarks/results/ui_console_eval.json`

### Static Asset & Telemetry Latency Benchmarks

| Endpoint Name | Path | HTTP Status | Minimum Latency | Average Latency | Status |
|---|---|---|---|---|---|
| UI HTML Shell | `/ui/` | `200` | 2.63 ms | **6.73 ms** | **PASS** |
| UI Style CSS | `/ui/style.css` | `200` | 2.37 ms | **2.53 ms** | **PASS** |
| UI Engine JS | `/ui/app.js` | `200` | 2.28 ms | **2.43 ms** | **PASS** |
| Console Redirect | `/console` | `200` (Redirect) | 3.94 ms | **4.13 ms** | **PASS** |
| Console State API | `/v1/console/state` | `200` | 2.14 ms | **2.37 ms** | **PASS** |

### Streaming Performance Summary

- **Static Asset Conformance**: 100.0% (5 / 5 endpoints)
- **Average Asset Delivery Latency**: **3.64 ms** (< 5ms SLA)
- **Streaming TTFT**: **159.2 ms**
- **Streaming Generation Rate**: **50.5 tokens/sec**
- **Tokens Streamed**: 41 tokens in 971.7 ms total duration

---

## 4. Verification and Test Suite

All 4 unit tests pass:

```powershell
python -m pytest tests/test_ui_endpoints.py -v
```

```
tests/test_ui_endpoints.py::test_ui_static_index_html PASSED             [ 25%]
tests/test_ui_endpoints.py::test_ui_static_css_and_js PASSED             [ 50%]
tests/test_ui_endpoints.py::test_console_redirect PASSED                 [ 75%]
tests/test_ui_endpoints.py::test_console_state_endpoint PASSED           [100%]
```

Full repository regression suite: **191 / 191 tests passing**.
Code lint status: **0 errors** (`ruff check .`).
