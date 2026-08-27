# Milestone 22: Native Server-Side Agentic Tool Execution Engine

## 1. Overview and Problem Statement

Standard client-side agent architectures require multiple HTTP network roundtrips:

1. **Client Dispatches Query**: `{"messages": [{"role": "user", "content": "What is 45 * 12 + 10?"}]}`
2. **Model Returns Tool Call**: `{"tool_calls": [{"function": {"name": "calculator", "arguments": "{\"expression\": \"45 * 12 + 10\"}"}}]}`
3. **Client Executes Tool Locally**: Computes `550`.
4. **Client Re-dispatches Context**: Appends tool output and calls gateway a second time.
5. **Model Synthesizes Final Answer**: Returns `"The result is 550."`

This roundtrip pattern incurs $2\times$ to $4\times$ network latency overhead and forces clients to maintain complex state machines and execution runtimes.

Milestone 22 implements `gateway/tool_engine.py`. The gateway intercepts model `tool_calls`, executes them in an internal isolated sandbox, appends the tool response, and re-invokes upstream inference in a closed loop, delivering the final synthesized answer to the client in a **single HTTP request**.

---

## 2. Technical Architecture

### Closed-Loop Server-Side Execution Flow

```
[ Client Request: X-Server-Tool-Execution: true ]
                       │
                       ▼
         [ Gateway Ingress Inspection ]
                       │
                       ▼
       [ Upstream vLLM (Generates tool_calls) ]
                       │
                       ▼
          [ Tool Engine Interceptor ]
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
  [ Calculator Sandbox ]       [ SQLite SQL Runner ]
  (AST-Isolated Math)         (In-Memory DB Engine)
        │                             │
        └──────────────┬──────────────┘
                       ▼
         [ Append Tool Result Message ]
                       │
                       ▼
       [ Re-invoke Upstream vLLM Synthesis ]
                       │
                       ▼
      [ Client Receives Final 200 OK Answer ]
```

---

## 3. Sandboxed Tool Runtimes

| Sandbox Runtime | Isolation Mechanism | Permitted Capabilities |
|---|---|---|
| `calculator` | AST Expression Visitor | Arithmetic (`+`, `-`, `*`, `/`, `**`), `sqrt`, `pow`, `log`, `sin`, `cos`, `abs`, `round`. No variable lookups. |
| `sql_runner` | In-Memory SQLite | Read-only SQL queries over in-memory database tables (`employees`, `metrics`, `orders`). 50-row limit. |
| `python_repl` | Sanitized Global Scope | List comprehensions, math, string manipulation. `__import__`, `open`, `os`, `sys`, and `subprocess` blocked. |

---

## 4. Empirical Tool Execution Evaluation

Evaluation executed against the live k3d Kubernetes gateway and cluster backend.

Source dataset: `benchmarks/results/tool_execution_eval.json`

### Benchmark Evaluation Matrix

| Scenario Name | Intended Tool | Tools Invoked | Status | Client Roundtrips | End-to-End Latency |
|---|---|---|---|---|---|
| `calculator_algebraic_computation` | `calculator` | `calculator` | **PASS** | **1 (Zero retry loops)** | 1,547.8 ms |
| `sql_database_query` | `sql_runner` | `sql_runner` | **PASS** | **1 (Zero retry loops)** | 2,056.6 ms |
| `python_repl_data_transformation` | `python_repl` | `python_repl` | **PASS** | **1 (Zero retry loops)** | 1,812.8 ms |
| `direct_conversational_no_tools` | `none` | `none` | **PASS** | **1 (Direct passthrough)** | 1,224.2 ms |

### Performance Summary

- **Total Scenarios Evaluated**: 4
- **Closed-Loop Execution Accuracy**: 100.0% (4 / 4)
- **Zero Client Retry Roundtrips**: 100%
- **Sandbox Latency Overhead**: < 2.0 ms per tool invocation
- **Average End-to-End Latency**: 1,660.4 ms (includes full multi-turn generation)

---

## 5. Verification and Test Suite

All 7 unit tests pass:

```powershell
python -m pytest tests/test_tool_engine.py -v
```

```
tests/test_tool_engine.py::test_calculator_valid_expressions PASSED      [ 14%]
tests/test_tool_engine.py::test_calculator_syntax_error_and_divzero PASSED [ 28%]
tests/test_tool_engine.py::test_sql_runner_in_memory_query PASSED        [ 42%]
tests/test_tool_engine.py::test_python_repl_safe_execution PASSED        [ 57%]
tests/test_tool_engine.py::test_python_repl_security_sandbox PASSED      [ 71%]
tests/test_tool_engine.py::test_tool_calls_extraction_openai_and_text PASSED [ 85%]
tests/test_tool_engine.py::test_tool_engine_metrics_tracking PASSED      [100%]
```

Full repository regression suite: **175 / 175 tests passing**.
Code lint status: **0 errors** (`ruff check .`).
