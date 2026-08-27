"""Unit tests for Native Server-Side Agentic Tool Execution Engine (gateway/tool_engine.py)."""

from __future__ import annotations

from gateway.tool_engine import ToolEngine


def test_calculator_valid_expressions() -> None:
    engine = ToolEngine()

    # Basic arithmetic
    ok, res = engine.execute_calculator("45 * 12 + 10")
    assert ok is True
    assert res == "550"

    # Mathematical functions
    ok, res = engine.execute_calculator("sqrt(144) + pow(2, 3)")
    assert ok is True
    assert float(res) == 20.0

    # Complex algebraic precedence
    ok, res = engine.execute_calculator("(100 - 25) / 5 * 2")
    assert ok is True
    assert res == "30"


def test_calculator_syntax_error_and_divzero() -> None:
    engine = ToolEngine()

    # Division by zero
    ok, res = engine.execute_calculator("10 / 0")
    assert ok is False
    assert "division by zero" in res.lower()

    # Syntax error
    ok, res = engine.execute_calculator("45 * + / 12")
    assert ok is False
    assert "calculation error" in res.lower()


def test_sql_runner_in_memory_query() -> None:
    engine = ToolEngine()

    query = "SELECT department, avg(salary) as avg_sal FROM employees GROUP BY department HAVING avg(salary) > 100000"
    ok, res = engine.execute_sql(query)
    assert ok is True
    assert "Engineering" in res
    assert "135000" in res or "135000.0" in res


def test_python_repl_safe_execution() -> None:
    engine = ToolEngine()

    code = (
        "items = [10, 20, 30, 40]\n"
        "total = sum(items)\n"
        "avg = total / len(items)\n"
        "print(f'Average: {avg}')"
    )
    ok, res = engine.execute_python_repl(code)
    assert ok is True
    assert "Average: 25.0" in res


def test_python_repl_security_sandbox() -> None:
    engine = ToolEngine()

    # Attempt file system breakout
    malicious_snippets = [
        "import os\nos.listdir('.')",
        "__import__('sys').exit(0)",
        "open('/etc/passwd', 'r').read()",
    ]
    for code in malicious_snippets:
        ok, res = engine.execute_python_repl(code)
        assert ok is False
        assert "Security violation" in res or "error" in res.lower()


def test_tool_calls_extraction_openai_and_text() -> None:
    engine = ToolEngine()

    # 1. OpenAI standard tool_calls
    openai_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "25 * 4"}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    extracted = engine.extract_tool_calls(openai_resp)
    assert len(extracted) == 1
    assert extracted[0]["function"]["name"] == "calculator"

    # 2. Raw text tags
    text_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Let me calculate that for you: <tool_call>{\"name\": \"calculator\", \"arguments\": {\"expression\": \"15 * 3\"}}</tool_call>",
                }
            }
        ]
    }
    extracted_text = engine.extract_tool_calls(text_resp)
    assert len(extracted_text) == 1
    assert extracted_text[0]["function"]["name"] == "calculator"


def test_tool_engine_metrics_tracking() -> None:
    engine = ToolEngine()

    res1 = engine.execute_tool_call("calculator", {"expression": "100 + 50"})
    res2 = engine.execute_tool_call("sql_runner", {"query": "SELECT count(*) FROM employees"})
    res3 = engine.execute_tool_call("calculator", {"expression": "10 / 0"})

    assert res1.success is True
    assert res2.success is True
    assert res3.success is False

    metrics = engine.get_metrics()
    assert metrics["enabled"] is True
    assert metrics["total_tool_calls_executed"] == 3
    assert metrics["successful_executions"] == 2
    assert metrics["failed_executions"] == 1
    assert metrics["success_rate_pct"] == 66.7
    assert metrics["tool_usage_breakdown"]["calculator"] == 2
    assert metrics["tool_usage_breakdown"]["sql_runner"] == 1
