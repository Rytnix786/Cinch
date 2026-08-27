"""Native Server-Side Agentic Tool Execution Engine (gateway/tool_engine.py).

Provides isolated sandbox runtimes for mathematical evaluation (AST calculator),
in-memory relational SQL queries (SQLite sandbox), and restricted Python data transformations,
enabling zero-roundtrip agentic reasoning loops inside the gateway.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import io
import json
import math
import operator
import re
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclasses.dataclass
class ToolExecutionResult:
    """Result of a sandboxed tool execution."""

    tool_name: str
    tool_call_id: str
    success: bool
    output: str
    error: Optional[str] = None
    execution_time_ms: float = 0.0

    def to_tool_message(self) -> Dict[str, Any]:
        """Convert to standard OpenAI tool response message format."""
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.tool_name,
            "content": self.output if self.success else f"Error executing {self.tool_name}: {self.error}",
        }


# ---------------------------------------------------------------------------
# Safe AST Calculator
# ---------------------------------------------------------------------------

SAFE_OPERATORS: Dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "pow": pow,
    "floor": math.floor,
    "ceil": math.ceil,
    "pi": lambda: math.pi,
    "e": lambda: math.e,
}


def _safe_eval_ast(node: ast.AST) -> Any:
    """Recursively evaluate an AST expression with strict node type whitelisting."""
    if isinstance(node, ast.Expression):
        return _safe_eval_ast(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type}")
        left = _safe_eval_ast(node.left)
        right = _safe_eval_ast(node.right)
        return SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type}")
        operand = _safe_eval_ast(node.operand)
        return SAFE_OPERATORS[op_type](operand)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct named mathematical functions are allowed")
        func_name = node.func.id
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(f"Function '{func_name}' is not permitted in sandbox")
        args = [_safe_eval_ast(arg) for arg in node.args]
        return SAFE_FUNCTIONS[func_name](*args)
    elif isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        elif node.id == "e":
            return math.e
        raise ValueError(f"Variable '{node.id}' is not defined")
    else:
        raise ValueError(f"Unsupported AST node: {type(node)}")


# ---------------------------------------------------------------------------
# Tool Engine Core
# ---------------------------------------------------------------------------

class ToolEngine:
    """
    Sandboxed tool execution engine and agentic closed-loop orchestrator.
    """

    def __init__(
        self,
        enabled: bool = True,
        max_iterations: int = 3,
        sandbox_timeout_seconds: float = 2.0,
    ) -> None:
        self.enabled = enabled
        self.max_iterations = max_iterations
        self.sandbox_timeout = sandbox_timeout_seconds

        self._total_executions: int = 0
        self._successful_executions: int = 0
        self._failed_executions: int = 0
        self._tool_usage: Dict[str, int] = {}
        self._total_exec_time_ms: float = 0.0

    def execute_calculator(self, expression: str) -> Tuple[bool, str]:
        """Safely compute mathematical expressions via AST evaluation."""
        clean_expr = expression.strip().rstrip("=")
        try:
            parsed = ast.parse(clean_expr, mode="eval")
            val = _safe_eval_ast(parsed)
            # Format float or int nicely
            if isinstance(val, float) and val.is_integer():
                val = int(val)
            return True, str(val)
        except Exception as exc:
            return False, f"Calculation error: {exc}"

    def execute_sql(self, query: str) -> Tuple[bool, str]:
        """Execute relational SQL query in an isolated in-memory SQLite sandbox."""
        try:
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()

            # Seed standard sample enterprise tables
            cursor.executescript("""
                CREATE TABLE employees (id INT, name TEXT, department TEXT, salary REAL, role TEXT);
                INSERT INTO employees VALUES (1, 'Alice Johnson', 'Engineering', 145000, 'Staff Engineer');
                INSERT INTO employees VALUES (2, 'Bob Smith', 'Engineering', 125000, 'Senior Engineer');
                INSERT INTO employees VALUES (3, 'Charlie Lee', 'Product', 135000, 'Principal PM');
                INSERT INTO employees VALUES (4, 'Dana White', 'Sales', 110000, 'Account Executive');
                INSERT INTO employees VALUES (5, 'Evan Wright', 'Finance', 95000, 'Financial Analyst');

                CREATE TABLE metrics (service TEXT, p99_latency_ms REAL, rps REAL, error_rate REAL);
                INSERT INTO metrics VALUES ('cinch-gateway', 12.4, 1850.0, 0.0001);
                INSERT INTO metrics VALUES ('cinch-vllm', 35.8, 420.0, 0.0005);
                INSERT INTO metrics VALUES ('cinch-cache', 3.2, 5200.0, 0.0000);
            """)

            cursor.execute(query)
            rows = cursor.fetchall()
            col_names = [d[0] for d in cursor.description] if cursor.description else []

            conn.close()

            if not col_names:
                return True, "Query executed successfully (0 rows returned)."

            results = [dict(zip(col_names, r)) for r in rows[:50]]
            return True, json.dumps(results, indent=2)
        except Exception as exc:
            return False, f"SQL execution error: {exc}"

    def execute_python_repl(self, code: str) -> Tuple[bool, str]:
        """Execute restricted Python code snippet with sanitized builtins."""
        # Block obvious shell/filesystem breakouts
        if re.search(r"\b(?:import\s+(?:os|sys|subprocess|shutil|socket|requests|urllib)|__import__|eval|exec|open)\b", code):
            return False, "Security violation: File system, network, and arbitrary code execution builtins are forbidden in sandbox."

        safe_globals: Dict[str, Any] = {
            "__builtins__": {
                "len": len, "range": range, "min": min, "max": max, "sum": sum,
                "sorted": sorted, "list": list, "dict": dict, "set": set, "tuple": tuple,
                "str": str, "int": int, "float": float, "bool": bool, "abs": abs,
                "round": round, "enumerate": enumerate, "zip": zip, "map": map,
                "filter": filter, "any": any, "all": all, "print": print,
            },
            "math": math,
            "json": json,
            "re": re,
        }
        safe_locals: Dict[str, Any] = {}

        stdout_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buf):
                # Try evaluating as single expression first
                try:
                    compiled = compile(code.strip(), "<sandbox>", "eval")
                    res = eval(compiled, safe_globals, safe_locals)
                    output = stdout_buf.getvalue()
                    if output and res is not None:
                        return True, f"{output}\n{res}".strip()
                    elif res is not None:
                        return True, str(res)
                    else:
                        return True, output.strip() or "None"
                except SyntaxError:
                    # Multi-line statement block
                    compiled = compile(code.strip(), "<sandbox>", "exec")
                    exec(compiled, safe_globals, safe_locals)
                    output = stdout_buf.getvalue()
                    if output:
                        return True, output.strip()
                    # Return modified local variables if any
                    clean_locals = {k: v for k, v in safe_locals.items() if not k.startswith("_")}
                    return True, json.dumps(clean_locals, default=str) if clean_locals else "Code executed successfully with no output."
        except Exception as exc:
            return False, f"Python execution error: {exc}"

    def execute_tool_call(
        self, tool_name: str, arguments: Dict[str, Any] | str, tool_call_id: str = ""
    ) -> ToolExecutionResult:
        """Dispatch tool call to appropriate sandboxed runtime."""
        t0 = time.perf_counter()
        self._total_executions += 1
        self._tool_usage[tool_name] = self._tool_usage.get(tool_name, 0) + 1

        # Parse string arguments if JSON string passed
        parsed_args: Dict[str, Any] = {}
        if isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments)
            except Exception:
                parsed_args = {"expression": arguments, "query": arguments, "code": arguments}
        elif isinstance(arguments, dict):
            parsed_args = arguments

        name = tool_name.lower().replace("-", "_")
        success = False
        output = ""
        error = None

        if name in ("calculator", "calc", "math_evaluator"):
            expr = parsed_args.get("expression") or parsed_args.get("expr") or str(parsed_args)
            success, output = self.execute_calculator(expr)
            if not success:
                error = output
        elif name in ("sql_runner", "sql", "database_query"):
            query = parsed_args.get("query") or parsed_args.get("sql") or str(parsed_args)
            success, output = self.execute_sql(query)
            if not success:
                error = output
        elif name in ("python_repl", "python", "code_interpreter"):
            code = parsed_args.get("code") or parsed_args.get("script") or str(parsed_args)
            success, output = self.execute_python_repl(code)
            if not success:
                error = output
        else:
            success = False
            error = f"Unknown tool '{tool_name}'. Available built-in tools: calculator, sql_runner, python_repl."
            output = error

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        self._total_exec_time_ms += elapsed_ms

        if success:
            self._successful_executions += 1
        else:
            self._failed_executions += 1

        return ToolExecutionResult(
            tool_name=tool_name,
            tool_call_id=tool_call_id or f"call_{int(time.time()*1000)}",
            success=success,
            output=output,
            error=error,
            execution_time_ms=elapsed_ms,
        )

    def extract_tool_calls(self, response_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool_calls array from standard OpenAI completion response or raw text tags."""
        choices = response_json.get("choices", [])
        if not choices or not isinstance(choices[0], dict):
            return []

        message = choices[0].get("message", {})
        # 1. Standard OpenAI tool_calls structure
        if "tool_calls" in message and isinstance(message["tool_calls"], list) and message["tool_calls"]:
            return message["tool_calls"]

        # 2. Raw text tags (e.g. <tool_call>{"name": "...", "arguments": ...}</tool_call>)
        content = message.get("content", "")
        if isinstance(content, str) and "<tool_call>" in content:
            tool_calls = []
            matches = re.findall(r"<tool_call>([\s\S]*?)</tool_call>", content)
            for i, match in enumerate(matches):
                try:
                    call_obj = json.loads(match.strip())
                    tool_calls.append({
                        "id": f"call_text_{i}_{int(time.time())}",
                        "type": "function",
                        "function": {
                            "name": call_obj.get("name", "calculator"),
                            "arguments": json.dumps(call_obj.get("arguments", {})),
                        },
                    })
                except Exception:
                    pass
            if tool_calls:
                return tool_calls

        # 3. Direct JSON structure with name and arguments
        if isinstance(content, str) and ('"name"' in content and '"arguments"' in content):
            try:
                clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
                start = clean.find("{")
                end = clean.rfind("}")
                if start != -1 and end != -1 and end > start:
                    obj = json.loads(clean[start : end + 1])
                    if "name" in obj and "arguments" in obj:
                        return [{
                            "id": f"call_json_{int(time.time())}",
                            "type": "function",
                            "function": {
                                "name": obj["name"],
                                "arguments": json.dumps(obj["arguments"]) if isinstance(obj["arguments"], dict) else str(obj["arguments"]),
                            },
                        }]
            except Exception:
                pass

        return []

    def prepare_upstream_request(self, body: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        Prepare request for upstream vLLM by converting OpenAI tools parameter into prompt instructions
        if vLLM does not have server-side auto-tool-choice enabled.
        """
        tools = body.get("tools")
        if not tools or not isinstance(tools, list):
            return body, False

        # Create copy of body without tools
        req_copy = {k: v for k, v in body.items() if k not in ("tools", "tool_choice", "server_tool_execution")}
        messages = [dict(m) for m in body.get("messages", [])]

        tools_prompt = (
            "\n# Tools\n\nYou have access to the following tools:\n"
            + json.dumps(tools, indent=2)
            + "\n\nIf you choose to call a tool, you MUST reply ONLY with a tool call block:\n"
            + "<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n"
        )

        # Inject into system prompt or prepend as system message
        sys_idx = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
        if sys_idx is not None:
            messages[sys_idx]["content"] = messages[sys_idx].get("content", "") + tools_prompt
        else:
            messages.insert(0, {"role": "system", "content": "You are a helpful assistant with access to tools." + tools_prompt})

        req_copy["messages"] = messages
        return req_copy, True

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational metrics for tool execution engine."""
        total = max(self._total_executions, 1)
        return {
            "enabled": self.enabled,
            "max_iterations": self.max_iterations,
            "sandbox_timeout_seconds": self.sandbox_timeout,
            "total_tool_calls_executed": self._total_executions,
            "successful_executions": self._successful_executions,
            "failed_executions": self._failed_executions,
            "success_rate_pct": round((self._successful_executions / total) * 100.0, 1),
            "tool_usage_breakdown": dict(self._tool_usage),
            "average_execution_time_ms": round(self._total_exec_time_ms / total, 3),
        }
