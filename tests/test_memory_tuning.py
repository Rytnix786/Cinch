"""Tests for memory tuning and analytical KV cache calculations."""

from __future__ import annotations

import json
from unittest.mock import patch
import pytest

from scripts.profile_memory import (
    QWEN2_5_7B,
    MemoryBudget,
    ModelArchitecture,
    evaluate_tuning_grid,
    format_tuning_table,
    main,
    query_host_gpu_telemetry,
)


def test_qwen2_5_7b_architecture_constants() -> None:
    """Verify Qwen2.5-7B architecture and KV cache geometry."""
    assert QWEN2_5_7B.num_layers == 28
    assert QWEN2_5_7B.num_kv_heads == 4
    assert QWEN2_5_7B.head_dim == 128
    assert QWEN2_5_7B.bytes_per_element == 2

    # 2 * 28 layers * 4 kv heads * 128 dim * 2 bytes = 57,344 bytes / token
    assert QWEN2_5_7B.bytes_per_token == 57344
    assert pytest.approx(QWEN2_5_7B.mib_per_token, 0.00001) == 57344 / (1024 * 1024)


def test_custom_architecture() -> None:
    """Verify custom model architecture KV calculation."""
    arch = ModelArchitecture(
        name="test-model",
        num_layers=32,
        num_kv_heads=8,
        head_dim=64,
        bytes_per_element=2,
    )
    # 2 * 32 * 8 * 64 * 2 = 65,536 bytes
    assert arch.bytes_per_token == 65536
    assert arch.mib_per_token == 65536 / (1024 * 1024)


def test_memory_budget_calculations() -> None:
    """Verify memory budget calculations at 0.85 utilization on 8192 MiB GPU."""
    budget = MemoryBudget(
        total_gpu_vram_mib=8192.0,
        gpu_memory_utilization=0.85,
        model_weights_mib=5416.96,  # 5.29 GiB
        peak_activation_mib=276.48,  # 0.27 GiB
        max_model_len=4096,
    )

    assert pytest.approx(budget.allocated_vram_mib, 0.01) == 6963.20
    assert pytest.approx(budget.host_headroom_mib, 0.01) == 1228.80

    # 6963.20 - 5416.96 - 276.48 = 1269.76 MiB
    assert pytest.approx(budget.available_kv_cache_mib, 0.01) == 1269.76

    token_capacity = budget.compute_token_capacity(QWEN2_5_7B)
    # 1269.76 * 1024 * 1024 / 57344 = 23223
    assert token_capacity == int(1269.76 * 1024 * 1024 // 57344)
    assert token_capacity > 20000

    max_concurrency = budget.compute_max_concurrency(QWEN2_5_7B)
    assert pytest.approx(max_concurrency, 0.01) == token_capacity / 4096


def test_memory_budget_insufficient_vram() -> None:
    """Verify behavior when model weights exceed allocated VRAM."""
    budget = MemoryBudget(
        total_gpu_vram_mib=8192.0,
        gpu_memory_utilization=0.50,  # 4096 MiB allocated
        model_weights_mib=5416.96,
        peak_activation_mib=276.48,
        max_model_len=4096,
    )
    assert budget.available_kv_cache_mib == 0.0
    assert budget.compute_token_capacity(QWEN2_5_7B) == 0
    assert budget.compute_max_concurrency(QWEN2_5_7B) == 0.0


def test_evaluate_tuning_grid() -> None:
    """Verify grid evaluation produces expected records."""
    grid = evaluate_tuning_grid(
        arch=QWEN2_5_7B,
        total_gpu_vram_mib=8192.0,
        utilization_sweep=[0.80, 0.85],
        context_len_sweep=[2048, 4096],
    )
    assert len(grid) == 4
    for record in grid:
        assert "gpu_memory_utilization" in record
        assert "max_model_len" in record
        assert "allocated_vram_mib" in record
        assert "available_kv_cache_gib" in record
        assert "token_capacity" in record
        assert "max_concurrency_at_max_len" in record


def test_format_tuning_table() -> None:
    """Verify markdown table formatting."""
    grid = evaluate_tuning_grid(
        arch=QWEN2_5_7B,
        total_gpu_vram_mib=8192.0,
        utilization_sweep=[0.85],
        context_len_sweep=[4096],
    )
    table = format_tuning_table(grid)
    assert "Utilization" in table
    assert "Allocated (MiB)" in table
    assert "0.85" in table
    assert "4096" in table


def test_query_host_gpu_telemetry_success() -> None:
    """Verify host telemetry parser when nvidia-smi succeeds."""
    mock_csv = "8192, 450, 7742, 1\n"
    with patch("subprocess.check_output", return_value=mock_csv):
        telemetry = query_host_gpu_telemetry()
        assert telemetry["status"] == "available"
        assert telemetry["total_mib"] == 8192.0
        assert telemetry["used_mib"] == 450.0
        assert telemetry["free_mib"] == 7742.0
        assert telemetry["gpu_utilization_pct"] == 1.0


def test_query_host_gpu_telemetry_failure() -> None:
    """Verify host telemetry fallback when nvidia-smi is unavailable."""
    with patch("subprocess.check_output", side_effect=FileNotFoundError("nvidia-smi not found")):
        telemetry = query_host_gpu_telemetry()
        assert telemetry["status"] == "unavailable"
        assert "not found" in telemetry["error"]


def test_cli_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI --json output format."""
    with patch("sys.argv", ["profile_memory.py", "--json", "--gpu-vram-mib", "8192"]):
        main()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["model_architecture"]["name"] == "Qwen/Qwen2.5-7B-Instruct-AWQ"
        assert len(data["tuning_grid"]) > 0


def test_cli_text_output(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI default text output."""
    with patch("sys.argv", ["profile_memory.py", "--gpu-vram-mib", "8192"]):
        main()
        captured = capsys.readouterr()
        assert "=== Memory Tuning Grid" in captured.out
        assert "KV Cache geometry" in captured.out
