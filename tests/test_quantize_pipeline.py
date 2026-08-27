"""Unit test suite for AutoAWQ Quantization Pipeline."""

from __future__ import annotations

import json
import os
import pytest
from scripts.quantize_awq import (
    calculate_quantization_statistics,
    generate_quantization_config,
    prepare_calibration_dataset,
    run_awq_quantization,
)


def test_prepare_calibration_dataset() -> None:
    """Verify calibration dataset batching."""
    data = prepare_calibration_dataset(samples=32)
    assert len(data) == 32
    assert all(isinstance(s, str) and len(s) > 10 for s in data)


def test_generate_quantization_config() -> None:
    """Verify quantization config layout for Marlin GEMM."""
    cfg = generate_quantization_config(w_bit=4, q_group_size=128, version="GEMM")
    assert cfg["w_bit"] == 4
    assert cfg["q_group_size"] == 128
    assert cfg["version"] == "GEMM"
    assert cfg["zero_point"] is True


def test_calculate_quantization_statistics() -> None:
    """Verify memory savings calculations and compression ratio."""
    stats = calculate_quantization_statistics(fp16_size_gb=14.4, w_bit=4, q_group_size=128)
    assert stats["fp16_size_gb"] == 14.4
    assert stats["quantized_size_gb"] < 4.5
    assert stats["compression_ratio"] >= 3.2
    assert stats["vram_saved_gb"] >= 9.5


def test_run_awq_quantization_dry_run(tmp_path: pytest.TempPathFactory) -> None:
    """Verify full dry-run execution creating config and manifest files."""
    dest_dir = str(tmp_path / "awq_test_model")

    summary = run_awq_quantization(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        quant_path=dest_dir,
        w_bit=4,
        q_group_size=128,
        calib_samples=16,
        dry_run=True,
    )

    assert summary["status"] == "dry_run_validated"
    assert os.path.exists(os.path.join(dest_dir, "quant_config.json"))
    assert os.path.exists(os.path.join(dest_dir, "quantization_summary.json"))

    with open(os.path.join(dest_dir, "quant_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["w_bit"] == 4
    assert cfg["version"] == "GEMM"
