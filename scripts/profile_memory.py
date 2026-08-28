"""Analytical memory profiler and KV cache geometry calculator for Cinch LLM serving."""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
from typing import Any, Dict, List, Optional


@dataclasses.dataclass(frozen=True)
class ModelArchitecture:
    """Model architecture parameters relevant to KV cache geometry."""

    name: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    bytes_per_element: int = 2  # FP16 / BF16 = 2 bytes

    @property
    def bytes_per_token(self) -> int:
        """KV cache bytes required per token across all layers.

        Formula: 2 (Key + Value) * num_layers * num_kv_heads * head_dim * bytes_per_element
        """
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.bytes_per_element

    @property
    def mib_per_token(self) -> float:
        """KV cache MiB required per token."""
        return self.bytes_per_token / (1024 * 1024)


# Predefined architectures
QWEN2_5_7B = ModelArchitecture(
    name="Qwen/Qwen2.5-7B-Instruct-AWQ",
    num_layers=28,
    num_kv_heads=4,
    head_dim=128,
    bytes_per_element=2,
)


@dataclasses.dataclass(frozen=True)
class MemoryBudget:
    """Memory budget and resulting KV cache capacity."""

    total_gpu_vram_mib: float
    gpu_memory_utilization: float
    model_weights_mib: float
    peak_activation_mib: float
    max_model_len: int

    @property
    def allocated_vram_mib(self) -> float:
        """Total VRAM allocated to vLLM based on utilization fraction."""
        return self.total_gpu_vram_mib * self.gpu_memory_utilization

    @property
    def host_headroom_mib(self) -> float:
        """Headroom left for OS, desktop window manager, and display overhead."""
        return self.total_gpu_vram_mib - self.allocated_vram_mib

    @property
    def available_kv_cache_mib(self) -> float:
        """Remaining VRAM allocated specifically to the KV cache."""
        usable = self.allocated_vram_mib - self.model_weights_mib - self.peak_activation_mib
        return max(0.0, usable)

    def compute_token_capacity(self, arch: ModelArchitecture) -> int:
        """Calculate total number of tokens the KV cache can hold."""
        if self.available_kv_cache_mib <= 0:
            return 0
        total_bytes = self.available_kv_cache_mib * 1024 * 1024
        return int(total_bytes // arch.bytes_per_token)

    def compute_max_concurrency(self, arch: ModelArchitecture) -> float:
        """Calculate theoretical max concurrent requests at max_model_len."""
        if self.max_model_len <= 0:
            return 0.0
        total_tokens = self.compute_token_capacity(arch)
        return total_tokens / self.max_model_len


def evaluate_tuning_grid(
    arch: ModelArchitecture = QWEN2_5_7B,
    total_gpu_vram_mib: float = 8192.0,
    model_weights_mib: float = 5416.96,  # 5.29 GiB
    peak_activation_mib: float = 276.48,  # 0.27 GiB
    utilization_sweep: Optional[List[float]] = None,
    context_len_sweep: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate theoretical KV cache metrics across a parameter grid."""
    if utilization_sweep is None:
        utilization_sweep = [0.75, 0.80, 0.85, 0.90]
    if context_len_sweep is None:
        context_len_sweep = [2048, 4096, 8192]

    results = []
    for util in utilization_sweep:
        for ctx_len in context_len_sweep:
            budget = MemoryBudget(
                total_gpu_vram_mib=total_gpu_vram_mib,
                gpu_memory_utilization=util,
                model_weights_mib=model_weights_mib,
                peak_activation_mib=peak_activation_mib,
                max_model_len=ctx_len,
            )
            token_capacity = budget.compute_token_capacity(arch)
            max_concurrency = budget.compute_max_concurrency(arch)
            results.append(
                {
                    "gpu_memory_utilization": util,
                    "max_model_len": ctx_len,
                    "allocated_vram_mib": round(budget.allocated_vram_mib, 2),
                    "host_headroom_mib": round(budget.host_headroom_mib, 2),
                    "available_kv_cache_mib": round(budget.available_kv_cache_mib, 2),
                    "available_kv_cache_gib": round(budget.available_kv_cache_mib / 1024, 3),
                    "token_capacity": token_capacity,
                    "max_concurrency_at_max_len": round(max_concurrency, 2),
                }
            )
    return results


def query_host_gpu_telemetry() -> Dict[str, Any]:
    """Query current host GPU VRAM usage via nvidia-smi."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,nounits,noheader",
        ]
        output = subprocess.check_output(cmd, text=True).strip()
        parts = [p.strip() for p in output.split(",")]
        return {
            "total_mib": float(parts[0]),
            "used_mib": float(parts[1]),
            "free_mib": float(parts[2]),
            "gpu_utilization_pct": float(parts[3]),
            "status": "available",
        }
    except Exception as e:
        return {
            "status": "unavailable",
            "error": str(e),
        }


def format_tuning_table(records: List[Dict[str, Any]]) -> str:
    """Format tuning evaluation records as a readable markdown table."""
    headers = [
        "Utilization",
        "Allocated (MiB)",
        "Headroom (MiB)",
        "KV Cache (GiB)",
        "Tokens Cap.",
        "Max Context",
        "Max Concurrency",
    ]
    rows = []
    for r in records:
        rows.append(
            [
                f"{r['gpu_memory_utilization']:.2f}",
                f"{r['allocated_vram_mib']:.1f}",
                f"{r['host_headroom_mib']:.1f}",
                f"{r['available_kv_cache_gib']:.2f}",
                f"{r['token_capacity']:,}",
                f"{r['max_model_len']}",
                f"{r['max_concurrency_at_max_len']:.2f}x",
            ]
        )

    # Format Markdown
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data_lines = ["| " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + data_lines)


def main() -> None:
    """CLI entrypoint for memory tuning evaluation."""
    parser = argparse.ArgumentParser(description="Analytical memory profiler for Cinch vLLM serving")
    parser.add_argument("--gpu-vram-mib", type=float, default=8192.0, help="Total GPU VRAM in MiB")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct-AWQ", help="Model name")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    arch = QWEN2_5_7B
    results = evaluate_tuning_grid(arch=arch, total_gpu_vram_mib=args.gpu_vram_mib)
    telemetry = query_host_gpu_telemetry()

    if args.json:
        payload = {
            "model_architecture": {
                "name": arch.name,
                "layers": arch.num_layers,
                "kv_heads": arch.num_kv_heads,
                "head_dim": arch.head_dim,
                "bytes_per_token": arch.bytes_per_token,
                "mib_per_token": arch.mib_per_token,
            },
            "host_telemetry": telemetry,
            "tuning_grid": results,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"=== Memory Tuning Grid — {arch.name} ===")
        print(f"KV Cache geometry: {arch.bytes_per_token:,} bytes/token ({arch.mib_per_token:.5f} MiB/token)")
        print(f"GPU Hardware: {args.gpu_vram_mib:.0f} MiB VRAM")
        if telemetry.get("status") == "available":
            print(
                f"Host Live: {telemetry['used_mib']:.0f} MiB used / {telemetry['total_mib']:.0f} MiB total ({telemetry['free_mib']:.0f} MiB free)\n"
            )
        else:
            print(f"Host Live: nvidia-smi unavailable ({telemetry.get('error')})\n")

        print(format_tuning_table(results))


if __name__ == "__main__":
    main()
