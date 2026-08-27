"""Automated benchmark comparison table generator for Cinch."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from typing import Any, Dict, List, Optional


def load_result_json(path: str) -> Dict[str, Any]:
    """Load benchmark result JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_comparison_data(
    baseline_payload: Dict[str, Any],
    optimized_payload: Dict[str, Any],
    gateway_payload: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build comparative metric dictionary per concurrency tier."""
    base_tiers = {t["concurrency"]: t for t in baseline_payload.get("tiers", [])}
    opt_tiers = {t["concurrency"]: t for t in optimized_payload.get("tiers", [])}
    gw_tiers = {t["concurrency"]: t for t in gateway_payload.get("tiers", [])} if gateway_payload else {}

    all_concurrencies = sorted(set(base_tiers.keys()) | set(opt_tiers.keys()))
    comparison_rows: List[Dict[str, Any]] = []

    for c in all_concurrencies:
        base = base_tiers.get(c, {})
        opt = opt_tiers.get(c, {})
        gw = gw_tiers.get(c, {})

        base_tps = base.get("tokens_per_second", 0.0)
        opt_tps = opt.get("tokens_per_second", 0.0)
        tps_speedup = (opt_tps / base_tps) if base_tps > 0 else 0.0

        base_p50 = base.get("latency_p50", 0.0)
        opt_p50 = opt.get("latency_p50", 0.0)
        p50_reduction = (base_p50 / opt_p50) if opt_p50 > 0 else 0.0

        base_p95 = base.get("latency_p95", 0.0)
        opt_p95 = opt.get("latency_p95", 0.0)
        p95_reduction = (base_p95 / opt_p95) if opt_p95 > 0 else 0.0

        gw_p50 = gw.get("latency_p50")
        gw_overhead_ms = ((gw_p50 - opt_p50) * 1000.0) if (gw_p50 is not None and opt_p50 > 0) else None

        row = {
            "concurrency": c,
            "baseline_engine": baseline_payload.get("engine", "Baseline"),
            "optimized_engine": optimized_payload.get("engine", "vLLM-AWQ"),
            "baseline_tps": round(base_tps, 2),
            "optimized_tps": round(opt_tps, 2),
            "throughput_speedup": round(tps_speedup, 2),
            "baseline_p50": round(base_p50, 4),
            "optimized_p50": round(opt_p50, 4),
            "p50_reduction": round(p50_reduction, 2),
            "baseline_p95": round(base_p95, 4),
            "optimized_p95": round(opt_p95, 4),
            "p95_reduction": round(p95_reduction, 2),
            "baseline_vram_peak": base.get("peak_vram_mib"),
            "optimized_vram_peak": opt.get("peak_vram_mib"),
            "gateway_p50": round(gw_p50, 4) if gw_p50 is not None else None,
            "gateway_overhead_ms": round(gw_overhead_ms, 2) if gw_overhead_ms is not None else None,
        }
        comparison_rows.append(row)

    return comparison_rows


def format_markdown_table(rows: List[Dict[str, Any]]) -> str:
    """Format comparison data as a GitHub Markdown table."""
    lines = [
        "| Concurrency | HF Baseline (tok/s) | vLLM+AWQ (tok/s) | Throughput Speedup | HF p50 Latency (s) | vLLM p50 Latency (s) | p50 Reduction | HF p95 Latency (s) | vLLM p95 Latency (s) | p95 Reduction | Peak VRAM (MiB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        vram_display = f"{r.get('optimized_vram_peak') or 0:.0f}"
        lines.append(
            f"| **{r['concurrency']}** | "
            f"{r['baseline_tps']:.1f} | "
            f"**{r['optimized_tps']:.1f}** | "
            f"**{r['throughput_speedup']:.2f}x** | "
            f"{r['baseline_p50']:.3f} | "
            f"**{r['optimized_p50']:.3f}** | "
            f"**{r['p50_reduction']:.2f}x** | "
            f"{r['baseline_p95']:.3f} | "
            f"**{r['optimized_p95']:.3f}** | "
            f"**{r['p95_reduction']:.2f}x** | "
            f"{vram_display} |"
        )
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint for benchmark comparison."""
    parser = argparse.ArgumentParser(description="Generate LLM Serving Benchmark Comparison Table")
    parser.add_argument("--baseline", type=str, default="benchmarks/results/baseline_hf.json", help="Path to baseline result JSON")
    parser.add_argument("--optimized", type=str, default="benchmarks/results/vllm_awq.json", help="Path to optimized result JSON")
    parser.add_argument("--gateway", type=str, default=None, help="Path to gateway result JSON (optional)")
    parser.add_argument("--output", type=str, default=None, help="Path to output summary JSON")
    args = parser.parse_args()

    base_path = pathlib.Path(args.baseline)
    opt_path = pathlib.Path(args.optimized)

    if not base_path.exists():
        print(f"Error: Baseline file not found at {base_path}")
        return
    if not opt_path.exists():
        print(f"Error: Optimized file not found at {opt_path}")
        return

    baseline_data = load_result_json(str(base_path))
    optimized_data = load_result_json(str(opt_path))
    gateway_data = load_result_json(args.gateway) if (args.gateway and pathlib.Path(args.gateway).exists()) else None

    rows = build_comparison_data(
        baseline_payload=baseline_data,
        optimized_payload=optimized_data,
        gateway_payload=gateway_data,
    )

    md_table = format_markdown_table(rows)
    print("\n=== Benchmark Comparison Table ===\n")
    print(md_table)
    print()

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"comparison": rows}, f, indent=2)
        print(f"[SAVED] Comparison summary written to {args.output}")


if __name__ == "__main__":
    main()
