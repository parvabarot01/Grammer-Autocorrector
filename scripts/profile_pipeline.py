"""Profile end-to-end correction latency across short, medium, and long inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.correction_pipeline import CorrectionPipeline
from src.utils.config import BASE_DIR, load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pipeline profiling."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--output_path",
        default=str(BASE_DIR / "results" / "performance_profile.md"),
    )
    return parser.parse_args()


def percentile(values: List[float], percentile_value: float) -> float:
    """Compute a percentile from a list of latency values."""

    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(int(round((percentile_value / 100.0) * len(ordered) + 0.5)) - 1, 0)
    return ordered[min(rank, len(ordered) - 1)]


def sample_inputs() -> Dict[str, str]:
    """Return profiling inputs across three text-length buckets."""

    return {
        "short": "She go to school everyday.",
        "medium": (
            "Yesterday he go to the market and buy a apple, "
            "then he say the shop is close too early."
        ),
        "long": (
            "Last year they goes to many different cities for work, and each time "
            "they write reports that contain article mistakes, tense problems, and "
            "small punctuation issues, so the team need a system that can review "
            "the full paragraph quickly while still keeping the original meaning."
        ),
    }


def main() -> int:
    """Run correction profiling and write a markdown report."""

    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = CorrectionPipeline(load_config())
    pipeline.load_all()

    rows: List[tuple[str, int, float, float, float, float, float]] = []
    for bucket, text in sample_inputs().items():
        latencies: List[float] = []
        started = perf_counter()
        for _ in range(args.iterations):
            result = pipeline.correct(text, mode="auto")
            latencies.append(result.processing_time_ms)
        total_duration = perf_counter() - started
        throughput = args.iterations / total_duration if total_duration else 0.0
        rows.append(
            (
                bucket,
                args.iterations,
                fmean(latencies),
                percentile(latencies, 50),
                percentile(latencies, 95),
                percentile(latencies, 99),
                throughput,
            )
        )

    timestamp = pipeline.utcnow()
    lines = [
        "# Performance Profile",
        "",
        "| Bucket | Requests | Mean Latency (ms) | P50 | P95 | P99 | Throughput (req/sec) | Timestamp |",
        "|--------|----------|-------------------|-----|-----|-----|----------------------|-----------|",
    ]
    for bucket, requests_count, mean_latency, p50, p95, p99, throughput in rows:
        lines.append(
            f"| {bucket} | {requests_count} | {mean_latency:.3f} | {p50:.3f} | "
            f"{p95:.3f} | {p99:.3f} | {throughput:.3f} | {timestamp} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
