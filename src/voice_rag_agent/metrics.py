from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


METRIC_FIELDS = [
    "retrieval_latency_ms",
    "llm_time_to_first_token_ms",
    "llm_total_latency_ms",
    "query_total_latency_ms",
    "tts_time_to_first_audio_ms",
    "turn_total_latency_ms",
    "tts_audio_chunks",
    "tts_audio_bytes",
]


@dataclass(frozen=True)
class MetricsReport:
    trace_dir: str
    total_trace_files: int
    total_metric_events: int
    mode_counts: dict[str, int]
    fields: dict[str, dict[str, float | int]]


def load_metric_events(
    trace_dir: Path,
    mode: str = "all",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if not trace_dir.exists():
        return []

    trace_files = sorted(trace_dir.glob("*.jsonl"), key=lambda path: path.stat().st_mtime)
    events: list[dict[str, Any]] = []

    for trace_file in trace_files:
        for line in trace_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("kind") != "event":
                continue

            if record.get("type") != "metrics.completed":
                continue

            payload = dict(record.get("payload") or {})
            payload["_trace_id"] = record.get("trace_id", "")
            payload["_created_at_ms"] = record.get("created_at_ms", 0)
            payload["_trace_file"] = str(trace_file)

            if mode != "all" and payload.get("mode") != mode:
                continue

            events.append(payload)

    if limit is not None and limit > 0:
        events = events[-limit:]

    return events


def summarize_metrics(
    trace_dir: Path,
    mode: str = "all",
    limit: int | None = None,
) -> MetricsReport:
    metric_events = load_metric_events(trace_dir, mode=mode, limit=limit)
    trace_file_count = len(list(trace_dir.glob("*.jsonl"))) if trace_dir.exists() else 0

    mode_counts: dict[str, int] = {}
    for event in metric_events:
        event_mode = str(event.get("mode", "unknown"))
        mode_counts[event_mode] = mode_counts.get(event_mode, 0) + 1

    field_summaries: dict[str, dict[str, float | int]] = {}

    for field in METRIC_FIELDS:
        values = [_as_float(event.get(field)) for event in metric_events]
        numeric_values = [value for value in values if value is not None]

        if not numeric_values:
            continue

        field_summaries[field] = _summarize_numbers(numeric_values)

    return MetricsReport(
        trace_dir=str(trace_dir),
        total_trace_files=trace_file_count,
        total_metric_events=len(metric_events),
        mode_counts=mode_counts,
        fields=field_summaries,
    )


def format_metrics_report(report: MetricsReport) -> str:
    lines: list[str] = []

    lines.append("Voice RAG Metrics Report")
    lines.append("=" * 24)
    lines.append(f"Trace dir: {report.trace_dir}")
    lines.append(f"Trace files: {report.total_trace_files}")
    lines.append(f"Metric events: {report.total_metric_events}")

    if report.mode_counts:
        mode_summary = ", ".join(
            f"{mode}={count}" for mode, count in sorted(report.mode_counts.items())
        )
        lines.append(f"Modes: {mode_summary}")

    if not report.fields:
        lines.append("")
        lines.append("No metric fields found yet.")
        lines.append("Run a streamed query or voice demo first, then run this command again.")
        return "\n".join(lines)

    lines.append("")
    lines.append("| Metric | Count | Avg | P50 | P95 | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for field, stats in report.fields.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    field,
                    str(stats["count"]),
                    _fmt(stats["avg"]),
                    _fmt(stats["p50"]),
                    _fmt(stats["p95"]),
                    _fmt(stats["min"]),
                    _fmt(stats["max"]),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def report_to_dict(report: MetricsReport) -> dict[str, Any]:
    return {
        "trace_dir": report.trace_dir,
        "total_trace_files": report.total_trace_files,
        "total_metric_events": report.total_metric_events,
        "mode_counts": report.mode_counts,
        "fields": report.fields,
    }


def _summarize_numbers(values: list[float]) -> dict[str, float | int]:
    sorted_values = sorted(values)

    return {
        "count": len(sorted_values),
        "avg": round(mean(sorted_values), 3),
        "p50": round(median(sorted_values), 3),
        "p95": round(_percentile(sorted_values, 95), 3),
        "min": round(sorted_values[0], 3),
        "max": round(sorted_values[-1], 3),
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0

    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (percentile / 100) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower

    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _as_float(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int | float):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None

    return None


def _fmt(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)

    if abs(value) >= 100:
        return f"{value:.1f}"

    return f"{value:.3f}"
