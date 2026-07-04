from __future__ import annotations

import asyncio
import cProfile
import io
import json
import pstats
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class ProfileFunction:
    function: str
    calls: int
    total_time_ms: float
    cumulative_time_ms: float
    per_call_ms: float


@dataclass(frozen=True)
class ProfileReport:
    mode: str
    query: str
    repeat: int
    warmup: int
    total_duration_ms: float
    avg_duration_ms: float
    peak_memory_kb: float
    current_memory_kb: float
    top_functions: list[ProfileFunction]


def run_profile(
    *,
    engine: Any,
    query: str,
    mode: str = "retrieve",
    repeat: int = 100,
    warmup: int = 5,
    top_k: int | None = None,
    top_functions: int = 15,
) -> ProfileReport:
    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    if mode not in {"retrieve", "query"}:
        raise ValueError("Profile mode must be retrieve or query.")

    if repeat <= 0:
        raise ValueError("Repeat must be positive.")

    if warmup < 0:
        raise ValueError("Warmup cannot be negative.")

    if top_functions <= 0:
        raise ValueError("Top functions must be positive.")

    for _ in range(warmup):
        _run_once(engine, query, mode, top_k)

    profiler = cProfile.Profile()
    tracemalloc.start()

    started = time.perf_counter()
    profiler.enable()

    for _ in range(repeat):
        _run_once(engine, query, mode, top_k)

    profiler.disable()
    total_duration_ms = (time.perf_counter() - started) * 1000
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return ProfileReport(
        mode=mode,
        query=query,
        repeat=repeat,
        warmup=warmup,
        total_duration_ms=round(total_duration_ms, 3),
        avg_duration_ms=round(total_duration_ms / repeat, 3),
        peak_memory_kb=round(peak_bytes / 1024, 3),
        current_memory_kb=round(current_bytes / 1024, 3),
        top_functions=_extract_top_functions(profiler, top_functions),
    )


def check_profile_thresholds(
    report: ProfileReport,
    *,
    max_avg_ms: float | None = None,
    max_peak_memory_kb: float | None = None,
) -> list[str]:
    failures: list[str] = []

    if max_avg_ms is not None and report.avg_duration_ms > max_avg_ms:
        failures.append(
            f"average duration {report.avg_duration_ms:.3f} ms exceeded threshold {max_avg_ms:.3f} ms"
        )

    if max_peak_memory_kb is not None and report.peak_memory_kb > max_peak_memory_kb:
        failures.append(
            f"peak memory {report.peak_memory_kb:.3f} KiB exceeded threshold {max_peak_memory_kb:.3f} KiB"
        )

    return failures


def format_profile_report(report: ProfileReport) -> str:
    lines = [
        "Voice RAG Profile Report",
        "========================",
        f"Mode: {report.mode}",
        f"Query: {report.query}",
        f"Repeat: {report.repeat}",
        f"Warmup: {report.warmup}",
        f"Total duration: {report.total_duration_ms:.3f} ms",
        f"Average duration: {report.avg_duration_ms:.3f} ms",
        f"Peak memory: {report.peak_memory_kb:.3f} KiB",
        f"Current memory: {report.current_memory_kb:.3f} KiB",
        "",
        "| Function | Calls | Total ms | Cumulative ms | Per call ms |",
        "|---|---:|---:|---:|---:|",
    ]

    for item in report.top_functions:
        lines.append(
            f"| `{item.function}` | {item.calls} | "
            f"{item.total_time_ms:.3f} | {item.cumulative_time_ms:.3f} | "
            f"{item.per_call_ms:.6f} |"
        )

    return "\n".join(lines)


def profile_report_to_dict(report: ProfileReport) -> dict[str, Any]:
    return {
        "mode": report.mode,
        "query": report.query,
        "repeat": report.repeat,
        "warmup": report.warmup,
        "total_duration_ms": report.total_duration_ms,
        "avg_duration_ms": report.avg_duration_ms,
        "peak_memory_kb": report.peak_memory_kb,
        "current_memory_kb": report.current_memory_kb,
        "top_functions": [
            {
                "function": item.function,
                "calls": item.calls,
                "total_time_ms": item.total_time_ms,
                "cumulative_time_ms": item.cumulative_time_ms,
                "per_call_ms": item.per_call_ms,
            }
            for item in report.top_functions
        ],
    }


def write_profile_report(report: ProfileReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile_report_to_dict(report), indent=2),
        encoding="utf-8",
    )


def _run_once(engine: Any, query: str, mode: str, top_k: int | None) -> None:
    if mode == "retrieve":
        if top_k is None:
            engine.retrieve(query)
        else:
            engine.retrieve(query, top_k=top_k)
        return

    asyncio.run(engine.query(query))


def _extract_top_functions(
    profiler: cProfile.Profile,
    limit: int,
) -> list[ProfileFunction]:
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")

    rows: list[ProfileFunction] = []

    raw_stats = cast(dict[Any, Any], stats.stats)  # type: ignore[attr-defined]

    for func, data in sorted(
        raw_stats.items(),
        key=lambda item: item[1][3],
        reverse=True,
    )[:limit]:
        primitive_calls, total_calls, total_time, cumulative_time, _callers = data
        filename, line_no, function_name = func
        function = f"{Path(filename).name}:{line_no}:{function_name}"
        per_call = cumulative_time / total_calls if total_calls else 0.0

        rows.append(
            ProfileFunction(
                function=function,
                calls=total_calls,
                total_time_ms=round(total_time * 1000, 3),
                cumulative_time_ms=round(cumulative_time * 1000, 3),
                per_call_ms=round(per_call * 1000, 6),
            )
        )

    return rows
