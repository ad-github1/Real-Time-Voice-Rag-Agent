from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


@dataclass(frozen=True)
class BenchmarkRequestResult:
    query: str
    ok: bool
    latency_ms: float
    trace_id: str
    citation_count: int
    error: str


@dataclass(frozen=True)
class BenchmarkReport:
    queries: list[str]
    concurrency: int
    repeat: int
    mode: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    total_duration_ms: float
    throughput_qps: float
    latency: dict[str, float | int]
    results: list[BenchmarkRequestResult]


def load_benchmark_queries(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark query file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError("Benchmark query file is empty.")

    if path.suffix.lower() == ".json":
        return _queries_from_json(json.loads(text))

    queries = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not queries:
        raise ValueError("No benchmark queries found.")

    return queries


async def run_query_benchmark(
    *,
    engine: Any,
    queries: list[str],
    concurrency: int = 4,
    repeat: int = 1,
    mode: str = "query",
    top_k: int | None = None,
) -> BenchmarkReport:
    if not queries:
        raise ValueError("At least one benchmark query is required.")

    if concurrency <= 0:
        raise ValueError("Concurrency must be positive.")

    if repeat <= 0:
        raise ValueError("Repeat must be positive.")

    if mode not in {"query", "retrieve"}:
        raise ValueError("Benchmark mode must be query or retrieve.")

    jobs = [query for _ in range(repeat) for query in queries]
    semaphore = asyncio.Semaphore(concurrency)
    started = time.perf_counter()

    async def run_one(query: str) -> BenchmarkRequestResult:
        async with semaphore:
            request_started = time.perf_counter()

            try:
                if mode == "query":
                    answer = await engine.query(query)
                    trace_id = str(getattr(answer, "trace_id", ""))
                    citation_count = len(getattr(answer, "citations", []) or [])
                else:
                    if top_k is None:
                        retrieved = engine.retrieve(query)
                    else:
                        retrieved = engine.retrieve(query, top_k=top_k)

                    trace_id = ""
                    citation_count = len(retrieved or [])

                latency_ms = (time.perf_counter() - request_started) * 1000

                return BenchmarkRequestResult(
                    query=query,
                    ok=True,
                    latency_ms=latency_ms,
                    trace_id=trace_id,
                    citation_count=citation_count,
                    error="",
                )
            except Exception as exc:  # noqa: BLE001
                latency_ms = (time.perf_counter() - request_started) * 1000

                return BenchmarkRequestResult(
                    query=query,
                    ok=False,
                    latency_ms=latency_ms,
                    trace_id="",
                    citation_count=0,
                    error=f"{type(exc).__name__}: {exc}",
                )

    results = await asyncio.gather(*(run_one(query) for query in jobs))
    total_duration_ms = (time.perf_counter() - started) * 1000
    successful = sum(1 for result in results if result.ok)
    failed = len(results) - successful

    return BenchmarkReport(
        queries=queries,
        concurrency=concurrency,
        repeat=repeat,
        mode=mode,
        total_requests=len(results),
        successful_requests=successful,
        failed_requests=failed,
        error_rate=round(failed / len(results), 6) if results else 0.0,
        total_duration_ms=round(total_duration_ms, 3),
        throughput_qps=round(len(results) / (total_duration_ms / 1000), 3)
        if total_duration_ms > 0
        else 0.0,
        latency=_summarize_latencies([result.latency_ms for result in results]),
        results=results,
    )


async def run_http_benchmark(
    *,
    url: str,
    queries: list[str],
    concurrency: int = 4,
    repeat: int = 1,
    api_key: str = "",
    timeout_ms: float = 5000.0,
) -> BenchmarkReport:
    if not queries:
        raise ValueError("At least one benchmark query is required.")

    if concurrency <= 0:
        raise ValueError("Concurrency must be positive.")

    if repeat <= 0:
        raise ValueError("Repeat must be positive.")

    if timeout_ms <= 0:
        raise ValueError("Timeout must be positive.")

    jobs = [query for _ in range(repeat) for query in queries]
    semaphore = asyncio.Semaphore(concurrency)
    started = time.perf_counter()
    timeout_seconds = timeout_ms / 1000

    async def run_one(query: str) -> BenchmarkRequestResult:
        async with semaphore:
            request_started = time.perf_counter()

            try:
                ok, trace_id, citation_count, error = await asyncio.to_thread(
                    _post_query,
                    url=url,
                    query=query,
                    api_key=api_key,
                    timeout_seconds=timeout_seconds,
                )
                latency_ms = (time.perf_counter() - request_started) * 1000

                return BenchmarkRequestResult(
                    query=query,
                    ok=ok,
                    latency_ms=latency_ms,
                    trace_id=trace_id,
                    citation_count=citation_count,
                    error=error,
                )
            except Exception as exc:  # noqa: BLE001
                latency_ms = (time.perf_counter() - request_started) * 1000

                return BenchmarkRequestResult(
                    query=query,
                    ok=False,
                    latency_ms=latency_ms,
                    trace_id="",
                    citation_count=0,
                    error=f"{type(exc).__name__}: {exc}",
                )

    results = await asyncio.gather(*(run_one(query) for query in jobs))
    total_duration_ms = (time.perf_counter() - started) * 1000
    successful = sum(1 for result in results if result.ok)
    failed = len(results) - successful

    return BenchmarkReport(
        queries=queries,
        concurrency=concurrency,
        repeat=repeat,
        mode="http",
        total_requests=len(results),
        successful_requests=successful,
        failed_requests=failed,
        error_rate=round(failed / len(results), 6) if results else 0.0,
        total_duration_ms=round(total_duration_ms, 3),
        throughput_qps=round(len(results) / (total_duration_ms / 1000), 3)
        if total_duration_ms > 0
        else 0.0,
        latency=_summarize_latencies([result.latency_ms for result in results]),
        results=results,
    )


def format_benchmark_report(report: BenchmarkReport) -> str:
    lines = [
        "Voice RAG Benchmark Report",
        "==========================",
        f"Unique queries: {len(report.queries)}",
        f"Repeat: {report.repeat}",
        f"Mode: {report.mode}",
        f"Concurrency: {report.concurrency}",
        f"Total requests: {report.total_requests}",
        f"Successful: {report.successful_requests}",
        f"Failed: {report.failed_requests}",
        f"Error rate: {report.error_rate:.2%}",
        f"Total duration: {report.total_duration_ms:.1f} ms",
        f"Throughput: {report.throughput_qps:.3f} qps",
        "",
        "| Metric | Value ms |",
        "|---|---:|",
    ]

    for key in ["avg", "p50", "p95", "p99", "min", "max"]:
        value = report.latency.get(key)
        if value is not None:
            lines.append(f"| latency_{key} | {_fmt_float(value)} |")

    if report.failed_requests:
        lines.append("")
        lines.append("Failures:")
        for result in report.results:
            if not result.ok:
                lines.append(f"- {result.query}: {result.error}")

    return "\n".join(lines)


def report_to_dict(report: BenchmarkReport) -> dict[str, Any]:
    return {
        "queries": report.queries,
        "concurrency": report.concurrency,
        "repeat": report.repeat,
        "mode": report.mode,
        "total_requests": report.total_requests,
        "successful_requests": report.successful_requests,
        "failed_requests": report.failed_requests,
        "error_rate": report.error_rate,
        "total_duration_ms": report.total_duration_ms,
        "throughput_qps": report.throughput_qps,
        "latency": report.latency,
        "results": [
            {
                "query": result.query,
                "ok": result.ok,
                "latency_ms": round(result.latency_ms, 3),
                "trace_id": result.trace_id,
                "citation_count": result.citation_count,
                "error": result.error,
            }
            for result in report.results
        ],
    }


def check_benchmark_thresholds(
    report: BenchmarkReport,
    *,
    max_p95_ms: float | None = None,
    max_error_rate: float | None = None,
    min_qps: float | None = None,
) -> list[str]:
    failures: list[str] = []

    p95 = report.latency.get("p95")

    if max_p95_ms is not None and isinstance(p95, int | float) and p95 > max_p95_ms:
        failures.append(f"p95 latency {p95:.3f} ms exceeded threshold {max_p95_ms:.3f} ms")

    if max_error_rate is not None and report.error_rate > max_error_rate:
        failures.append(
            f"error rate {report.error_rate:.6f} exceeded threshold {max_error_rate:.6f}"
        )

    if min_qps is not None and report.throughput_qps < min_qps:
        failures.append(
            f"throughput {report.throughput_qps:.3f} qps below threshold {min_qps:.3f} qps"
        )

    return failures


def load_benchmark_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Baseline benchmark report not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("Benchmark report must be a JSON object.")

    return payload


def benchmark_baseline_summary(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float | None]:
    baseline_p95 = _latency_metric(baseline, "p95")
    current_p95 = _latency_metric(current, "p95")
    baseline_qps = _numeric_metric(baseline, "throughput_qps")
    current_qps = _numeric_metric(current, "throughput_qps")
    baseline_error_rate = _numeric_metric(baseline, "error_rate")
    current_error_rate = _numeric_metric(current, "error_rate")

    return {
        "baseline_p95_ms": baseline_p95,
        "current_p95_ms": current_p95,
        "p95_regression_percent": _percent_change(current_p95, baseline_p95),
        "baseline_qps": baseline_qps,
        "current_qps": current_qps,
        "qps_ratio": _ratio(current_qps, baseline_qps),
        "baseline_error_rate": baseline_error_rate,
        "current_error_rate": current_error_rate,
        "error_rate_delta": _delta(current_error_rate, baseline_error_rate),
    }


def compare_benchmark_reports(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    max_p95_regression_percent: float | None = None,
    max_error_rate_delta: float | None = None,
    min_qps_ratio: float | None = None,
) -> list[str]:
    failures: list[str] = []

    baseline_p95 = _latency_metric(baseline, "p95")
    current_p95 = _latency_metric(current, "p95")

    if (
        max_p95_regression_percent is not None
        and baseline_p95 is not None
        and current_p95 is not None
        and baseline_p95 > 0
    ):
        regression = ((current_p95 - baseline_p95) / baseline_p95) * 100

        if regression > max_p95_regression_percent:
            failures.append(
                "p95 latency regressed by "
                f"{regression:.2f}% "
                f"({baseline_p95:.3f} ms -> {current_p95:.3f} ms), "
                f"threshold {max_p95_regression_percent:.2f}%"
            )

    baseline_error_rate = _numeric_metric(baseline, "error_rate")
    current_error_rate = _numeric_metric(current, "error_rate")

    if (
        max_error_rate_delta is not None
        and baseline_error_rate is not None
        and current_error_rate is not None
    ):
        delta = current_error_rate - baseline_error_rate

        if delta > max_error_rate_delta:
            failures.append(
                "error rate increased by "
                f"{delta:.6f} "
                f"({baseline_error_rate:.6f} -> {current_error_rate:.6f}), "
                f"threshold {max_error_rate_delta:.6f}"
            )

    baseline_qps = _numeric_metric(baseline, "throughput_qps")
    current_qps = _numeric_metric(current, "throughput_qps")

    if (
        min_qps_ratio is not None
        and baseline_qps is not None
        and current_qps is not None
        and baseline_qps > 0
    ):
        ratio = current_qps / baseline_qps

        if ratio < min_qps_ratio:
            failures.append(
                "throughput ratio "
                f"{ratio:.3f} "
                f"({baseline_qps:.3f} qps -> {current_qps:.3f} qps), "
                f"minimum {min_qps_ratio:.3f}"
            )

    return failures


def _latency_metric(payload: dict[str, Any], key: str) -> float | None:
    latency = payload.get("latency")

    if not isinstance(latency, dict):
        return None

    value = latency.get(key)

    return float(value) if isinstance(value, int | float) else None


def _numeric_metric(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)

    return float(value) if isinstance(value, int | float) else None


def _percent_change(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None or baseline <= 0:
        return None

    return round(((current - baseline) / baseline) * 100, 6)


def _ratio(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None or baseline <= 0:
        return None

    return round(current / baseline, 6)


def _delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None

    return round(current - baseline, 6)


def _post_query(
    *,
    url: str,
    query: str,
    api_key: str,
    timeout_seconds: float,
) -> tuple[bool, str, int, str]:
    payload = json.dumps({"query": query}).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "voice-rag-agent-benchmark",
    }

    if api_key:
        headers["X-API-Key"] = api_key

    request = urllib_request.Request(
        url=url,
        data=payload,
        headers=headers,
        method="POST",
    )

    try:
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            status_code = response.getcode()
            body = response.read().decode("utf-8", errors="replace")
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return False, "", 0, f"HTTP {exc.code}: {body[:300]}"
    except urllib_error.URLError as exc:
        return False, "", 0, f"URLError: {exc.reason}"

    if status_code < 200 or status_code >= 300:
        return False, "", 0, f"HTTP {status_code}: {body[:300]}"

    try:
        response_payload = json.loads(body) if body.strip() else {}
    except json.JSONDecodeError:
        response_payload = {}

    citations = response_payload.get("citations") or []

    return (
        True,
        str(response_payload.get("trace_id", "")),
        len(citations) if isinstance(citations, list) else 0,
        "",
    )


def _queries_from_json(payload: Any) -> list[str]:
    if isinstance(payload, list):
        queries = []
        for item in payload:
            if isinstance(item, str):
                queries.append(item)
            elif isinstance(item, dict) and isinstance(item.get("query"), str):
                queries.append(item["query"])

        cleaned = [query.strip() for query in queries if query.strip()]
        if cleaned:
            return cleaned

    if isinstance(payload, dict):
        raw_queries = payload.get("queries")
        if isinstance(raw_queries, list):
            dict_queries: list[str] = []
            for item in raw_queries:
                if isinstance(item, str):
                    dict_queries.append(item)
                elif isinstance(item, dict):
                    query = item.get("query")
                    if isinstance(query, str):
                        dict_queries.append(query)

            cleaned = [query.strip() for query in dict_queries if query.strip()]
            if cleaned:
                return cleaned

    raise ValueError(
        "JSON benchmark file must be a list of queries or an object with a queries list."
    )


def _summarize_latencies(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}

    sorted_values = sorted(values)

    return {
        "count": len(sorted_values),
        "avg": round(mean(sorted_values), 3),
        "p50": round(median(sorted_values), 3),
        "p95": round(_percentile(sorted_values, 95), 3),
        "p99": round(_percentile(sorted_values, 99), 3),
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


def _fmt_float(value: object) -> str:
    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        return f"{value:.1f}" if abs(value) >= 100 else f"{value:.3f}"

    return str(value)
