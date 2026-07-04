from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from voice_rag_agent.benchmark import (
    compare_benchmark_reports,
    load_benchmark_queries,
    load_benchmark_report,
    report_to_dict,
    run_query_benchmark,
    run_http_benchmark,
)


@dataclass(frozen=True)
class FakeAnswer:
    trace_id: str
    citations: list[dict[str, str]]


class FakeEngine:
    async def query(self, query: str) -> FakeAnswer:
        if query == "fail":
            raise ValueError("forced failure")

        return FakeAnswer(
            trace_id=f"trace-{query}",
            citations=[{"title": "doc"}],
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[object]:
        if query == "fail":
            raise ValueError("forced failure")

        return [object() for _ in range(top_k or 1)]


def test_load_benchmark_queries_from_json(tmp_path: Path) -> None:
    path = tmp_path / "queries.json"
    path.write_text(
        '{"queries": [{"query": "hello"}, "world"]}',
        encoding="utf-8",
    )

    assert load_benchmark_queries(path) == ["hello", "world"]


def test_run_query_benchmark_reports_latency_and_errors() -> None:
    report = asyncio.run(
        run_query_benchmark(
            engine=FakeEngine(),
            queries=["ok", "fail"],
            concurrency=2,
            repeat=2,
        )
    )

    payload = report_to_dict(report)

    assert payload["total_requests"] == 4
    assert payload["successful_requests"] == 2
    assert payload["failed_requests"] == 2
    assert payload["error_rate"] == 0.5
    assert payload["latency"]["count"] == 4
    assert "p95" in payload["latency"]
    assert "p99" in payload["latency"]


def test_benchmark_threshold_gate_passes_and_fails() -> None:
    report = asyncio.run(
        run_query_benchmark(
            engine=FakeEngine(),
            queries=["ok"],
            concurrency=1,
            repeat=1,
        )
    )

    from voice_rag_agent.benchmark import check_benchmark_thresholds

    assert (
        check_benchmark_thresholds(
            report,
            max_p95_ms=1000,
            max_error_rate=0,
            min_qps=0.1,
        )
        == []
    )

    failures = check_benchmark_thresholds(
        report,
        max_p95_ms=0.0001,
        max_error_rate=0,
        min_qps=1_000_000,
    )

    assert any("p95 latency" in failure for failure in failures)
    assert any("throughput" in failure for failure in failures)


def test_run_retrieval_benchmark_mode_reports_hot_path_latency() -> None:
    report = asyncio.run(
        run_query_benchmark(
            engine=FakeEngine(),
            queries=["ok"],
            concurrency=1,
            repeat=2,
            mode="retrieve",
            top_k=3,
        )
    )

    payload = report_to_dict(report)

    assert payload["mode"] == "retrieve"
    assert payload["total_requests"] == 2
    assert payload["successful_requests"] == 2
    assert payload["failed_requests"] == 0
    assert payload["results"][0]["citation_count"] == 3
    assert "p95" in payload["latency"]


def test_run_http_benchmark_against_local_server() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/query":
                self.send_response(404)
                self.end_headers()
                return

            if self.headers.get("X-API-Key") != "secret":
                self.send_response(403)
                self.end_headers()
                self.wfile.write(b'{"detail":"forbidden"}')
                return

            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            payload = {
                "trace_id": f"http-{body['query']}",
                "answer": "ok",
                "citations": [{"title": "doc"}],
            }
            response = json.dumps(payload).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host = str(server.server_address[0])
        port = int(server.server_address[1])
        report = asyncio.run(
            run_http_benchmark(
                url=f"http://{host}:{port}/query",
                queries=["alpha", "beta"],
                concurrency=2,
                repeat=1,
                api_key="secret",
                timeout_ms=1000,
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    payload = report_to_dict(report)

    assert payload["mode"] == "http"
    assert payload["total_requests"] == 2
    assert payload["successful_requests"] == 2
    assert payload["failed_requests"] == 0
    assert payload["results"][0]["citation_count"] == 1
    assert "p95" in payload["latency"]


def test_benchmark_baseline_comparison_passes_and_fails(tmp_path: Path) -> None:
    baseline = {
        "latency": {"p95": 100.0},
        "throughput_qps": 100.0,
        "error_rate": 0.0,
    }
    current_good = {
        "latency": {"p95": 110.0},
        "throughput_qps": 95.0,
        "error_rate": 0.0,
    }
    current_bad = {
        "latency": {"p95": 140.0},
        "throughput_qps": 70.0,
        "error_rate": 0.05,
    }

    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")

    assert load_benchmark_report(path) == baseline

    assert (
        compare_benchmark_reports(
            current_good,
            baseline,
            max_p95_regression_percent=25,
            max_error_rate_delta=0,
            min_qps_ratio=0.9,
        )
        == []
    )

    failures = compare_benchmark_reports(
        current_bad,
        baseline,
        max_p95_regression_percent=25,
        max_error_rate_delta=0,
        min_qps_ratio=0.9,
    )

    assert any("p95 latency regressed" in failure for failure in failures)
    assert any("error rate increased" in failure for failure in failures)
    assert any("throughput ratio" in failure for failure in failures)
