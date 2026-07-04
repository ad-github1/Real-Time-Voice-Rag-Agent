from __future__ import annotations

from dataclasses import dataclass

from voice_rag_agent.profiling import (
    check_profile_thresholds,
    format_profile_report,
    profile_report_to_dict,
    run_profile,
)


@dataclass(frozen=True)
class FakeAnswer:
    trace_id: str
    citations: list[dict[str, str]]


class FakeEngine:
    async def query(self, query: str) -> FakeAnswer:
        return FakeAnswer(trace_id=f"trace-{query}", citations=[{"title": "doc"}])

    def retrieve(self, query: str, top_k: int | None = None) -> list[object]:
        return [object() for _ in range(top_k or 1)]


def test_run_retrieval_profile_reports_cpu_and_memory() -> None:
    report = run_profile(
        engine=FakeEngine(),
        query="voice latency streaming",
        mode="retrieve",
        repeat=3,
        warmup=1,
        top_k=2,
        top_functions=5,
    )

    payload = profile_report_to_dict(report)
    formatted = format_profile_report(report)

    assert payload["mode"] == "retrieve"
    assert payload["repeat"] == 3
    assert payload["avg_duration_ms"] >= 0
    assert payload["peak_memory_kb"] >= 0
    assert payload["top_functions"]
    assert "Voice RAG Profile Report" in formatted
    assert "Peak memory" in formatted


def test_profile_rejects_empty_query() -> None:
    try:
        run_profile(
            engine=FakeEngine(),
            query="   ",
            mode="retrieve",
            repeat=1,
        )
    except ValueError as exc:
        assert "Query cannot be empty" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_profile_threshold_gate_passes_and_fails() -> None:
    report = run_profile(
        engine=FakeEngine(),
        query="voice latency streaming",
        mode="retrieve",
        repeat=2,
        warmup=0,
        top_k=1,
    )

    assert (
        check_profile_thresholds(
            report,
            max_avg_ms=1000,
            max_peak_memory_kb=100000,
        )
        == []
    )

    failures = check_profile_thresholds(
        report,
        max_avg_ms=-1,
        max_peak_memory_kb=-1,
    )

    assert any("average duration" in failure for failure in failures)
    assert any("peak memory" in failure for failure in failures)
