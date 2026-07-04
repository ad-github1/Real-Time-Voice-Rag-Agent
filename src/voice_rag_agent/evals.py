from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .rag.engine import RagEngine


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    must_include: tuple[str, ...]
    expected_citations: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    passed: bool
    answer: str
    missing_terms: tuple[str, ...]
    citation_count: int
    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    retrieval_recall: float = 1.0
    retrieval_mrr: float = 1.0
    trace_id: str = ""
    included_terms: tuple[str, ...] = ()
    term_recall: float = 1.0
    citation_titles: tuple[str, ...] = ()
    citation_paths: tuple[str, ...] = ()
    retrieved_citation_titles: tuple[str, ...] = ()
    retrieved_citation_paths: tuple[str, ...] = ()
    expected_citations: tuple[str, ...] = ()
    missing_expected_citations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "answer": self.answer,
            "missing_terms": list(self.missing_terms),
            "included_terms": list(self.included_terms),
            "term_recall": self.term_recall,
            "citation_count": self.citation_count,
            "citation_titles": list(self.citation_titles),
            "citation_paths": list(self.citation_paths),
            "expected_citations": list(self.expected_citations),
            "missing_expected_citations": list(self.missing_expected_citations),
            "latency_ms": self.latency_ms,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "retrieval_recall": self.retrieval_recall,
            "retrieval_mrr": self.retrieval_mrr,
            "retrieved_citation_titles": list(self.retrieved_citation_titles),
            "retrieved_citation_paths": list(self.retrieved_citation_paths),
            "trace_id": self.trace_id,
        }


@dataclass(frozen=True)
class EvalReport:
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    citation_coverage_rate: float
    average_term_recall: float
    average_latency_ms: float
    p95_latency_ms: float
    average_retrieval_latency_ms: float
    average_retrieval_recall: float
    average_retrieval_mrr: float
    results: list[EvalResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total_cases": self.total_cases,
                "passed_cases": self.passed_cases,
                "failed_cases": self.failed_cases,
                "pass_rate": self.pass_rate,
                "citation_coverage_rate": self.citation_coverage_rate,
                "average_term_recall": self.average_term_recall,
                "average_latency_ms": self.average_latency_ms,
                "p95_latency_ms": self.p95_latency_ms,
                "average_retrieval_latency_ms": self.average_retrieval_latency_ms,
                "average_retrieval_recall": self.average_retrieval_recall,
                "average_retrieval_mrr": self.average_retrieval_mrr,
            },
            "results": [result.to_dict() for result in self.results],
        }


async def evaluate(engine: RagEngine, cases: list[EvalCase]) -> list[EvalResult]:
    results: list[EvalResult] = []

    for case in cases:
        retrieval_started = time.perf_counter()
        retrieved = engine.retrieve(case.query)
        retrieval_latency_ms = round((time.perf_counter() - retrieval_started) * 1000, 3)

        retrieved_citations = [result.citation() for result in retrieved]
        retrieved_citation_titles = tuple(
            str(citation.get("title", "")) for citation in retrieved_citations
        )
        retrieved_citation_paths = tuple(
            str(citation.get("path", "")) for citation in retrieved_citations
        )
        retrieved_blob = " ".join([*retrieved_citation_titles, *retrieved_citation_paths]).lower()
        retrieval_recall = _expected_citation_recall(
            case.expected_citations,
            retrieved_blob,
        )
        retrieval_mrr = _expected_citation_mrr(
            case.expected_citations,
            retrieved_citations,
        )

        started = time.perf_counter()
        answer = await engine.query(case.query)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)

        lower_answer = answer.text.lower()
        included_terms = tuple(term for term in case.must_include if term.lower() in lower_answer)
        missing_terms = tuple(
            term for term in case.must_include if term.lower() not in lower_answer
        )
        term_recall = len(included_terms) / len(case.must_include) if case.must_include else 1.0

        citation_titles = tuple(str(citation.get("title", "")) for citation in answer.citations)
        citation_paths = tuple(str(citation.get("path", "")) for citation in answer.citations)
        citation_blob = " ".join([*citation_titles, *citation_paths]).lower()
        missing_expected_citations = tuple(
            expected
            for expected in case.expected_citations
            if expected.lower() not in citation_blob
        )

        passed = not missing_terms and bool(answer.citations) and not missing_expected_citations

        results.append(
            EvalResult(
                case_id=case.id,
                passed=passed,
                answer=answer.text,
                missing_terms=missing_terms,
                citation_count=len(answer.citations),
                latency_ms=latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                retrieval_recall=round(retrieval_recall, 6),
                retrieval_mrr=round(retrieval_mrr, 6),
                trace_id=answer.trace_id,
                included_terms=included_terms,
                term_recall=round(term_recall, 6),
                citation_titles=citation_titles,
                citation_paths=citation_paths,
                retrieved_citation_titles=retrieved_citation_titles,
                retrieved_citation_paths=retrieved_citation_paths,
                expected_citations=case.expected_citations,
                missing_expected_citations=missing_expected_citations,
            )
        )

    return results


def build_eval_report(results: list[EvalResult]) -> EvalReport:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    citation_covered = sum(1 for result in results if result.citation_count > 0)
    latencies = [result.latency_ms for result in results]
    retrieval_latencies = [result.retrieval_latency_ms for result in results]

    return EvalReport(
        total_cases=total,
        passed_cases=passed,
        failed_cases=total - passed,
        pass_rate=round(passed / total, 6) if total else 0.0,
        citation_coverage_rate=round(citation_covered / total, 6) if total else 0.0,
        average_term_recall=round(mean([result.term_recall for result in results]), 6)
        if results
        else 0.0,
        average_latency_ms=round(mean(latencies), 3) if latencies else 0.0,
        p95_latency_ms=round(_percentile(sorted(latencies), 95), 3) if latencies else 0.0,
        average_retrieval_latency_ms=round(mean(retrieval_latencies), 3)
        if retrieval_latencies
        else 0.0,
        average_retrieval_recall=round(mean([result.retrieval_recall for result in results]), 6)
        if results
        else 0.0,
        average_retrieval_mrr=round(mean([result.retrieval_mrr for result in results]), 6)
        if results
        else 0.0,
        results=results,
    )


def format_eval_report(report: EvalReport) -> str:
    lines = [
        "# Voice RAG Evaluation Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total cases | {report.total_cases} |",
        f"| Passed cases | {report.passed_cases} |",
        f"| Failed cases | {report.failed_cases} |",
        f"| Pass rate | {report.pass_rate:.2%} |",
        f"| Citation coverage | {report.citation_coverage_rate:.2%} |",
        f"| Average term recall | {report.average_term_recall:.2%} |",
        f"| Average latency | {report.average_latency_ms:.3f} ms |",
        f"| P95 latency | {report.p95_latency_ms:.3f} ms |",
        f"| Average Retrieval Latency | {report.average_retrieval_latency_ms:.3f} ms |",
        f"| Average Retrieval Recall | {report.average_retrieval_recall:.2%} |",
        f"| Average Retrieval MRR | {report.average_retrieval_mrr:.3f} |",
        "",
        "## Cases",
        "",
        "| Case | Status | Term Recall | Retrieval Recall | Retrieval MRR | Citations | Retrieval ms | Latency ms | Missing Terms | Missing Expected Citations |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        missing_terms = ", ".join(result.missing_terms) if result.missing_terms else "-"
        missing_citations = (
            ", ".join(result.missing_expected_citations)
            if result.missing_expected_citations
            else "-"
        )

        lines.append(
            "| "
            + " | ".join(
                [
                    result.case_id,
                    status,
                    f"{result.term_recall:.2%}",
                    f"{result.retrieval_recall:.2%}",
                    f"{result.retrieval_mrr:.3f}",
                    str(result.citation_count),
                    f"{result.retrieval_latency_ms:.3f}",
                    f"{result.latency_ms:.3f}",
                    missing_terms,
                    missing_citations,
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- A case passes only when all required answer terms are present, at least one citation is returned, and any expected citation hints are found."
    )
    lines.append(
        "- Latency is measured around the full in-process RAG query call, including retrieval and answer generation."
    )

    return "\n".join(lines)


def load_eval_cases(path: Path) -> list[EvalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["cases"] if isinstance(payload, dict) else payload

    return [
        EvalCase(
            id=str(item["id"]),
            query=str(item["query"]),
            must_include=tuple(str(term) for term in item.get("must_include", [])),
            expected_citations=tuple(str(term) for term in item.get("expected_citations", [])),
        )
        for item in cases
    ]


def run_eval(
    engine: RagEngine,
    cases_path: Path,
    output_path: Path | None = None,
    report_path: Path | None = None,
) -> list[EvalResult]:
    results = asyncio.run(evaluate(engine, load_eval_cases(cases_path)))
    report = build_eval_report(results)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report.to_dict(), indent=2),
            encoding="utf-8",
        )

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            format_eval_report(report),
            encoding="utf-8",
        )

    return results


def _expected_citation_recall(expected_citations: tuple[str, ...], blob: str) -> float:
    if not expected_citations:
        return 1.0

    hits = sum(1 for expected in expected_citations if expected.lower() in blob)

    return hits / len(expected_citations)


def _expected_citation_mrr(
    expected_citations: tuple[str, ...],
    retrieved_citations: list[dict[str, Any]],
) -> float:
    if not expected_citations:
        return 1.0

    for rank, citation in enumerate(retrieved_citations, start=1):
        citation_blob = (
            str(citation.get("title", "")) + " " + str(citation.get("path", ""))
        ).lower()

        if any(expected.lower() in citation_blob for expected in expected_citations):
            return 1 / rank

    return 0.0


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
