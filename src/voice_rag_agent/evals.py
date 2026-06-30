from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rag.engine import RagEngine


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    must_include: tuple[str, ...]


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    passed: bool
    answer: str
    missing_terms: tuple[str, ...]
    citation_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "answer": self.answer,
            "missing_terms": list(self.missing_terms),
            "citation_count": self.citation_count,
        }


async def evaluate(engine: RagEngine, cases: list[EvalCase]) -> list[EvalResult]:
    results: list[EvalResult] = []
    for case in cases:
        answer = await engine.query(case.query)
        lower = answer.text.lower()
        missing = tuple(term for term in case.must_include if term.lower() not in lower)
        results.append(
            EvalResult(
                case_id=case.id,
                passed=not missing and bool(answer.citations),
                answer=answer.text,
                missing_terms=missing,
                citation_count=len(answer.citations),
            )
        )
    return results


def load_eval_cases(path: Path) -> list[EvalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["cases"] if isinstance(payload, dict) else payload
    return [
        EvalCase(
            id=str(item["id"]),
            query=str(item["query"]),
            must_include=tuple(str(term) for term in item.get("must_include", [])),
        )
        for item in cases
    ]


def run_eval(engine: RagEngine, cases_path: Path, output_path: Path | None = None) -> list[EvalResult]:
    results = asyncio.run(evaluate(engine, load_eval_cases(cases_path)))
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([result.to_dict() for result in results], indent=2),
            encoding="utf-8",
        )
    return results
