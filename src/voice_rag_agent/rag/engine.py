from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from voice_rag_agent.config import Settings
from voice_rag_agent.events import RagEvent, new_trace_id
from voice_rag_agent.llm.base import Reasoner, TemplateReasoner
from voice_rag_agent.observability import TraceRecorder

from .documents import load_documents
from .prompts import build_grounded_prompt
from .retriever import BM25Retriever, CachedRetriever, RetrievedChunk, Retriever


@dataclass(frozen=True)
class RagAnswer:
    text: str
    citations: list[dict[str, str | int | float]]
    trace_id: str


class RagEngine:
    def __init__(
        self,
        retriever: Retriever,
        reasoner: Reasoner,
        tracer: TraceRecorder,
        top_k: int = 4,
    ) -> None:
        self.retriever = retriever
        self.reasoner = reasoner
        self.tracer = tracer
        self.top_k = top_k

    async def stream_query(self, query: str, trace_id: str | None = None) -> AsyncIterator[RagEvent]:
        trace_id = trace_id or new_trace_id()
        query = _normalize_query(query)
        yield self._event("query.received", trace_id, {"text": query})

        with self.tracer.span(trace_id, "retrieval", query=query, top_k=self.top_k):
            results = self.retriever.retrieve(query, self.top_k)

        yield self._event(
            "retrieval.completed",
            trace_id,
            {
                "count": len(results),
                "citations": [result.citation() for result in results],
            },
        )
        for index, result in enumerate(results):
            yield self._event(
                "retrieval.chunk",
                trace_id,
                {
                    "rank": index + 1,
                    "score": round(result.score, 4),
                    "text": _preview(result.chunk.text),
                    "citation": result.citation(),
                },
            )

        prompt = build_grounded_prompt(query, results)
        answer_parts: list[str] = []
        with self.tracer.span(trace_id, "reasoning", provider=self.reasoner.name):
            async for delta in self.reasoner.stream_answer(query, results, prompt):
                answer_parts.append(delta)
                yield self._event("answer.delta", trace_id, {"text": delta})

        answer_text = "".join(answer_parts).strip()
        yield self._event(
            "answer.completed",
            trace_id,
            {
                "text": answer_text,
                "citations": [result.citation() for result in results],
                "grounded": bool(results),
            },
        )

    async def query(self, query: str, trace_id: str | None = None) -> RagAnswer:
        answer = ""
        citations: list[dict[str, str | int | float]] = []
        last_trace_id = trace_id or new_trace_id()
        async for event in self.stream_query(query, trace_id=last_trace_id):
            if event.type == "answer.completed":
                answer = str(event.payload["text"])
                citations = list(event.payload["citations"])  # type: ignore[arg-type]
                last_trace_id = event.trace_id
        return RagAnswer(text=answer, citations=citations, trace_id=last_trace_id)

    def _event(self, event_type: str, trace_id: str, payload: dict[str, object]) -> RagEvent:
        event = RagEvent(type=event_type, trace_id=trace_id, payload=payload)
        self.tracer.write_event(event)
        return event


def build_engine(settings: Settings, reasoner: Reasoner | None = None) -> RagEngine:
    settings.ensure_dirs()
    chunks = load_documents(settings.data_dir, settings.chunk_size, settings.chunk_overlap)
    retriever = CachedRetriever(
        BM25Retriever(chunks),
        max_entries=settings.max_cache_entries,
    )
    tracer = TraceRecorder(settings.trace_dir, enabled=settings.enable_tracing)
    return RagEngine(
        retriever=retriever,
        reasoner=reasoner or TemplateReasoner(),
        tracer=tracer,
        top_k=settings.top_k,
    )


def _normalize_query(query: str) -> str:
    normalized = re.sub(r"\s+", " ", query).strip()
    if not normalized:
        raise ValueError("Query cannot be empty.")
    return normalized


def _preview(text: str, max_chars: int = 360) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= max_chars else f"{compact[: max_chars - 3]}..."


async def collect_events(events: AsyncIterator[RagEvent]) -> list[RagEvent]:
    collected: list[RagEvent] = []
    async for event in events:
        collected.append(event)
        await asyncio.sleep(0)
    return collected
