from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import cast, AsyncIterator

from voice_rag_agent.config import Settings
from voice_rag_agent.events import RagEvent, new_trace_id
from voice_rag_agent.llm.base import Reasoner, TemplateReasoner
from voice_rag_agent.observability import TraceRecorder

from .documents import load_documents
from .index_store import load_index_if_valid
from .prompts import build_grounded_prompt

from .retriever import (
    CachedRetriever,
    RerankingRetriever,
    RetrievedChunk,
    Retriever,
    build_reranker,
    build_retriever,
)


WHITESPACE_RE = re.compile(r"\s+")


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
        retriever_name: str = "bm25",
        reranker_name: str = "none",
    ) -> None:
        self.retriever = retriever
        self.reasoner = reasoner
        self.tracer = tracer
        self.top_k = top_k
        self.retriever_name = retriever_name
        self.reranker_name = reranker_name

    async def stream_query(
        self, query: str, trace_id: str | None = None
    ) -> AsyncIterator[RagEvent]:
        trace_id = trace_id or new_trace_id()
        query_started_ns = time.perf_counter_ns()
        retrieval_started_ns: int | None = None
        reasoning_started_ns: int | None = None
        retrieval_latency_ms: float | None = None
        llm_time_to_first_token_ms: float | None = None
        llm_total_latency_ms: float | None = None
        stage = "validation"

        query = _normalize_query(query)
        yield self._event("query.received", trace_id, {"text": query})

        try:
            stage = "retrieval"
            retrieval_started_ns = time.perf_counter_ns()

            with self.tracer.span(
                trace_id,
                "retrieval",
                query=query,
                top_k=self.top_k,
                retriever=self.retriever_name,
                reranker=self.reranker_name,
            ):
                results = self.retriever.retrieve(query, self.top_k)

            retrieval_latency_ms = _elapsed_ms(retrieval_started_ns)

            yield self._event(
                "retrieval.completed",
                trace_id,
                {
                    "count": len(results),
                    "retriever": self.retriever_name,
                    "reranker": self.reranker_name,
                    "citations": [result.citation() for result in results],
                    "retrieval_latency_ms": retrieval_latency_ms,
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

            stage = "prompt"
            prompt = build_grounded_prompt(query, results)
            answer_parts: list[str] = []

            stage = "reasoning"
            reasoning_started_ns = time.perf_counter_ns()

            with self.tracer.span(trace_id, "reasoning", provider=self.reasoner.name):
                async for delta in self.reasoner.stream_answer(query, results, prompt):
                    if delta and llm_time_to_first_token_ms is None:
                        llm_time_to_first_token_ms = _elapsed_ms(reasoning_started_ns)

                    answer_parts.append(delta)
                    yield self._event("answer.delta", trace_id, {"text": delta})

            llm_total_latency_ms = _elapsed_ms(reasoning_started_ns)
            query_total_latency_ms = _elapsed_ms(query_started_ns)
            answer_text = "".join(answer_parts).strip()

            yield self._event(
                "answer.completed",
                trace_id,
                {
                    "text": answer_text,
                    "citations": [result.citation() for result in results],
                    "grounded": bool(results),
                    "retriever": self.retriever_name,
                    "reranker": self.reranker_name,
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "llm_time_to_first_token_ms": llm_time_to_first_token_ms,
                    "llm_total_latency_ms": llm_total_latency_ms,
                    "query_total_latency_ms": query_total_latency_ms,
                },
            )

            yield self._event(
                "metrics.completed",
                trace_id,
                {
                    "mode": "text",
                    "failed": False,
                    "retriever": self.retriever_name,
                    "reranker": self.reranker_name,
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "llm_time_to_first_token_ms": llm_time_to_first_token_ms,
                    "llm_total_latency_ms": llm_total_latency_ms,
                    "query_total_latency_ms": query_total_latency_ms,
                    "retrieved_chunks": len(results),
                    "reasoner": self.reasoner.name,
                },
            )
        except Exception as exc:  # noqa: BLE001
            query_total_latency_ms = _elapsed_ms(query_started_ns)

            if retrieval_latency_ms is None and retrieval_started_ns is not None:
                retrieval_latency_ms = _elapsed_ms(retrieval_started_ns)

            if llm_total_latency_ms is None and reasoning_started_ns is not None:
                llm_total_latency_ms = _elapsed_ms(reasoning_started_ns)

            error_payload = {
                "code": "stream_failed",
                "message": str(exc),
                "stage": stage,
                "error_type": type(exc).__name__,
                "retriever": self.retriever_name,
                "reranker": self.reranker_name,
                "reasoner": self.reasoner.name,
                "query_total_latency_ms": query_total_latency_ms,
            }

            yield self._event("error", trace_id, error_payload)

            yield self._event(
                "metrics.completed",
                trace_id,
                {
                    "mode": "text",
                    "failed": True,
                    "error_code": "stream_failed",
                    "error_stage": stage,
                    "error_type": type(exc).__name__,
                    "retriever": self.retriever_name,
                    "reranker": self.reranker_name,
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "llm_time_to_first_token_ms": llm_time_to_first_token_ms,
                    "llm_total_latency_ms": llm_total_latency_ms,
                    "query_total_latency_ms": query_total_latency_ms,
                    "retrieved_chunks": 0,
                    "reasoner": self.reasoner.name,
                },
            )

    async def query(self, query: str, trace_id: str | None = None) -> RagAnswer:
        answer = ""
        citations: list[dict[str, str | int | float]] = []
        last_trace_id = trace_id or new_trace_id()
        stream_error: dict[str, object] | None = None

        async for event in self.stream_query(query, trace_id=last_trace_id):
            if event.type == "answer.completed":
                answer = str(event.payload["text"])
                citations = cast(list[dict[str, str | int | float]], event.payload["citations"])
                last_trace_id = event.trace_id
            elif event.type == "error":
                stream_error = dict(event.payload)
                last_trace_id = event.trace_id

        if stream_error is not None:
            message = str(stream_error.get("message") or "Query failed.")
            raise RuntimeError(message)

        return RagAnswer(text=answer, citations=citations, trace_id=last_trace_id)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        normalized_query = _normalize_query(query)

        return self.retriever.retrieve(normalized_query, top_k or self.top_k)

    def _event(self, event_type: str, trace_id: str, payload: dict[str, object]) -> RagEvent:
        event = RagEvent(type=event_type, trace_id=trace_id, payload=payload)
        self.tracer.write_event(event)

        return event


def build_engine(settings: Settings, reasoner: Reasoner | None = None) -> RagEngine:
    settings.ensure_dirs()

    chunks = load_index_if_valid(
        data_dir=settings.data_dir,
        index_dir=settings.index_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if chunks is None:
        chunks = load_documents(settings.data_dir, settings.chunk_size, settings.chunk_overlap)

    base_retriever = build_retriever(chunks, settings.retriever)
    reranker = build_reranker(settings.reranker)

    if reranker.name != "none":
        base_retriever = RerankingRetriever(base_retriever, reranker)

    retriever = CachedRetriever(
        base_retriever,
        max_entries=settings.max_cache_entries,
    )

    tracer = TraceRecorder(settings.trace_dir, enabled=settings.enable_tracing)

    return RagEngine(
        retriever=retriever,
        reasoner=reasoner or TemplateReasoner(),
        tracer=tracer,
        top_k=settings.top_k,
        retriever_name=settings.retriever,
        reranker_name=reranker.name,
    )


def _normalize_query(query: str) -> str:
    normalized = WHITESPACE_RE.sub(" ", query).strip()

    if not normalized:
        raise ValueError("Query cannot be empty.")

    return normalized


def _preview(text: str, max_chars: int = 360) -> str:
    compact = WHITESPACE_RE.sub(" ", text).strip()

    return compact if len(compact) <= max_chars else f"{compact[: max_chars - 3]}..."


def _elapsed_ms(start_ns: int, end_ns: int | None = None) -> float:
    end = end_ns if end_ns is not None else time.perf_counter_ns()

    return round((end - start_ns) / 1_000_000, 3)


async def collect_events(events: AsyncIterator[RagEvent]) -> list[RagEvent]:
    collected: list[RagEvent] = []

    async for event in events:
        collected.append(event)
        await asyncio.sleep(0)

    return collected
