from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator, Protocol

from voice_rag_agent.rag.retriever import RetrievedChunk


class Reasoner(Protocol):
    name: str

    async def stream_answer(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        prompt: str,
    ) -> AsyncIterator[str]:
        ...


class TemplateReasoner:
    """Deterministic answerer for offline demos, tests, and eval baselines."""

    name = "template"

    async def stream_answer(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        prompt: str,
    ) -> AsyncIterator[str]:
        del prompt
        answer = self._answer(query, chunks)
        for token in _voice_tokens(answer):
            await asyncio.sleep(0)
            yield token

    def _answer(self, query: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return (
                "I do not have enough document context to answer that yet. "
                "Share a relevant document or ask a narrower question."
            )

        first = chunks[0]
        support = _first_sentence(first.chunk.text)
        title = first.chunk.metadata.get("title", "the retrieved document")
        lower_query = query.lower()
        if any(term in lower_query for term in ("latency", "fast", "real-time", "realtime")):
            lead = "The agent keeps latency low by streaming each stage and caching retrieval results."
        elif any(
            term in lower_query
            for term in ("voice", "speech", "audio", "transcription", "asr", "tts")
        ):
            lead = (
                "The voice path uses AssemblyAI for ASR and Cartesia for TTS, "
                "with RAG and reasoning in between."
            )
        elif any(term in lower_query for term in ("mcp", "a2a", "protocol")):
            lead = "The protocol layer exposes the same grounded query capability through MCP and A2A."
        else:
            lead = "The best match in the document set points to this answer."

        return f"{lead} In {title}, the supporting detail is: {support}"


def _first_sentence(text: str) -> str:
    body_lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    compact = re.sub(r"\s+", " ", " ".join(body_lines)).strip()
    match = re.search(r"(.+?[.!?])(?:\s|$)", compact)
    sentence = match.group(1) if match else compact
    return sentence[:420]


def _voice_tokens(text: str) -> list[str]:
    words = text.split(" ")
    tokens: list[str] = []
    for index, word in enumerate(words):
        suffix = "" if index == len(words) - 1 else " "
        tokens.append(f"{word}{suffix}")
    return tokens
