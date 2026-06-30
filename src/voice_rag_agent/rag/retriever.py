from __future__ import annotations

import math
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Iterable, Protocol

from .documents import DocumentChunk


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+-]*", re.IGNORECASE)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float

    def citation(self) -> dict[str, str | int | float]:
        return {
            "id": self.chunk.id,
            "title": self.chunk.metadata.get("title", ""),
            "path": self.chunk.metadata.get("path", self.chunk.source_path),
            "chunk_index": self.chunk.metadata.get("chunk_index", 0),
            "score": round(self.score, 4),
        }


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        ...


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


class BM25Retriever:
    """Small, fast lexical retriever for local development and deterministic tests."""

    def __init__(self, chunks: Iterable[DocumentChunk], k1: float = 1.4, b: float = 0.72) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freqs: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freqs.update(set(tokens))

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        query_terms = tokenize(query)
        if not query_terms or not self.chunks:
            return []

        scored: list[RetrievedChunk] = []
        query_counts = Counter(query_terms)
        for index, chunk in enumerate(self.chunks):
            score = self._score_doc(index, query_counts)
            if score > 0:
                scored.append(RetrievedChunk(chunk=chunk, score=score))

        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def _score_doc(self, index: int, query_counts: Counter[str]) -> float:
        score = 0.0
        doc_len = self.doc_lengths[index] or 1
        freqs = self.term_freqs[index]
        total_docs = len(self.chunks)
        for term, query_weight in query_counts.items():
            term_frequency = freqs.get(term, 0)
            if term_frequency == 0:
                continue
            doc_frequency = self.doc_freqs.get(term, 0)
            idf = math.log(1 + (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
            numerator = term_frequency * (self.k1 + 1)
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1)
            )
            score += query_weight * idf * numerator / denominator
        return score


class CachedRetriever:
    """Simple LRU cache around any retriever."""

    def __init__(self, retriever: Retriever, max_entries: int = 128) -> None:
        self.retriever = retriever
        self.max_entries = max_entries
        self._cache: OrderedDict[tuple[str, int], list[RetrievedChunk]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        key = (" ".join(tokenize(query)), top_k)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return list(self._cache[key])

        self.misses += 1
        results = self.retriever.retrieve(query, top_k)
        self._cache[key] = list(results)
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return results
