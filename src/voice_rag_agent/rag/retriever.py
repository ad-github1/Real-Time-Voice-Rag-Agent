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
    diagnostics: dict[str, str | int | float] | None = None

    def citation(self) -> dict[str, str | int | float]:
        citation: dict[str, str | int | float] = {
            "id": self.chunk.id,
            "title": self.chunk.metadata.get("title", ""),
            "path": self.chunk.metadata.get("path", self.chunk.source_path),
            "chunk_index": self.chunk.metadata.get("chunk_index", 0),
            "score": round(self.score, 4),
        }

        if self.diagnostics:
            for key, value in self.diagnostics.items():
                citation[key] = round(value, 4) if isinstance(value, float) else value

        return citation


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        ...


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def build_retriever(chunks: Iterable[DocumentChunk], mode: str) -> Retriever:
    normalized_mode = mode.strip().lower()

    if normalized_mode == "bm25":
        return BM25Retriever(chunks)

    if normalized_mode == "hybrid":
        return HybridRetriever(chunks)

    raise ValueError(f"Unsupported retriever mode: {mode}. Use 'bm25' or 'hybrid'.")


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
                scored.append(
                    RetrievedChunk(
                        chunk=chunk,
                        score=score,
                        diagnostics={
                            "retriever": "bm25",
                            "bm25_score": score,
                        },
                    )
                )

        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def score_all(self, query: str) -> dict[str, float]:
        query_terms = tokenize(query)

        if not query_terms or not self.chunks:
            return {}

        query_counts = Counter(query_terms)
        scores: dict[str, float] = {}

        for index, chunk in enumerate(self.chunks):
            score = self._score_doc(index, query_counts)

            if score > 0:
                scores[chunk.id] = score

        return scores

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


class LightweightSemanticRetriever:
    """Dependency-free semantic-ish retriever using hashed lexical features and cosine similarity.

    This is not a neural embedding model. It is a lightweight local approximation that
    captures token overlap, token bigrams, and character shingles.
    """

    def __init__(self, chunks: Iterable[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        self.doc_features = [_features(chunk.text) for chunk in self.chunks]
        self.doc_freqs: Counter[str] = Counter()

        for features in self.doc_features:
            self.doc_freqs.update(set(features))

        self.total_docs = len(self.chunks)
        self.doc_vectors = [self._tfidf(features) for features in self.doc_features]
        self.doc_norms = [_norm(vector) for vector in self.doc_vectors]

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        scores = self.score_all(query)
        by_id = {chunk.id: chunk for chunk in self.chunks}

        results = [
            RetrievedChunk(
                chunk=by_id[chunk_id],
                score=score,
                diagnostics={
                    "retriever": "semantic-lite",
                    "semantic_score": score,
                },
            )
            for chunk_id, score in scores.items()
        ]

        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    def score_all(self, query: str) -> dict[str, float]:
        query_vector = self._tfidf(_features(query))
        query_norm = _norm(query_vector)

        if query_norm == 0:
            return {}

        scores: dict[str, float] = {}

        for index, chunk in enumerate(self.chunks):
            doc_norm = self.doc_norms[index]

            if doc_norm == 0:
                continue

            score = _dot(query_vector, self.doc_vectors[index]) / (query_norm * doc_norm)

            if score > 0:
                scores[chunk.id] = score

        return scores

    def _tfidf(self, features: Counter[str]) -> dict[str, float]:
        vector: dict[str, float] = {}

        for feature, count in features.items():
            doc_frequency = self.doc_freqs.get(feature, 0)
            idf = math.log(1 + (self.total_docs + 1) / (doc_frequency + 1))
            vector[feature] = (1 + math.log(count)) * idf

        return vector


class HybridRetriever:
    """Combines BM25 lexical scoring and lightweight semantic similarity."""

    def __init__(self, chunks: Iterable[DocumentChunk], bm25_weight: float = 0.65) -> None:
        self.chunks = list(chunks)
        self.bm25_weight = bm25_weight
        self.semantic_weight = 1.0 - bm25_weight
        self.bm25 = BM25Retriever(self.chunks)
        self.semantic = LightweightSemanticRetriever(self.chunks)
        self.by_id = {chunk.id: chunk for chunk in self.chunks}

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        bm25_scores = self.bm25.score_all(query)
        semantic_scores = self.semantic.score_all(query)

        bm25_norm = _normalize_scores(bm25_scores)
        semantic_norm = _normalize_scores(semantic_scores)

        candidate_ids = set(bm25_scores) | set(semantic_scores)
        scored: list[RetrievedChunk] = []

        for chunk_id in candidate_ids:
            hybrid_score = (
                self.bm25_weight * bm25_norm.get(chunk_id, 0.0)
                + self.semantic_weight * semantic_norm.get(chunk_id, 0.0)
            )

            if hybrid_score <= 0:
                continue

            scored.append(
                RetrievedChunk(
                    chunk=self.by_id[chunk_id],
                    score=hybrid_score,
                    diagnostics={
                        "retriever": "hybrid",
                        "bm25_score": bm25_scores.get(chunk_id, 0.0),
                        "semantic_score": semantic_scores.get(chunk_id, 0.0),
                        "bm25_norm": bm25_norm.get(chunk_id, 0.0),
                        "semantic_norm": semantic_norm.get(chunk_id, 0.0),
                        "bm25_weight": self.bm25_weight,
                        "semantic_weight": self.semantic_weight,
                    },
                )
            )

        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


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


def _features(text: str) -> Counter[str]:
    tokens = tokenize(text)
    features: Counter[str] = Counter()

    for token in tokens:
        features[f"tok:{token}"] += 2

        if len(token) >= 4:
            for index in range(len(token) - 2):
                features[f"tri:{token[index:index + 3]}"] += 1

    for left, right in zip(tokens, tokens[1:]):
        features[f"bi:{left}_{right}"] += 2

    return features


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}

    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)

    if math.isclose(max_score, min_score):
        return {key: 1.0 for key in scores}

    return {
        key: (value - min_score) / (max_score - min_score)
        for key, value in scores.items()
    }


def _dot(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left

    return sum(value * right.get(feature, 0.0) for feature, value in left.items())


def _norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))
