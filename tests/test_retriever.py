from pathlib import Path
import unittest

from voice_rag_agent.rag.documents import DocumentChunk, load_documents
from voice_rag_agent.rag.retriever import (
    BM25Retriever,
    CachedRetriever,
    LexicalReranker,
    RerankingRetriever,
    RetrievedChunk,
    tokenize,
)


ROOT = Path(__file__).resolve().parents[1]


class RetrieverTests(unittest.TestCase):
    def test_retrieves_latency_context(self) -> None:
        chunks = load_documents(ROOT / "data/sample_docs")
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("How is latency optimized?", top_k=2)

        self.assertTrue(results)
        title = str(results[0].chunk.metadata["title"])
        self.assertIn("Latency", title)

    def test_cache_hits_repeat_queries(self) -> None:
        chunks = load_documents(ROOT / "data/sample_docs")
        cached = CachedRetriever(BM25Retriever(chunks), max_entries=2)

        cached.retrieve("Tell me about MCP", top_k=1)
        cached.retrieve("Tell me about MCP", top_k=1)

        self.assertEqual(cached.hits, 1)
        self.assertEqual(cached.misses, 1)

    def test_lexical_reranker_promotes_query_term_coverage(self) -> None:
        weak = DocumentChunk(
            id="weak",
            text="latency",
            source_path="weak.md",
            metadata={"title": "Weak", "path": "weak.md", "chunk_index": 0},
        )
        strong = DocumentChunk(
            id="strong",
            text="latency streaming websocket audio pipeline",
            source_path="strong.md",
            metadata={"title": "Strong", "path": "strong.md", "chunk_index": 0},
        )

        results = [
            RetrievedChunk(chunk=weak, score=1.0, diagnostics={"retriever": "test"}),
            RetrievedChunk(chunk=strong, score=1.0, diagnostics={"retriever": "test"}),
        ]

        reranked = LexicalReranker().rerank(
            "latency streaming audio",
            results,
            top_k=2,
        )

        self.assertEqual(reranked[0].chunk.id, "strong")
        reranked_diagnostics = reranked[0].diagnostics
        assert reranked_diagnostics is not None
        self.assertEqual(reranked_diagnostics["reranker"], "lexical")
        self.assertEqual(reranked_diagnostics["pre_rerank_rank"], 2)
        self.assertGreater(float(reranked_diagnostics["reranker_score"]), 0)

    def test_reranking_retriever_adds_reranker_diagnostics(self) -> None:
        chunks = load_documents(ROOT / "data/sample_docs")
        retriever = RerankingRetriever(
            BM25Retriever(chunks),
            LexicalReranker(),
        )
        results = retriever.retrieve("voice latency streaming", top_k=2)

        self.assertTrue(results)
        diagnostics = results[0].diagnostics
        assert diagnostics is not None
        self.assertIn("reranker", diagnostics)
        self.assertEqual(diagnostics["reranker"], "lexical")

    def test_tokenize_uses_stable_cached_output(self) -> None:
        first = tokenize("Voice latency streaming")
        second = tokenize("Voice latency streaming")

        self.assertEqual(first, ["voice", "latency", "streaming"])
        self.assertEqual(second, first)
        self.assertIsNot(second, first)


if __name__ == "__main__":
    unittest.main()
