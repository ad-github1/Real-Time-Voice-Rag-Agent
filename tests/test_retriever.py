from pathlib import Path
import unittest

from voice_rag_agent.rag.documents import load_documents
from voice_rag_agent.rag.retriever import BM25Retriever, CachedRetriever


ROOT = Path(__file__).resolve().parents[1]


class RetrieverTests(unittest.TestCase):
    def test_retrieves_latency_context(self) -> None:
        chunks = load_documents(ROOT / "data/sample_docs")
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("How is latency optimized?", top_k=2)
        self.assertTrue(results)
        self.assertIn("Latency", results[0].chunk.metadata["title"])

    def test_cache_hits_repeat_queries(self) -> None:
        chunks = load_documents(ROOT / "data/sample_docs")
        cached = CachedRetriever(BM25Retriever(chunks), max_entries=2)
        cached.retrieve("Tell me about MCP", top_k=1)
        cached.retrieve("Tell me about MCP", top_k=1)
        self.assertEqual(cached.hits, 1)
        self.assertEqual(cached.misses, 1)


if __name__ == "__main__":
    unittest.main()
