from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from voice_rag_agent.rag.documents import load_documents, split_text


class DocumentTests(unittest.TestCase):
    def test_split_text_respects_non_empty_chunks(self) -> None:
        chunks = split_text("First paragraph.\n\nSecond paragraph.", max_chars=20, overlap=5)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_load_documents_extracts_title(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.md"
            path.write_text("# Handbook\n\nLatency and streaming matter.", encoding="utf-8")
            chunks = load_documents(Path(tmp), chunk_size=200, chunk_overlap=0)
        self.assertEqual(chunks[0].metadata["title"], "Handbook")
        self.assertIn("Latency", chunks[0].text)


if __name__ == "__main__":
    unittest.main()
