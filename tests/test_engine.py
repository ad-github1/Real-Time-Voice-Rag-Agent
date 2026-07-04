import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from voice_rag_agent.config import Settings
from voice_rag_agent.events import events_to_ndjson
from voice_rag_agent.llm.base import TemplateReasoner
from voice_rag_agent.observability import TraceRecorder
from voice_rag_agent.rag.retriever import RetrievedChunk
from voice_rag_agent.rag.engine import RagEngine, build_engine, collect_events


class FailingRetriever:
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        raise RuntimeError("retrieval exploded")


ROOT = Path(__file__).resolve().parents[1]


class EngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_query_emits_answer_and_citations(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = Settings.from_env(ROOT)
            settings = _replace(
                settings,
                trace_dir=Path(tmp),
                data_dir=ROOT / "data/sample_docs",
                enable_tracing=True,
            )
            engine = build_engine(settings)
            events = await collect_events(engine.stream_query("How does streaming reduce latency?"))

        event_types = [event.type for event in events]
        self.assertIn("retrieval.completed", event_types)
        self.assertIn("answer.completed", event_types)
        final = next(event for event in events if event.type == "answer.completed")
        self.assertTrue(final.payload["citations"])
        for line in events_to_ndjson(events).splitlines():
            self.assertIn("type", json.loads(line))

    async def test_engine_applies_lexical_reranker(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = Settings.from_env(ROOT)
            settings = _replace(
                settings,
                trace_dir=Path(tmp),
                data_dir=ROOT / "data/sample_docs",
                enable_tracing=False,
                reranker="lexical",
            )
            engine = build_engine(settings)
            results = engine.retrieve("voice latency streaming", top_k=2)

        self.assertTrue(results)
        self.assertEqual(engine.reranker_name, "lexical")
        diagnostics = results[0].diagnostics
        assert diagnostics is not None
        self.assertEqual(diagnostics["reranker"], "lexical")

    async def test_stream_query_emits_error_and_metrics_on_retrieval_failure(self) -> None:
        with TemporaryDirectory() as tmp:
            engine = RagEngine(
                retriever=FailingRetriever(),
                reasoner=TemplateReasoner(),
                tracer=TraceRecorder(Path(tmp), enabled=False),
                top_k=2,
                retriever_name="failing",
            )
            events = await collect_events(engine.stream_query("trigger failure"))

        event_types = [event.type for event in events]

        self.assertIn("query.received", event_types)
        self.assertIn("error", event_types)
        self.assertIn("metrics.completed", event_types)

        error = next(event for event in events if event.type == "error")
        self.assertEqual(error.payload["code"], "stream_failed")
        self.assertEqual(error.payload["stage"], "retrieval")
        self.assertEqual(error.payload["error_type"], "RuntimeError")

        metrics = events[-1]
        self.assertEqual(metrics.type, "metrics.completed")
        self.assertTrue(metrics.payload["failed"])
        self.assertEqual(metrics.payload["error_code"], "stream_failed")


def _replace(settings: Settings, **changes: object) -> Settings:
    values = settings.__dict__.copy()
    values.update(changes)
    return Settings(**values)


if __name__ == "__main__":
    unittest.main()
