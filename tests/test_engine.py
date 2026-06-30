import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from voice_rag_agent.config import Settings
from voice_rag_agent.events import events_to_ndjson
from voice_rag_agent.rag.engine import build_engine, collect_events


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


def _replace(settings: Settings, **changes: object) -> Settings:
    values = settings.__dict__.copy()
    values.update(changes)
    return Settings(**values)


if __name__ == "__main__":
    unittest.main()
