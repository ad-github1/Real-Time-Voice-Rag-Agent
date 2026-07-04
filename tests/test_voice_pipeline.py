from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from voice_rag_agent.config import Settings
from voice_rag_agent.rag.engine import build_engine, collect_events
from voice_rag_agent.voice.asr import IterableTranscriptSource
from voice_rag_agent.voice.pipeline import VoiceRagPipeline


ROOT = Path(__file__).resolve().parents[1]


class VoicePipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_voice_demo_emits_tts_audio_event(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = Settings.from_env(ROOT)
            settings = _replace(
                settings,
                trace_dir=Path(tmp),
                data_dir=ROOT / "data/sample_docs",
                enable_tracing=False,
            )
            engine = build_engine(settings)
            pipeline = VoiceRagPipeline(
                engine,
                IterableTranscriptSource(["What does the voice path do?"]),
            )
            events = await collect_events(pipeline.stream())
        self.assertIn("asr.transcript", [event.type for event in events])
        self.assertIn("tts.audio", [event.type for event in events])


def _replace(settings: Settings, **changes: object) -> Settings:
    values = settings.__dict__.copy()
    values.update(changes)
    return Settings(**values)


if __name__ == "__main__":
    unittest.main()
