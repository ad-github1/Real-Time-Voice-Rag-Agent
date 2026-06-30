from __future__ import annotations

from typing import AsyncIterator

from voice_rag_agent.events import RagEvent
from voice_rag_agent.rag.engine import RagEngine

from .asr import TranscriptEvent, TranscriptSource
from .tts import ConsoleTTS, SpeechSynthesizer


class VoiceRagPipeline:
    def __init__(
        self,
        engine: RagEngine,
        transcript_source: TranscriptSource,
        tts: SpeechSynthesizer | None = None,
    ) -> None:
        self.engine = engine
        self.transcript_source = transcript_source
        self.tts = tts or ConsoleTTS()

    async def stream(self) -> AsyncIterator[RagEvent]:
        async for transcript in self.transcript_source.stream():
            yield self._transcript_event(transcript)
            if not transcript.is_final:
                continue

            completed_text = ""
            turn_trace_id = "voice_pipeline"
            async for event in self.engine.stream_query(transcript.text):
                turn_trace_id = event.trace_id
                yield event
                if event.type == "answer.completed":
                    completed_text = str(event.payload["text"])

            async for audio in self.tts.synthesize(completed_text):
                yield RagEvent(
                    type="tts.audio",
                    trace_id=turn_trace_id,
                    payload={
                        "provider": self.tts.name,
                        "bytes": len(audio),
                        "preview": audio[:80].decode("utf-8", errors="ignore"),
                    },
                )

    def _transcript_event(self, transcript: TranscriptEvent) -> RagEvent:
        return RagEvent(
            type="asr.transcript",
            trace_id="voice_pipeline",
            payload={
                "text": transcript.text,
                "is_final": transcript.is_final,
                "confidence": transcript.confidence,
                "speaker": transcript.speaker,
            },
        )
