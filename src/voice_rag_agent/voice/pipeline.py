from __future__ import annotations

from typing import AsyncIterator

from voice_rag_agent.events import RagEvent
from voice_rag_agent.rag.engine import RagEngine

from .asr import TranscriptEvent, TranscriptSource
from .audio_sink import AudioSink
from .tts import ConsoleTTS, SpeechSynthesizer


class VoiceRagPipeline:
    def __init__(
        self,
        engine: RagEngine,
        transcript_source: TranscriptSource,
        tts: SpeechSynthesizer | None = None,
        audio_sink: AudioSink | None = None,
    ) -> None:
        self.engine = engine
        self.transcript_source = transcript_source
        self.tts = tts or ConsoleTTS()
        self.audio_sink = audio_sink

    async def stream(self) -> AsyncIterator[RagEvent]:
        async for transcript in self.transcript_source.stream():
            yield self._transcript_event(transcript)

            if not transcript.is_final:
                continue

            turn_trace_id = "voice_pipeline"
            sentence_buffer = ""

            async for event in self.engine.stream_query(transcript.text):
                turn_trace_id = event.trace_id
                yield event

                if event.type == "answer.delta":
                    delta = str(event.payload["text"])
                    sentence_buffer += delta

                    ready_sentences, sentence_buffer = _pop_ready_sentences(sentence_buffer)

                    for sentence in ready_sentences:
                        async for audio in self.tts.synthesize(sentence):
                            if self.audio_sink:
                                await self.audio_sink.write(audio)

                            yield RagEvent(
                                type="tts.audio",
                                trace_id=turn_trace_id,
                                payload={
                                    "provider": self.tts.name,
                                    "bytes": len(audio),
                                    "text": sentence,
                                },
                            )

                if event.type == "answer.completed":
                    tail = sentence_buffer.strip()

                    if tail:
                        async for audio in self.tts.synthesize(tail):
                            if self.audio_sink:
                                await self.audio_sink.write(audio)

                            yield RagEvent(
                                type="tts.audio",
                                trace_id=turn_trace_id,
                                payload={
                                    "provider": self.tts.name,
                                    "bytes": len(audio),
                                    "text": tail,
                                },
                            )

    @staticmethod
    def _transcript_event(transcript: TranscriptEvent) -> RagEvent:
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


def _pop_ready_sentences(buffer: str) -> tuple[list[str], str]:
    ready: list[str] = []
    start = 0

    for index, char in enumerate(buffer):
        if char in ".!?":
            sentence = buffer[start : index + 1].strip()
            if sentence:
                ready.append(sentence)
            start = index + 1

    remaining = buffer[start:]
    return ready, remaining
