from __future__ import annotations

import time
from typing import AsyncIterator

from voice_rag_agent.events import RagEvent
from voice_rag_agent.rag.engine import RagEngine

from .asr import TranscriptSource
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
            transcript_event = self._event(
                "asr.transcript",
                "voice_pipeline",
                {
                    "text": transcript.text,
                    "is_final": transcript.is_final,
                    "confidence": transcript.confidence,
                    "speaker": transcript.speaker,
                },
            )
            yield transcript_event

            if not transcript.is_final:
                continue

            turn_started_ns = time.perf_counter_ns()
            turn_trace_id = "voice_pipeline"
            sentence_buffer = ""

            engine_metrics: dict[str, object] = {}
            answer_completed = False

            tts_started_ns: int | None = None
            first_tts_audio_ns: int | None = None
            tts_audio_chunks = 0
            tts_audio_bytes = 0

            async for event in self.engine.stream_query(transcript.text):
                turn_trace_id = event.trace_id

                if event.type == "metrics.completed":
                    engine_metrics = dict(event.payload)
                    continue

                yield event

                if event.type == "answer.delta":
                    delta = str(event.payload["text"])
                    sentence_buffer += delta

                    ready_sentences, sentence_buffer = _pop_ready_sentences(sentence_buffer)

                    for sentence in ready_sentences:
                        if tts_started_ns is None:
                            tts_started_ns = time.perf_counter_ns()

                        async for audio in self.tts.synthesize(sentence):
                            if first_tts_audio_ns is None:
                                first_tts_audio_ns = time.perf_counter_ns()

                            tts_audio_chunks += 1
                            tts_audio_bytes += len(audio)

                            if self.audio_sink:
                                await self.audio_sink.write(audio)

                            yield self._event(
                                "tts.audio",
                                turn_trace_id,
                                {
                                    "provider": self.tts.name,
                                    "bytes": len(audio),
                                    "text": sentence,
                                },
                            )

                if event.type == "answer.completed":
                    answer_completed = True
                    tail = sentence_buffer.strip()

                    if tail:
                        if tts_started_ns is None:
                            tts_started_ns = time.perf_counter_ns()

                        async for audio in self.tts.synthesize(tail):
                            if first_tts_audio_ns is None:
                                first_tts_audio_ns = time.perf_counter_ns()

                            tts_audio_chunks += 1
                            tts_audio_bytes += len(audio)

                            if self.audio_sink:
                                await self.audio_sink.write(audio)

                            yield self._event(
                                "tts.audio",
                                turn_trace_id,
                                {
                                    "provider": self.tts.name,
                                    "bytes": len(audio),
                                    "text": tail,
                                },
                            )

            if answer_completed:
                turn_total_latency_ms = _elapsed_ms(turn_started_ns)
                tts_time_to_first_audio_ms = (
                    _elapsed_ms(tts_started_ns, first_tts_audio_ns)
                    if tts_started_ns is not None and first_tts_audio_ns is not None
                    else None
                )

                yield self._event(
                    "metrics.completed",
                    turn_trace_id,
                    {
                        "mode": "voice",
                        "retrieval_latency_ms": engine_metrics.get("retrieval_latency_ms"),
                        "llm_time_to_first_token_ms": engine_metrics.get(
                            "llm_time_to_first_token_ms"
                        ),
                        "llm_total_latency_ms": engine_metrics.get("llm_total_latency_ms"),
                        "query_total_latency_ms": engine_metrics.get("query_total_latency_ms"),
                        "tts_time_to_first_audio_ms": tts_time_to_first_audio_ms,
                        "turn_total_latency_ms": turn_total_latency_ms,
                        "tts_audio_chunks": tts_audio_chunks,
                        "tts_audio_bytes": tts_audio_bytes,
                        "tts_provider": self.tts.name,
                    },
                )

    def _event(self, event_type: str, trace_id: str, payload: dict[str, object]) -> RagEvent:
        event = RagEvent(type=event_type, trace_id=trace_id, payload=payload)
        self.engine.tracer.write_event(event)
        return event


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


def _elapsed_ms(start_ns: int, end_ns: int | None = None) -> float:
    end = end_ns if end_ns is not None else time.perf_counter_ns()
    return round((end - start_ns) / 1_000_000, 3)
