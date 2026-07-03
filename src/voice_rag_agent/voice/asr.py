from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, Protocol


@dataclass(frozen=True)
class TranscriptEvent:
    text: str
    is_final: bool
    confidence: float = 1.0
    speaker: str = "user"


class TranscriptSource(Protocol):
    async def stream(self) -> AsyncIterator[TranscriptEvent]:
        ...


class IterableTranscriptSource:
    """Test/demo transcript source that behaves like final ASR messages."""

    def __init__(self, utterances: Iterable[str]) -> None:
        self.utterances = list(utterances)

    async def stream(self) -> AsyncIterator[TranscriptEvent]:
        for utterance in self.utterances:
            await asyncio.sleep(0)
            yield TranscriptEvent(text=utterance, is_final=True)


class AssemblyAITranscriptSource:
    """Real microphone ASR source using AssemblyAI Universal Streaming."""

    def __init__(
        self,
        api_key: str,
        sample_rate: int = 16000,
        speech_model: str = "universal-3-5-pro",
        one_turn: bool = True,
    ) -> None:
        if not api_key:
            raise RuntimeError("ASSEMBLYAI_API_KEY is required for AssemblyAITranscriptSource.")
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.speech_model = speech_model
        self.one_turn = one_turn

    async def stream(self) -> AsyncIterator[TranscriptEvent]:
        import queue
        import threading

        import sounddevice as sd

        import typing
        from typing_extensions import TypeAliasType

        if not hasattr(typing, "TypeAliasType"):
            typing.TypeAliasType = TypeAliasType
        from assemblyai.streaming.v3 import (
        BeginEvent,
        StreamingClient,
        StreamingClientOptions,
        StreamingError,
        StreamingEvents,
        StreamingParameters,
        TerminationEvent,
        TurnEvent,
        )

        events: queue.Queue[TranscriptEvent | None] = queue.Queue()
        stop = threading.Event()

        def mic_stream():
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=800,
            ) as mic:
                while not stop.is_set():
                    frames, _ = mic.read(800)
                    yield bytes(frames)

        def on_begin(client: StreamingClient, event: BeginEvent) -> None:
            del client, event
            print("AssemblyAI connected. Speak your question...")

        def on_turn(client: StreamingClient, event: TurnEvent) -> None:
            del client
            text = (getattr(event, "transcript", "") or "").strip()
            if not text:
                return

            is_final = bool(getattr(event, "end_of_turn", False))
            confidence = float(getattr(event, "end_of_turn_confidence", 1.0) or 1.0)

            events.put(
                TranscriptEvent(
                    text=text,
                    is_final=is_final,
                    confidence=confidence,
                    speaker="user",
                )
            )

            if is_final and self.one_turn:
                stop.set()

        def on_terminated(client: StreamingClient, event: TerminationEvent) -> None:
            del client, event
            events.put(None)

        def on_error(client: StreamingClient, error: StreamingError) -> None:
            del client
            print(f"AssemblyAI error: {error}")
            stop.set()
            events.put(None)

        def worker() -> None:
            client = StreamingClient(
                StreamingClientOptions(api_key=self.api_key)
            )

            client.on(StreamingEvents.Begin, on_begin)
            client.on(StreamingEvents.Turn, on_turn)
            client.on(StreamingEvents.Termination, on_terminated)
            client.on(StreamingEvents.Error, on_error)

            try:
                client.connect(
                    StreamingParameters(
                        speech_model=self.speech_model,
                        sample_rate=self.sample_rate,
                    )
                )
                client.stream(mic_stream())
            finally:
                stop.set()
                client.disconnect(terminate=True)
                events.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        try:
            while True:
                item = await asyncio.to_thread(events.get)
                if item is None:
                    break
                yield item
        finally:
            stop.set()
    
