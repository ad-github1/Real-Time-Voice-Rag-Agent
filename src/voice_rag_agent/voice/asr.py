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
    """Provider placeholder for AssemblyAI Universal Streaming."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise RuntimeError("ASSEMBLYAI_API_KEY is required for AssemblyAITranscriptSource.")
        self.api_key = api_key

    async def stream(self) -> AsyncIterator[TranscriptEvent]:
        raise NotImplementedError(
            "Attach AssemblyAI's streaming client here and yield TranscriptEvent objects. "
            "The rest of the agent pipeline is provider-agnostic."
        )
