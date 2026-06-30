from __future__ import annotations

import asyncio
from typing import AsyncIterator, Protocol


class SpeechSynthesizer(Protocol):
    name: str

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        ...


class ConsoleTTS:
    """Offline TTS stand-in that emits text bytes for stream plumbing tests."""

    name = "console"

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        for sentence in _sentences(text):
            await asyncio.sleep(0)
            yield sentence.encode("utf-8")


class CartesiaTTS:
    """Provider adapter boundary for Cartesia TTS."""

    name = "cartesia"

    def __init__(self, api_key: str, voice_id: str) -> None:
        if not api_key:
            raise RuntimeError("CARTESIA_API_KEY is required for CartesiaTTS.")
        if not voice_id:
            raise RuntimeError("CARTESIA_VOICE_ID is required for CartesiaTTS.")
        self.api_key = api_key
        self.voice_id = voice_id

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        del text
        raise NotImplementedError(
            "Install the Cartesia SDK and stream PCM/Opus chunks from this adapter into LiveKit."
        )


def _sentences(text: str) -> list[str]:
    pieces: list[str] = []
    current = ""
    for char in text:
        current += char
        if char in ".!?":
            pieces.append(current.strip())
            current = ""
    if current.strip():
        pieces.append(current.strip())
    return pieces or [text]
