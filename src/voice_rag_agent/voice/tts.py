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
    """Real Cartesia TTS adapter that streams PCM audio chunks."""

    name = "cartesia"

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "sonic-3.5",
        sample_rate: int = 44100,
    ) -> None:
        if not api_key:
            raise RuntimeError("CARTESIA_API_KEY is required for CartesiaTTS.")
        if not voice_id:
            raise RuntimeError("CARTESIA_VOICE_ID is required for CartesiaTTS.")

        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        import queue
        import threading

        chunks: queue.Queue[bytes | None] = queue.Queue()
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                from cartesia import Cartesia

                client = Cartesia(api_key=self.api_key)

                with client.tts.websocket_connect() as connection:
                    ctx = connection.context(
                        model_id=self.model_id,
                        voice={"mode": "id", "id": self.voice_id},
                        output_format={
                            "container": "raw",
                            "encoding": "pcm_f32le",
                            "sample_rate": self.sample_rate,
                        },
                        language="en",
                    )

                    for sentence in _sentences(text):
                        ctx.push(sentence + " ")

                    ctx.no_more_inputs()

                    for response in ctx.receive():
                        if response.type == "chunk" and response.audio:
                            chunks.put(response.audio)
                        elif response.type == "done":
                            break

            except BaseException as exc:
                errors.append(exc)
            finally:
                chunks.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            chunk = await asyncio.to_thread(chunks.get)
            if chunk is None:
                break
            yield chunk

        if errors:
            raise RuntimeError(f"Cartesia TTS failed: {errors[0]}") from errors[0]


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
