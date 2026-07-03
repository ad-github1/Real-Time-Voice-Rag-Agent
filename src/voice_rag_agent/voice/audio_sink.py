from __future__ import annotations

import asyncio
import subprocess
from typing import Protocol


class AudioSink(Protocol):
    async def write(self, chunk: bytes) -> None: ...

    async def close(self) -> None: ...


class FfplayPCMFloat32Sink:
    """Plays raw Cartesia pcm_f32le audio through ffplay."""

    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate
        self.process: subprocess.Popen[bytes] | None = None

    def _ensure_process(self) -> subprocess.Popen[bytes]:
        if self.process is None:
            self.process = subprocess.Popen(
                [
                    "ffplay",
                    "-f",
                    "f32le",
                    "-ar",
                    str(self.sample_rate),
                    "-probesize",
                    "32",
                    "-analyzeduration",
                    "0",
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    "-",
                ],
                stdin=subprocess.PIPE,
                bufsize=0,
            )
        return self.process

    async def write(self, chunk: bytes) -> None:
        process = self._ensure_process()
        if process.stdin is None:
            return
        await asyncio.to_thread(process.stdin.write, chunk)
        await asyncio.to_thread(process.stdin.flush)

    async def close(self) -> None:
        if self.process and self.process.stdin:
            await asyncio.to_thread(self.process.stdin.close)
            await asyncio.to_thread(self.process.wait)
