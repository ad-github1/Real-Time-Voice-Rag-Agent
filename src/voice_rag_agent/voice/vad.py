from __future__ import annotations

import audioop
from dataclasses import dataclass


@dataclass(frozen=True)
class VadDecision:
    is_speech: bool
    rms: int
    threshold: int


class EnergyVadGate:
    """Dependency-free VAD fallback useful for local smoke tests."""

    def __init__(self, threshold: int = 450, sample_width: int = 2) -> None:
        self.threshold = threshold
        self.sample_width = sample_width

    def inspect(self, pcm: bytes) -> VadDecision:
        if not pcm:
            return VadDecision(is_speech=False, rms=0, threshold=self.threshold)
        rms = audioop.rms(pcm, self.sample_width)
        return VadDecision(is_speech=rms >= self.threshold, rms=rms, threshold=self.threshold)


class SileroVadGate:
    """Optional production VAD hook; import is deferred to keep the base package light."""

    def __init__(self) -> None:
        try:
            import silero_vad  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Install the voice extra to use Silero VAD.") from exc
        self._silero_vad = silero_vad

    def inspect(self, pcm: bytes) -> VadDecision:
        del pcm
        raise NotImplementedError("Wire Silero frame scoring here for microphone/live room audio.")
