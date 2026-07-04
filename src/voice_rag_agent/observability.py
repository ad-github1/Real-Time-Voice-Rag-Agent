from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .events import RagEvent


@dataclass(frozen=True)
class SpanRecord:
    trace_id: str
    name: str
    started_at_ms: int
    ended_at_ms: int
    duration_ms: int
    attributes: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "started_at_ms": self.started_at_ms,
            "ended_at_ms": self.ended_at_ms,
            "duration_ms": self.duration_ms,
            "attributes": dict(self.attributes),
        }


class TraceRecorder:
    """Tiny JSONL trace recorder that mirrors the provider events."""

    def __init__(self, trace_dir: Path, enabled: bool = True) -> None:
        self.trace_dir = trace_dir
        self.enabled = enabled
        if enabled:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def span(self, trace_id: str, name: str, **attributes: Any) -> Iterator[None]:
        started = int(time.time() * 1000)
        try:
            yield
        finally:
            ended = int(time.time() * 1000)
            self.write_span(
                SpanRecord(
                    trace_id=trace_id,
                    name=name,
                    started_at_ms=started,
                    ended_at_ms=ended,
                    duration_ms=ended - started,
                    attributes=attributes,
                )
            )

    def write_event(self, event: RagEvent) -> None:
        if not self.enabled:
            return
        self._append(event.trace_id, {"kind": "event", **event.to_dict()})

    def write_span(self, span: SpanRecord) -> None:
        if not self.enabled:
            return
        self._append(span.trace_id, {"kind": "span", **span.to_dict()})

    def _append(self, trace_id: str, record: Mapping[str, Any]) -> None:
        path = self.trace_dir / f"{trace_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")
