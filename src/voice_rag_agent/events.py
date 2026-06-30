from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable, Mapping


def new_trace_id(prefix: str = "trace") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass(frozen=True)
class RagEvent:
    """A small event envelope for NDJSON, SSE, WebSocket, and LiveKit metadata."""

    type: str
    trace_id: str
    payload: Mapping[str, Any]
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "trace_id": self.trace_id,
            "created_at_ms": self.created_at_ms,
            "payload": dict(self.payload),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, separators=(",", ":"))


def events_to_ndjson(events: Iterable[RagEvent]) -> str:
    return "".join(f"{event.to_json()}\n" for event in events)


async def async_events_to_ndjson(events: AsyncIterator[RagEvent]) -> AsyncIterator[str]:
    async for event in events:
        yield f"{event.to_json()}\n"
