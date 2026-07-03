from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceExportResult:
    trace_dir: str
    output_path: str
    records_exported: int
    spans_exported: int
    trace_count: int
    kind_counts: dict[str, int]


def export_traces_to_otel_json(
    *,
    trace_dir: Path,
    output_path: Path,
    service_name: str = "voice-rag-agent",
    trace_id: str | None = None,
    limit: int | None = None,
) -> TraceExportResult:
    records = load_trace_records(trace_dir, trace_id=trace_id, limit=limit)
    spans = [_record_to_otel_span(record, index) for index, record in enumerate(records)]

    trace_ids = {str(record.get("trace_id", "unknown")) for record in records}
    kind_counts: dict[str, int] = {}

    for record in records:
        kind = str(record.get("kind", "unknown"))
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    result = TraceExportResult(
        trace_dir=str(trace_dir),
        output_path=str(output_path),
        records_exported=len(records),
        spans_exported=len(spans),
        trace_count=len(trace_ids),
        kind_counts=kind_counts,
    )

    payload = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        _attribute("service.name", service_name),
                        _attribute("telemetry.sdk.language", "python"),
                        _attribute("voice_rag.exporter", "jsonl-to-otel-json"),
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {
                            "name": "voice_rag_agent.trace_export",
                            "version": "0.1.0",
                        },
                        "spans": spans,
                    }
                ],
            }
        ],
        "summary": result_to_dict(result),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return result


def load_trace_records(
    trace_dir: Path,
    trace_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if not trace_dir.exists():
        return []

    records: list[dict[str, Any]] = []

    for trace_file in sorted(trace_dir.glob("*.jsonl"), key=lambda path: path.stat().st_mtime):
        for line in trace_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            record_trace_id = str(record.get("trace_id", trace_file.stem))

            if trace_id and record_trace_id != trace_id:
                continue

            record["_trace_file"] = str(trace_file)
            records.append(record)

    if limit is not None and limit > 0:
        records = records[-limit:]

    return records


def format_trace_export_summary(result: TraceExportResult) -> str:
    kind_summary = ", ".join(
        f"{kind}={count}" for kind, count in sorted(result.kind_counts.items())
    )

    lines = [
        "OpenTelemetry Trace Export",
        "==========================",
        f"Trace dir: {result.trace_dir}",
        f"Output: {result.output_path}",
        f"Records exported: {result.records_exported}",
        f"Spans exported: {result.spans_exported}",
        f"Traces exported: {result.trace_count}",
    ]

    if kind_summary:
        lines.append(f"Kinds: {kind_summary}")

    return "\n".join(lines)


def result_to_dict(result: TraceExportResult) -> dict[str, Any]:
    return {
        "trace_dir": result.trace_dir,
        "output_path": result.output_path,
        "records_exported": result.records_exported,
        "spans_exported": result.spans_exported,
        "trace_count": result.trace_count,
        "kind_counts": result.kind_counts,
    }


def _record_to_otel_span(record: dict[str, Any], index: int) -> dict[str, Any]:
    kind = str(record.get("kind", "unknown"))
    raw_trace_id = str(record.get("trace_id", "unknown"))

    if kind == "span":
        name = str(record.get("name", "span.unknown"))
        start_ms = _as_int(record.get("started_at_ms"))
        end_ms = _as_int(record.get("ended_at_ms"))
        attributes = {
            "voice_rag.record.kind": kind,
            "voice_rag.trace_id": raw_trace_id,
            "voice_rag.trace_file": str(record.get("_trace_file", "")),
            "voice_rag.duration_ms": record.get("duration_ms", 0),
            **_prefix_mapping("attr.", record.get("attributes")),
        }
    elif kind == "event":
        event_type = str(record.get("type", "unknown"))
        name = f"event.{event_type}"
        start_ms = _as_int(record.get("created_at_ms"))
        end_ms = start_ms
        attributes = {
            "voice_rag.record.kind": kind,
            "voice_rag.trace_id": raw_trace_id,
            "voice_rag.trace_file": str(record.get("_trace_file", "")),
            "event.type": event_type,
            **_prefix_mapping("payload.", record.get("payload")),
        }
    else:
        name = f"record.{kind}"
        start_ms = 0
        end_ms = 0
        attributes = {
            "voice_rag.record.kind": kind,
            "voice_rag.trace_id": raw_trace_id,
            "voice_rag.trace_file": str(record.get("_trace_file", "")),
        }

    return {
        "traceId": _otel_trace_id(raw_trace_id),
        "spanId": _otel_span_id(raw_trace_id, index, name, start_ms),
        "name": name,
        "kind": "SPAN_KIND_INTERNAL",
        "startTimeUnixNano": str(start_ms * 1_000_000),
        "endTimeUnixNano": str(end_ms * 1_000_000),
        "attributes": [_attribute(key, value) for key, value in sorted(attributes.items())],
    }


def _prefix_mapping(prefix: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    return {f"{prefix}{key}": item for key, item in value.items()}


def _attribute(key: str, value: Any) -> dict[str, Any]:
    return {"key": key, "value": _otel_value(value)}


def _otel_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}

    if isinstance(value, int) and not isinstance(value, bool):
        return {"intValue": str(value)}

    if isinstance(value, float):
        return {"doubleValue": value}

    if value is None:
        return {"stringValue": ""}

    if isinstance(value, dict | list | tuple):
        return {"stringValue": json.dumps(value, sort_keys=True)}

    return {"stringValue": str(value)}


def _otel_trace_id(raw_trace_id: str) -> str:
    return hashlib.sha256(raw_trace_id.encode("utf-8")).hexdigest()[:32]


def _otel_span_id(raw_trace_id: str, index: int, name: str, start_ms: int) -> str:
    raw = f"{raw_trace_id}:{index}:{name}:{start_ms}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0

    return 0
