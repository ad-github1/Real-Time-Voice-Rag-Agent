from __future__ import annotations

import json
from pathlib import Path

from voice_rag_agent.trace_export import export_traces_to_otel_json


def test_export_traces_to_otel_json(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    output_path = tmp_path / "outputs" / "otel_traces.json"
    trace_dir.mkdir()

    trace_file = trace_dir / "trace_test.jsonl"
    trace_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "span",
                        "trace_id": "trace_test",
                        "name": "retrieval",
                        "started_at_ms": 1000,
                        "ended_at_ms": 1025,
                        "duration_ms": 25,
                        "attributes": {"query": "hello", "top_k": 4},
                    }
                ),
                json.dumps(
                    {
                        "kind": "event",
                        "trace_id": "trace_test",
                        "type": "metrics.completed",
                        "created_at_ms": 1030,
                        "payload": {
                            "mode": "text",
                            "retrieval_latency_ms": 25.0,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = export_traces_to_otel_json(
        trace_dir=trace_dir,
        output_path=output_path,
        service_name="test-service",
    )

    assert result.records_exported == 2
    assert result.spans_exported == 2
    assert result.trace_count == 1
    assert result.kind_counts == {"event": 1, "span": 1}

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

    assert payload["summary"]["records_exported"] == 2
    assert spans[0]["name"] == "retrieval"
    assert spans[1]["name"] == "event.metrics.completed"
    assert spans[0]["traceId"]
    assert spans[0]["spanId"]
