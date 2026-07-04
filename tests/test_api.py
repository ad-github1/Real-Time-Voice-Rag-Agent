from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from voice_rag_agent.api import create_app
from voice_rag_agent.config import Settings
from voice_rag_agent.events import RagEvent


def _settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    trace_dir = tmp_path / "traces"
    index_dir = tmp_path / "index"

    data_dir.mkdir(parents=True)
    (data_dir / "base.md").write_text(
        "# Base Knowledge\n\n"
        "AssemblyAI handles transcription and Cartesia handles text to speech.\n",
        encoding="utf-8",
    )

    return Settings(
        root_dir=tmp_path,
        env="test",
        data_dir=data_dir,
        trace_dir=trace_dir,
        index_dir=index_dir,
        chunk_size=900,
        chunk_overlap=120,
        top_k=4,
        max_cache_entries=128,
        enable_tracing=True,
        reasoner="template",
        retriever="bm25",
        ollama_base_url="http://localhost:11434",
        ollama_model="gemma3:4b",
        openai_api_key="",
        openai_model="",
        assemblyai_api_key="",
        cartesia_api_key="",
        cartesia_voice_id="",
        livekit_url="",
        livekit_api_key="",
        livekit_api_secret="",
    )


def test_ready_endpoint_reports_query_readiness(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["ready"] is True
    assert payload["env"] == "test"
    assert payload["document_count"] == 1
    assert payload["top_k"] == 4
    assert payload["reasoner"] == "template"
    assert payload["retriever"] == "bm25"
    assert payload["reranker"] == "none"
    assert payload["index"]["current_source_count"] == 1


def test_health_documents_and_index_lifecycle(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200

    ready = client.get("/ready")
    assert ready.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["env"] == "test"

    documents = client.get("/documents")
    assert documents.status_code == 200
    assert documents.json()["count"] == 1

    initial_stats = client.get("/index/stats")
    assert initial_stats.status_code == 200
    assert initial_stats.json()["exists"] is False
    assert initial_stats.json()["current_source_count"] == 1

    rebuild = client.post("/index/rebuild")
    assert rebuild.status_code == 200
    assert rebuild.json()["source_count"] == 1
    assert rebuild.json()["chunk_count"] >= 1

    final_stats = client.get("/index/stats")
    assert final_stats.status_code == 200
    assert final_stats.json()["exists"] is True
    assert final_stats.json()["is_valid_for_current_sources"] is True


def test_upload_rebuilds_index_and_query_finds_uploaded_document(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    upload_content = (
        "# Upload Integration Note\n\n"
        "The API upload route saves documents safely, rebuilds the persistent index, "
        "and reloads the RAG engine.\n"
    ).encode("utf-8")

    upload = client.post(
        "/documents/upload?rebuild=true",
        files={"file": ("upload_note.md", upload_content, "text/markdown")},
    )

    assert upload.status_code == 200
    payload = upload.json()
    assert payload["original_filename"] == "upload_note.md"
    assert payload["size_bytes"] == len(upload_content)
    assert payload["index"]["source_count"] == 2
    assert payload["index"]["chunk_count"] >= 2

    documents = client.get("/documents")
    assert documents.status_code == 200
    assert documents.json()["count"] == 2
    assert any(
        document["filename"].startswith("upload_note_")
        for document in documents.json()["documents"]
    )

    answer = client.post(
        "/query",
        json={"query": "What does the upload integration note say?"},
    )

    assert answer.status_code == 200
    answer_payload = answer.json()
    assert "saves documents safely" in answer_payload["answer"]
    assert answer_payload["citations"]
    assert answer_payload["citations"][0]["title"] == "Upload Integration Note"


def test_upload_rejects_unsupported_file_type(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/documents/upload?rebuild=true",
        files={"file": ("bad.exe", b"not allowed", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_rejects_empty_file(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/documents/upload?rebuild=true",
        files={"file": ("empty.md", b"", "text/markdown")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."


def test_api_adds_request_id_and_structured_error_response(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/documents/upload?rebuild=true",
        files={"file": ("bad.exe", b"not allowed", "application/octet-stream")},
        headers={"X-Request-ID": "test-request-123"},
    )

    assert response.status_code == 400
    assert response.headers["X-Request-ID"] == "test-request-123"

    payload = response.json()
    assert "Unsupported file type" in payload["detail"]
    assert payload["error"]["code"] == "bad_request"
    assert payload["error"]["request_id"] == "test-request-123"
    assert payload["error"]["status_code"] == 400


def test_api_rate_limit_returns_structured_429(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    app.state.rate_limiter.max_requests = 1
    client = TestClient(app)

    first = client.get("/health", headers={"X-Forwarded-For": "203.0.113.10"})
    assert first.status_code == 200
    assert first.headers["X-RateLimit-Limit"] == "1"
    assert first.headers["X-RateLimit-Remaining"] == "0"

    second = client.get(
        "/health",
        headers={
            "X-Forwarded-For": "203.0.113.10",
            "X-Request-ID": "rate-limit-test",
        },
    )

    assert second.status_code == 429
    assert second.headers["X-Request-ID"] == "rate-limit-test"

    payload = second.json()
    assert payload["detail"] == "Too many requests. Please retry later."
    assert payload["error"]["code"] == "rate_limit_exceeded"
    assert payload["error"]["request_id"] == "rate-limit-test"
    assert payload["error"]["status_code"] == 429


def test_optional_api_key_auth_protects_non_health_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOICE_RAG_API_KEY", "secret-test-key")

    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200

    ready = client.get("/ready")
    assert ready.status_code == 200

    missing = client.get(
        "/documents",
        headers={"X-Request-ID": "missing-key-test"},
    )
    assert missing.status_code == 401
    assert missing.headers["X-Request-ID"] == "missing-key-test"
    assert missing.json()["error"]["code"] == "api_key_missing"
    assert missing.json()["error"]["request_id"] == "missing-key-test"

    invalid = client.get(
        "/documents",
        headers={
            "X-Request-ID": "invalid-key-test",
            "X-API-Key": "wrong-key",
        },
    )
    assert invalid.status_code == 403
    assert invalid.headers["X-Request-ID"] == "invalid-key-test"
    assert invalid.json()["error"]["code"] == "invalid_api_key"
    assert invalid.json()["error"]["request_id"] == "invalid-key-test"

    valid = client.get(
        "/documents",
        headers={"X-API-Key": "secret-test-key"},
    )
    assert valid.status_code == 200
    assert valid.json()["count"] == 1


def test_stream_query_returns_error_event_when_iterator_crashes(tmp_path: Path) -> None:
    class CrashingStreamEngine:
        async def stream_query(
            self,
            query: str,
            trace_id: str | None = None,
        ) -> AsyncIterator[RagEvent]:
            yield RagEvent(
                type="query.received",
                trace_id=trace_id or "missing-trace",
                payload={"text": query},
            )
            raise RuntimeError("crashed during streaming")

    app = create_app(_settings(tmp_path))
    app.state.engine_holder["engine"] = CrashingStreamEngine()
    client = TestClient(app)

    response = client.post(
        "/query/stream",
        json={"query": "please crash", "trace_id": "api-stream-test"},
    )

    assert response.status_code == 200

    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]

    assert events[0]["type"] == "query.received"
    assert events[-1]["type"] == "error"
    assert events[-1]["trace_id"] == "api-stream-test"
    assert events[-1]["payload"]["code"] == "stream_failed"
    assert events[-1]["payload"]["error_type"] == "RuntimeError"
    assert "crashed during streaming" in events[-1]["payload"]["message"]
