from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from voice_rag_agent.api import create_app
from voice_rag_agent.config import Settings


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


def test_health_documents_and_index_lifecycle(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
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
