from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError as exc:
    raise RuntimeError("Install the api extra to run the FastAPI server.") from exc

from .config import Settings
from .events import async_events_to_ndjson
from .llm import build_reasoner
from .rag.documents import SOURCE_SUFFIXES, iter_source_files
from .rag.engine import build_engine
from .rag.index_store import build_persistent_index, index_stats


MAX_UPLOAD_BYTES = 15 * 1024 * 1024
UPLOAD_SUBDIR = "api_uploads"


class QueryRequest(BaseModel):
    query: str
    trace_id: str | None = None


def create_app(settings: Settings | None = None) -> Any:
    resolved = settings or Settings.from_env(Path.cwd())
    resolved.ensure_dirs()

    upload_dir = resolved.data_dir / UPLOAD_SUBDIR
    upload_dir.mkdir(parents=True, exist_ok=True)

    engine_holder: dict[str, Any] = {
        "engine": build_engine(resolved, reasoner=build_reasoner(resolved))
    }

    def reload_engine() -> None:
        engine_holder["engine"] = build_engine(resolved, reasoner=build_reasoner(resolved))

    app = FastAPI(title="Real-Time Voice RAG Agent", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str | int]:
        return {
            "status": "ok",
            "env": resolved.env,
            "top_k": resolved.top_k,
            "reasoner": resolved.reasoner,
            "retriever": resolved.retriever,
        }

    @app.post("/query")
    async def query(request: QueryRequest) -> dict[str, Any]:
        try:
            answer = await engine_holder["engine"].query(
                request.query,
                trace_id=request.trace_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "trace_id": answer.trace_id,
            "answer": answer.text,
            "citations": answer.citations,
        }

    @app.post("/query/stream")
    async def stream_query(request: QueryRequest) -> StreamingResponse:
        async def body() -> Any:
            async for line in async_events_to_ndjson(
                engine_holder["engine"].stream_query(
                    request.query,
                    trace_id=request.trace_id,
                )
            ):
                yield line

        return StreamingResponse(body(), media_type="application/x-ndjson")

    @app.get("/documents")
    async def list_documents() -> dict[str, Any]:
        documents = [
            _document_record(path, resolved.data_dir)
            for path in iter_source_files(resolved.data_dir)
        ]

        return {
            "data_dir": str(resolved.data_dir),
            "count": len(documents),
            "documents": documents,
        }

    @app.post("/documents/upload")
    async def upload_document(
        file: UploadFile = File(...),
        rebuild: bool = True,
    ) -> dict[str, Any]:
        original_filename = file.filename or ""

        try:
            safe_filename = _safe_upload_filename(original_filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        suffix = Path(safe_filename).suffix.lower()

        if suffix not in SOURCE_SUFFIXES:
            supported = ", ".join(sorted(SOURCE_SUFFIXES))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix}'. Supported types: {supported}",
            )

        content = await file.read()

        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File is too large. Max upload size is {MAX_UPLOAD_BYTES} bytes.",
            )

        digest = hashlib.sha256(content).hexdigest()
        destination = _upload_destination(upload_dir, safe_filename, digest)
        destination.write_bytes(content)

        payload: dict[str, Any] = {
            "filename": destination.name,
            "original_filename": original_filename,
            "path": str(destination),
            "size_bytes": len(content),
            "sha256": digest,
            "rebuild": rebuild,
        }

        if rebuild:
            result = build_persistent_index(
                data_dir=resolved.data_dir,
                index_dir=resolved.index_dir,
                chunk_size=resolved.chunk_size,
                chunk_overlap=resolved.chunk_overlap,
            )
            reload_engine()
            payload["index"] = _index_build_payload(result)

        return payload

    @app.post("/index/rebuild")
    async def rebuild_index() -> dict[str, Any]:
        result = build_persistent_index(
            data_dir=resolved.data_dir,
            index_dir=resolved.index_dir,
            chunk_size=resolved.chunk_size,
            chunk_overlap=resolved.chunk_overlap,
        )
        reload_engine()

        return _index_build_payload(result)

    @app.get("/index/stats")
    async def get_index_stats() -> dict[str, Any]:
        return index_stats(resolved.index_dir, resolved.data_dir)

    return app


def _safe_upload_filename(filename: str) -> str:
    name = Path(filename).name.strip()

    if not name:
        raise ValueError("Upload filename is required.")

    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = name.strip("._")

    if not name:
        raise ValueError("Upload filename is invalid.")

    if Path(name).suffix.lower() not in SOURCE_SUFFIXES:
        supported = ", ".join(sorted(SOURCE_SUFFIXES))
        raise ValueError(f"Unsupported file type. Supported types: {supported}")

    return name


def _upload_destination(upload_dir: Path, safe_filename: str, digest: str) -> Path:
    path = Path(safe_filename)
    stem = path.stem[:80]
    suffix = path.suffix.lower()

    return upload_dir / f"{stem}_{digest[:12]}{suffix}"


def _document_record(path: Path, data_dir: Path) -> dict[str, Any]:
    stat = path.stat()

    try:
        relative_path = path.resolve().relative_to(data_dir.resolve()).as_posix()
    except ValueError:
        relative_path = str(path)

    return {
        "path": str(path),
        "relative_path": relative_path,
        "filename": path.name,
        "suffix": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "modified_at_ms": int(stat.st_mtime * 1000),
    }


def _index_build_payload(result: Any) -> dict[str, Any]:
    return {
        "index_dir": str(result.index_dir),
        "manifest_path": str(result.manifest_path),
        "chunks_path": str(result.chunks_path),
        "source_count": result.source_count,
        "chunk_count": result.chunk_count,
        "duplicate_chunks_skipped": result.duplicate_chunks_skipped,
        "created_at_ms": result.created_at_ms,
        "created_at_iso": _iso_from_ms(result.created_at_ms),
    }


def _iso_from_ms(timestamp_ms: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp_ms / 1000))
