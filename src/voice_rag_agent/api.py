from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, File, HTTPException, Request, UploadFile
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel
except ImportError as exc:
    raise RuntimeError("Install the api extra to run the FastAPI server.") from exc

from .config import Settings
from .events import RagEvent, async_events_to_ndjson, new_trace_id
from .llm import build_reasoner
from .rag.documents import SOURCE_SUFFIXES, iter_source_files
from .rag.engine import build_engine
from .rag.index_store import build_persistent_index, index_stats


MAX_UPLOAD_BYTES = 15 * 1024 * 1024
UPLOAD_SUBDIR = "api_uploads"
API_KEY_HEADER = "X-API-Key"
PUBLIC_AUTH_PATHS = {"/health", "/ready", "/docs", "/redoc", "/openapi.json"}
PUBLIC_AUTH_PREFIXES = ("/docs/",)
REQUEST_ID_HEADER = "X-Request-ID"
RATE_LIMIT_HEADER = "X-RateLimit-Limit"
RATE_LIMIT_REMAINING_HEADER = "X-RateLimit-Remaining"
RATE_LIMIT_RESET_HEADER = "X-RateLimit-Reset"


class QueryRequest(BaseModel):
    query: str
    trace_id: str | None = None


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_seconds: int


class InMemoryRateLimiter:
    def __init__(self, max_requests: int = 120, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, now: float | None = None) -> RateLimitDecision:
        current = time.monotonic() if now is None else now
        bucket = self._buckets[key]
        cutoff = current - self.window_seconds

        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= self.max_requests:
            reset_seconds = max(1, int(bucket[0] + self.window_seconds - current))
            return RateLimitDecision(
                allowed=False,
                limit=self.max_requests,
                remaining=0,
                reset_seconds=reset_seconds,
            )

        bucket.append(current)

        return RateLimitDecision(
            allowed=True,
            limit=self.max_requests,
            remaining=max(0, self.max_requests - len(bucket)),
            reset_seconds=self.window_seconds,
        )


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
    app.state.engine_holder = engine_holder
    app.state.rate_limiter = InMemoryRateLimiter(
        max_requests=_env_int("VOICE_RAG_API_RATE_LIMIT_REQUESTS", 120),
        window_seconds=_env_int("VOICE_RAG_API_RATE_LIMIT_WINDOW_SECONDS", 60),
    )
    app.state.api_key = os.getenv("VOICE_RAG_API_KEY", "").strip()

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id

        client_key = _client_key(request)
        decision = app.state.rate_limiter.check(client_key)

        if not decision.allowed:
            return _rate_limit_response(request_id, decision)

        auth_response = _authenticate_request(
            request=request,
            request_id=request_id,
            decision=decision,
            expected_api_key=app.state.api_key,
        )

        if auth_response is not None:
            return auth_response

        response = await call_next(request)
        _set_common_headers(response.headers, request_id, decision)

        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = _request_id(request)
        message = str(exc.detail)
        code = _status_code_to_error_code(exc.status_code)

        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                code=code,
                message=message,
                request_id=request_id,
                status_code=exc.status_code,
                detail=message,
            ),
            headers=dict(exc.headers or {}),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        request_id = _request_id(request)

        return JSONResponse(
            status_code=422,
            content=_error_payload(
                code="validation_error",
                message="Request validation failed.",
                request_id=request_id,
                status_code=422,
                detail="Request validation failed.",
                details=exc.errors(),
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id(request)

        return JSONResponse(
            status_code=500,
            content=_error_payload(
                code="internal_server_error",
                message="Internal server error.",
                request_id=request_id,
                status_code=500,
                detail="Internal server error.",
            ),
        )

    @app.get("/health")
    async def health() -> dict[str, str | int]:
        return {
            "status": "ok",
            "env": resolved.env,
            "top_k": resolved.top_k,
            "reasoner": resolved.reasoner,
            "retriever": resolved.retriever,
        }

    @app.get("/ready")
    async def ready() -> JSONResponse:
        documents = list(iter_source_files(resolved.data_dir))
        stats = index_stats(resolved.index_dir, resolved.data_dir)
        is_ready = (
            resolved.data_dir.exists()
            and len(documents) > 0
            and engine_holder.get("engine") is not None
        )
        status_code = 200 if is_ready else 503

        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ready" if is_ready else "not_ready",
                "ready": is_ready,
                "env": resolved.env,
                "data_dir": str(resolved.data_dir),
                "document_count": len(documents),
                "index": stats,
                "top_k": resolved.top_k,
                "reasoner": resolved.reasoner,
                "retriever": resolved.retriever,
                "reranker": resolved.reranker,
            },
        )

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
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        trace_id = request.trace_id or new_trace_id()

        async def body() -> Any:
            events = engine_holder["engine"].stream_query(
                request.query,
                trace_id=trace_id,
            )
            async for line in _stream_events_with_error_boundary(events, trace_id):
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


async def _stream_events_with_error_boundary(events: Any, trace_id: str) -> Any:
    try:
        async for line in async_events_to_ndjson(events):
            yield line
    except Exception as exc:  # noqa: BLE001
        event = RagEvent(
            type="error",
            trace_id=trace_id,
            payload={
                "code": "stream_failed",
                "message": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        yield f"{event.to_json()}\n"


def _authenticate_request(
    *,
    request: Request,
    request_id: str,
    decision: RateLimitDecision,
    expected_api_key: str,
) -> JSONResponse | None:
    if not expected_api_key:
        return None

    if _is_public_auth_path(request.url.path):
        return None

    provided_api_key = request.headers.get(API_KEY_HEADER, "")

    if not provided_api_key:
        return _auth_error_response(
            status_code=401,
            code="api_key_missing",
            message="Missing API key.",
            request_id=request_id,
            decision=decision,
        )

    if not hmac.compare_digest(provided_api_key, expected_api_key):
        return _auth_error_response(
            status_code=403,
            code="invalid_api_key",
            message="Invalid API key.",
            request_id=request_id,
            decision=decision,
        )

    return None


def _is_public_auth_path(path: str) -> bool:
    return path in PUBLIC_AUTH_PATHS or any(
        path.startswith(prefix) for prefix in PUBLIC_AUTH_PREFIXES
    )


def _auth_error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    decision: RateLimitDecision,
) -> JSONResponse:
    response = JSONResponse(
        status_code=status_code,
        content=_error_payload(
            code=code,
            message=message,
            request_id=request_id,
            status_code=status_code,
            detail=message,
        ),
    )
    _set_common_headers(response.headers, request_id, decision)

    return response


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


def _rate_limit_response(request_id: str, decision: RateLimitDecision) -> JSONResponse:
    response = JSONResponse(
        status_code=429,
        content=_error_payload(
            code="rate_limit_exceeded",
            message="Too many requests. Please retry later.",
            request_id=request_id,
            status_code=429,
            detail="Too many requests. Please retry later.",
        ),
    )
    _set_common_headers(response.headers, request_id, decision)

    return response


def _set_common_headers(
    headers: Any,
    request_id: str,
    decision: RateLimitDecision,
) -> None:
    headers[REQUEST_ID_HEADER] = request_id
    headers[RATE_LIMIT_HEADER] = str(decision.limit)
    headers[RATE_LIMIT_REMAINING_HEADER] = str(decision.remaining)
    headers[RATE_LIMIT_RESET_HEADER] = str(decision.reset_seconds)


def _error_payload(
    *,
    code: str,
    message: str,
    request_id: str,
    status_code: int,
    detail: str,
    details: Any | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "request_id": request_id,
        "status_code": status_code,
    }

    if details is not None:
        error["details"] = details

    return {
        "detail": detail,
        "error": error,
    }


def _status_code_to_error_code(status_code: int) -> str:
    if status_code == 400:
        return "bad_request"

    if status_code == 401:
        return "unauthorized"

    if status_code == 403:
        return "forbidden"

    if status_code == 404:
        return "not_found"

    if status_code == 413:
        return "payload_too_large"

    if status_code == 429:
        return "rate_limit_exceeded"

    if status_code >= 500:
        return "internal_server_error"

    return "http_error"


def _client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")

    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    if request.client:
        return request.client.host

    return "unknown"


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "")) or uuid.uuid4().hex


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)

    if raw is None or raw.strip() == "":
        return default

    try:
        return int(raw)
    except ValueError:
        return default


def _iso_from_ms(timestamp_ms: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp_ms / 1000))
